"""Graph neural network actor-critic.

Two message-passing backends are available:
  GraphLayer  — original weighted-aggregation layer (fast, default)
  GATLayer    — multi-head graph attention layer (richer, --use_gat)

Actor and critic both read [h_current_node || mean(h_all_nodes)].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Message-passing layers
# ---------------------------------------------------------------------------

class GraphLayer(nn.Module):
    """h' = h + GELU(LN(W_s·h + W_n·(A·h)))
    A is the similarity-weighted, row-normalised adjacency matrix."""

    def __init__(self, dim: int):
        super().__init__()
        self.ws = nn.Linear(dim, dim)
        self.wn = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return h + F.gelu(self.norm(self.ws(h) + self.wn(torch.bmm(adj, h))))


class GATLayer(nn.Module):
    """Multi-head graph attention layer (Veličković et al., 2018).

    Attention scores are computed between every pair of neighbours and
    softmax-normalised over the neighbourhood (adj > 0), making the
    aggregation data-driven rather than fixed by appearance similarity.

    h'_i = h_i + GELU(LN( W_o · concat_heads( sum_j α_ij · W_v · h_j ) ))

    where  α_ij = softmax_j [ (W_q·h_i)·(W_k·h_j) / sqrt(d_head) ]
                  masked to actual neighbours via adj.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.d_head = dim // heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = math.sqrt(self.d_head)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        H, dh = self.heads, self.d_head

        # Project → [B, H, N, dh]
        q = self.wq(h).view(B, N, H, dh).permute(0, 2, 1, 3)
        k = self.wk(h).view(B, N, H, dh).permute(0, 2, 1, 3)
        v = self.wv(h).view(B, N, H, dh).permute(0, 2, 1, 3)

        # Scaled dot-product attention  [B, H, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Mask: nodes with no edge get -inf (keeping self-loop)
        no_edge = (adj == 0)                           # [B, N, N]
        eye = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        mask = no_edge & ~eye                          # zero-out self-loop mask
        attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
        attn = torch.softmax(attn, dim=-1)             # [B, H, N, N]

        # Aggregate values
        out = torch.matmul(attn, v)                    # [B, H, N, dh]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return h + F.gelu(self.norm(self.wo(out)))


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

class GraphActorCritic(nn.Module):
    """GNN-based actor-critic for sequential superpixel labelling.

    Args:
        in_dim   : node feature dimension (NODE_FEAT_DIM + 3 state flags,
                   or NODE_FEAT_DIM + CNN_FEAT_DIM + 3 when --cnn_features).
        hidden   : width of all GNN layers and MLP heads.
        layers   : number of message-passing layers.
        use_gat  : if True use GATLayer instead of GraphLayer.
        gat_heads: number of attention heads (only used when use_gat=True).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        layers: int = 3,
        use_gat: bool = False,
        gat_heads: int = 4,
    ):
        super().__init__()
        # +2 for explicit neighbour-label context (mean label, fraction labelled)
        self.inp = nn.Linear(in_dim + 2, hidden)

        layer_cls = GATLayer if use_gat else GraphLayer
        if use_gat:
            self.gnn_layers = nn.ModuleList(
                GATLayer(hidden, heads=gat_heads) for _ in range(layers)
            )
        else:
            self.gnn_layers = nn.ModuleList(
                GraphLayer(hidden) for _ in range(layers)
            )

        self.pi = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        self.v = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

        # Orthogonal init for stable early learning
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)
        nn.init.zeros_(self.v[-1].bias)

    def forward(
        self,
        x: torch.Tensor,          # [B, N, in_dim]
        adj: torch.Tensor,         # [B, N, N]
        valid: torch.Tensor,       # [B, N]  1 = real node
        cur_idx: torch.Tensor,     # [B]     index of current node
    ):
        # Explicit neighbour-label context (not learned from adj alone)
        labeled, label = x[..., -3:-2], x[..., -2:-1]
        nb_lab = torch.bmm(adj, labeled * label)   # mean label of labelled neighbours
        nb_cnt = torch.bmm(adj, labeled)            # fraction of neighbours labelled

        h = F.gelu(self.inp(torch.cat([x, nb_lab, nb_cnt], -1)))
        for layer in self.gnn_layers:
            h = layer(h, adj)

        h = h * valid[..., None]                   # zero-out padding nodes
        h_cur = h[torch.arange(h.shape[0], device=h.device), cur_idx]
        g = h.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)  # graph-mean embedding
        z = torch.cat([h_cur, g], -1)              # [B, 2*hidden]

        return self.pi(z), self.v(z).squeeze(-1)
