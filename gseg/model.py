"""Graph neural network actor-critic."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    """h' = h + act(LN(W_s h + W_n (A h))) with A the similarity-weighted, row-normalised adjacency."""

    def __init__(self, dim):
        super().__init__()
        self.ws, self.wn, self.norm = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.LayerNorm(dim)

    def forward(self, h, adj):
        return h + F.gelu(self.norm(self.ws(h) + self.wn(torch.bmm(adj, h))))


class GraphActorCritic(nn.Module):
    """Input x: [B, N, F] whose last 3 columns are (is_labelled, label, is_current).
    Besides the raw state, each node gets the (weighted) mean label of its labelled neighbours
    and the fraction of neighbours already labelled -> explicit graph-guided label context."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(in_dim + 2, hidden)
        self.layers = nn.ModuleList(GraphLayer(hidden) for _ in range(layers))
        self.pi = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.v = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)
        nn.init.zeros_(self.v[-1].bias)

    def forward(self, x, adj, valid, cur_idx):
        labeled, label = x[..., -3:-2], x[..., -2:-1]
        nb_lab = torch.bmm(adj, labeled * label)
        nb_cnt = torch.bmm(adj, labeled)
        h = F.gelu(self.inp(torch.cat([x, nb_lab, nb_cnt], -1)))
        for layer in self.layers:
            h = layer(h, adj)
        h = h * valid[..., None]
        h_cur = h[torch.arange(h.shape[0], device=h.device), cur_idx]
        g = h.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        z = torch.cat([h_cur, g], -1)
        return self.pi(z), self.v(z).squeeze(-1)
