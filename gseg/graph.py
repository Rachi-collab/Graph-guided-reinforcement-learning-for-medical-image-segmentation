"""Image -> superpixel region adjacency graph (RAG).

Each superpixel becomes a graph node. Two superpixels are connected when they
share a boundary. Edge weights encode appearance similarity, so message passing
(and the traversal order used by the agent) is guided by the image structure.

Optional CNN features (--cnn_features):
  A pretrained ResNet-18 (torchvision) is used as a frozen feature extractor.
  Feature maps from layer2 (128-ch, stride-8) are average-pooled per superpixel,
  giving each node a 128-dim descriptor that captures texture and local context
  far beyond hand-crafted colour statistics.  NODE_FEAT_DIM_CNN = 12 + 128.
"""
import heapq

import numpy as np
from skimage.segmentation import slic

NODE_FEAT_DIM = 12       # hand-crafted: 3 mean + 3 std + 3 contrast + cy + cx + area
CNN_FEAT_DIM  = 128      # ResNet-18 layer2 features, projected to this size
NODE_FEAT_DIM_CNN = NODE_FEAT_DIM + CNN_FEAT_DIM  # 140 when --cnn_features

# Lazy-loaded CNN encoder (singleton so we don't reload on every graph build)
_cnn_encoder = None


def _get_cnn_encoder(device="cpu"):
    """Return a frozen ResNet-18 feature extractor (up to layer2, 128-ch)."""
    global _cnn_encoder
    if _cnn_encoder is not None:
        return _cnn_encoder
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # Keep only stem + layer1 + layer2  (output: 128-ch, stride 8)
        encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
        )
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder = encoder.to(device)
        _cnn_encoder = encoder
        return encoder
    except ImportError:
        raise ImportError("torchvision is required for --cnn_features. "
                          "Install it with: pip install torchvision")


def extract_cnn_features(img: np.ndarray, seg: np.ndarray, N: int) -> np.ndarray:
    """Pool pretrained ResNet-18 (layer2) features per superpixel.

    Args:
        img : float32 [H, W, 3] in [0, 1]
        seg : int32   [H, W] superpixel label map (0 … N-1)
        N   : number of superpixels

    Returns:
        feats : float32 [N, CNN_FEAT_DIM]
    """
    import torch

    device = "cpu"
    encoder = _get_cnn_encoder(device)

    # Normalise with ImageNet stats and add batch dim → [1, 3, H, W]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((img - mean) / std).transpose(2, 0, 1)          # [3, H, W]
    x = torch.from_numpy(x).unsqueeze(0).to(device)       # [1, 3, H, W]

    with torch.no_grad():
        fmap = encoder(x)                                  # [1, 128, H/8, W/8]

    fmap = fmap.squeeze(0).cpu().numpy()                   # [128, Hf, Wf]
    C, Hf, Wf = fmap.shape
    H, W = seg.shape

    # Resize seg to feature-map resolution (nearest)
    import PIL.Image as PILImage
    seg_small = np.array(
        PILImage.fromarray(seg.astype(np.int32)).resize(
            (Wf, Hf), PILImage.NEAREST
        )
    )

    flat_seg = seg_small.ravel()
    feats = np.zeros((N, C), dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)
    for c in range(C):
        ch = fmap[c].ravel()
        np.add.at(feats[:, c], flat_seg, ch)
    np.add.at(counts, flat_seg, 1)
    counts = np.maximum(counts, 1)
    feats /= counts[:, None]

    # L2-normalise each node's feature vector for stability
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
    return (feats / norms).astype(np.float32)


def build_graph(img: np.ndarray, mask: np.ndarray, n_segments: int = 64,
                compactness: float = 0.3, cnn_features: bool = False) -> dict:
    """img: float32 [H, W, 3] in [0, 1]; mask: {0, 1} [H, W].

    Args:
        cnn_features: if True, append pretrained ResNet-18 (layer2) features
                      to each node's hand-crafted feature vector, raising
                      x0 width from NODE_FEAT_DIM (12) to NODE_FEAT_DIM_CNN (140).
    """
    H, W, _ = img.shape
    seg = slic(img, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1, convert2lab=False)
    _, seg = np.unique(seg, return_inverse=True)  # make labels contiguous
    seg = seg.reshape(H, W)
    N = int(seg.max()) + 1
    flat = seg.ravel()

    area = np.bincount(flat, minlength=N).astype(np.float64)
    ys, xs = np.mgrid[0:H, 0:W]
    cy = np.bincount(flat, weights=ys.ravel(), minlength=N) / area / (H - 1)
    cx = np.bincount(flat, weights=xs.ravel(), minlength=N) / area / (W - 1)

    mean, std = [], []
    for c in range(3):
        ch = img[..., c].ravel().astype(np.float64)
        m = np.bincount(flat, weights=ch, minlength=N) / area
        s2 = np.bincount(flat, weights=ch * ch, minlength=N) / area - m * m
        mean.append(m)
        std.append(np.sqrt(np.clip(s2, 0, None)))
    mean = np.stack(mean, 1)
    std = np.stack(std, 1)
    rel = mean - img.reshape(-1, 3).mean(0, keepdims=True)

    x0 = np.concatenate(
        [mean, std, rel, cy[:, None], cx[:, None], (area / (H * W) * N)[:, None]], axis=1
    ).astype(np.float32)

    # Optional: append pretrained CNN features (ResNet-18 layer2, pooled per superpixel)
    if cnn_features:
        cnn_feats = extract_cnn_features(img, seg, N)   # [N, CNN_FEAT_DIM]
        x0 = np.concatenate([x0, cnn_feats], axis=1)   # [N, NODE_FEAT_DIM_CNN]

    # adjacency from neighbouring pixels with different labels
    A = np.zeros((N, N), dtype=bool)
    for a, b in ((seg[:, :-1], seg[:, 1:]), (seg[:-1, :], seg[1:, :])):
        diff = a != b
        A[a[diff], b[diff]] = True
    A |= A.T
    np.fill_diagonal(A, False)

    # appearance-similarity edge weights
    d2 = ((mean[:, None, :] - mean[None, :, :]) ** 2).sum(-1)
    sigma2 = np.median(d2[A]) + 1e-6 if A.any() else 1.0
    Wt = np.exp(-d2 / (2 * sigma2)) * A
    adj = (Wt / (Wt.sum(1, keepdims=True) + 1e-6)).astype(np.float32)

    order = _guided_order(Wt, cy, cx)

    frac = (np.bincount(flat, weights=mask.ravel().astype(np.float64), minlength=N) / area).astype(np.float32)
    return dict(
        x0=x0, adj=adj, order=order.astype(np.int64), frac=frac, area=area.astype(np.float32),
        seg=seg.astype(np.int32), N=N, img=img.astype(np.float32), mask=mask.astype(np.uint8),
    )


def _guided_order(Wt: np.ndarray, cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
    """Prim-style traversal: start near the image centre, always expand to the most
    similar unvisited neighbour of the already-visited region. This lets the agent see
    labels of adjacent regions before deciding on the next one."""
    N = Wt.shape[0]
    dist_c = (cy - 0.5) ** 2 + (cx - 0.5) ** 2
    visited = np.zeros(N, dtype=bool)
    order = []
    while len(order) < N:
        remaining = np.where(~visited)[0]
        start = remaining[np.argmin(dist_c[remaining])]
        heap = [(0.0, int(start))]
        while heap:
            _, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            order.append(u)
            for v in np.nonzero(Wt[u])[0]:
                if not visited[v]:
                    heapq.heappush(heap, (-float(Wt[u, v]), int(v)))
    return np.array(order)
