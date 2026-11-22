"""Image -> superpixel region adjacency graph (RAG).

Each superpixel becomes a graph node. Two superpixels are connected when they
share a boundary. Edge weights encode appearance similarity, so message passing
(and the traversal order used by the agent) is guided by the image structure.
"""
import heapq

import numpy as np
from skimage.segmentation import slic

NODE_FEAT_DIM = 12  # 3 mean + 3 std + 3 (mean - image mean) + cy + cx + rel. area


def build_graph(img: np.ndarray, mask: np.ndarray, n_segments: int = 64, compactness: float = 0.3) -> dict:
    """img: float32 [H, W, 3] in [0, 1]; mask: {0, 1} [H, W]."""
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
