"""Datasets: a synthetic lesion generator (for smoke tests) and a generic image/mask folder loader."""
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .graph import build_graph


# --------------------------------------------------------------------------- synthetic
def synth_sample(rng: np.random.Generator, size: int = 128):
    yy, xx = np.mgrid[0:size, 0:size] / size

    bg = gaussian_filter(rng.standard_normal((size, size)), size / 12)
    bg = (bg - bg.mean()) / (bg.std() + 1e-6) * 0.05 + rng.uniform(0.35, 0.6)

    def blob(cy, cx, ry, rx, theta):
        dy, dx = yy - cy, xx - cx
        u = np.cos(theta) * dx + np.sin(theta) * dy
        v = -np.sin(theta) * dx + np.cos(theta) * dy
        ang = np.arctan2(v, u)
        wob = 1 + sum(rng.uniform(0, 0.12) * np.sin(k * ang + rng.uniform(0, 6.28)) for k in (2, 3, 4))
        d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        return d, wob

    cy, cx = rng.uniform(0.35, 0.65, 2)
    ry, rx = rng.uniform(0.12, 0.26, 2)
    d, wob = blob(cy, cx, ry, rx, rng.uniform(0, np.pi))
    mask = (d < wob).astype(np.uint8)
    soft = 1 / (1 + np.exp((d - wob) / 0.06))
    contrast = rng.uniform(0.15, 0.32) * rng.choice([-1, 1])
    tex = gaussian_filter(rng.standard_normal((size, size)), 2) * 0.05
    gray = bg + soft * (contrast + tex)

    if rng.random() < 0.5:  # distractor: weaker, opposite-sign blob that is NOT part of the mask
        dcy, dcx = rng.uniform(0.1, 0.9, 2)
        d2, w2 = blob(dcy, dcx, rng.uniform(0.05, 0.1), rng.uniform(0.05, 0.1), rng.uniform(0, np.pi))
        gray = gray + 1 / (1 + np.exp((d2 - w2) / 0.06)) * (-0.5 * contrast) * (1 - soft)

    tint = rng.uniform(0.88, 1.1, 3)
    img = gray[..., None] * tint + rng.normal(0, 0.04, (size, size, 3))
    return np.clip(img, 0, 1).astype(np.float32), mask


def make_synthetic(n: int, size: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [synth_sample(rng, size) for _ in range(n)]


# --------------------------------------------------------------------------- real data
_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _stem(path):
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_segmentation", "_mask", "_seg", "_gt", "-mask"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def load_folder(root: str, size: int = 128):
    """Expects root/images/* and root/masks/* (matched by file stem; common suffixes such as
    `_mask` or `_segmentation` are ignored). Masks are binarised (> 0). 2-D only; for 3-D
    volumes, export slices to PNG first."""
    imgs = sorted(p for p in glob.glob(os.path.join(root, "images", "*")) if p.lower().endswith(_IMG_EXT))
    masks = {_stem(p): p for p in glob.glob(os.path.join(root, "masks", "*")) if p.lower().endswith(_IMG_EXT)}
    pairs = []
    for p in imgs:
        s = _stem(p)
        if s not in masks:
            continue
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        mk = Image.open(masks[s]).convert("L").resize((size, size), Image.NEAREST)
        pairs.append((np.asarray(im, dtype=np.float32) / 255.0, (np.asarray(mk) > 0).astype(np.uint8)))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {root}/images and {root}/masks")
    return pairs


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Light augmentation: horizontal/vertical flip + brightness/contrast jitter.
    Applied only to training pairs; val pairs are never augmented."""
    # Random horizontal flip
    if rng.random() < 0.5:
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
    # Random vertical flip
    if rng.random() < 0.5:
        img, mask = img[::-1].copy(), mask[::-1].copy()
    # Brightness jitter ±15 %
    img = np.clip(img * rng.uniform(0.85, 1.15), 0, 1).astype(np.float32)
    # Contrast jitter: scale around mean
    m = img.mean()
    img = np.clip((img - m) * rng.uniform(0.85, 1.15) + m, 0, 1).astype(np.float32)
    return img, mask


def build_graphs(pairs, n_segments=64, compactness=0.3, verbose=True, augment_fn=None, rng=None):
    graphs = []
    for i, (img, mask) in enumerate(pairs):
        if augment_fn is not None and rng is not None:
            img, mask = augment_fn(img, mask, rng)
        graphs.append(build_graph(img, mask, n_segments, compactness))
        if verbose and (i + 1) % 100 == 0:
            print(f"  built {i + 1}/{len(pairs)} graphs")
    return graphs


def get_graphs(args):
    """Returns (train_graphs, val_graphs) according to the CLI args."""
    aug = getattr(args, "augment", False)
    if args.data == "synthetic":
        tr = make_synthetic(args.n_train, args.size, seed=args.seed)
        va = make_synthetic(args.n_val, args.size, seed=10_000 + args.seed)
    else:
        pairs = load_folder(args.data_dir, args.size)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(pairs))
        n_val = max(1, int(len(pairs) * args.val_frac))
        va = [pairs[i] for i in idx[:n_val]]
        tr = [pairs[i] for i in idx[n_val:]]
    print(f"Building superpixel graphs: {len(tr)} train / {len(va)} val ...")
    aug_rng = np.random.default_rng(args.seed + 1) if aug else None
    return (
        build_graphs(tr, args.n_segments, args.compactness,
                     augment_fn=augment if aug else None, rng=aug_rng),
        build_graphs(va, args.n_segments, args.compactness),  # val: no augmentation
    )
