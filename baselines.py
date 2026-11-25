"""
Classical baselines on CVC-ClinicDB validation set:
  - Otsu thresholding (grayscale)
  - K-Means clustering (k=2, Lab colour)
Metrics: Dice, IoU, HD95
"""
import glob, os, random
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans

# ------------------------------------------------------------------ metrics
def dice(pred, gt):
    inter = (pred & gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)

def iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-8)

def hd95(pred, gt):
    """95th-percentile Hausdorff distance (pixels). Returns 0 if both empty."""
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(max(pred.shape))  # max possible distance

    dt_pred = distance_transform_edt(~pred)
    dt_gt   = distance_transform_edt(~gt)
    # surface-to-surface distances
    d1 = dt_gt[pred]      # pred surface → gt
    d2 = dt_pred[gt]      # gt surface → pred
    # surface = boundary pixels; approximate by using all non-zero pixels
    # for small images this is a reasonable approximation
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))

# ------------------------------------------------------------------ load val set (same seed=0 split as training)
IMG_DIR  = "dataset/images"
MASK_DIR = "dataset/masks"
SIZE     = 128    # same resize as training
VAL_FRAC = 0.20
SEED     = 0

_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def _stem(p):
    s = os.path.splitext(os.path.basename(p))[0]
    for suf in ("_segmentation", "_mask", "_seg", "_gt", "-mask"):
        if s.endswith(suf): s = s[:-len(suf)]
    return s

imgs  = sorted(p for p in glob.glob(os.path.join(IMG_DIR, "*"))  if p.lower().endswith(_EXT))
masks = {_stem(p): p for p in glob.glob(os.path.join(MASK_DIR, "*")) if p.lower().endswith(_EXT)}

pairs = []
for p in imgs:
    s = _stem(p)
    if s not in masks: continue
    im = np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.
    mk = (np.asarray(Image.open(masks[s]).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 0)
    pairs.append((im, mk))

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(pairs))
n_val = max(1, int(len(pairs) * VAL_FRAC))
val_pairs = [pairs[i] for i in idx[:n_val]]
print(f"Val set: {len(val_pairs)} images (seed={SEED}, val_frac={VAL_FRAC})")
print(f"Image size after resize: {SIZE}×{SIZE}")
print(f"Total dataset: {len(pairs)} images | Train: {len(pairs)-n_val} | Val: {n_val}")

# ------------------------------------------------------------------ Otsu
def otsu_segment(img):
    gray = img.mean(axis=2)
    thr  = threshold_otsu(gray)
    pred = gray > thr
    # decide which class is foreground: pick the smaller region (polyp heuristic)
    # but just try both and pick the one with better Dice vs majority class
    # Actually: return both options; caller picks best vs GT
    return pred, ~pred

otsu_dice, otsu_iou, otsu_hd95 = [], [], []
for img, gt in val_pairs:
    p1, p2 = otsu_segment(img)
    d1, d2 = dice(p1, gt), dice(p2, gt)
    best   = p1 if d1 >= d2 else p2
    otsu_dice.append(max(d1, d2))
    otsu_iou.append(iou(best, gt))
    otsu_hd95.append(hd95(best, gt))

print(f"\nOtsu thresholding (grayscale):")
print(f"  Dice  = {np.mean(otsu_dice):.4f} ± {np.std(otsu_dice):.4f}")
print(f"  IoU   = {np.mean(otsu_iou):.4f} ± {np.std(otsu_iou):.4f}")
print(f"  HD95  = {np.mean(otsu_hd95):.2f} ± {np.std(otsu_hd95):.2f} px")

# ------------------------------------------------------------------ K-Means (k=2, Lab)
def kmeans_segment(img):
    lab  = rgb2lab(img).reshape(-1, 3)
    km   = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3)
    lbls = km.fit_predict(lab).reshape(img.shape[:2])
    p1   = lbls == 0
    p2   = lbls == 1
    return p1, p2

km_dice, km_iou, km_hd95 = [], [], []
for img, gt in val_pairs:
    p1, p2 = kmeans_segment(img)
    d1, d2 = dice(p1, gt), dice(p2, gt)
    best   = p1 if d1 >= d2 else p2
    km_dice.append(max(d1, d2))
    km_iou.append(iou(best, gt))
    km_hd95.append(hd95(best, gt))

print(f"\nK-Means (k=2, CIE-Lab colour space):")
print(f"  Dice  = {np.mean(km_dice):.4f} ± {np.std(km_dice):.4f}")
print(f"  IoU   = {np.mean(km_iou):.4f} ± {np.std(km_iou):.4f}")
print(f"  HD95  = {np.mean(km_hd95):.2f} ± {np.std(km_hd95):.2f} px")

# ------------------------------------------------------------------ summary
print("\n========== SUMMARY ==========")
print(f"{'Method':<30} {'Dice':>7} {'IoU':>7} {'HD95':>8}")
print("-"*56)
print(f"{'Otsu thresholding':<30} {np.mean(otsu_dice):>7.4f} {np.mean(otsu_iou):>7.4f} {np.mean(otsu_hd95):>8.2f}")
print(f"{'K-Means (k=2, Lab)':<30} {np.mean(km_dice):>7.4f} {np.mean(km_iou):>7.4f} {np.mean(km_hd95):>8.2f}")
print(f"{'GRL-PPO (ours, val best)':<30} {'0.4758':>7} {'0.3604':>7} {'—':>8}")
print(f"{'Oracle superpixel (200 sp)':<30} {'0.8649':>7} {'—':>7} {'—':>8}")
