"""
demo_figures.py — Generate high-quality presentation figures for the
Graph-Guided RL Medical Image Segmentation project.

Saves all figures to runs/smoke/final_figures/
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from skimage.segmentation import mark_boundaries
from torch.distributions import Categorical

# ---- project imports -------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from gseg.data import make_synthetic
from gseg.env import BatchedSegEnv
from gseg.graph import NODE_FEAT_DIM, build_graph
from gseg.model import GraphActorCritic
from gseg.utils import pick_device

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT = "runs/smoke/best.pt"
LOG_CSV = "runs/smoke/log.csv"
METRICS_JSON = "runs/smoke/metrics.json"
OUT_DIR = "runs/smoke/final_figures"
N_SAMPLES = 8
IMG_SIZE = 128
N_SEG = 64
SEED = 42
IN_DIM = NODE_FEAT_DIM + 3  # +3 for is_labelled, label, is_current

os.makedirs(OUT_DIR, exist_ok=True)
device = pick_device("auto")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {CKPT}")
cp = torch.load(CKPT, map_location="cpu", weights_only=False)
args = cp.get("args", {})
hidden = args.get("hidden", 64) if isinstance(args, dict) else getattr(args, "hidden", 64)
layers = args.get("layers", 3) if isinstance(args, dict) else getattr(args, "layers", 3)
model = GraphActorCritic(IN_DIM, hidden=hidden, layers=layers)
model.load_state_dict(cp["model"])
model.eval()
model.to(device)
print("Model loaded.")

# ---------------------------------------------------------------------------
# Generate synthetic test samples & build graphs
# ---------------------------------------------------------------------------
print(f"Generating {N_SAMPLES} synthetic test images …")
samples = make_synthetic(N_SAMPLES, size=IMG_SIZE, seed=SEED)
graphs = []
for img, mask in samples:
    g = build_graph(img, mask, n_segments=N_SEG)
    graphs.append(g)
print("Graphs built.")


# ---------------------------------------------------------------------------
# Run agent on all graphs
# ---------------------------------------------------------------------------
def run_agent_single(g):
    """Return (labels [N], probs_fg [N]) for a single graph."""
    N = g["N"]
    t_adj = torch.from_numpy(g["adj"]).unsqueeze(0).to(device)
    env = BatchedSegEnv([g], device)
    labels = torch.zeros(N, dtype=torch.long)
    probs_fg = torch.zeros(N)
    with torch.no_grad():
        obs, cur = env.reset()
        for _ in range(N):
            x = obs
            valid = torch.ones(1, env.N, device=device)
            logits, _ = model(x, t_adj, valid, cur)
            dist = Categorical(logits=logits)
            action = dist.probs.argmax(-1)
            p_fg = dist.probs[:, 1]
            node_idx = int(cur[0].item())
            if node_idx < N:
                labels[node_idx] = int(action[0].item())
                probs_fg[node_idx] = float(p_fg[0].item())
            env.step(action)
            obs, cur = env.obs()
    return labels.numpy(), probs_fg.numpy()


def build_pred_mask(seg, labels):
    pred = np.zeros_like(seg, dtype=np.uint8)
    for n in range(len(labels)):
        if labels[n] == 1:
            pred[seg == n] = 1
    return pred


def dice(pred, gt):
    p, g = pred.ravel().astype(float), gt.ravel().astype(float)
    return 2 * (p * g).sum() / (p.sum() + g.sum() + 1e-6)


print("Running agent inference on all samples …")
all_labels, all_probs = [], []
all_preds, all_dices = [], []
for i, g in enumerate(graphs):
    lbl, prb = run_agent_single(g)
    pred = build_pred_mask(g["seg"], lbl)
    d = dice(pred, g["mask"])
    all_labels.append(lbl)
    all_probs.append(prb)
    all_preds.append(pred)
    all_dices.append(d)
    print(f"  sample {i+1}: Dice={d:.4f}")

print(f"Mean Dice: {np.mean(all_dices):.4f}")

# ---------------------------------------------------------------------------
# Load training log
# ---------------------------------------------------------------------------
log_rows = []
with open(LOG_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        log_rows.append(row)

iters = [int(r["iter"]) for r in log_rows]
train_dice = [float(r["train_dice"]) for r in log_rows]
entropy = [float(r["ent"]) for r in log_rows]

val_iters, val_dice_log = [], []
for r in log_rows:
    if r.get("val_dice", ""):
        val_iters.append(int(r["iter"]))
        val_dice_log.append(float(r["val_dice"]))

with open(METRICS_JSON) as f:
    metrics = json.load(f)
oracle_dice = metrics.get("oracle_dice", 0.9576)
val_dice_final = metrics.get("val_dice", 0.9472)
val_iou_final = metrics.get("val_iou", 0.9015)

# ===========================================================================
# FIGURE 1: Architecture Diagram
# ===========================================================================
print("Generating architecture_diagram.png …")
fig1, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig1.patch.set_facecolor("#f8f9fa")

BOXES = [
    (1.0, "Input\nImage\nH×W×3", "#4285f4"),
    (3.5, "SLIC\nSuperpixels\nN≈64 nodes", "#0f9d58"),
    (6.0, "RAG\nGraph\nN×N edges", "#f4b400"),
    (8.5, "GNN\nActor-Critic\n3 GraphLayers", "#db4437"),
    (11.0, "Binary\nMask\nH×W", "#7b1fa2"),
]

box_w, box_h = 2.0, 1.8
box_y = 1.6
arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=2.0,
                mutation_scale=18, connectionstyle="arc3,rad=0")

for cx, label, color in BOXES:
    rect = mpatches.FancyBboxPatch(
        (cx - box_w / 2, box_y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.1", linewidth=2, edgecolor=color,
        facecolor=color + "22"
    )
    ax.add_patch(rect)
    ax.text(cx, box_y, label, ha="center", va="center", fontsize=10,
            fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=1, foreground="white")])

# Arrows between boxes
arrow_xs = [(2.0, 2.5), (4.5, 5.0), (7.0, 7.5), (9.5, 10.0)]
for x1, x2 in arrow_xs:
    ax.annotate("", xy=(x2, box_y), xytext=(x1, box_y),
                arrowprops=arrow_kw)

# Sub-labels below boxes
sub_labels = [
    (1.0, 1.0, "pixel grid"),
    (3.5, 1.0, "12-dim node features\nappearance weights"),
    (6.0, 1.0, "guided traversal\nPrim-style order"),
    (8.5, 1.0, "binary action\nbg / fg per node"),
    (11.0, 1.0, "pixel-level output"),
]
for cx, y, txt in sub_labels:
    ax.text(cx, y, txt, ha="center", va="top", fontsize=8, color="#555555", style="italic")

# Title
ax.text(7.0, 4.6, "Graph-Guided RL Segmentation — Pipeline Architecture",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#202124")

# PPO loop arrow
ax.annotate("", xy=(8.5, box_y + 1.2), xytext=(11.0, box_y + 1.2),
            arrowprops=dict(arrowstyle="-|>", color="#db4437", lw=1.5,
                            connectionstyle="arc3,rad=-0.3", linestyle="dashed",
                            mutation_scale=14))
ax.text(9.75, box_y + 1.6, "PPO reward\n(Dice-delta)", ha="center", va="center",
        fontsize=8, color="#db4437")

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "architecture_diagram.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  Saved architecture_diagram.png")

# ===========================================================================
# FIGURE 2: Performance Summary (2×2 grid)
# ===========================================================================
print("Generating performance_summary.png …")
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Performance Summary", fontsize=16, fontweight="bold", y=0.98)

# (a) Training curves
ax = axes[0, 0]
ax.plot(iters, train_dice, label="Train Dice", color="#4285f4", alpha=0.8, linewidth=1.5)
if val_iters:
    ax.plot(val_iters, val_dice_log, label="Val Dice", color="#0f9d58",
            marker="o", linewidth=2.0, markersize=5)
ax.axhline(oracle_dice, linestyle="--", color="#db4437", linewidth=1.5,
           label=f"Oracle ({oracle_dice:.3f})")
ax.set_title("(a) Training Curves", fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel("Dice")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (b) Val dice distribution (across test samples)
ax = axes[0, 1]
ax.hist(all_dices, bins=8, color="#4285f4", edgecolor="white", linewidth=1.5)
ax.axvline(np.mean(all_dices), color="#db4437", linestyle="--",
           linewidth=2, label=f"Mean={np.mean(all_dices):.3f}")
ax.axvline(oracle_dice, color="#f4b400", linestyle="--",
           linewidth=2, label=f"Oracle={oracle_dice:.3f}")
ax.set_title("(b) Val Dice Distribution (test samples)", fontweight="bold")
ax.set_xlabel("Dice Score")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (c) Sample predictions grid (4 samples, row: GT / pred)
ax = axes[1, 0]
ax.axis("off")
ax.set_title("(c) Sample Predictions (top=GT, bottom=Pred)", fontweight="bold")

n_show = min(4, N_SAMPLES)
sub_axes_c = []
for j in range(n_show * 2):
    sub_ax = fig2.add_axes([
        0.07 + (j % n_show) * 0.11,
        0.08 + (1 - j // n_show) * 0.17,
        0.09, 0.14
    ])
    sub_axes_c.append(sub_ax)

for j in range(n_show):
    img_j = samples[j][0]
    gt_j = graphs[j]["mask"]
    pred_j = all_preds[j]

    # top row: GT overlay
    sub_axes_c[j].imshow(img_j)
    sub_axes_c[j].imshow(gt_j, alpha=0.4, cmap="Greens", vmin=0, vmax=1)
    sub_axes_c[j].axis("off")
    sub_axes_c[j].set_title(f"S{j+1}\nGT", fontsize=7)

    # bottom row: pred overlay
    sub_axes_c[n_show + j].imshow(img_j)
    sub_axes_c[n_show + j].imshow(pred_j, alpha=0.4, cmap="Blues", vmin=0, vmax=1)
    sub_axes_c[n_show + j].axis("off")
    sub_axes_c[n_show + j].set_title(f"D={all_dices[j]:.2f}", fontsize=7)

# (d) Oracle vs Agent Dice scatter
ax = axes[1, 1]
# Compute oracle dice for each sample
oracle_dices_per = []
for g in graphs:
    oracle_per_node = g["frac"]  # fraction of GT pixels in each superpixel
    oracle_pred = (oracle_per_node > 0.5).astype(np.uint8)
    pred_px = build_pred_mask(g["seg"], oracle_pred)
    od = dice(pred_px, g["mask"])
    oracle_dices_per.append(od)

ax.scatter(oracle_dices_per, all_dices, color="#4285f4", s=80, alpha=0.8, edgecolors="white",
           linewidths=1, zorder=3)
lim_min = min(min(oracle_dices_per), min(all_dices)) - 0.02
lim_max = max(max(oracle_dices_per), max(all_dices)) + 0.02
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, label="y=x")
ax.set_xlabel("Oracle Dice (per-image ceiling)")
ax.set_ylabel("Agent Dice")
ax.set_title("(d) Oracle vs Agent Dice", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "performance_summary.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  Saved performance_summary.png")

# ===========================================================================
# FIGURE 3: Sample Segmentations (2×4 grid)
# ===========================================================================
print("Generating sample_segmentations.png …")
fig3, axes3 = plt.subplots(N_SAMPLES, 4, figsize=(16, N_SAMPLES * 2.2))
fig3.suptitle("Sample Segmentations — 8 Test Images", fontsize=15, fontweight="bold")

col_titles = ["Input Image", "Superpixel Overlay", "Ground Truth", "Prediction"]
for j, ct in enumerate(col_titles):
    axes3[0, j].set_title(ct, fontsize=11, fontweight="bold")

for i in range(N_SAMPLES):
    img = samples[i][0]
    gt = graphs[i]["mask"]
    pred = all_preds[i]
    seg = graphs[i]["seg"]

    # Input
    axes3[i, 0].imshow(img)
    axes3[i, 0].axis("off")
    axes3[i, 0].set_ylabel(f"Sample {i+1}", fontsize=9, rotation=90, va="center")

    # Superpixel overlay
    sp_img = mark_boundaries(img, seg, color=(0.1, 0.6, 1.0), mode="thick")
    axes3[i, 1].imshow(sp_img)
    axes3[i, 1].axis("off")

    # GT
    axes3[i, 2].imshow(img)
    axes3[i, 2].imshow(gt, alpha=0.5, cmap="Greens", vmin=0, vmax=1)
    axes3[i, 2].axis("off")

    # Prediction with TP/FP/FN overlay
    result = img.copy()
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    alpha = 0.5
    result[tp] = result[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha
    result[fp] = result[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    result[fn] = result[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha
    result = np.clip(result, 0, 1)
    axes3[i, 3].imshow(result)
    axes3[i, 3].axis("off")
    axes3[i, 3].set_title(f"Dice={all_dices[i]:.3f}", fontsize=8, pad=1)

# Legend
legend_handles = [
    mpatches.Patch(color="#00cc00", label="TP (Green)"),
    mpatches.Patch(color="#ff0000", label="FP (Red)"),
    mpatches.Patch(color="#0000ff", label="FN (Blue)"),
]
fig3.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
            bbox_to_anchor=(0.75, 0.0))

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
fig3.savefig(os.path.join(OUT_DIR, "sample_segmentations.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)
print("  Saved sample_segmentations.png")

# ===========================================================================
# FIGURE 4: Traversal Visualisation (one image)
# ===========================================================================
print("Generating traversal_visualization.png …")
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle("Graph-Guided Traversal Order Visualisation (Sample 1)", fontsize=14, fontweight="bold")

g0 = graphs[0]
img0 = samples[0][0]
seg0 = g0["seg"]
order0 = g0["order"]  # shape [N] — the traversal order (node index at step t)
N0 = g0["N"]

# Build traversal position map (position t in traversal for each node n)
trav_pos = np.zeros(N0, dtype=float)
for step, node in enumerate(order0):
    trav_pos[int(node)] = step / max(1, N0 - 1)  # normalised 0→1

# Pixel-level traversal map
trav_map = np.zeros(seg0.shape, dtype=float)
for n in range(N0):
    trav_map[seg0 == n] = trav_pos[n]

# Panel (a): Input image with superpixel boundaries
axes4[0].imshow(img0)
sp_img0 = mark_boundaries(img0, seg0, color=(1, 1, 1), mode="thick")
axes4[0].imshow(sp_img0, alpha=0.5)
axes4[0].set_title("Input + Superpixels", fontweight="bold")
axes4[0].axis("off")

# Panel (b): Traversal heatmap
im_trav = axes4[1].imshow(trav_map, cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
axes4[1].imshow(mark_boundaries(np.ones((*seg0.shape, 3)), seg0, color=(0, 0, 0), mode="thin"),
                alpha=0.5)
axes4[1].set_title("Traversal Order\n(Blue=first, Red=last)", fontweight="bold")
axes4[1].axis("off")
plt.colorbar(im_trav, ax=axes4[1], fraction=0.046, pad=0.04, label="Normalised visit order")

# Panel (c): Confidence map
prbs0 = all_probs[0]
conf_map0 = np.zeros(seg0.shape, dtype=float)
for n in range(N0):
    if n < len(prbs0):
        conf_map0[seg0 == n] = float(prbs0[n])

im_conf = axes4[2].imshow(conf_map0, cmap="plasma", vmin=0, vmax=1, interpolation="nearest")
axes4[2].imshow(mark_boundaries(np.ones((*seg0.shape, 3)), seg0, color=(0, 0, 0), mode="thin"),
                alpha=0.4)
axes4[2].set_title("GNN Confidence\nP(foreground)", fontweight="bold")
axes4[2].axis("off")
plt.colorbar(im_conf, ax=axes4[2], fraction=0.046, pad=0.04, label="P(fg)")

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "traversal_visualization.png"), dpi=150, bbox_inches="tight")
plt.close(fig4)
print("  Saved traversal_visualization.png")

# ---------------------------------------------------------------------------
print("\nAll figures saved to:", os.path.abspath(OUT_DIR))
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {f:40s}  {size_kb:.1f} KB")
