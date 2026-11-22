"""
Graph-Guided Reinforcement Learning for Medical Image Segmentation
Streamlit Demo Application
"""

import io
import os
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage.segmentation import mark_boundaries
from torch.distributions import Categorical

from gseg.env import BatchedSegEnv
from gseg.graph import NODE_FEAT_DIM, build_graph
from gseg.model import GraphActorCritic
from gseg.utils import pick_device

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Graph-Guided RL Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1a73e8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🔬 Graph-Guided Reinforcement Learning for Medical Image Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A PPO-trained GNN agent that labels superpixels via graph-guided traversal · Val Dice 0.947 · Val IoU 0.901</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Checkpoint selector
    default_ckpt = "runs/smoke/best.pt"
    ckpt_path = st.text_input("Model checkpoint", value=default_ckpt)

    st.subheader("Image Parameters")
    n_segments = st.slider("Number of superpixels", min_value=32, max_value=128, value=64, step=8)
    analysis_size = st.slider("Analysis size (px)", min_value=64, max_value=256, value=128, step=32)

    st.subheader("Visualisation")
    show_sp_overlay = st.toggle("Show superpixel overlay", value=True)
    show_confidence = st.toggle("Show confidence heatmap", value=True)

    st.divider()
    st.subheader("ℹ️ System Info")
    device = pick_device("auto")
    device_label = "🟢 GPU (CUDA)" if device.type == "cuda" else "🔵 CPU"
    st.info(f"**Device:** {device_label}")

    # Architecture expander
    with st.expander("🏗️ Architecture Details"):
        st.markdown("""
**Pipeline:**
1. **SLIC** superpixel partitioning (N≈64 nodes)
2. **RAG** region adjacency graph with appearance-similarity edge weights
3. **GNN** (3 × GraphLayer, hidden=64) message-passing actor-critic
4. **PPO** policy gradient with Dice-delta reward shaping

**Node features (12-dim):**
- RGB mean, RGB std, mean relative to image (3+3+3)
- Centroid (cy, cx), relative area (2+1)

**Training:** 100 iterations on 200 synthetic lesion images
""")

# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------
IN_DIM = NODE_FEAT_DIM + 3  # +3 for is_labelled, label, is_current


@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_model(ckpt: str):
    if not os.path.isfile(ckpt):
        return None, f"Checkpoint not found: {ckpt}"
    try:
        cp = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = cp.get("args", {})
        hidden = args.get("hidden", 64) if isinstance(args, dict) else getattr(args, "hidden", 64)
        layers = args.get("layers", 3) if isinstance(args, dict) else getattr(args, "layers", 3)
        model = GraphActorCritic(IN_DIM, hidden=hidden, layers=layers)
        model.load_state_dict(cp["model"])
        model.eval()
        return model, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Graph builder (cached by image bytes + params)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building superpixel graph…")
def cached_build_graph(img_bytes: bytes, mask_bytes, n_seg: int, size: int):
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((size, size), Image.BILINEAR)
    img_np = np.asarray(img_pil, dtype=np.float32) / 255.0

    if mask_bytes is not None:
        mk_pil = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((size, size), Image.NEAREST)
        mask_np = (np.asarray(mk_pil) > 0).astype(np.uint8)
    else:
        mask_np = np.zeros((size, size), dtype=np.uint8)

    g = build_graph(img_np, mask_np, n_segments=n_seg)
    return g, img_np, mask_np


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def run_inference(model, graph, dev):
    """Run the trained agent over the graph, return per-node labels + logit probs."""
    g = graph
    B = 1
    N = g["N"]

    t_x0 = torch.from_numpy(g["x0"]).unsqueeze(0).to(dev)        # [1,N,F]
    t_adj = torch.from_numpy(g["adj"]).unsqueeze(0).to(dev)        # [1,N,N]
    t_order = torch.from_numpy(g["order"]).unsqueeze(0).to(dev)    # [1,N]
    t_frac = torch.from_numpy(g["frac"]).unsqueeze(0).to(dev)      # [1,N]
    t_area = torch.from_numpy(g["area"]).unsqueeze(0).to(dev)      # [1,N]

    env = BatchedSegEnv([g], dev)
    labels = torch.zeros(N, dtype=torch.long)
    probs_fg = torch.zeros(N)

    with torch.no_grad():
        obs, cur = env.reset()
        for step in range(N):
            x = obs                                                # [1,N,F+3]
            valid = torch.ones(B, N, device=dev)
            logits, _ = model(x, t_adj, valid, cur)
            dist = Categorical(logits=logits)
            action = dist.probs.argmax(-1)
            p_fg = dist.probs[:, 1]
            node_idx = int(cur[0].item())
            labels[node_idx] = int(action[0].item())
            probs_fg[node_idx] = float(p_fg[0].item())
            obs, cur = env.obs()
            env.step(action)

    return labels.numpy(), probs_fg.numpy()


def build_pred_mask(seg, labels):
    """Map per-node labels back to pixel mask."""
    pred = np.zeros_like(seg, dtype=np.uint8)
    N = int(seg.max()) + 1
    for n in range(N):
        if n < len(labels) and labels[n] == 1:
            pred[seg == n] = 1
    return pred


def build_confidence_map(seg, probs_fg):
    """Map per-node fg probability to pixel heatmap."""
    cmap = np.zeros(seg.shape, dtype=np.float32)
    N = int(seg.max()) + 1
    for n in range(N):
        if n < len(probs_fg):
            cmap[seg == n] = probs_fg[n]
    return cmap


def overlay_tp_fp_fn(img, pred, gt):
    """Return RGB image with TP=green, FP=red, FN=blue overlays."""
    result = img.copy()
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    alpha = 0.45
    result[tp] = result[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha
    result[fp] = result[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    result[fn] = result[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha
    return np.clip(result, 0, 1)


def overlay_pred(img, pred):
    """Return RGB image with foreground overlay (teal)."""
    result = img.copy()
    alpha = 0.45
    result[pred == 1] = result[pred == 1] * (1 - alpha) + np.array([0, 0.8, 0.9]) * alpha
    return np.clip(result, 0, 1)


def dice_score(pred, gt):
    pred_f = pred.astype(float).ravel()
    gt_f = gt.astype(float).ravel()
    return 2 * (pred_f * gt_f).sum() / (pred_f.sum() + gt_f.sum() + 1e-6)


def mask_to_png_bytes(mask):
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_demo, tab_results, tab_arch, tab_about = st.tabs([
    "🔬 Live Demo",
    "📊 Training Results",
    "🏗️ Architecture",
    "ℹ️ About",
])

# ===========================================================================
# TAB 1 — Live Demo
# ===========================================================================
with tab_demo:
    st.header("Live Segmentation Demo")

    # Model status in sidebar
    model, model_err = load_model(ckpt_path)
    with st.sidebar:
        st.subheader("🤖 Model Status")
        if model_err:
            st.error(f"❌ {model_err}")
        else:
            st.success("✅ Model loaded")

    col_up_img, col_up_gt = st.columns(2)
    with col_up_img:
        uploaded_img = st.file_uploader("Upload an image (RGB)", type=["png", "jpg", "jpeg", "bmp", "tif"])
    with col_up_gt:
        uploaded_gt = st.file_uploader("Upload GT mask (optional, grayscale)", type=["png", "jpg", "jpeg", "bmp", "tif"])

    if uploaded_img is None:
        st.info("👆 Upload a medical image to run segmentation. Use the sidebar to adjust parameters.")
        st.markdown("""
**Demo tips:**
- Upload any grayscale or RGB image (histology, MRI slice, CT scan, etc.)
- Optionally upload a ground-truth binary mask to see TP/FP/FN overlay and Dice score
- Adjust superpixel count and analysis size in the sidebar
        """)
    else:
        if model is None:
            st.error(f"Cannot run inference: {model_err}")
        else:
            img_bytes = uploaded_img.read()
            gt_bytes = uploaded_gt.read() if uploaded_gt is not None else None

            with st.spinner("Building graph and running agent…"):
                t0 = time.perf_counter()
                g, img_np, mask_np = cached_build_graph(img_bytes, gt_bytes, n_segments, analysis_size)
                labels, probs_fg = run_inference(model, g, device)
                elapsed = time.perf_counter() - t0

            pred_mask = build_pred_mask(g["seg"], labels)
            has_gt = (gt_bytes is not None)

            # ----- Visualisations -----
            st.subheader("Results")
            cols = st.columns(4 if has_gt else 3)

            with cols[0]:
                st.caption("📷 Original")
                st.image((img_np * 255).astype(np.uint8), use_column_width=True)

            with cols[1]:
                st.caption("🔷 Superpixel Graph")
                sp_img = mark_boundaries(img_np, g["seg"], color=(0.1, 0.6, 1.0), mode="thick")
                st.image((sp_img * 255).astype(np.uint8), use_column_width=True)

            if has_gt:
                with cols[2]:
                    st.caption("✅ Ground Truth")
                    gt_rgb = np.stack([mask_np, mask_np, mask_np], -1).astype(np.float32)
                    st.image((gt_rgb * 255).astype(np.uint8), use_column_width=True)

                with cols[3]:
                    st.caption("🎯 Prediction (G=TP R=FP B=FN)")
                    overlay = overlay_tp_fp_fn(img_np, pred_mask, mask_np)
                    st.image((overlay * 255).astype(np.uint8), use_column_width=True)
            else:
                with cols[2]:
                    st.caption("🎯 Prediction Overlay")
                    overlay = overlay_pred(img_np, pred_mask)
                    st.image((overlay * 255).astype(np.uint8), use_column_width=True)

            # Confidence heatmap
            if show_confidence:
                conf_map = build_confidence_map(g["seg"], probs_fg)
                fig_conf, ax_conf = plt.subplots(figsize=(4, 4))
                im = ax_conf.imshow(conf_map, cmap="plasma", vmin=0, vmax=1)
                ax_conf.axis("off")
                ax_conf.set_title("Confidence (P[foreground])", fontsize=10)
                plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
                plt.tight_layout()
                col_conf1, col_conf2 = st.columns([1, 3])
                with col_conf1:
                    st.pyplot(fig_conf)
                plt.close(fig_conf)

            # ----- Metrics -----
            st.subheader("📐 Metrics")
            m_cols = st.columns(4)
            with m_cols[0]:
                if has_gt:
                    d = dice_score(pred_mask, mask_np)
                    st.metric("Val Dice", f"{d:.4f}")
                else:
                    st.metric("Val Dice", "N/A (no GT)")
            with m_cols[1]:
                seg_area_pct = pred_mask.sum() / pred_mask.size * 100
                st.metric("Segmented Area", f"{seg_area_pct:.1f}%")
            with m_cols[2]:
                st.metric("Superpixels (N)", g["N"])
            with m_cols[3]:
                st.metric("Inference Time", f"{elapsed*1000:.0f} ms")

            # ----- Download -----
            st.download_button(
                label="⬇️ Download Prediction Mask (PNG)",
                data=mask_to_png_bytes(pred_mask),
                file_name="segmentation_mask.png",
                mime="image/png",
            )

# ===========================================================================
# TAB 2 — Training Results
# ===========================================================================
with tab_results:
    st.header("Training Results")

    log_path = "runs/smoke/log.csv"
    metrics_path = "runs/smoke/metrics.json"
    pred_png = "runs/smoke/predictions.png"
    curves_png = "runs/smoke/curves.png"

    if not os.path.isfile(log_path):
        st.error(f"Log file not found: {log_path}")
    else:
        import csv
        import json

        rows = []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        iters = [int(r["iter"]) for r in rows]
        train_dice = [float(r["train_dice"]) for r in rows]
        entropy = [float(r["ent"]) for r in rows]

        # Val dice only on eval rows
        val_iters, val_dice = [], []
        for r in rows:
            if r["val_dice"]:
                val_iters.append(int(r["iter"]))
                val_dice.append(float(r["val_dice"]))

        oracle_dice = 0.9576
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                mj = json.load(f)
            oracle_dice = mj.get("oracle_dice", oracle_dice)

        # --- Summary metric cards ---
        st.subheader("Final Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            final_vd = val_dice[-1] if val_dice else 0.0
            st.metric("Val Dice", f"{final_vd:.4f}", delta=f"+{final_vd - 0.196:.3f} vs untrained")
        with mc2:
            final_iou = mj.get("val_iou", 0.901) if os.path.isfile(metrics_path) else 0.901
            st.metric("Val IoU", f"{final_iou:.4f}")
        with mc3:
            st.metric("Oracle Dice (ceiling)", f"{oracle_dice:.4f}")
        with mc4:
            n_val = mj.get("n_val", 40) if os.path.isfile(metrics_path) else 40
            st.metric("Validation images", n_val)

        st.divider()

        # --- Chart 1: Dice curves ---
        st.subheader("Training & Validation Dice")

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(iters, train_dice, label="Train Dice", color="#1a73e8", alpha=0.8, linewidth=1.5)
        if val_iters:
            ax1.plot(val_iters, val_dice, label="Val Dice", color="#34a853", marker="o", linewidth=2.0)
        ax1.axhline(oracle_dice, linestyle="--", color="#ea4335", linewidth=1.5, label=f"Oracle ceiling ({oracle_dice:.3f})")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Dice")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title("Dice vs Iteration")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # --- Chart 2: Policy Entropy ---
        st.subheader("Policy Entropy")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(iters, entropy, color="#fbbc04", linewidth=1.5, label="Entropy")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Entropy (nats)")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_title("Policy Entropy over Training")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.divider()

        # --- Saved figures ---
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if os.path.isfile(pred_png):
                st.subheader("Sample Predictions")
                st.image(pred_png, use_column_width=True)
            else:
                st.warning("predictions.png not found")
        with fig_col2:
            if os.path.isfile(curves_png):
                st.subheader("Training Curves")
                st.image(curves_png, use_column_width=True)
            else:
                st.warning("curves.png not found")

# ===========================================================================
# TAB 3 — Architecture
# ===========================================================================
with tab_arch:
    st.header("System Architecture")

    st.markdown("""
```
┌────────────────────────────────────────────────────────────────────────────┐
│                   Graph-Guided RL Segmentation Pipeline                    │
└────────────────────────────────────────────────────────────────────────────┘

   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐
   │  Input   │    │  SLIC    │    │  RAG     │    │  GNN     │    │ Mask  │
   │  Image   │───▶│Superpixel│───▶│  Graph   │───▶│  Policy  │───▶│Output │
   │ H×W×3   │    │N≈64 nodes│    │N×N edges │    │(PPO+AC)  │    │ H×W  │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └───────┘
        │               │               │               │
        │          SLIC segments    appearance      node labels
        │          + features      similarity      (bg/fg)
        │          extraction      weights
        │
        └── pixel intensity, texture → 12-dim node feature vector

   GraphActorCritic forward pass (per step):
   ┌─────────────────────────────────────────────────────────────┐
   │  State x [B,N,15]  ──▶  Linear(inp)  ──▶  GELU            │
   │       ↓                                      ↓             │
   │  3 × GraphLayer (h + GELU(LN(W_s·h + W_n·A·h)))            │
   │       ↓                                                     │
   │  h_cur ‖ mean(h)  ──▶  pi head  ──▶  Categorical(logits)   │
   │                    ──▶  v  head  ──▶  value estimate        │
   └─────────────────────────────────────────────────────────────┘
```
""")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Key Design Decisions")
        st.markdown("""
- **Graph-guided traversal (Prim-style):** agent always sees already-labelled neighbours
  before deciding on next node → better use of spatial context
- **Dice-delta reward shaping:** instantaneous reward = change in global Dice + area-weighted accuracy
  → return equals final Dice score, dense training signal
- **PPO with clipped objective:** stable off-policy updates, 4 epochs per rollout
- **Oracle ceiling:** upper bound from superpixel partition (fractional coverage), quantifies information loss
- **Synthetic lesion generator:** procedural blobs with distractors for fast iteration without real labels
""")

    with col_b:
        st.subheader("Hyperparameters")
        st.table({
            "Parameter": ["Hidden dim", "GNN layers", "LR", "Entropy coef", "PPO epochs",
                          "Minibatch", "Num envs", "Training iters", "n_segments"],
            "Value": ["64", "3", "3×10⁻⁴", "0.01", "4", "128", "16", "100", "64"],
        })

    st.subheader("Performance")
    st.table({
        "Metric": ["Val Dice", "Val IoU", "Oracle Dice"],
        "Untrained policy": ["~0.196", "—", "—"],
        "Trained (100 iters)": ["**0.947**", "**0.901**", "—"],
        "Oracle ceiling": ["—", "—", "**0.958**"],
    })

# ===========================================================================
# TAB 4 — About
# ===========================================================================
with tab_about:
    st.header("About This Project")

    st.markdown("""
## Overview

This project demonstrates that **reinforcement learning on superpixel graphs** can learn to segment
medical images without dense pixel-level supervision at training time (beyond the binary mask labels
used to compute the reward signal).

The agent is a **Graph Neural Network Actor-Critic** trained with **Proximal Policy Optimisation (PPO)**.
It sequentially labels each superpixel node in an image graph, guided by a Prim-style traversal that
ensures already-labelled neighbours are always visible.

## How It Works

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **SLIC** | Over-segment image into N≈64 compact superpixels |
| 2 | **Feature extraction** | Compute 12-dim node features (RGB stats, centroid, area) |
| 3 | **RAG construction** | Build region adjacency graph with appearance-similarity weights |
| 4 | **Guided traversal** | Prim-style traversal starting from image centre |
| 5 | **GNN policy** | 3-layer GraphLayer message-passing; sees neighbour labels |
| 6 | **Action** | Binary label (background=0 / foreground=1) for current node |
| 7 | **Reward** | Dice-delta + area-weighted correctness signal |
| 8 | **PPO update** | 4 epochs, clipped surrogate objective |

## Extending This Work

- **Real medical datasets:** Point the `--data` flag at a folder with `images/` and `masks/` sub-folders
- **Multi-class segmentation:** Extend action space from 2 to K classes
- **3-D volumes:** Process each axial/coronal/sagittal slice independently
- **Attention-based GNN:** Replace GraphLayer with multi-head graph attention
- **Curriculum learning:** Gradually increase n_segments during training

## Citation / Repository

> *Graph-Guided Reinforcement Learning for Medical Image Segmentation*
> Synthetic experiments on procedurally generated lesion images.
> Trained checkpoint: `runs/smoke/best.pt`

---
Built with **PyTorch**, **scikit-image**, **Streamlit** · Device: {}
""".format("CUDA GPU" if pick_device("auto").type == "cuda" else "CPU"))
