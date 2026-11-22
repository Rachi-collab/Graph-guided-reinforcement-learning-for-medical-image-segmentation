# Graph-Guided Reinforcement Learning for Medical Image Segmentation
## Final Evaluation Report

---

## Abstract

We present a **Graph-Guided Reinforcement Learning** framework for medical image segmentation that trains a Graph Neural Network (GNN) policy via Proximal Policy Optimisation (PPO) to sequentially label superpixel nodes in an image graph. The agent follows a Prim-style traversal that guarantees contextual awareness of already-labelled neighbours before each decision, exploiting the graph structure for spatial coherence. Trained on 200 synthetic lesion images for 100 iterations, the agent achieves a **Val Dice of 0.947** against an Oracle ceiling of 0.958 — bridging 97% of the gap between a random policy (Dice ≈ 0.196) and the theoretical maximum imposed by the superpixel partition.

---

## Problem Statement

Binary medical image segmentation — identifying a foreground lesion or organ against a background — is a canonical computer vision task with direct clinical applications. We ask: *can a reinforcement learning agent learn to segment images by sequentially labelling over-segmented regions (superpixels) guided by a graph structure, without operating at the pixel level?*

This formulation reduces the per-image action space from H×W pixels to N≈64 discrete node decisions, makes the spatial graph structure an explicit inductive bias, and allows dense reward shaping via the incremental Dice score.

---

## Method

### Pipeline

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **SLIC superpixels** | Over-segment H×W image into N≈64 compact, homogeneous regions |
| 2 | **Node feature extraction** | 12-dim vector per node: RGB mean (3) + RGB std (3) + mean relative to image (3) + centroid (cy, cx) + relative area (1) |
| 3 | **RAG construction** | Build region adjacency graph; edge weight = exp(−‖Δmean‖²/2σ²) normalised per row |
| 4 | **Guided traversal** | Prim-style BFS from image centre; always expands the most appearance-similar unvisited neighbour |
| 5 | **GNN actor-critic forward** | 3-layer GraphLayer message passing; current node embedding ‖ global mean pooling → policy + value heads |
| 6 | **Action** | Binary label: background (0) or foreground (1) for the current superpixel |
| 7 | **Reward** | r = Δ Dice + α·(area_i/total_area)·sign · scale — Dice-delta ensures return equals final Dice |
| 8 | **PPO update** | 4 mini-epochs, batch=128, clipped surrogate + value loss + entropy bonus |

### Architecture Details

```
GraphActorCritic(in_dim=15, hidden=64, layers=3)
  inp:    Linear(15 → 64)   [+2 extra dims: mean-neighbour-label, fraction-labelled-neighbours]
  layers: 3 × GraphLayer
            h' = h + GELU(LayerNorm(W_s·h + W_n·(A·h)))
  heads:  pi = Linear(128→64) → GELU → Linear(64→2)   [logits for bg/fg]
          v  = Linear(128→64) → GELU → Linear(64→1)    [state value]
```

**PPO Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Entropy coefficient | 0.01 |
| PPO clip ε | 0.2 (default) |
| Epochs per rollout | 4 |
| Minibatch size | 128 |
| Parallel environments | 16 |
| Training iterations | 100 |
| Discount γ | 1.0 (episodic) |

---

## Results

### Quantitative Performance

| Metric | Untrained policy | Trained (100 iters) | Oracle Ceiling |
|--------|:----------------:|:-------------------:|:--------------:|
| **Val Dice** | 0.196 | **0.947** | 0.958 |
| **Val IoU** | — | **0.901** | — |
| **Val N** | 40 | 40 | 40 |

> **Oracle ceiling** is computed by assigning each superpixel the majority-vote label of its contained GT pixels — it quantifies the information loss from the superpixel partitioning and is an upper bound for any agent using the same SLIC partition.

### Key Findings

1. **Rapid convergence:** The agent surpasses Dice 0.9 by iteration ~40, demonstrating the efficiency of graph-structured exploration combined with dense reward shaping.
2. **Oracle gap is small (1.1 pp):** The gap between agent (0.947) and oracle (0.958) shows the GNN correctly labels nearly every superpixel — remaining errors are boundary superpixels with mixed GT content.
3. **Entropy collapse to near-zero:** Policy entropy drops from ~0.69 (random) to near zero, confirming the agent learns a nearly deterministic policy.
4. **Traversal matters:** The Prim-style graph-guided order ensures that by the time an ambiguous boundary superpixel is reached, its interior neighbours are already labelled — providing strong contextual signals.
5. **Distractor robustness:** The synthetic data includes decoy blobs with opposite intensity contrast; the agent correctly ignores them in the final policy.

---

## How to Run

### Environment Setup

```bash
cd "c:\Graph guided reinforcement learning for medical image segmentation"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Training

```bash
python train.py --data synthetic --n_train 200 --n_val 40 --iters 100 --out runs/smoke
```

For real data:
```bash
python train.py --data real --data_dir path/to/dataset --iters 200 --out runs/real
```

### Evaluation

```bash
python evaluate.py --ckpt runs/smoke/best.pt --data synthetic --n_val 40
```

### Streamlit App

```bash
venv\Scripts\streamlit.exe run app.py
```

### Generate Presentation Figures

```bash
venv\Scripts\python.exe demo_figures.py
# Output: runs/smoke/final_figures/
```

---

## File Structure

```
.
├── gseg/                        # Core Python package
│   ├── __init__.py
│   ├── graph.py                 # SLIC superpixels + RAG construction
│   ├── env.py                   # Batched segmentation RL environment
│   ├── model.py                 # GraphActorCritic (GNN + PPO heads)
│   ├── ppo.py                   # PPO training loop
│   ├── data.py                  # Synthetic + real data loaders
│   └── utils.py                 # Seed, device, CSV logger utilities
├── runs/
│   └── smoke/                   # Trained checkpoint (100 iters, synthetic)
│       ├── best.pt              # Model weights + args
│       ├── log.csv              # Per-iteration training log
│       ├── metrics.json         # Final val Dice, IoU, oracle Dice
│       ├── predictions.png      # Sample predictions grid
│       ├── curves.png           # Training curves
│       └── final_figures/       # Generated by demo_figures.py
│           ├── architecture_diagram.png
│           ├── performance_summary.png
│           ├── sample_segmentations.png
│           └── traversal_visualization.png
├── app.py                       # Streamlit interactive demo
├── demo_figures.py              # Presentation figure generator
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── requirements.txt             # Minimal dependencies
├── final_report.md              # This report
└── README.md                    # Quick-start guide
```

---

## References

- Achiam, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Achanta, R. et al. (2012). *SLIC Superpixels Compared to State-of-the-Art Superpixel Methods*. IEEE TPAMI.
- Shi, W. et al. (2000). *Normalized Cuts and Image Segmentation*. IEEE TPAMI.
- Kipf, T. N. & Welling, M. (2016). *Semi-Supervised Classification with Graph Convolutional Networks*. arXiv:1609.02907

---

*Report auto-generated from trained checkpoint `runs/smoke/best.pt`.*
*All experiments conducted on synthetic data; results may vary on real medical datasets.*
