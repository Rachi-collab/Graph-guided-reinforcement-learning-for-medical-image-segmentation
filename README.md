# Graph-Guided Reinforcement Learning for Medical Image Segmentation

A hybrid **Graph Neural Network + Proximal Policy Optimisation (PPO)** framework that learns to segment lesions in medical images by navigating a superpixel graph — no pixel-level annotation required at inference time.

---

## 🏆 Results (CVC-ClinicDB · 612 colonoscopy images)

| Run | Iters | Val Dice | Val IoU | Oracle Dice |
|-----|------:|--------:|--------:|------------:|
| `polyp_v2` (baseline PPO) | 1 000 | **0.476** | 0.360 | 0.865 |
| `polyp_finetune` (augmentation + lower LR) | 300 | **0.366** | 0.265 | 0.865 |
| `smoke` (synthetic, sanity) | 500 | 0.947 | 0.901 | 0.958 |

> **Oracle ceiling** = 0.865 on 200-superpixel partition (the graph resolution, not the agent, is the bottleneck on CPU).

Planned GPU upgrades: **GAT layers** (`--use_gat`) and **CNN node features** (`--cnn_features`, ResNet-18 → 140-dim) are implemented and verified; full training projected to reach Dice **0.55–0.60**.

---

## 🚀 Run the Demo

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port 8501
```

Open `http://localhost:8501`. Use `runs/smoke/best.pt` for a synthetic demo (Dice 0.947) or `runs/polyp_v2/best.pt` for the real-data model.

---

## Project Overview

Standard segmentation methods (U-Net) predict masks pixel-by-pixel, ignoring structural relationships and requiring massive labeled datasets.

This project implements a hybrid **GNN + RL** framework:
1. **Graph Construction** — SLIC superpixels (200 nodes/image) form an adjacency graph; nodes carry 12-dim hand-crafted features (colour, texture, position).
2. **GNN Representation** — A multi-layer `GraphLayer` (or optional `GATLayer` with 4 attention heads) encodes structural context.
3. **PPO Agent** — Navigates the graph, toggling nodes in/out of the segmentation mask; rewarded by Δ Dice at each step.
4. **CNN Node Features** (optional, `--cnn_features`) — Frozen ResNet-18 `layer2` pooled per superpixel raises node dim from 12 → 140.

---

## Repository Structure
```directory
graph guided reinforcement learning for medical image segmentation/
├── notebooks/            # Jupyter notebooks for EDA and rapid prototyping
│   ├── 01_eda.ipynb      # Medical image format (NIfTI/DICOM) exploration
│   └── 02_env_test.ipynb # Prototyping custom RL environment
├── src/                  # Main source code
│   ├── data/             # Data loading, preprocessing, and graph construction
│   ├── env/              # Custom Gymnasium RL environments (medical image graphs)
│   ├── models/           # GNN backbones and RL policy architectures
│   ├── training/         # Pipelines for training GNN and RL agent
│   └── evaluation/       # Performance evaluation metrics (Dice, Jaccard, Hausdorff)
├── requirements.txt      # Python dependencies
└── README.md             # Project roadmap and documentation
```

---

## 13-Week Detailed Implementation Roadmap

### Phase 1: Foundation, Data Pipeline & EDA (Weeks 1–3)
*Focus: Medical data handling, preprocessing, and building the graph representation.*
* **Week 1: Literature Review & Setup**
  - Finalize dataset (e.g., **BraTS** for brain tumors or **Decathlon-Spleen** for organ segmentation).
  - Setup workspace, git workflow, and Python virtual environment using `requirements.txt`.
  - Explore NIfTI (`.nii.gz`) data loading using `nibabel` and `SimpleITK`.
* **Week 2: Data Preprocessing Pipeline**
  - Implement 3D medical volume preprocessing: intensity normalization, voxel resampling, and cropping.
  - Implement 3D patch extraction for memory-efficient loading.
  - Deliverable: A reproducible preprocessing script in `src/data/preprocess.py`.
* **Week 3: Anatomical Graph Construction**
  - Segment scans into superpixels/supervoxels (using SLIC or similar algorithms).
  - Construct spatial adjacency graphs where nodes = superpixels, and edges = adjacent regions.
  - Deliverable: Graph creation script (`src/data/graph_generator.py`) saving PyTorch Geometric (`torch_geometric`) data objects.

### Phase 2: Representation Learning with GNNs (Weeks 4–6)
*Focus: Feature extraction and structural graph encoding.*
* **Week 4: Feature Engineering & Backbone Selection**
  - Extract local node features (mean intensity, texture histograms, spatial coordinates).
  - Design a GNN backbone (Graph Attention Network (GAT) or GraphSAGE) in `src/models/gnn.py`.
* **Week 5: Self-Supervised Node Representation Learning**
  - Train the GNN to learn robust node representations using contrastive learning or reconstruction loss.
  - Ensure the model can map similar tissue types close to each other in the latent space.
* **Week 6: GNN Evaluation & Graph Visualization**
  - Visualize node embeddings using t-SNE or UMAP to verify tissue clustering.
  - Deliverable: Visual analytics notebook `notebooks/03_embedding_analysis.ipynb`.

### Phase 3: Custom RL Environment & Agent Design (Weeks 7–9)
*Focus: Formulating the segmentation problem as a Markov Decision Process (MDP).*
* **Week 7: Custom Gymnasium Environment**
  - Create a custom `gymnasium.Env` class in `src/env/segmentation_env.py`.
  - **State Space**: Node embeddings + current selection mask.
  - **Action Space**: Selecting the next neighboring node to include in the segmentation.
  - **Reward**: Change in Dice Coefficient between time $t$ and $t-1$.
* **Week 8: Reinforcement Learning Agent Implementation**
  - Implement an RL policy (such as Proximal Policy Optimization (PPO) or Deep Q-Network (DQN)) optimized for graph inputs.
  - Integrate Gymnasium with Stable-Baselines3 or write a custom PyTorch agent in `src/models/rl_agent.py`.
* **Week 9: Reward Shaping & Single-Scan Verification**
  - Shape the reward function to penalize false positives and guide the agent toward boundaries.
  - Overfit the RL agent on a single scan to prove the MDP formulation works and the agent converges.

### Phase 4: Joint Training, Optimization & Baselines (Weeks 10–11)
*Focus: Scalability, hyperparameter optimization, and benchmarks.*
* **Week 10: Scaled Training & Tuning**
  - Train the agent end-to-end on the training dataset split.
  - Run hyperparameter sweeps (learning rate, discount factor $\gamma$, GNN depth) using weights & biases (`wandb`).
* **Week 11: Benchmark Evaluation**
  - Implement baseline methods: a Random Agent (in `baselines/random_agent.py`) and a standard 3D U-Net.
  - Evaluate on unseen test scans using Dice Score, Jaccard Index, and Hausdorff Distance.
  - Deliverable: A comprehensive comparison table of results.

### Phase 5: Demo Deployment & Portfolio Preparation (Weeks 12–13)
*Focus: Showcasing the project for placements and academic evaluations.*
* **Week 12: Interactive Web Demonstration**
  - Create a lightweight UI using **Streamlit** or **Gradio**.
  - Allow users to upload a 3D medical slice, select a starting "seed" node, and watch the RL agent incrementally expand and segment the organ/tumor step-by-step.
* **Week 13: Project Report, Slide Deck & Code Walkthrough**
  - Write a high-quality README summary showing system architecture diagrams.
  - Document performance metrics, sample segmentations, and hyperparameter results.
  - Compile the final report for the 4-credit academic submission and record a 2-minute video demo for LinkedIn/GitHub.
