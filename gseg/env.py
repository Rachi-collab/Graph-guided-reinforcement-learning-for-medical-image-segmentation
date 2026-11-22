"""Batched segmentation environment on superpixel graphs.

Episode  : one image. The agent labels every superpixel (node) exactly once.
Order    : graph-guided traversal (see graph._guided_order), so already-labelled neighbours
           are visible when a node is decided.
State    : node features + [is_labelled, label, is_current] per node, plus the graph.
Action   : {0 = background, 1 = foreground} for the current node.
Reward   : delta of the global Dice of the painted mask  +  area-weighted per-node correctness.
           (Dice term makes the return equal the final Dice; correctness term is dense shaping.)
"""
import numpy as np
import torch


class BatchedSegEnv:
    def __init__(self, graphs, device, acc_coef: float = 1.0, reward_scale: float = 10.0):
        self.device = device
        self.B = len(graphs)
        self.N = max(g["N"] for g in graphs)
        B, N = self.B, self.N
        F0 = graphs[0]["x0"].shape[1]

        x0 = np.zeros((B, N, F0), np.float32)
        adj = np.zeros((B, N, N), np.float32)
        order = np.zeros((B, N), np.int64)
        frac = np.zeros((B, N), np.float32)
        area = np.zeros((B, N), np.float32)
        valid = np.zeros((B, N), np.float32)
        n = np.zeros(B, np.int64)
        for i, g in enumerate(graphs):
            k = g["N"]
            x0[i, :k] = g["x0"]
            adj[i, :k, :k] = g["adj"]
            order[i, :k] = g["order"]
            frac[i, :k] = g["frac"]
            area[i, :k] = g["area"]
            valid[i, :k] = 1
            n[i] = k
        t = lambda a: torch.from_numpy(a).to(device)
        self.x0, self.adj, self.order, self.frac = t(x0), t(adj), t(order), t(frac)
        self.area, self.valid, self.n = t(area), t(valid), t(n)
        self.ar = torch.arange(B, device=device)

        self.acc_coef, self.reward_scale = acc_coef, reward_scale
        self.gt_area = (self.frac * self.area).sum(1)  # |G| (pixels)
        self.tot_area = self.area.sum(1)
        self.max_steps = N
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        B, N = self.B, self.N
        self.t = 0
        self.labeled = torch.zeros(B, N, device=self.device)
        self.label = torch.zeros(B, N, device=self.device)
        self.tp = torch.zeros(B, device=self.device)  # |P ∩ G|
        self.pa = torch.zeros(B, device=self.device)  # |P|
        return self.obs()

    def _cur_idx(self):
        return self.order[:, min(self.t, self.N - 1)]

    def obs(self):
        cur = self._cur_idx()
        is_cur = torch.zeros_like(self.labeled)
        active = (self.t < self.n).float()
        is_cur[self.ar, cur] = active
        x = torch.cat([self.x0, self.labeled[..., None], self.label[..., None], is_cur[..., None]], -1)
        return x, cur

    def step_valid_mask(self):
        """[T, B] bool: True where step t is a real decision for env b."""
        t = torch.arange(self.max_steps, device=self.device)[:, None]
        return t < self.n[None, :]

    # ------------------------------------------------------------------ metrics
    def dice(self):
        return 2 * self.tp / (self.pa + self.gt_area + 1e-6)

    def iou(self):
        return self.tp / (self.pa + self.gt_area - self.tp + 1e-6)

    # ------------------------------------------------------------------ dynamics
    def step(self, action: torch.Tensor):
        ar, idx = self.ar, self._cur_idx()
        active = (self.t < self.n).float()
        lab = action.float() * active
        a_i, f_i = self.area[ar, idx], self.frac[ar, idx]

        d_old = self.dice()
        self.labeled[ar, idx] = torch.where(active > 0, torch.ones_like(active), self.labeled[ar, idx])
        self.label[ar, idx] = torch.where(active > 0, lab, self.label[ar, idx])
        self.tp = self.tp + lab * f_i * a_i
        self.pa = self.pa + lab * a_i
        d_new = self.dice()

        signed = (2 * lab - 1) * (2 * f_i - 1)
        r = (d_new - d_old) + self.acc_coef * (a_i / self.tot_area) * signed
        r = r * active * self.reward_scale
        self.t += 1
        return r
