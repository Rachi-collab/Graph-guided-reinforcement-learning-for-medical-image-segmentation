"""PPO (clipped objective, GAE) for the batched graph segmentation environment."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import BatchedSegEnv


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 128
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@torch.no_grad()
def collect_rollout(model, env: BatchedSegEnv, cfg: PPOConfig):
    T, B = env.max_steps, env.B
    xs, curs, acts, lps, vals, rews = [], [], [], [], [], []
    env.reset()
    for _ in range(T):
        x, cur = env.obs()
        logits, v = model(x, env.adj, env.valid, cur)
        dist = Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)
        r = env.step(a)
        xs.append(x); curs.append(cur); acts.append(a); lps.append(lp); vals.append(v); rews.append(r)
    buf = {k: torch.stack(v) for k, v in dict(x=xs, cur=curs, act=acts, logp=lps, val=vals, rew=rews).items()}
    buf["valid"] = env.step_valid_mask()
    buf["adv"], buf["ret"] = compute_gae(buf["rew"], buf["val"], buf["valid"], cfg.gamma, cfg.lam)
    return buf


def compute_gae(rew, val, valid, gamma, lam):
    T, B = rew.shape
    adv = torch.zeros_like(rew)
    last = torch.zeros(B, device=rew.device)
    for t in reversed(range(T)):
        if t + 1 < T:
            nonterm, nextv = valid[t + 1].float(), val[t + 1]
        else:
            nonterm, nextv = torch.zeros(B, device=rew.device), torch.zeros(B, device=rew.device)
        delta = rew[t] + gamma * nextv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def ppo_update(model, opt, buf, env: BatchedSegEnv, cfg: PPOConfig):
    tt, bb = buf["valid"].nonzero(as_tuple=True)
    adv_all = buf["adv"][tt, bb]
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
    stats = dict(pi=[], v=[], ent=[], kl=[], clipfrac=[])
    M = tt.numel()
    for _ in range(cfg.epochs):
        perm = torch.randperm(M, device=tt.device)
        for s in range(0, M, cfg.minibatch):
            ix = perm[s : s + cfg.minibatch]
            t_i, b_i = tt[ix], bb[ix]
            logits, v = model(buf["x"][t_i, b_i], env.adj[b_i], env.valid[b_i], buf["cur"][t_i, b_i])
            dist = Categorical(logits=logits)
            lp = dist.log_prob(buf["act"][t_i, b_i])
            ratio = torch.exp(lp - buf["logp"][t_i, b_i])
            a = adv_all[ix]
            pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            v_loss = F.mse_loss(v, buf["ret"][t_i, b_i])
            ent = dist.entropy().mean()
            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["pi"].append(pi_loss.item()); stats["v"].append(v_loss.item()); stats["ent"].append(ent.item())
                stats["kl"].append((buf["logp"][t_i, b_i] - lp).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


@torch.no_grad()
def evaluate_policy(model, graphs, device, batch: int = 32, greedy: bool = True, return_labels: bool = False):
    """Runs the policy on `graphs`; returns mean pixel Dice / IoU (exact, computed from superpixel areas)."""
    model.eval()
    dices, ious, labels = [], [], []
    for s in range(0, len(graphs), batch):
        env = BatchedSegEnv(graphs[s : s + batch], device)
        for _ in range(env.max_steps):
            x, cur = env.obs()
            logits, _ = model(x, env.adj, env.valid, cur)
            a = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
            env.step(a)
        dices += env.dice().tolist()
        ious += env.iou().tolist()
        if return_labels:
            labels += [env.label[i, : env.n[i]].cpu().numpy() for i in range(env.B)]
    model.train()
    out = dict(dice=float(np.mean(dices)), iou=float(np.mean(ious)), per_image_dice=dices)
    if return_labels:
        out["labels"] = labels
    return out


def oracle_dice(graphs):
    """Upper bound reachable with this superpixel partition (label = majority GT of each node)."""
    out = []
    for g in graphs:
        lab = (g["frac"] > 0.5).astype(np.float64)
        tp = (lab * g["frac"] * g["area"]).sum()
        pa = (lab * g["area"]).sum()
        ga = (g["frac"] * g["area"]).sum()
        out.append(2 * tp / (pa + ga + 1e-6))
    return float(np.mean(out))
