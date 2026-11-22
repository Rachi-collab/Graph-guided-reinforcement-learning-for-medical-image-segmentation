"""Train a graph-guided PPO agent for binary medical image segmentation.

Examples
  python train.py --data synthetic --iters 300
  python train.py --data folder --data_dir /path/to/dataset --size 128 --iters 1000
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from gseg.data import get_graphs
from gseg.env import BatchedSegEnv
from gseg.graph import NODE_FEAT_DIM
from gseg.model import GraphActorCritic
from gseg.ppo import PPOConfig, collect_rollout, evaluate_policy, oracle_dice, ppo_update
from gseg.utils import CSVLogger, pick_device, set_seed


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "folder"], default="synthetic")
    p.add_argument("--data_dir", default=None, help="folder with images/ and masks/ (for --data folder)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_val", type=int, default=60)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--n_segments", type=int, default=64)
    p.add_argument("--compactness", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--num_envs", type=int, default=16, help="images per PPO rollout")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/exp1")
    return p.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device: {device}")

    train_g, val_g = get_graphs(args)
    print(f"oracle superpixel Dice (upper bound): train {oracle_dice(train_g):.3f} | val {oracle_dice(val_g):.3f}")

    model = GraphActorCritic(NODE_FEAT_DIM + 3, args.hidden, args.layers).to(device)
    cfg = PPOConfig(lr=args.lr, epochs=args.epochs, minibatch=args.minibatch, ent_coef=args.ent_coef)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    log = CSVLogger(os.path.join(args.out, "log.csv"))
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)

    r0 = evaluate_policy(model, val_g, device)
    print(f"untrained policy val Dice: {r0['dice']:.3f}")

    rng = np.random.default_rng(args.seed)
    best, t0 = -1.0, time.time()
    for it in range(1, args.iters + 1):
        idx = rng.choice(len(train_g), args.num_envs, replace=len(train_g) < args.num_envs)
        env = BatchedSegEnv([train_g[i] for i in idx], device)
        buf = collect_rollout(model, env, cfg)
        train_dice = env.dice().mean().item()
        st = ppo_update(model, opt, buf, env, cfg)
        row = dict(iter=it, train_dice=train_dice, **st, val_dice="", val_iou="", time=time.time() - t0)

        if it % args.eval_every == 0 or it == args.iters:
            r = evaluate_policy(model, val_g, device)
            row["val_dice"], row["val_iou"] = r["dice"], r["iou"]
            tag = ""
            if r["dice"] > best:
                best = r["dice"]
                torch.save(dict(model=model.state_dict(), args=vars(args)), os.path.join(args.out, "best.pt"))
                tag = " *"
            print(f"it {it:4d} | train Dice {train_dice:.3f} | val Dice {r['dice']:.3f} IoU {r['iou']:.3f} "
                  f"| ent {st['ent']:.3f} kl {st['kl']:.4f} | {time.time() - t0:.0f}s{tag}")
        log.log(row)

    torch.save(dict(model=model.state_dict(), args=vars(args)), os.path.join(args.out, "last.pt"))
    print(f"done. best val Dice {best:.3f}. Checkpoints + log.csv in {args.out}")


if __name__ == "__main__":
    main()
