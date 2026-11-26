"""Evaluate a trained checkpoint on the validation split, save metrics + qualitative overlays + training curves.

python evaluate.py --ckpt runs/exp1/best.pt
"""

import argparse
import json
import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import mark_boundaries

from gseg.data import get_graphs
from gseg.graph import NODE_FEAT_DIM, NODE_FEAT_DIM_CNN
from gseg.model import GraphActorCritic
from gseg.ppo import evaluate_policy, oracle_dice
from gseg.utils import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_vis", type=int, default=6)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = pick_device(a.device)
    ck = torch.load(a.ckpt, map_location=device)
    args = SimpleNamespace(**ck["args"])
    out = os.path.dirname(a.ckpt)

    _, val_g = get_graphs(args)
    node_feat_dim = NODE_FEAT_DIM_CNN if args.cnn_features else NODE_FEAT_DIM
    model = GraphActorCritic(
        node_feat_dim + 3,
        args.hidden,
        args.layers,
        use_gat=args.use_gat,
        gat_heads=args.gat_heads,
    ).to(device)
    model.load_state_dict(ck["model"])

    r = evaluate_policy(model, val_g, device, return_labels=True)
    metrics = dict(
        val_dice=r["dice"],
        val_iou=r["iou"],
        oracle_dice=oracle_dice(val_g),
        n_val=len(val_g),
    )
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(metrics)

    # qualitative results: image | ground truth | prediction (green = TP, red = FP, blue = FN)
    k = min(a.n_vis, len(val_g))
    fig, ax = plt.subplots(k, 4, figsize=(11, 2.8 * k))
    ax = np.atleast_2d(ax)
    for i in range(k):
        g, lab = val_g[i], r["labels"][i]
        pred = lab[g["seg"]] > 0.5
        gt = g["mask"] > 0
        ov = np.zeros((*pred.shape, 3))
        ov[pred & gt] = (0, 0.8, 0)
        ov[pred & ~gt] = (0.9, 0, 0)
        ov[~pred & gt] = (0, 0.3, 1)
        ax[i, 0].imshow(g["img"])
        ax[i, 1].imshow(mark_boundaries(g["img"], g["seg"], color=(1, 1, 0)))
        ax[i, 2].imshow(gt, cmap="gray")
        ax[i, 3].imshow(ov)
        ax[i, 3].set_title(f"Dice {r['per_image_dice'][i]:.2f}", fontsize=9)
        if i == 0:
            for j, t in enumerate(
                [
                    "image",
                    "superpixel graph nodes",
                    "ground truth",
                    "prediction (G=TP R=FP B=FN)",
                ]
            ):
                ax[0, j].set_title(
                    t if j < 3 else t + f"\nDice {r['per_image_dice'][0]:.2f}",
                    fontsize=9,
                )
        for j in range(4):
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "predictions.png"), dpi=110)
    plt.close()

    # training curves
    log = os.path.join(out, "log.csv")
    if os.path.exists(log):
        import csv

        rows = list(csv.DictReader(open(log)))
        it = [int(x["iter"]) for x in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(
            it, [float(x["train_dice"]) for x in rows], label="train (stochastic)"
        )
        ev = [
            (int(x["iter"]), float(x["val_dice"])) for x in rows if x["val_dice"] != ""
        ]
        ax[0].plot(*zip(*ev), "o-", label="val (greedy)")
        ax[0].axhline(
            metrics["oracle_dice"], ls="--", c="gray", label="superpixel oracle"
        )
        ax[0].set_xlabel("PPO iteration")
        ax[0].set_ylabel("Dice")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(it, [float(x["ent"]) for x in rows])
        ax[1].set_xlabel("PPO iteration")
        ax[1].set_ylabel("policy entropy")
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "curves.png"), dpi=110)
        plt.close()
    print(f"saved metrics.json, predictions.png, curves.png in {out}")


if __name__ == "__main__":
    main()
