import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._fields = None

    def log(self, row: dict):
        new = self._fields is None
        if new:
            self._fields = list(row.keys())
        with open(self.path, "a" if not new else "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self._fields})
