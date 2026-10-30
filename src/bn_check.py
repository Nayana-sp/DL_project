"""Check whether BatchNorm running statistics explain the erratic validation scores.

Scores a saved checkpoint on the train and val clips twice: as saved, and after re-estimating the
BatchNorm statistics on un-augmented training clips. No weights are changed and nothing is saved.
The test split is not used.

Usage (project folder, .venv active):
    python src/bn_check.py --name image_cnn_v2 --which last
    python src/bn_check.py --name image_cnn_v2 --which best
"""
import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import EMOTIONS, load_index
from train import build, predict_probs

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"


def scores(model, DS, idx):
    loader = DataLoader(DS(index=idx), batch_size=32, shuffle=False)
    _, y, p = predict_probs(model, loader)
    pred = p.argmax(1)
    f1 = f1_score(y, pred, average="macro", labels=list(range(len(EMOTIONS))), zero_division=0)
    return accuracy_score(y, pred), f1


@torch.no_grad()
def recalibrate(model, DS, train_idx):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # None = plain average over all batches
    model.train()
    for x, _, _ in DataLoader(DS(index=train_idx), batch_size=32, shuffle=False):
        model(x)
    model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--which", choices=["last", "best"], default="last")
    a = ap.parse_args()

    best = CKPT / f"{a.name}_best.pt"
    path = CKPT / f"{a.name}_{a.which}.pt"
    if not best.exists() or not path.exists():
        sys.exit(f"Checkpoint not found: {best.name} or {path.name}")
    targs = argparse.Namespace(**torch.load(best, map_location="cpu", weights_only=False)["args"])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    epoch = ck["epoch"] if "epoch" in ck else ck["state"]["epoch"]

    model, DS = build(targs.model, targs)
    model.load_state_dict(ck["model"])
    model.eval()
    train_idx, val_idx = load_index("train"), load_index("val")

    print(f"{path.name}: epoch {epoch}")
    for step, tag in enumerate(("as saved", "BN re-estimated")):
        if step == 1:
            recalibrate(model, DS, train_idx)
        tr, va = scores(model, DS, train_idx), scores(model, DS, val_idx)
        print(f"{tag:16s} | train acc {tr[0]:.3f} F1 {tr[1]:.3f} | val acc {va[0]:.3f} F1 {va[1]:.3f}")


if __name__ == "__main__":
    main()