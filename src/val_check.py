"""Why does a saved checkpoint's validation score differ a little from the score in the training log?

Re-scores one checkpoint on the validation split under several conditions. No training, no test split.
Works for checkpoints trained with the --bn-recal option (image_cnn_v3 and later).

Usage (project folder, .venv active):
    python src/val_check.py --name early_cnn
"""
import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import EMOTIONS, load_index
from train import build, predict_probs, recalibrate_bn

ROOT = Path(__file__).resolve().parents[1]


def val_scores(model, DS, val_idx, batch_size):
    _, y, p = predict_probs(model, DataLoader(DS(index=val_idx), batch_size=batch_size, shuffle=False))
    pred = p.argmax(1)
    f1 = f1_score(y, pred, average="macro", labels=list(range(len(EMOTIONS))), zero_division=0)
    return f1, float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    path = ROOT / "checkpoints" / f"{a.name}_best.pt"
    if not path.exists():
        sys.exit(f"{path} not found.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    model, DS = build(targs.model, targs)
    model.load_state_dict(ck["model"])
    model.eval()
    val_idx = load_index("val")

    print(f"{path.name}: epoch {ck['epoch']} | val macro-F1 stored at selection: {ck['val_macro_f1']:.4f}")
    for threads in (10, 12):
        torch.set_num_threads(threads)
        for bs in (16, 32):
            f1, acc = val_scores(model, DS, val_idx, bs)
            print(f"as saved              | threads {threads} | batch {bs:2d} | val macro-F1 {f1:.4f} | acc {acc:.4f}")

    torch.set_num_threads(10)
    tr_idx = load_index("train")
    recal_idx = tr_idx.sample(min(targs.bn_recal, len(tr_idx)), random_state=targs.seed)
    recalibrate_bn(model, DataLoader(DS(index=recal_idx), batch_size=targs.batch_size, shuffle=False))
    f1, acc = val_scores(model, DS, val_idx, targs.batch_size)
    print(f"training recal re-run | threads 10 | batch {targs.batch_size:2d} | val macro-F1 {f1:.4f} | acc {acc:.4f}")


if __name__ == "__main__":
    main()