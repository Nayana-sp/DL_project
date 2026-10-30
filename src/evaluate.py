"""Evaluate a trained model on the val and test splits.

Usage (project folder, .venv active):
    python src/evaluate.py --name image_cnn
Writes metrics, per-class tables, confusion matrices, per-clip probabilities and figures.
The model is chosen by validation macro-F1 in train.py; the test split is only read here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from dataset import EMOTIONS, load_index
from train import build, predict_probs

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
METRICS = ROOT / "outputs" / "metrics"
FIGS = ROOT / "outputs" / "figures"


def report_split(name, split, ids, y, p, secs):
    labels = list(range(len(EMOTIONS)))
    pred = p.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"emotion": EMOTIONS, "precision": pr, "recall": rc, "f1": f1, "support": sup})
    cm = confusion_matrix(y, pred, labels=labels)
    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, pred, average="macro", labels=labels, zero_division=0))

    per_class.round(4).to_csv(METRICS / f"eval_{name}_{split}_per_class.csv", index=False)
    pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(METRICS / f"eval_{name}_{split}_confusion.csv")
    preds = pd.DataFrame({"sample_id": ids, "true": y, "pred": pred})
    for i, e in enumerate(EMOTIONS):
        preds[f"p_{e}"] = p[:, i]
    preds.to_csv(METRICS / f"eval_{name}_{split}_predictions.csv", index=False)

    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, xticklabels=EMOTIONS,
                yticklabels=EMOTIONS, ax=ax, cbar_kws={"label": "fraction of true class"})
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name} - {split} (acc {acc:.3f}, macro-F1 {mf1:.3f})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / f"confusion_{name}_{split}.png", dpi=110)
    plt.close(fig)

    print("=" * 60)
    print(f"{split.upper()}: {len(y)} clips | accuracy {acc:.3f} | macro-F1 {mf1:.3f} "
          f"| chance 0.125 | inference {1000 * secs / len(y):.1f} ms/clip (incl. data loading)")
    print(per_class.round(3).to_string(index=False))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.argsort(-off, axis=None)[:5]
    print("Most frequent confusions (true -> predicted):")
    for k in top:
        i, j = np.unravel_index(k, off.shape)
        if off[i, j] > 0:
            print(f"  {EMOTIONS[i]} -> {EMOTIONS[j]}: {off[i, j]} clips")
    return {"accuracy": acc, "macro_f1": mf1, "n_clips": int(len(y)),
            "inference_seconds_total": round(secs, 2),
            "inference_ms_per_clip": round(1000 * secs / len(y), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    best = CKPT / f"{a.name}_best.pt"
    if not best.exists():
        sys.exit(f"{best} not found. Train first with train.py --name {a.name}")
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    ck = torch.load(best, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    model, DS = build(targs.model, targs)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {best.name}: model={targs.model}, best epoch {ck['epoch']}, "
          f"val macro-F1 at selection {ck['val_macro_f1']:.3f}")

    summary = {"name": a.name, "model": targs.model, "best_epoch": ck["epoch"],
               "selection_val_macro_f1": ck["val_macro_f1"]}
    for split in ("val", "test"):
        loader = DataLoader(DS(index=load_index(split)), batch_size=a.batch_size, shuffle=False)
        t0 = time.time()
        ids, y, p = predict_probs(model, loader)
        summary[split] = report_split(a.name, split, ids, y, p, time.time() - t0)
    (METRICS / f"eval_{a.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 60)
    print("Report the TEST numbers. Validation was used to pick the epoch, so it is optimistic.")
    print(f"Saved eval_{a.name}_* files in outputs\\metrics and confusion figures in outputs\\figures")


if __name__ == "__main__":
    main()