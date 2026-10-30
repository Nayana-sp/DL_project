"""Diagnostic: accuracy per split and per actor for a trained model. No training, nothing is changed.

Usage (project folder, .venv active):
    python src/diagnose.py --name image_cnn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import EMOTIONS, load_index
from train import build, predict_probs

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    best = ROOT / "checkpoints" / f"{a.name}_best.pt"
    if not best.exists():
        sys.exit(f"{best} not found.")
    ck = torch.load(best, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    model, DS = build(targs.model, targs)
    model.load_state_dict(ck["model"])
    model.eval()

    frames = []
    for split in ("train", "val", "test"):
        idx = load_index(split)
        loader = DataLoader(DS(index=idx), batch_size=32, shuffle=False)
        ids, y, p = predict_probs(model, loader)
        df = pd.DataFrame({"sample_id": list(ids), "true": np.asarray(y), "pred": p.argmax(1)})
        df = df.merge(idx[["sample_id", "actor_id"]], on="sample_id")
        df["split"] = split
        df["correct"] = df["true"] == df["pred"]
        frames.append(df)
        print(f"{split:5s}: {len(df)} clips | accuracy {df['correct'].mean():.3f}")
        counts = df["pred"].value_counts().reindex(range(len(EMOTIONS)), fill_value=0)
        print("   predicted class counts:", {EMOTIONS[i]: int(c) for i, c in counts.items()})

    allp = pd.concat(frames)
    per_actor = (allp.groupby(["split", "actor_id"])["correct"].mean().round(3).reset_index()
                 .rename(columns={"correct": "accuracy"}))
    print("\nAccuracy per actor (chance = 0.125):")
    print(per_actor.to_string(index=False))
    out = ROOT / "outputs" / "metrics" / f"diagnose_{a.name}_per_actor.csv"
    per_actor.to_csv(out, index=False)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()