"""Phase 5: actor-independent train/val/test split.

Usage (project folder, .venv active):
    python src/make_splits.py                 # default seed 42, 2+2 val, 2+2 test actors
Writes data/processed/splits.csv (per actor) and dataset_index.csv (per clip).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-per-gender", type=int, default=2)
    ap.add_argument("--test-per-gender", type=int, default=2)
    a = ap.parse_args()

    meta = pd.read_csv(PROC / "metadata.csv")
    meta = meta[meta["selected"] == True].drop(columns=["audio_path", "preprocessing_status"])  # noqa: E712
    prep = pd.read_csv(PROC / "preprocess_results.csv")
    prep = prep[prep["preprocessing_status"] == "ok"][["sample_id", "audio_path", "frames_path"]]
    df = meta.merge(prep, on="sample_id", how="inner")
    if len(df) != len(meta):
        sys.exit(f"{len(meta) - len(df)} selected clips have no preprocessed files")

    # RAVDESS documentation: odd actor IDs are male, even are female
    df["gender"] = np.where(df["actor_id"] % 2 == 1, "male", "female")

    rng = np.random.default_rng(a.seed)
    actor_split = {}
    for g in ("male", "female"):
        actors = sorted(df.loc[df["gender"] == g, "actor_id"].unique().tolist())
        perm = [int(x) for x in rng.permutation(actors)]
        t, v = a.test_per_gender, a.val_per_gender
        if t + v >= len(actors):
            sys.exit("not enough actors for this split size")
        for x in perm[:t]:
            actor_split[x] = "test"
        for x in perm[t:t + v]:
            actor_split[x] = "val"
        for x in perm[t + v:]:
            actor_split[x] = "train"

    df["split"] = df["actor_id"].map(actor_split)

    # leakage checks
    assert df["split"].notna().all(), "clip without split"
    assert df.groupby("actor_id")["split"].nunique().max() == 1, "actor in more than one split"
    assert df["sample_id"].is_unique, "duplicate sample_id"

    splits = (df.groupby("actor_id")
                .agg(gender=("gender", "first"), split=("split", "first"), clips=("sample_id", "count"))
                .reset_index())
    splits.to_csv(PROC / "splits.csv", index=False)

    cols = ["sample_id", "video_path", "audio_path", "frames_path", "recording_type", "emotion_id",
            "emotion", "intensity", "statement", "repetition", "actor_id", "gender", "split"]
    df = df[cols].assign(preprocessing_status="ok").sort_values("sample_id")
    df.to_csv(PROC / "dataset_index.csv", index=False)
    (PROC / "split_config.json").write_text(json.dumps(vars(a), indent=2))

    print("Actors per split:")
    for s in ("train", "val", "test"):
        ids = sorted(splits.loc[splits.split == s, "actor_id"])
        print(f"  {s:5s} ({len(ids)}): {ids}")
    print("\nClips per split:\n" + df["split"].value_counts().to_string())
    print("\nGender per split (actors):\n" + pd.crosstab(splits["split"], splits["gender"]).to_string())
    tab = pd.crosstab(df["emotion"], df["split"]).reindex(EMOTIONS)[["train", "val", "test"]]
    print("\nClips per emotion and split:\n" + tab.to_string())
    print("\nLeakage checks passed: no actor in two splits, every clip has one split.")
    print("Saved splits.csv, dataset_index.csv, split_config.json in data/processed/")


if __name__ == "__main__":
    main()