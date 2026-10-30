"""Image vs audio disagreement analysis (Phase 10, part 1). Uses saved predictions only; no model is run.

Usage (project folder, .venv active):
    python src/disagreement.py --image image_cnn_v3 --audio audio_cnn --fusion early_cnn late_fusion --split test
"""
import argparse
from pathlib import Path

import pandas as pd

from dataset import EMOTIONS

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"


def load(name, split):
    df = pd.read_csv(METRICS / f"eval_{name}_{split}_predictions.csv")
    return df[["sample_id", "true", "pred"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--fusion", nargs="*", default=[])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    df = load(a.image, a.split).rename(columns={"pred": "img"})
    df = df.merge(load(a.audio, a.split).drop(columns="true").rename(columns={"pred": "aud"}), on="sample_id")
    for f in a.fusion:
        df = df.merge(load(f, a.split).drop(columns="true").rename(columns={"pred": "f_" + f}), on="sample_id")
    df["img_ok"] = df["img"] == df["true"]
    df["aud_ok"] = df["aud"] == df["true"]
    df["agree"] = df["img"] == df["aud"]

    n = len(df)
    ag, dis = df[df["agree"]], df[~df["agree"]]
    print(f"{a.split}: {n} clips | image and audio give the same answer on {len(ag)} ({len(ag) / n:.1%}) "
          f"and different answers on {len(dis)} ({len(dis) / n:.1%})")
    print(f"When they agree: the shared answer is correct for {ag['img_ok'].mean():.1%} of {len(ag)} clips")
    print(f"When they differ ({len(dis)} clips): only image right {int((dis['img_ok'] & ~dis['aud_ok']).sum())}, "
          f"only audio right {int((~dis['img_ok'] & dis['aud_ok']).sum())}, "
          f"neither right {int((~dis['img_ok'] & ~dis['aud_ok']).sum())}")
    print(f"If one could always pick the right modality, accuracy would be {(df['img_ok'] | df['aud_ok']).mean():.3f}")
    one = (~df["agree"]) & (df["img_ok"] ^ df["aud_ok"])
    for f in a.fusion:
        ok = df["f_" + f] == df["true"]
        print(f"{f}: accuracy {ok.mean():.3f} | when modalities agree {ok[df['agree']].mean():.3f} | "
              f"when they differ {ok[~df['agree']].mean():.3f} | right on {ok[one].mean():.1%} of the "
              f"{int(one.sum())} clips where exactly one modality is right")

    rows = []
    for i, e in enumerate(EMOTIONS):
        d = df[df["true"] == i]
        rows.append({"emotion": e, "clips": len(d), "disagree_rate": round(float((~d["agree"]).mean()), 3),
                     "both_right": int((d["img_ok"] & d["aud_ok"]).sum()),
                     "only_image_right": int((d["img_ok"] & ~d["aud_ok"]).sum()),
                     "only_audio_right": int((~d["img_ok"] & d["aud_ok"]).sum()),
                     "neither_right": int((~d["img_ok"] & ~d["aud_ok"]).sum())})
    per = pd.DataFrame(rows)
    print("\nBy true emotion:")
    print(per.to_string(index=False))
    out = METRICS / f"disagreement_{a.split}_per_class.csv"
    per.to_csv(out, index=False)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()