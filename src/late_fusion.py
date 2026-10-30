"""Late (decision-level) fusion: weighted average of image-only and audio-only class probabilities.

Uses the saved predictions of two trained baselines (no training, no data loading). The audio weight is chosen
on the validation split only; the test split is scored once with that weight.

Usage (project folder, .venv active):
    python src/late_fusion.py --image image_cnn_v3 --audio audio_cnn --name late_fusion
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dataset import EMOTIONS
from evaluate import report_split

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
PCOLS = [f"p_{e}" for e in EMOTIONS]


def load_pair(image_name, audio_name, split):
    im = pd.read_csv(METRICS / f"eval_{image_name}_{split}_predictions.csv")
    au = pd.read_csv(METRICS / f"eval_{audio_name}_{split}_predictions.csv")
    m = im.merge(au, on="sample_id", suffixes=("_img", "_aud"))
    assert len(m) == len(im) == len(au), "image and audio predictions cover different clips"
    assert (m["true_img"] == m["true_aud"]).all(), "labels differ between the two files"
    p_img = m[[c + "_img" for c in PCOLS]].to_numpy()
    p_aud = m[[c + "_aud" for c in PCOLS]].to_numpy()
    return m["sample_id"].tolist(), m["true_img"].to_numpy(), p_img, p_aud


def macro_f1(y, p):
    return f1_score(y, p.argmax(1), average="macro", labels=list(range(len(EMOTIONS))), zero_division=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", required=True, help="name of the image-only run, e.g. image_cnn_v3")
    ap.add_argument("--audio", required=True, help="name of the audio-only run, e.g. audio_cnn")
    ap.add_argument("--name", default="late_fusion")
    a = ap.parse_args()

    ids_v, y_v, pi_v, pa_v = load_pair(a.image, a.audio, "val")
    ids_t, y_t, pi_t, pa_t = load_pair(a.image, a.audio, "test")

    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    val_f1 = np.array([macro_f1(y_v, w * pa_v + (1 - w) * pi_v) for w in grid])
    top = val_f1.max()
    w = float(min(grid[val_f1 >= top - 1e-9], key=lambda g: abs(g - 0.5)))  # ties: closest to equal weights
    pd.DataFrame({"audio_weight": grid, "val_macro_f1": val_f1.round(4)}).to_csv(
        METRICS / f"{a.name}_weight_sweep.csv", index=False)
    print("audio weight -> val macro-F1:", ", ".join(f"{g:.2f}:{f:.3f}" for g, f in zip(grid, val_f1)))
    print(f"Chosen audio weight (best val macro-F1, ties toward 0.5): {w:.2f}")
    print(f"For reference, test macro-F1 with equal weights (0.5): {macro_f1(y_t, 0.5 * pa_t + 0.5 * pi_t):.3f}")

    summary = {"name": a.name, "image_model": a.image, "audio_model": a.audio, "audio_weight": w,
               "val_sweep_best_macro_f1": float(top)}
    for split, ids, y, pi, pa in (("val", ids_v, y_v, pi_v, pa_v), ("test", ids_t, y_t, pi_t, pa_t)):
        summary[split] = report_split(a.name, split, ids, y, w * pa + (1 - w) * pi, float("nan"))
    (METRICS / f"eval_{a.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 60)
    print("Ignore 'inference nan ms/clip' (no model is run). Report the TEST numbers.")


if __name__ == "__main__":
    main()