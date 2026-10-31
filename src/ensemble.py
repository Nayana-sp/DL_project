"""Average the class probabilities of several runs (e.g. two seeds) into one ensemble 'run'.

No training and no data loading: it only reads the saved predictions of the runs.
Usage (project folder, .venv active):
    python src/ensemble.py --runs image_pre_v1 image_pre_s2 --name image_pre_ens
    python src/ensemble.py --runs audio_pre_v1 audio_pre_s2 --name audio_pre_ens
The output files have the same format as evaluate.py's, so late_fusion.py can use the ensemble by name:
    python src/late_fusion.py --image image_pre_ens --audio audio_pre_ens --name late_fusion_ens
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import EMOTIONS
from evaluate import report_split

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
PCOLS = [f"p_{e}" for e in EMOTIONS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="names of the runs to average, e.g. image_pre_v1 image_pre_s2")
    ap.add_argument("--name", required=True, help="name of the ensemble, e.g. image_pre_ens")
    a = ap.parse_args()
    if len(a.runs) < 2:
        raise SystemExit("give at least two runs")

    summary = {"name": a.name, "runs": a.runs, "method": "mean of class probabilities"}
    for split in ("val", "test"):
        dfs = [pd.read_csv(METRICS / f"eval_{r}_{split}_predictions.csv").sort_values("sample_id").reset_index(drop=True)
               for r in a.runs]
        ids = dfs[0]["sample_id"].tolist()
        for r, d in zip(a.runs[1:], dfs[1:]):
            assert d["sample_id"].tolist() == ids, f"{r} covers different clips"
            assert (d["true"].to_numpy() == dfs[0]["true"].to_numpy()).all(), f"{r} has different labels"
        p = np.mean([d[PCOLS].to_numpy() for d in dfs], axis=0)
        summary[split] = report_split(a.name, split, ids, dfs[0]["true"].to_numpy(), p, float("nan"))
    (METRICS / f"eval_{a.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 60)
    print("Ignore 'inference nan ms/clip' (no model is run). Report the TEST numbers.")


if __name__ == "__main__":
    main()