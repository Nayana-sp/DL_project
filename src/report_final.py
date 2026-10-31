"""Final comparison from saved predictions only (no model is run, nothing is tuned).

Usage (project folder, .venv active):
    python src/report_final.py
Writes outputs/metrics/final_comparison.csv, final_bootstrap.csv, final_per_class.csv, final_per_actor.csv,
final_tables.md and outputs/figures/fig_final_comparison.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIGS = ROOT / "outputs" / "figures"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SEED = 42

# (label, saved run name). Runs without saved predictions are skipped with a message.
SYSTEMS = [
    ("Image only, from scratch", "image_cnn_v3"),
    ("Audio only, from scratch", "audio_cnn"),
    ("Early fusion, from scratch", "early_cnn"),
    ("Late fusion, from scratch", "late_fusion"),
    ("Image only, pretrained seed 1", "image_pre_v1"),
    ("Image only, pretrained seed 2", "image_pre_s2"),
    ("Audio only, pretrained seed 1", "audio_pre_v1"),
    ("Audio only, pretrained seed 2", "audio_pre_s2"),
    ("Image only, pretrained 2-seed average", "image_pre_ens"),
    ("Audio only, pretrained 2-seed average", "audio_pre_ens"),
    ("Late fusion, pretrained seed 1", "late_fusion_pre_b"),
    ("Late fusion, pretrained 2-seed average (final)", "late_fusion_ens"),
]
KEY = ["image_cnn_v3", "audio_cnn", "late_fusion", "image_pre_ens", "audio_pre_ens", "late_fusion_ens"]
PAIRS = [("late_fusion_ens", "late_fusion"), ("late_fusion_ens", "late_fusion_pre_b"),
         ("late_fusion_ens", "audio_pre_ens"), ("late_fusion_ens", "early_cnn"),
         ("audio_pre_ens", "audio_cnn"), ("image_pre_ens", "image_cnn_v3"),
         ("audio_pre_ens", "audio_pre_s2"), ("image_pre_ens", "image_pre_s2")]


def load(name, split):
    p = METRICS / f"eval_{name}_{split}_predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("sample_id").reset_index(drop=True)


def scores(d):
    acc = float((d["pred"] == d["true"]).mean())
    f1 = float(f1_score(d["true"], d["pred"], average="macro", labels=list(range(8)), zero_division=0))
    return acc, f1


def fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def main():
    label = {name: lab for lab, name in SYSTEMS}
    test, val, rows = {}, {}, []
    for lab, name in SYSTEMS:
        t, v = load(name, "test"), load(name, "val")
        if t is None or v is None:
            print(f"skipped (no saved predictions): {name}")
            continue
        test[name], val[name] = t, v
        ta, tf, va, vf = *scores(t), *scores(v)
        rows.append({"system": lab, "run": name, "val acc": va, "val macro-F1": vf,
                     "test acc": ta, "test macro-F1": tf})
    ref = next(iter(test.values()))["sample_id"].tolist()
    for name, d in test.items():
        assert d["sample_id"].tolist() == ref, f"clip order differs in {name}"
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(METRICS / "final_comparison.csv", index=False)

    # paired bootstrap on test clips (clips of one actor are correlated and there are only 4 actors)
    n = len(ref)
    idx = np.random.default_rng(SEED).integers(0, n, size=(2000, n))
    ok = {k: (d["pred"] == d["true"]).to_numpy().astype(float) for k, d in test.items()}
    brow = []
    for a, b in PAIRS:
        if a in ok and b in ok:
            diff = ok[a] - ok[b]
            lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
            brow.append({"comparison": f"{label[a]} - {label[b]}", "accuracy diff": float(diff.mean()),
                         "95% CI low": float(lo), "95% CI high": float(hi),
                         "excludes 0": "yes" if (lo > 0 or hi < 0) else "no"})
    boot = pd.DataFrame(brow)
    boot.round(4).to_csv(METRICS / "final_bootstrap.csv", index=False)

    # per-class F1 and per-actor accuracy (test) for the key systems
    keys = [k for k in KEY if k in test]
    pc = pd.DataFrame({"emotion": EMOTIONS})
    pa = pd.DataFrame()
    for k in keys:
        d = test[k]
        pc[label[k]] = f1_score(d["true"], d["pred"], average=None, labels=list(range(8)), zero_division=0)
        actor = d["sample_id"].str.split("-").str[-1].astype(int)
        pa[label[k]] = (d["pred"] == d["true"]).groupby(actor).mean()
    pa = pa.reset_index().rename(columns={"sample_id": "test actor", "index": "test actor"})
    pc.round(4).to_csv(METRICS / "final_per_class.csv", index=False)
    pa.round(4).to_csv(METRICS / "final_per_actor.csv", index=False)

    # figure
    show = [k for k in ["image_cnn_v3", "audio_cnn", "early_cnn", "late_fusion", "image_pre_ens",
                        "audio_pre_ens", "late_fusion_ens"] if k in test]
    sub = comp.set_index("run").loc[show]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(show))
    ax.bar(x - 0.2, sub["test acc"], 0.4, label="accuracy", color="#4c78a8")
    ax.bar(x + 0.2, sub["test macro-F1"], 0.4, label="macro-F1", color="#f58518")
    for i, (a_, f_) in enumerate(zip(sub["test acc"], sub["test macro-F1"])):
        ax.text(i - 0.2, a_ + 0.01, f"{a_:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f_ + 0.01, f"{f_:.2f}", ha="center", fontsize=8)
    ax.axhline(0.125, color="gray", ls=":", label="chance (0.125)")
    ax.set_xticks(x, [label[k].replace(", ", "\n") for k in show], fontsize=7)
    ax.set_ylim(0, 0.85)
    ax.set_title("Test set (4 unseen actors, 240 clips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_final_comparison.png", dpi=120)
    plt.close(fig)

    parts = ["## All systems (accuracy and macro-F1)", md_table(comp.drop(columns="run")),
             "## Paired bootstrap of test accuracy differences (2000 resamples of clips)", md_table(boot),
             "Clips from one actor are correlated and there are only 4 test actors, so these intervals are too narrow.",
             "## Per-class F1 (test)", md_table(pc), "## Accuracy per test actor", md_table(pa)]
    (METRICS / "final_tables.md").write_text("\n\n".join(parts), encoding="utf-8")

    print(comp.round(3).drop(columns="run").to_string(index=False))
    print("\nPaired bootstrap (accuracy difference, 95% CI):\n" + boot.round(3).to_string(index=False))
    print("\nAccuracy per test actor:\n" + pa.round(3).to_string(index=False))
    print("\nSaved final_comparison.csv, final_bootstrap.csv, final_per_class.csv, final_per_actor.csv, "
          "final_tables.md in outputs\\metrics and fig_final_comparison.png in outputs\\figures")


if __name__ == "__main__":
    main()