"""Phase 11: final tables, statistics and figures from saved results. No model is run.

Usage (project folder, .venv active):
    python src/report.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIGS = ROOT / "outputs" / "figures"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
RUNS = {"image": "image_cnn_v3", "audio": "audio_cnn", "early": "early_cnn", "late": "late_fusion"}
LABEL = {"image": "Image only", "audio": "Audio only", "early": "Early fusion", "late": "Late fusion"}
COLOR = {"image": "#d95f02", "audio": "#1b9e77", "early": "#7570b3", "late": "#e7298a"}
SEED = 42


def read_json(p):
    return json.loads(Path(p).read_text())


def fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:.3f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def summary_table():
    tr = {k: read_json(METRICS / f"train_{v}_summary.json") for k, v in RUNS.items() if k != "late"}
    ev = {k: read_json(METRICS / f"eval_{v}_summary.json") for k, v in RUNS.items()}
    rows = []
    for k in RUNS:
        if k == "late":
            params = tr["image"]["params"] + tr["audio"]["params"]
            mins = (tr["image"]["train_seconds"] + tr["audio"]["train_seconds"]) / 60
            epochs = best = np.nan
            ms = ev["image"]["test"]["inference_ms_per_clip"] + ev["audio"]["test"]["inference_ms_per_clip"]
        else:
            params, mins = tr[k]["params"], tr[k]["train_seconds"] / 60
            epochs, best = tr[k]["epochs_run"], tr[k]["best_epoch"]
            ms = ev[k]["test"]["inference_ms_per_clip"]
        rows.append({"system": LABEL[k], "params": int(params), "epochs run": epochs, "best epoch": best,
                     "train min": mins, "val acc": ev[k]["val"]["accuracy"], "val macro-F1": ev[k]["val"]["macro_f1"],
                     "test acc": ev[k]["test"]["accuracy"], "test macro-F1": ev[k]["test"]["macro_f1"],
                     "ms/clip": ms})
    return pd.DataFrame(rows)


def per_class_table():
    t = pd.DataFrame({"emotion": EMOTIONS})
    for k, name in RUNS.items():
        d = pd.read_csv(METRICS / f"eval_{name}_test_per_class.csv")
        t[LABEL[k]] = d["f1"].to_numpy()
    return t


def bootstrap(n_boot=2000):
    ok = {}
    ids = None
    for k, name in RUNS.items():
        d = pd.read_csv(METRICS / f"eval_{name}_test_predictions.csv").sort_values("sample_id")
        if ids is None:
            ids = d["sample_id"].tolist()
        assert d["sample_id"].tolist() == ids, "clip order differs between systems"
        ok[k] = (d["pred"].to_numpy() == d["true"].to_numpy()).astype(float)
    n = len(ids)
    idx = np.random.default_rng(SEED).integers(0, n, size=(n_boot, n))
    rows = []
    for a, b in [("late", "audio"), ("early", "audio"), ("late", "early"),
                 ("late", "image"), ("audio", "image"), ("early", "image")]:
        diff = ok[a] - ok[b]
        lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
        rows.append({"comparison": f"{LABEL[a]} - {LABEL[b]}", "accuracy diff": diff.mean(),
                     "95% CI low": lo, "95% CI high": hi,
                     "excludes 0": "yes" if (lo > 0 or hi < 0) else "no"})
    return pd.DataFrame(rows)


def fig_main(summ):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(summ))
    ax.bar(x - 0.2, summ["test acc"], 0.4, label="accuracy", color="#4c78a8")
    ax.bar(x + 0.2, summ["test macro-F1"], 0.4, label="macro-F1", color="#f58518")
    for i, (a, f) in enumerate(zip(summ["test acc"], summ["test macro-F1"])):
        ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    ax.axhline(0.125, color="gray", ls=":", label="chance (0.125)")
    ax.set_xticks(x, summ["system"])
    ax.set_ylim(0, 0.8)
    ax.set_title("Test set (4 unseen actors, 240 clips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_main_comparison.png", dpi=120)
    plt.close(fig)


def fig_per_class(pc):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.2
    for j, k in enumerate(RUNS):
        ax.bar(np.arange(8) + (j - 1.5) * w, pc[LABEL[k]], w, label=LABEL[k], color=COLOR[k])
    ax.set_xticks(np.arange(8), EMOTIONS)
    ax.set_ylabel("F1 (test)")
    ax.set_title("Per-class F1 (32 test clips per class, 16 for neutral)")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_per_class_f1.png", dpi=120)
    plt.close(fig)


def fig_confusions():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (k, name) in zip(axes.ravel(), RUNS.items()):
        cm = pd.read_csv(METRICS / f"eval_{name}_test_confusion.csv", index_col=0)
        norm = cm.to_numpy() / np.clip(cm.to_numpy().sum(1, keepdims=True), 1, None)
        sns.heatmap(norm, annot=cm.to_numpy(), fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=EMOTIONS, yticklabels=EMOTIONS, annot_kws={"size": 7})
        ax.set_title(LABEL[k])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_confusion_grid.png", dpi=110)
    plt.close(fig)


def num(s):
    return float(re.findall(r"[-+]?\d*\.?\d+", s)[0])


def robustness_figs():
    r = pd.read_csv(METRICS / "robustness_test_fine.csv")
    clean = r[r["condition"] == "clean"].set_index("system")["accuracy"]
    specs = [("audio_noise", "audio SNR (dB), noisier to the right", True, "fig_robust_audio_noise.png"),
             ("image_dark", "brightness factor, darker to the right", True, "fig_robust_brightness.png"),
             ("image_noise", "image noise sigma, noisier to the right", False, "fig_robust_image_noise.png")]
    for cond, xlabel, desc, fname in specs:
        d = r[r["condition"] == cond].copy()
        d["x"] = d["severity"].map(num)
        order = sorted(d["x"].unique(), reverse=desc)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for k in RUNS:
            ys = [clean[k]] + [d[(d["system"] == k) & (d["x"] == x)]["accuracy"].iloc[0] for x in order]
            ax.plot(range(len(ys)), ys, marker="o", color=COLOR[k], label=LABEL[k],
                    ls="--" if np.allclose(ys, ys[0]) else "-")
        ax.axhline(0.125, color="gray", ls=":")
        ax.set_xticks(range(len(order) + 1), ["clean"] + [f"{x:g}" for x in order])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 0.75)
        ax.set_title("Dashed = system does not use the corrupted input")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / fname, dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    conds = [("clean", "clean"), ("missing_audio", "audio absent"), ("missing_image", "face absent")]
    for j, k in enumerate(RUNS):
        vals = [r[(r["condition"] == c) & (r["system"] == k)]["accuracy"].iloc[0] for c, _ in conds]
        pos = np.arange(3) + (j - 1.5) * 0.2
        ax.bar(pos, np.nan_to_num(vals), 0.2, color=COLOR[k], label=LABEL[k])
        for p, v in zip(pos, vals):
            if np.isnan(v):
                ax.text(p, 0.01, "n/a", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(range(3), [n for _, n in conds])
    ax.set_ylabel("test accuracy")
    ax.set_title("Missing modality (late = fallback, early = naive zero-fill)")
    ax.legend(fontsize=7)
    ax = axes[1]
    c = r[r["condition"] == "conflict"].set_index("system")
    x = np.arange(len(RUNS))
    ax.bar(x - 0.2, [c.loc[k, "follows_image"] for k in RUNS], 0.4, label="equals face's emotion", color="#d95f02")
    ax.bar(x + 0.2, [c.loc[k, "follows_audio"] for k in RUNS], 0.4, label="equals voice's emotion", color="#1b9e77")
    ax.set_xticks(x, [LABEL[k] for k in RUNS])
    ax.set_title("Conflict: face and voice from different emotions")
    ax.set_ylabel("share of predictions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_robust_missing_conflict.png", dpi=120)
    plt.close(fig)
    return r


def fig_disagreement():
    d = pd.read_csv(METRICS / "disagreement_test_per_class.csv")
    cols = [("both_right", "both right", "#2ca02c"), ("only_image_right", "only image right", "#d95f02"),
            ("only_audio_right", "only audio right", "#1b9e77"), ("neither_right", "neither right", "#999999")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = np.zeros(len(d))
    for c, lab, col in cols:
        ax.barh(d["emotion"], d[c], left=left, label=lab, color=col)
        left += d[c].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("test clips")
    ax.set_title("Image vs audio model: who is right (test)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_disagreement.png", dpi=120)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    summ, pc, boot = summary_table(), per_class_table(), bootstrap()
    summ.round(4).to_csv(METRICS / "summary_table.csv", index=False)
    boot.round(4).to_csv(METRICS / "bootstrap_differences.csv", index=False)
    fig_main(summ)
    fig_per_class(pc)
    fig_confusions()
    r = robustness_figs()
    fig_disagreement()

    rob = r.copy()
    rob["cell"] = rob.apply(lambda x: "n/a" if np.isnan(x["accuracy"]) and np.isnan(x["follows_image"])
                            else (f"{x['accuracy']:.3f}" if not np.isnan(x["accuracy"])
                                  else f"face {x['follows_image']:.2f} / voice {x['follows_audio']:.2f}"), axis=1)
    rob_t = rob.pivot_table(index=["condition", "severity"], columns="system", values="cell",
                            aggfunc="first", sort=False).reset_index()
    rob_t = rob_t[["condition", "severity"] + [k for k in RUNS if k in rob_t.columns]]
    rob_t.columns = ["condition", "severity"] + [LABEL[c] for c in rob_t.columns[2:]]

    parts = ["## Main results (test: 4 unseen actors, 240 clips)", md_table(summ),
             "Late fusion has no training of its own: parameters and training time are the sum of the image and "
             "audio models, and ms/clip is the sum of both models' inference time (includes data loading).",
             "## Per-class F1 (test)", md_table(pc),
             "## Paired bootstrap of accuracy differences (2000 resamples of test clips)", md_table(boot),
             "Clips from the same actor are correlated and there are only 4 test actors, so these intervals are "
             "too narrow.",
             "## Test accuracy under corruption", md_table(rob_t),
             "Conflict rows: share of predictions equal to the face's emotion / the voice's emotion. Missing input: "
             "late fusion falls back to the available model; early fusion zeroes that embedding (never trained for it)."]
    (METRICS / "results_tables.md").write_text("\n\n".join(parts), encoding="utf-8")

    print(summ.round(3).to_string(index=False))
    print("\nPer-class test F1:\n" + pc.round(3).to_string(index=False))
    print("\nPaired bootstrap (accuracy difference, 95% CI):\n" + boot.round(3).to_string(index=False))
    print("\nSaved summary_table.csv, bootstrap_differences.csv, results_tables.md in outputs\\metrics")
    print("Saved 8 figures (fig_*.png) in outputs\\figures")


if __name__ == "__main__":
    main()