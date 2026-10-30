"""Phase 10 (second half): robustness tests. No model is trained or tuned here.

Systems: image-only, audio-only, early fusion, late fusion (probability blend with the saved audio weight).
Conditions: clean | audio noise at fixed SNRs | darker or noisy faces | missing audio | missing face |
conflict (face of one clip + voice of a clip with a different emotion).

Missing input:
  late fusion  -> falls back to the one available model (genuine handling)
  early fusion -> that branch's embedding is set to zero; the model never saw this in training (naive zero-fill)
  single-modality systems -> no input, shown as n/a

Usage (project folder, .venv active):
    python src/robustness.py --split val      # check that everything runs
    python src/robustness.py --split test     # numbers to report
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from dataset import AudioDataset, FusionDataset, ImageDataset, load_index, load_wave
from train import build, predict_probs

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
METRICS = ROOT / "outputs" / "metrics"
LABELS = list(range(8))


def load_model(name):
    path = CKPT / f"{name}_best.pt"
    if not path.exists():
        sys.exit(f"{path} not found")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    model, _ = build(targs.model, targs)
    model.load_state_dict(ck["model"])
    return model.eval()


def loader(ds, bs):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


# ---------- corruptions ----------
def dark_fn(factor):
    return lambda x, sid: ((x + 1) / 2 * factor).clamp(0, 1) * 2 - 1


def img_noise_fn(sigma, seed):
    def fn(x, sid):
        g = torch.Generator().manual_seed(zlib.crc32(sid.encode()) + seed)
        x01 = (x + 1) / 2 + sigma * torch.randn(x.shape, generator=g)
        return x01.clamp(0, 1) * 2 - 1
    return fn


def audio_noise_fn(snr_db, seed):
    def fn(x):
        nz = np.flatnonzero(x)
        if nz.size == 0:
            return x
        n = int(nz[-1]) + 1  # real clip length; the zero padding after it stays zero
        power = float(np.mean(x[:n].astype(np.float64) ** 2))
        sigma = np.sqrt(power / 10 ** (snr_db / 10))
        rng = np.random.default_rng(zlib.crc32(x.tobytes()) + seed)
        y = x.copy()
        y[:n] += rng.normal(0.0, sigma, n).astype(np.float32)
        return y
    return fn


class ImgCorrupt(Dataset):
    """Applies fn(frames, sample_id) to the face frames of an image or fusion dataset."""

    def __init__(self, base, fn, fusion):
        self.base, self.fn, self.fusion = base, fn, fusion

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, sid = self.base[i]
        x = (self.fn(x[0], sid), x[1]) if self.fusion else self.fn(x, sid)
        return x, y, sid


class ConflictFusion(Dataset):
    """Face of clip i with the voice of clip partner[i] (a different emotion). Label = face's emotion."""

    def __init__(self, df, partner):
        self.img, self.aud, self.partner = ImageDataset(index=df), AudioDataset(index=df), partner

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        xi, y, sid = self.img[i]
        xa, _, _ = self.aud[int(self.partner[i])]
        return (xi, xa), y, sid


@torch.no_grad()
def early_probs(model, ld, drops=(None,)):
    """Early-fusion probabilities. drops: None, 'audio' or 'image' (that embedding is set to zero)."""
    out = {d: [] for d in drops}
    ys, ids = [], []
    for (xi, xa), y, sid in ld:
        ei, ea = model.image.forward_embedding(xi), model.audio.forward_embedding(xa)
        for d in drops:
            a_emb = torch.zeros_like(ea) if d == "audio" else ea
            i_emb = torch.zeros_like(ei) if d == "image" else ei
            out[d].append(torch.softmax(model.head(torch.cat([i_emb, a_emb], 1)), 1).numpy())
        ys.extend(y.tolist())
        ids.extend(sid)
    return ids, np.array(ys), {d: np.concatenate(v) for d, v in out.items()}


def make_row(system, cond, sev, note, acc=np.nan, f1=np.nan, fi=np.nan, fa=np.nan):
    return dict(system=system, condition=cond, severity=sev, accuracy=acc, macro_f1=f1,
                follows_image=fi, follows_audio=fa, note=note)


def cell(r):
    if not np.isnan(r["accuracy"]):
        return f"{r['accuracy']:.3f} / {r['macro_f1']:.3f}"
    if not np.isnan(r["follows_image"]):
        return f"face {r['follows_image']:.2f} | voice {r['follows_audio']:.2f}"
    return "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", default="image_cnn_v3")
    ap.add_argument("--audio", default="audio_cnn")
    ap.add_argument("--early", default="early_cnn")
    ap.add_argument("--late", default="late_fusion")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--snr", type=float, nargs="+", default=[20, 10, 0])
    ap.add_argument("--dark", type=float, nargs="+", default=[0.5, 0.25])
    ap.add_argument("--img-noise", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="suffix for the output file name, e.g. _fine")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    bs = a.batch_size

    df = load_index(a.split)
    img_m, aud_m, early_m = load_model(a.image), load_model(a.audio), load_model(a.early)
    w = float(json.loads((METRICS / f"eval_{a.late}_summary.json").read_text())["audio_weight"])
    print(f"split={a.split} | {len(df)} clips | late-fusion audio weight {w:.2f}")

    def blend(pi, pa):
        return w * pa + (1 - w) * pi

    # ---- clean passes ----
    ids0, y, pi0 = predict_probs(img_m, loader(ImageDataset(index=df), bs))
    ids1, _, pa0 = predict_probs(aud_m, loader(AudioDataset(index=df), bs))
    ids2, _, pe = early_probs(early_m, loader(FusionDataset(index=df), bs), drops=(None, "audio", "image"))
    assert ids0 == ids1 == ids2, "clip order differs between passes"

    rows = []

    def add(system, cond, sev, p, note=""):
        pred = p.argmax(1)
        rows.append(make_row(system, cond, sev, note, accuracy_score(y, pred),
                             f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)))

    def add_na(system, cond, sev, note):
        rows.append(make_row(system, cond, sev, note))

    clean = {"image": pi0, "audio": pa0, "early": pe[None], "late": blend(pi0, pa0)}
    names = {"image": a.image, "audio": a.audio, "early": a.early, "late": a.late}
    print("Clean check against the saved evaluation (must say OK):")
    for s, p in clean.items():
        add(s, "clean", "-", p)
        saved = json.loads((METRICS / f"eval_{names[s]}_summary.json").read_text())[a.split]["accuracy"]
        ok = abs(saved - rows[-1]["accuracy"]) < 1e-4
        print(f"  {s:5s} accuracy {rows[-1]['accuracy']:.4f} vs saved {saved:.4f} -> {'OK' if ok else 'MISMATCH'}")

    # ---- audio noise ----
    x0 = load_wave(ROOT / df.iloc[0]["audio_path"])
    n0 = int(np.flatnonzero(x0)[-1]) + 1
    for snr in a.snr:
        fn = audio_noise_fn(snr, a.seed)
        y0 = fn(x0)
        meas = 10 * np.log10(np.mean(x0[:n0].astype(np.float64) ** 2)
                             / np.mean((y0[:n0] - x0[:n0]).astype(np.float64) ** 2))
        print(f"SNR check on first clip: target {snr:g} dB, measured {meas:.1f} dB")
        _, _, pa = predict_probs(aud_m, loader(AudioDataset(index=df, wave_fn=fn), bs))
        _, _, pe_n = early_probs(early_m, loader(FusionDataset(index=df, wave_fn=fn), bs))
        sev = f"SNR {snr:g} dB"
        add("image", "audio_noise", sev, pi0, "unaffected: no audio input")
        add("audio", "audio_noise", sev, pa)
        add("early", "audio_noise", sev, pe_n[None])
        add("late", "audio_noise", sev, blend(pi0, pa))

    # ---- image corruption ----
    conds = [("image_dark", f"brightness x{f:g}", dark_fn(f)) for f in a.dark]
    conds += [("image_noise", f"sigma {s:g}", img_noise_fn(s, a.seed)) for s in a.img_noise]
    for cond, sev, fn in conds:
        _, _, pi = predict_probs(img_m, loader(ImgCorrupt(ImageDataset(index=df), fn, False), bs))
        _, _, pe_c = early_probs(early_m, loader(ImgCorrupt(FusionDataset(index=df), fn, True), bs))
        add("image", cond, sev, pi)
        add("audio", cond, sev, pa0, "unaffected: no image input")
        add("early", cond, sev, pe_c[None])
        add("late", cond, sev, blend(pi, pa0))

    # ---- missing modality ----
    add("image", "missing_audio", "audio absent", pi0, "unaffected: no audio input")
    add_na("audio", "missing_audio", "audio absent", "no input left")
    add("early", "missing_audio", "audio absent", pe["audio"], "NAIVE zero-fill: never trained for this")
    add("late", "missing_audio", "audio absent", pi0, "fallback to image model (genuine handling)")
    add_na("image", "missing_image", "face absent", "no input left")
    add("audio", "missing_image", "face absent", pa0, "unaffected: no image input")
    add("early", "missing_image", "face absent", pe["image"], "NAIVE zero-fill: never trained for this")
    add("late", "missing_image", "face absent", pa0, "fallback to audio model (genuine handling)")

    # ---- conflicting modalities ----
    rng = np.random.default_rng(a.seed)
    partner = np.array([rng.choice(np.flatnonzero(y != y[i])) for i in range(len(y))])
    _, _, pe_x = early_probs(early_m, loader(ConflictFusion(df, partner), bs))

    def add_conflict(system, p, note=""):
        pred = p.argmax(1)
        rows.append(make_row(system, "conflict", "voice from another emotion", note,
                             fi=float((pred == y).mean()), fa=float((pred == y[partner]).mean())))

    add_conflict("image", pi0, "sees only the face")
    add_conflict("audio", pa0[partner], "sees only the voice")
    add_conflict("early", pe_x[None])
    add_conflict("late", blend(pi0, pa0[partner]))

    res = pd.DataFrame(rows)
    out = METRICS / f"robustness_{a.split}{a.tag}.csv"
    res.round(4).to_csv(out, index=False)
    res["cell"] = res.apply(cell, axis=1)
    table = res.pivot_table(index=["condition", "severity"], columns="system", values="cell",
                            aggfunc="first", sort=False)
    table = table[[c for c in ["image", "audio", "early", "late"] if c in table.columns]]
    print("\nCells show accuracy / macro-F1. Conflict rows show the share of predictions equal to the "
          "face's emotion | the voice's emotion.")
    print(table.to_string())
    print("\nMissing input: late fusion = fallback to the available model (genuine). "
          "Early fusion = zeroed embedding, never trained for it (naive).")
    print(f"Saved {out.relative_to(ROOT)} | total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()