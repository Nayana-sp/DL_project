"""Phase 4: preprocess audio and face frames for the selected RAVDESS clips.

Usage (project folder, .venv active):
    python src/prepare_data.py --limit 20     # test run, writes *_test files
    python src/prepare_data.py                # full run (frames are resumable)
Raw videos are only read. Outputs go to data/processed/.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
META_CSV = PROC / "metadata.csv"
AUDIO_DIR = PROC / "audio"
FRAME_DIR = PROC / "frames"
LOG_FILE = ROOT / "outputs" / "logs" / "prepare_data.log"
FIG_FILE = ROOT / "outputs" / "figures" / "crop_check.png"

_local = threading.local()


def get_cascade():
    """One classifier per thread (a shared one is not thread-safe)."""
    if not hasattr(_local, "c"):
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        c = cv2.CascadeClassifier(str(path))
        if c.empty():
            raise RuntimeError(f"could not load {path}")
        _local.c = c
    return _local.c


# ---------------- audio ----------------
def process_audio(video_path, out_path, sr, n_samples):
    cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr.decode(errors="replace")[:200])
    x = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if x.size == 0:
        raise RuntimeError("empty audio")
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        raise RuntimeError("silent audio")
    x = x / peak * 0.9
    n_orig = int(x.size)
    x = x[:n_samples] if n_orig >= n_samples else np.pad(x, (0, n_samples - n_orig))
    tmp = out_path.with_name(out_path.stem + ".tmp.wav")
    sf.write(str(tmp), x, sr, subtype="PCM_16")
    os.replace(tmp, out_path)
    return {"audio_orig_samples": n_orig,
            "audio_valid_samples": min(n_orig, n_samples),
            "audio_trimmed": n_orig > n_samples}


# ---------------- video ----------------
def read_frames(path, n_frames, frame_count, lo, hi):
    last = max(frame_count - 1, 1)
    targets = sorted(set(np.linspace(lo * last, hi * last, n_frames).round().astype(int).tolist()))
    want = set(targets)
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for i in range(targets[-1] + 1):
            if not cap.grab():
                break
            if i in want:
                ok, f = cap.retrieve()
                if ok:
                    frames.append(f)
    finally:
        cap.release()
    return frames


def detect_face(frame, min_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_cascade().detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def crop_square(frame, box, expand, shift_up, size):
    H, W = frame.shape[:2]
    x, y, w, h = box
    side = min(expand * w, H, W)
    cx, cy = x + w / 2, y + h / 2 - shift_up * w
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    s = int(round(side))
    crop = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def param_vector(a):
    return np.array([a.n_frames, a.img_size, a.expand, a.shift_up,
                     a.frame_lo, a.frame_hi, a.min_face], dtype=float)


def process_video(video_path, out_path, a, frame_count):
    params = param_vector(a)
    if out_path.exists() and not a.force:  # reuse only if made with identical settings
        try:
            with np.load(out_path) as z:
                if (np.allclose(z["params"], params)
                        and z["frames"].shape == (a.n_frames, a.img_size, a.img_size, 3)):
                    return {"n_faces_detected": int(z["n_detections"]), "face_box": z["box"].tolist()}
        except Exception:
            pass
    frames = read_frames(video_path, a.n_frames, frame_count, a.frame_lo, a.frame_hi)
    if len(frames) < a.n_frames:
        raise RuntimeError(f"only {len(frames)} frames readable")
    boxes = [b for b in (detect_face(f, a.min_face) for f in frames) if b is not None]
    if not boxes:
        return {"n_faces_detected": 0, "face_box": None}
    box = np.median(np.array(boxes, dtype=float), axis=0)
    crops = np.stack([crop_square(f, box, a.expand, a.shift_up, a.img_size) for f in frames])
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, frames=crops, box=box, n_detections=len(boxes), params=params)
    os.replace(tmp, out_path)
    return {"n_faces_detected": len(boxes), "face_box": box.tolist()}


# ---------------- per clip ----------------
def process_sample(row, a):
    sid = row["sample_id"]
    rec = {"sample_id": sid, "audio_path": "", "frames_path": ""}
    problems = []
    video = ROOT / row["video_path"]
    wav, npz = AUDIO_DIR / f"{sid}.wav", FRAME_DIR / f"{sid}.npz"
    try:
        rec.update(process_audio(video, wav, a.sample_rate, int(round(a.sample_rate * a.duration))))
        rec["audio_path"] = wav.relative_to(ROOT).as_posix()
    except Exception as e:
        problems.append("audio_failed")
        logging.error("audio failed %s: %s", sid, e)
    try:
        v = process_video(video, npz, a, int(row["frame_count"]))
        box = v.pop("face_box")
        rec.update(v)
        if box is None:
            problems.append("face_not_found")
            logging.error("face not found %s", sid)
        else:
            rec.update(face_x=box[0], face_y=box[1], face_w=box[2], face_h=box[3])
            rec["frames_path"] = npz.relative_to(ROOT).as_posix()
    except Exception as e:
        problems.append("video_failed")
        logging.error("video failed %s: %s", sid, e)
    rec["preprocessing_status"] = "ok" if not problems else ";".join(problems)
    return rec


# ---------------- reporting ----------------
def contact_sheet(res, meta, a):
    ok = res[res.preprocessing_status == "ok"]
    if ok.empty:
        return
    pick = ok.sample(n=min(24, len(ok)), random_state=42)
    lab = meta.set_index("sample_id")
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        with np.load(ROOT / r["frames_path"]) as z:
            ax.imshow(z["frames"][a.n_frames // 2])
        m = lab.loc[r["sample_id"]]
        ax.set_title(f"A{int(m['actor_id']):02d} {m['emotion']}", fontsize=9)
    fig.tight_layout()
    FIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_FILE, dpi=100)
    plt.close(fig)


def report(res, a, elapsed):
    log = logging.info
    ok = res[res.preprocessing_status == "ok"]
    log("=" * 60)
    log(f"clips processed: {len(res)} | ok: {len(ok)} | failed: {len(res) - len(ok)} | time: {elapsed:.0f}s")
    bad = res[res.preprocessing_status != "ok"]
    if len(bad):
        log("Failure types:\n" + bad.preprocessing_status.value_counts().to_string())
        log("First failed ids: " + ", ".join(bad.sample_id.head(10)))
    log(f"Faces found per clip (out of {a.n_frames} frames):\n"
        + res.n_faces_detected.value_counts().sort_index().to_string())
    if len(ok):
        w = ok.face_w.describe()
        log(f"Face box width px: min {w['min']:.0f}, median {w['50%']:.0f}, max {w['max']:.0f}")
        secs = ok.audio_valid_samples / a.sample_rate
        log(f"Audio length s: min {secs.min():.2f}, mean {secs.mean():.2f}, max {secs.max():.2f}")
        log(f"Audio clips trimmed at {a.duration}s: {int(ok.audio_trimmed.sum())}")
    size = sum(f.stat().st_size for d in (AUDIO_DIR, FRAME_DIR) for f in d.glob("*") if f.is_file())
    log(f"Disk used by processed audio+frames: {size / 1e6:.0f} MB")
    mi = psutil.Process().memory_info()
    log(f"RAM now: {mi.rss / 1e6:.0f} MB | peak: {getattr(mi, 'peak_wset', mi.rss) / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--expand", type=float, default=1.45)
    ap.add_argument("--shift-up", type=float, default=0.125)
    ap.add_argument("--frame-lo", type=float, default=0.25)
    ap.add_argument("--frame-hi", type=float, default=0.75)
    ap.add_argument("--min-face", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="test on N clips spread over the dataset")
    ap.add_argument("--force", action="store_true", help="recompute cached frames")
    a = ap.parse_args()

    if not META_CSV.exists():
        sys.exit(f"{META_CSV} not found - run inspect_dataset.py first")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")
    for d in (AUDIO_DIR, FRAME_DIR, LOG_FILE.parent):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")])
    cv2.setNumThreads(1)

    meta = pd.read_csv(META_CSV)
    sel = meta[meta["selected"] == True].reset_index(drop=True)  # noqa: E712
    if a.limit:
        sel = sel.iloc[np.linspace(0, len(sel) - 1, a.limit).round().astype(int)]
    rows = sel.to_dict("records")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(tqdm(pool.map(lambda r: process_sample(r, a), rows),
                            total=len(rows), desc="Preprocessing"))
    res = pd.DataFrame(results)
    for c in ("n_faces_detected", "face_w", "audio_valid_samples", "audio_trimmed"):
        if c not in res.columns:
            res[c] = np.nan

    suffix = "_test" if a.limit else ""
    res.to_csv(PROC / f"preprocess_results{suffix}.csv", index=False)
    cfg = {**vars(a), "opencv": cv2.__version__}
    (PROC / f"preprocess_config{suffix}.json").write_text(json.dumps(cfg, indent=2))
    report(res, a, time.time() - t0)
    contact_sheet(res, meta, a)
    logging.info(f"Saved preprocess_results{suffix}.csv, preprocess_config{suffix}.json, "
                 f"{FIG_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()