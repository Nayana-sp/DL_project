"""Phase 3: inspect RAVDESS videos and build data/processed/metadata.csv.

Usage (from the project folder, .venv active):
    python src/inspect_dataset.py --limit 20          # small test run
    python src/inspect_dataset.py --check-volume      # full run
Raw data is only read, never changed.
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "RAVDESS"
OUT_CSV = ROOT / "data" / "processed" / "metadata.csv"
TEST_CSV = ROOT / "data" / "processed" / "metadata_test.csv"
TABLE_CSV = ROOT / "outputs" / "metrics" / "emotion_by_actor.csv"
LOG_FILE = ROOT / "outputs" / "logs" / "inspect_dataset.log"

# Official RAVDESS filename codes: modality-vocal-emotion-intensity-statement-repetition-actor
FIELD_KEYS = ["modality", "vocal_channel", "emotion_id", "intensity",
              "statement", "repetition", "actor_id"]
EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
MODALITY = {1: "audio-video", 2: "video-only", 3: "audio-only"}
VOCAL = {1: "speech", 2: "song"}
INTENSITY = {1: "normal", 2: "strong"}
STATEMENT = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}
NAME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.mp4$")
ACTOR_DIR_RE = re.compile(r"^Actor_(\d{2})$")
VOL_RE = re.compile(r"(mean|max)_volume:\s*(-?inf|-?[\d.]+)\s*dB")
SILENT_DB = -80.0  # audio-video clips with a peak below this are flagged as silent

COLUMNS = ["sample_id", "video_path", "audio_source", "audio_path",
           "recording_type", "modality", "vocal_channel", "emotion_id", "emotion",
           "intensity", "statement", "repetition", "actor_id",
           "has_video", "has_audio", "duration_s", "width", "height", "fps",
           "frame_count", "first_frame_ok", "audio_sample_rate", "audio_channels",
           "mean_volume_db", "max_volume_db",
           "usable", "selected", "preprocessing_status", "issues"]


def parse_filename(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    return dict(zip(FIELD_KEYS, (int(g) for g in m.groups())))


def to_float(frac):
    try:
        num, den = frac.split("/")
        return float(num) / float(den) if float(den) else None
    except Exception:
        return None


def run_probe(path):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=60)
        if out.returncode != 0:
            return None, "probe_failed"
        return json.loads(out.stdout), None
    except Exception as e:
        return None, f"probe_failed:{type(e).__name__}"


def run_volume(path):
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
           "-vn", "-af", "volumedetect", "-f", "null", os.devnull]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        vals = {k: float(v) for k, v in VOL_RE.findall(out.stderr)}
        return vals.get("mean"), vals.get("max")
    except Exception:
        return None, None


def check_first_frame(path):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return bool(ok and frame is not None and frame.size > 0), n
    finally:
        cap.release()


def build_record(path, check_volume):
    rel = path.relative_to(ROOT).as_posix()
    rec = {"sample_id": path.stem, "video_path": rel, "audio_source": rel}
    issues = []

    fields = parse_filename(path.name)
    if fields is None:
        issues.append("bad_filename")
        fields = {k: None for k in FIELD_KEYS}
    rec.update(fields)

    if fields["modality"] is not None:
        valid = (fields["modality"] in MODALITY and fields["vocal_channel"] in VOCAL
                 and fields["emotion_id"] in EMOTIONS and fields["intensity"] in INTENSITY
                 and fields["statement"] in STATEMENT and fields["repetition"] in (1, 2)
                 and 1 <= fields["actor_id"] <= 24)
        if not valid:
            issues.append("invalid_code")

    m = ACTOR_DIR_RE.match(path.parent.name)
    if not m:
        issues.append("unexpected_folder")
    elif fields["actor_id"] is not None and int(m.group(1)) != fields["actor_id"]:
        issues.append("actor_folder_mismatch")

    rec["emotion"] = EMOTIONS.get(fields["emotion_id"])
    mod, voc = MODALITY.get(fields["modality"]), VOCAL.get(fields["vocal_channel"])
    rec["recording_type"] = f"{mod} {voc}" if mod and voc else None

    info, err = run_probe(path)
    streams = info.get("streams", []) if info else []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    rec["has_video"] = video is not None
    rec["has_audio"] = audio is not None
    if err:
        issues.append(err)
    else:
        if video is None:
            issues.append("no_video_stream")
        if audio is None and fields["modality"] == 1:
            issues.append("no_audio_stream")

    try:
        rec["duration_s"] = float(info["format"]["duration"])
    except Exception:
        rec["duration_s"] = None
    rec["width"] = video.get("width") if video else None
    rec["height"] = video.get("height") if video else None
    rec["fps"] = to_float(video.get("r_frame_rate", "")) if video else None
    rec["audio_sample_rate"] = int(audio["sample_rate"]) if audio and "sample_rate" in audio else None
    rec["audio_channels"] = audio.get("channels") if audio else None

    ok, n_frames = check_first_frame(path)
    rec["first_frame_ok"], rec["frame_count"] = ok, n_frames
    if not ok:
        issues.append("frame_unreadable")

    rec["mean_volume_db"] = rec["max_volume_db"] = None
    if check_volume and audio is not None:
        mean_db, max_db = run_volume(path)
        rec["mean_volume_db"], rec["max_volume_db"] = mean_db, max_db
        if max_db is None:
            issues.append("volume_check_failed")
        elif fields["modality"] == 1 and max_db < SILENT_DB:
            issues.append("silent_audio")

    rec["issues"] = ";".join(issues)
    rec["usable"] = not issues
    rec["selected"] = (not issues) and fields["modality"] == 1 and fields["vocal_channel"] == 1
    rec["audio_path"] = f"data/processed/audio/{path.stem}.wav" if rec["selected"] else ""
    rec["preprocessing_status"] = ("pending" if rec["selected"]
                                   else "failed_check" if issues else "not_selected")
    return rec


def safe_build(path, check_volume):
    try:
        return build_record(path, check_volume)
    except Exception as e:
        logging.exception("Unexpected failure on %s", path)
        return {"sample_id": path.stem, "video_path": path.relative_to(ROOT).as_posix(),
                "issues": f"exception:{type(e).__name__}", "usable": False,
                "selected": False, "preprocessing_status": "failed_check"}


def report(df, check_volume, ignored, limited):
    log = logging.info
    log("=" * 60)
    log(f"mp4 files inspected: {len(df)} | other non-zip files ignored: {ignored}")
    log("Recording types:\n" + df["recording_type"].value_counts(dropna=False).to_string())

    bad = df[df["issues"] != ""]
    log(f"Files with issues: {len(bad)}")
    if len(bad):
        log("Issue types:\n" + bad["issues"].str.split(";").explode().value_counts().to_string())
        log("First problem files:\n" + bad[["video_path", "issues"]].head(20).to_string(index=False))

    sel = df[df["selected"]]
    log(f"Selected (audio-video speech, no issues): {len(sel)}")
    if sel.empty:
        return
    log(f"Actors in selection: {sel['actor_id'].nunique()}")
    per_emotion = sel.groupby("emotion").size().reindex(list(EMOTIONS.values()), fill_value=0)
    log("Selected clips per emotion:\n" + per_emotion.to_string())
    table = pd.crosstab(sel["actor_id"], sel["emotion"]).reindex(
        columns=list(EMOTIONS.values()), fill_value=0)
    log(f"Clips per actor: min {table.sum(axis=1).min()}, max {table.sum(axis=1).max()}")
    log(f"Actor-emotion cells with zero clips: {int((table == 0).sum().sum())}")
    log("Video sizes:\n" + sel.groupby(["width", "height"]).size().to_string())
    log("FPS values:\n" + sel["fps"].round(2).value_counts().to_string())
    log("Duration (s) and frame count:\n"
        + sel[["duration_s", "frame_count"]].describe().loc[["min", "mean", "max"]].round(2).to_string())
    if check_volume:
        log("Volume dB (selected):\n"
            + sel[["mean_volume_db", "max_volume_db"]].describe().loc[["min", "mean", "max"]].round(1).to_string())
        vo = df[(df["modality"] == 2) & df["max_volume_db"].notna()]
        if len(vo):
            log(f"Video-only files volume-checked: {len(vo)}; "
                f"loudest peak among them: {vo['max_volume_db'].max():.1f} dB")
    else:
        log("Volume was NOT checked in this run (use --check-volume).")
    if not limited:
        TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLE_CSV)
        log(f"Saved {TABLE_CSV.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="inspect only the first N files (test run)")
    ap.add_argument("--check-volume", action="store_true", help="measure audio loudness (slower)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not RAW_DIR.is_dir():
        sys.exit(f"Dataset folder not found: {RAW_DIR}")
    for tool in ("ffprobe", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")])

    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    mp4s = [p for p in files if p.suffix.lower() == ".mp4"]
    ignored = sum(1 for p in files if p.suffix.lower() not in (".mp4", ".zip"))
    if args.limit:
        mp4s = mp4s[:args.limit]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(tqdm(pool.map(lambda p: safe_build(p, args.check_volume), mp4s),
                            total=len(mp4s), desc="Inspecting"))

    df = pd.DataFrame(records)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]

    dup = df.duplicated("sample_id", keep=False)
    if dup.any():
        df.loc[dup, "issues"] = df.loc[dup, "issues"].apply(
            lambda s: (s + ";" if s else "") + "duplicate_sample_id")
        df.loc[dup, ["usable", "selected"]] = False
        df.loc[dup, "preprocessing_status"] = "failed_check"
        df.loc[dup, "audio_path"] = ""
    for col in FIELD_KEYS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")

    out = TEST_CSV if args.limit else OUT_CSV
    df.to_csv(out, index=False)
    report(df, args.check_volume, ignored, limited=bool(args.limit))
    logging.info(f"Saved {out.relative_to(ROOT)} and {LOG_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()