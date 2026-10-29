import argparse, subprocess, tempfile, re
from pathlib import Path
import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    k
}

def parse_ravdess_name(path: Path):
    """Return metadata for RAVDESS names: modality-vocal-emotion-intensity-statement-repetition-actor."""
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    if modality != "01" or vocal != "01" or emotion not in EMOTIONS:
        return None
    return {
        "emotion_code": emotion, "emotion": EMOTIONS[emotion],
        "actor_id": int(actor), "gender": "female" if int(actor) % 2 == 0 else "male",
        "statement": int(statement), "intensity": int(intensity),
        "repetition": int(repetition), "clip_id": path.stem
    }

def extract_audio(video_path: Path, output_path: Path, sr=16000):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn",
           "-ac", "1", "-ar", str(sr), "-t", "10", str(output_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def face_frames(video_path: Path, n_frames=8, size=224):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames")
    indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    detector_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(detector_path)
    crops, previous = [], None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if previous is not None:
                crops.append(previous.copy())
                continue
            raise RuntimeError("Failed to decode sampled frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            pad_x, pad_y = int(w * .15), int(h * .15)
            x0, y0 = max(0, x-pad_x), max(0, y-pad_y)
            x1, y1 = min(frame.shape[1], x+w+pad_x), min(frame.shape[0], y+h+pad_y)
            crop = frame[y0:y1, x0:x1]
        elif previous is not None:
            crop = previous
        else:
            # Fallback center crop if Haar detection fails on the first frame.
            h, w = frame.shape[:2]
            side = min(h, w)
            crop = frame[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        previous = crop
        crops.append(crop)
    cap.release()
    return np.asarray(crops, dtype=np.uint8)

def audio_logmel(wav_path: Path, sr=16000, n_mels=96, duration=4.0):
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    target = int(sr * duration)
    y = y[:target] if len(y) > target else np.pad(y, (0, target-len(y)))
    if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-5:
        raise ValueError("Silent or invalid audio")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256,
                                         n_mels=n_mels, fmin=50, fmax=sr//2, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = np.clip(logmel, -80, 0).astype(np.float32)
    # Per-clip normalisation is only a representation transform; training-set stats are not used here.
    return (logmel / 40.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="RAVDESS extracted directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Optional number of clips for a smoke test")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    (out/"audio").mkdir(parents=True, exist_ok=True)
    (out/"video").mkdir(parents=True, exist_ok=True)
    videos = sorted(inp.rglob("*.mp4"))
    rows = []
    valid = [p for p in videos if parse_ravdess_name(p)]
    if args.limit:
        valid = valid[:args.limit]
    if not valid:
        raise SystemExit("No matching RAVDESS audio-visual speech .mp4 files found. Check extracted folder and filename codes.")
    for video in tqdm(valid, desc="Preprocessing clips"):
        meta = parse_ravdess_name(video)
        wav = out/"audio"/f"{video.stem}.wav"
        audio_npy = out/"audio"/f"{video.stem}.npy"
        video_npy = out/"video"/f"{video.stem}.npy"
        status, error = "ok", ""
        try:
            extract_audio(video, wav)
            mel = audio_logmel(wav)
            frames = face_frames(video, n_frames=args.frames)
            np.save(audio_npy, mel)
            np.save(video_npy, frames)
        except Exception as exc:
            status, error = "error", str(exc)[:300]
        rows.append({**meta, "video_path": str(video), "audio_feature": str(audio_npy),
                     "video_feature": str(video_npy), "preprocessing_status": status, "error": error})
        # Do not retain extracted waveform files after the log-Mel representation is saved.
        if wav.exists():
            wav.unlink()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"manifest.csv", index=False)
    print(f"Manifest: {out/'manifest.csv'}; successful={sum(r['preprocessing_status']=='ok' for r in rows)} / {len(rows)}")

if __name__ == "__main__":
    main()
