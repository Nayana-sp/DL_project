"""Datasets shared by all experiments (image part now; audio is added in later phases)."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
INDEX_CSV = ROOT / "data" / "processed" / "dataset_index.csv"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
# label = emotion_id - 1, so index i in EMOTIONS is class i everywhere in the project


def load_index(split=None):
    df = pd.read_csv(INDEX_CSV)
    if split is not None:
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


def class_weights(df):
    """Inverse-frequency weights from the given (training) rows."""
    counts = np.bincount(df["emotion_id"] - 1, minlength=len(EMOTIONS)).astype(float)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def augment_frames(x, pad=6):
    """x: (n, 3, H, W) in [0, 1]. Flip, small shift, brightness and contrast jitter per frame."""
    n, _, h, w = x.shape
    out = torch.empty_like(x)
    for k in range(n):
        f = x[k]
        if torch.rand(1).item() < 0.5:
            f = torch.flip(f, dims=[2])
        f = F.pad(f.unsqueeze(0), (pad, pad, pad, pad), mode="replicate")[0]
        dy, dx = torch.randint(0, 2 * pad + 1, (2,)).tolist()
        f = f[:, dy:dy + h, dx:dx + w]
        contrast = 0.8 + 0.4 * torch.rand(1)
        brightness = (torch.rand(1) - 0.5) * 0.2
        out[k] = ((f - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
    return out


class ImageDataset(Dataset):
    """One item = one clip: frames (n_frames, 3, H, W) in [-1, 1], label (int), sample_id (str)."""

    def __init__(self, split=None, augment=False, index=None):
        self.df = load_index(split) if index is None else index.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with np.load(ROOT / r["frames_path"]) as z:
            x = torch.from_numpy(z["frames"]).permute(0, 3, 1, 2).float() / 255.0
        if self.augment:
            x = augment_frames(x)
        x = (x - 0.5) / 0.5
        return x, int(r["emotion_id"]) - 1, r["sample_id"]

# ---- Audio (Phase 7) --------------------------------------------------------
# These must match prepare_data.py (16 kHz mono WAV, fixed 5.0 s).
AUDIO_SR = 16000
AUDIO_SECONDS = 5.0
N_MELS = 64
N_FFT = 1024       # FFT size (large enough that no Mel band is empty)
WIN_LENGTH = 400   # 25 ms window at 16 kHz
HOP_LENGTH = 160   # 10 ms hop
N_FRAMES = 1 + int(AUDIO_SR * AUDIO_SECONDS) // HOP_LENGTH  # 501 time steps


def load_wave(path):
    """Read a WAV as mono float32 and cut it to AUDIO_SECONDS (shorter clips stay short)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != AUDIO_SR:
        raise ValueError(f"{path}: sample rate {sr}, expected {AUDIO_SR}")
    return x.mean(axis=1)[: int(AUDIO_SR * AUDIO_SECONDS)]


def log_mel(x):
    """Waveform -> log-Mel spectrogram (N_MELS, frames), standardised per clip."""
    mel = librosa.feature.melspectrogram(y=x, sr=AUDIO_SR, n_fft=N_FFT, win_length=WIN_LENGTH,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    s = np.log(mel + 1e-6).astype(np.float32)
    return (s - s.mean()) / (s.std() + 1e-5)


def spec_augment(s, max_freq=8, max_time=40):
    """Zero one random band of Mel bins and one random span of frames (train only)."""
    s = s.copy()
    n_mels, n_t = s.shape
    f = int(torch.randint(0, max_freq + 1, (1,)))
    f0 = int(torch.randint(0, n_mels - f + 1, (1,)))
    s[f0:f0 + f, :] = 0.0
    t = int(torch.randint(0, min(max_time, n_t // 4) + 1, (1,)))
    t0 = int(torch.randint(0, n_t - t + 1, (1,)))
    s[:, t0:t0 + t] = 0.0
    return s


class AudioDataset(Dataset):
    """One item = one clip: log-Mel (1, N_MELS, N_FRAMES) float32, label (int), sample_id (str).

    Frames after the real end of the clip are 0 (the clip mean after standardising).
    wave_fn: optional function applied to the waveform before the spectrogram
    (used later for the noise experiments).
    """

    def __init__(self, split=None, augment=False, index=None, wave_fn=None):
        self.df = load_index(split) if index is None else index.reset_index(drop=True)
        self.augment = augment
        self.wave_fn = wave_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = load_wave(ROOT / r["audio_path"])
        if self.wave_fn is not None:
            x = self.wave_fn(x)
        s = log_mel(x)
        if self.augment:
            s = spec_augment(s)
        out = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
        n = min(s.shape[1], N_FRAMES)
        out[:, :n] = s[:, :n]
        return torch.from_numpy(out).unsqueeze(0), int(r["emotion_id"]) - 1, r["sample_id"]

class FusionDataset(Dataset):
    """One item = one clip with both modalities: ((frames, log-Mel), label, sample_id).

    frames: (5, 3, 112, 112); log-Mel: (1, N_MELS, N_FRAMES).
    wave_fn is passed to the audio side (used later for the noise experiments).
    """

    def __init__(self, split=None, augment=False, index=None, wave_fn=None):
        df = load_index(split) if index is None else index
        df = df.reset_index(drop=True)
        self.img = ImageDataset(index=df, augment=augment)
        self.aud = AudioDataset(index=df, augment=augment, wave_fn=wave_fn)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        xi, y, sid = self.img[i]
        xa, _, _ = self.aud[i]
        return (xi, xa), y, sid


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        print(s, len(load_index(s)), "clips")
    ds = ImageDataset("train", augment=True)
    x, y, sid = ds[0]
    print("one item:", tuple(x.shape), x.dtype, "| label", y, EMOTIONS[y], "|", sid)
    print("value range: %.2f to %.2f" % (x.min(), x.max()))
    print("train class weights:", [round(v, 2) for v in class_weights(load_index("train")).tolist()])

    a = AudioDataset("train", augment=True)
    s, y, sid = a[0]
    print("audio item:", tuple(s.shape), s.dtype, "| label", y, EMOTIONS[y], "|", sid)
    print("audio value range: %.2f to %.2f" % (s.min(), s.max()))
    t0 = time.time()
    for k in range(20):
        a[k]
    print("audio loading: %.1f ms per clip (20 clips)" % (50 * (time.time() - t0)))

