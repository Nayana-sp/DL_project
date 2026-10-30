"""Datasets shared by all experiments (image part now; audio is added in later phases)."""
from pathlib import Path

import numpy as np
import pandas as pd
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


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        print(s, len(load_index(s)), "clips")
    ds = ImageDataset("train", augment=True)
    x, y, sid = ds[0]
    print("one item:", tuple(x.shape), x.dtype, "| label", y, EMOTIONS[y], "|", sid)
    print("value range: %.2f to %.2f" % (x.min(), x.max()))
    print("train class weights:", [round(v, 2) for v in class_weights(load_index("train")).tolist()])