"""Train a model on the actor-independent split (image-only now; audio and fusion added later).

Usage (project folder, .venv active):
    python src/train.py --model image --name image_smoke --epochs 1 --limit 64   # smoke test
    python src/train.py --model image --name image_cnn                           # full run
    python src/train.py --model image --name image_cnn --resume                  # continue after a stop
The test split is never used here.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EMOTIONS, AudioDataset, FusionDataset, ImageDataset, class_weights, load_index
from models import (AudioCNN, ClipImageModel, ClipImageModelPre, EarlyFusionModel,
                    PretrainedAudioCNN, count_params)
ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
METRICS = ROOT / "outputs" / "metrics"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_dev(x, device):
    """Move a tensor, or a list/tuple of tensors (fusion models), to the device."""
    if isinstance(x, (list, tuple)):
        return [t.to(device) for t in x]
    return x.to(device)

def build(kind, a):
    if kind == "image":
        return ClipImageModel(len(EMOTIONS), a.emb_dim, a.dropout), ImageDataset
    if kind == "audio":
        return AudioCNN(len(EMOTIONS), a.emb_dim, a.dropout), AudioDataset
    if kind == "early":
        return EarlyFusionModel(len(EMOTIONS), a.emb_dim, a.dropout), FusionDataset
    if kind == "image_pre":
        return ClipImageModelPre(len(EMOTIONS), a.emb_dim, a.dropout), ImageDataset
    if kind == "audio_pre":
        return PretrainedAudioCNN(len(EMOTIONS), a.emb_dim, a.dropout), AudioDataset
    sys.exit(f"unknown model: {kind}")


@torch.no_grad()
def predict_probs(model, loader, device="cpu"):
    """Returns (sample_ids, true labels, clip probabilities) for every clip in the loader."""
    model.eval()
    ids, ys, ps = [], [], []
    for x, y, sid in loader:
        ps.append(torch.softmax(model(to_dev(x, device)), dim=1).cpu().numpy())
        ys.extend(y.tolist())
        ids.extend(sid)
    return ids, np.array(ys), np.concatenate(ps)


def score(y, p):
    pred = p.argmax(1)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    return {"loss": float(nll), "acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro",
                                       labels=list(range(len(EMOTIONS))), zero_division=0))}


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

@torch.no_grad()
def recalibrate_bn(model, loader, device="cpu"):
    """Re-estimate BatchNorm running statistics from clean (un-augmented) clips; weights are not changed.

    The running averages collected during training lag behind fast-changing weights and are built from
    augmented batches, so they can be off at evaluation time (see bn_check.py).
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        return
    old = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # None = plain average over all batches
    model.train()
    for x, _, _ in loader:
        model(to_dev(x, device))
    for m, mom in zip(bns, old):
        m.momentum = mom
    model.eval()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, y, _ in tqdm(loader, desc="  train", leave=False):
        x, y = to_dev(x, device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=["image", "audio", "early", "image_pre", "audio_pre"], default="image")
    ap.add_argument("--name", default="image_cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only N train and N val clips (smoke test)")
    ap.add_argument("--bn-recal", type=int, default=480,
                    help="clips used to re-estimate BatchNorm statistics before each validation (0 = off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing checkpoints of this name")
    a = ap.parse_args()

    seed_everything(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    CKPT.mkdir(exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    last, best = CKPT / f"{a.name}_last.pt", CKPT / f"{a.name}_best.pt"
    if last.exists() and not (a.resume or a.force):
        sys.exit(f"{last.name} exists. Use --resume to continue, --force to overwrite, or another --name.")

    model, DS = build(a.model, a)
    tr_idx, va_idx = load_index("train"), load_index("val")
    weights = class_weights(tr_idx)  # always from the full training split
    if a.limit:
        tr_idx = tr_idx.sample(min(a.limit, len(tr_idx)), random_state=a.seed)
        va_idx = va_idx.sample(min(a.limit, len(va_idx)), random_state=a.seed)
    train_loader = DataLoader(DS(index=tr_idx, augment=True), batch_size=a.batch_size, shuffle=True,
                              num_workers=a.workers, generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(DS(index=va_idx), batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    recal_loader = None
    if a.bn_recal > 0:
        recal_idx = tr_idx.sample(min(a.bn_recal, len(tr_idx)), random_state=a.seed)
        recal_loader = DataLoader(DS(index=recal_idx), batch_size=a.batch_size, shuffle=False,
                                  num_workers=a.workers)
        
    criterion = nn.CrossEntropyLoss(weight=weights)
    params = model.param_groups(a.lr) if hasattr(model, "param_groups") else model.parameters()
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    state = dict(epoch=0, best_f1=-1.0, best_epoch=0, no_improve=0, history=[], train_seconds=0.0)

    if a.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        state = ck["state"]
        print(f"Resumed from epoch {state['epoch']} (best val macro-F1 so far {state['best_f1']:.3f})")
    elif a.resume:
        print("No checkpoint to resume from; starting fresh.")

    print(f"model={a.model} | params={count_params(model)} | train clips={len(tr_idx)} | "
          f"val clips={len(va_idx)} | threads={torch.get_num_threads()}")

    for epoch in range(state["epoch"] + 1, a.epochs + 1):
        if state["no_improve"] >= a.patience:
            break
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if recal_loader is not None:
            recalibrate_bn(model, recal_loader, device)
        _, yv, pv = predict_probs(model, val_loader, device)
        val = score(yv, pv)
        scheduler.step()
        dt = time.time() - t0
        state["train_seconds"] += dt
        state["epoch"] = epoch
        rss = psutil.Process().memory_info().rss / 1e6

        if val["macro_f1"] > state["best_f1"] + 1e-4:
            state.update(best_f1=val["macro_f1"], best_epoch=epoch, no_improve=0)
            save_atomic({"model": model.state_dict(), "args": vars(a), "epoch": epoch,
                         "val_macro_f1": val["macro_f1"]}, best)
        else:
            state["no_improve"] += 1

        state["history"].append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val["loss"],
                                 "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
                                 "lr": optimizer.param_groups[0]["lr"], "seconds": dt, "ram_mb": rss})
        save_atomic({"model": model.state_dict(), "opt": optimizer.state_dict(),
                     "sched": scheduler.state_dict(), "state": state}, last)
        pd.DataFrame(state["history"]).to_csv(METRICS / f"train_{a.name}_history.csv", index=False)
        print(f"epoch {epoch:02d}/{a.epochs} | train loss {tr_loss:.3f} | val loss {val['loss']:.3f} | "
              f"val acc {val['acc']:.3f} | val macro-F1 {val['macro_f1']:.3f} | "
              f"{dt:.0f}s | RAM {rss:.0f} MB" + ("  *best*" if state["no_improve"] == 0 else ""))

    stopped = "early stopping" if state["no_improve"] >= a.patience else "reached max epochs"
    summary = {"name": a.name, "model": a.model, "params": count_params(model),
               "epochs_run": state["epoch"], "stopped_by": stopped, "best_epoch": state["best_epoch"],
               "best_val_macro_f1": state["best_f1"], "train_seconds": round(state["train_seconds"], 1),
               "seed": a.seed, "torch": torch.__version__, "args": vars(a)}
    (METRICS / f"train_{a.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone ({stopped}). Best epoch {state['best_epoch']}, val macro-F1 {state['best_f1']:.3f}, "
          f"total training time {state['train_seconds']:.0f}s")
    print(f"Best checkpoint: checkpoints\\{best.name} | Test split was not used.")


if __name__ == "__main__":
    main()