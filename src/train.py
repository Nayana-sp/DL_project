from pathlib import Path

code = '''import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from .models import AudioCNN, VisualCNN, MultimodalModel

CLASSES = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def actor_folds(actor_ids, n_splits=6, seed=13):
    # Corrected: distribute unique actors evenly across non-empty folds.
    actors = sorted(set(int(x) for x in actor_ids))
    if len(actors) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique actors, but found {len(actors)}: {actors}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    buckets = [[] for _ in range(n_splits)]
    for i, actor in enumerate(actors):
        buckets[i % n_splits].append(actor)
    return buckets

class FeatureDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        audio = np.load(row.audio_feature).astype(np.float32)
        video = np.load(row.video_feature).astype(np.float32)  # T,H,W,C uint8 originally
        audio = torch.from_numpy(audio).unsqueeze(0)
        video = torch.from_numpy(video).permute(0,3,1,2).float()/255.0
        # ImageNet normalisation for ResNet.
        mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
        std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
        video = (video-mean)/std
        if self.augment:
            if torch.rand(1).item() < .35:
                audio = audio + torch.randn_like(audio)*.08
            if torch.rand(1).item() < .25:
                audio = audio * (0.5 + torch.rand(1).item())
        label = int(row.emotion_code)-1
        return audio, video, label, row.clip_id

def run_epoch(model, loader, criterion, optimizer, device, train, mode, condition='clean'):
    model.train(train)
    total_loss, ys, preds, ids = 0., [], [], []
    for audio, video, labels, clip_ids in tqdm(loader, leave=False):
        audio, video, labels = audio.to(device), video.to(device), labels.to(device)
        if not train:
            # Fixed test-only perturbations; the same condition is applied to every model.
            if condition == "audio_missing":
                audio = torch.zeros_like(audio)
            elif condition == "video_missing":
                video = torch.zeros_like(video)
            elif condition == "audio_noise":
                generator = torch.Generator(device=audio.device).manual_seed(12345)
                audio = audio + torch.randn(audio.shape, generator=generator, device=audio.device) * 0.15
            elif condition == "video_blur":
                b, t, c, h, w = video.shape
                flat = video.reshape(b*t, c, h, w)
                flat = torch.nn.functional.avg_pool2d(flat, kernel_size=7, stride=1, padding=3)
                video = flat.reshape(b, t, c, h, w)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(audio, video) if mode not in ("audio","visual") else (model(audio) if mode=="audio" else model(video))
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item()*len(labels)
        ys.extend(labels.cpu().numpy().tolist())
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
        ids.extend(list(clip_ids))
    return total_loss/max(1,len(loader.dataset)), np.array(ys), np.array(preds), ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", choices=["audio","visual","early","late","gated","early_dropout","gated_dropout"], default="gated")
    ap.add_argument("--fold", type=int, default=0, choices=range(6))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output", default="results")
    args=ap.parse_args()
    seed_everything(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=pd.read_csv(args.manifest)
    df=df[df.preprocessing_status=="ok"].copy()
    if df.empty: raise SystemExit("Manifest has no successful rows. Run preprocessing first.")
    folds=actor_folds(df.actor_id.tolist(), n_splits=6, seed=args.seed)
    test_actors=set(folds[args.fold])
    val_actors=set(folds[(args.fold+1)%6])
    train_actors=set(a for i,b in enumerate(folds) if i not in (args.fold,(args.fold+1)%6) for a in b)
    train_df=df[df.actor_id.isin(train_actors)]
    val_df=df[df.actor_id.isin(val_actors)]
    test_df=df[df.actor_id.isin(test_actors)]
    if min(len(train_df),len(val_df),len(test_df))==0:
        raise SystemExit("Empty split. Ensure the manifest contains enough actors and all folds.")
    print(f"Device={device}; train actors={sorted(train_actors)}; validation actors={sorted(val_actors)}; test actors={sorted(test_actors)}")
    print(f"Rows train/val/test={len(train_df)}/{len(val_df)}/{len(test_df)}")
    train_loader=DataLoader(FeatureDataset(train_df, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader=DataLoader(FeatureDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader=DataLoader(FeatureDataset(test_df), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.model in ("audio","visual"):
        model=AudioCNN() if args.model=="audio" else VisualCNN()
    else:
        model=MultimodalModel(args.model, modality_dropout=args.model.endswith("_dropout"))
    model.to(device)
    counts=np.bincount(train_df.emotion_code.astype(int).values-1, minlength=8)
    weights=counts.sum()/np.maximum(counts,1)
    weights=weights/weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)
    out=Path(args.output)/f"{args.model}_fold{args.fold}_seed{args.seed}"
    out.mkdir(parents=True,exist_ok=True)
    best_f1=-1; best_epoch=0; bad=0; history={"train_loss":[],"val_loss":[],"val_macro_f1":[]}
    start=time.perf_counter()
    for epoch in range(args.epochs):
        tr_loss,_,_,_=run_epoch(model,train_loader,criterion,optimizer,device,True,args.model)
        va_loss,yv,pv,_=run_epoch(model,val_loader,criterion,None,device,False,args.model)
        vf1=f1_score(yv,pv,average="macro",zero_division=0)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss); history["val_macro_f1"].append(vf1)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_macro_f1={vf1:.4f}")
        if vf1>best_f1:
            best_f1,best_epoch,bad=vf1,epoch+1,0
            torch.save(model.state_dict(),out/"best_model.pt")
        else:
            bad+=1
            if bad>=args.patience:
                print("Early stopping.")
                break
    model.load_state_dict(torch.load(out/"best_model.pt",map_location=device))
    test_loss,yt,pt,ids=run_epoch(model,test_loader,criterion,None,device,False,args.model,condition="clean")
    report=classification_report(yt,pt,labels=list(range(8)),target_names=CLASSES,output_dict=True,zero_division=0)
    robustness = {}
    for condition in ("clean", "audio_missing", "video_missing", "audio_noise", "video_blur"):
        _, cy, cp, _ = run_epoch(model, test_loader, criterion, None, device, False, args.model, condition=condition)
        robustness[condition] = {
            "accuracy": float(accuracy_score(cy, cp)),
            "macro_f1": float(f1_score(cy, cp, average="macro", zero_division=0))
        }
    (out/"robustness_metrics.json").write_text(json.dumps(robustness, indent=2))
    cm=confusion_matrix(yt,pt,labels=list(range(8)))
    metrics={"model":args.model,"fold":args.fold,"seed":args.seed,"best_epoch":best_epoch,
             "test_loss":test_loss,"accuracy":accuracy_score(yt,pt),
             "macro_f1":f1_score(yt,pt,average="macro",zero_division=0),
             "test_actors":sorted(test_actors),"validation_actors":sorted(val_actors),
             "train_actors":sorted(train_actors),"device":str(device),"n_train":len(train_df),
             "n_validation":len(val_df),"n_test":len(test_df)}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(report).T.to_csv(out/"classification_report.csv")
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/"confusion_matrix.csv")
    pd.DataFrame({"clip_id":ids,"true":yt,"pred":pt}).to_csv(out/"test_predictions.csv",index=False)
    (out/"history.json").write_text(json.dumps(history,indent=2))
    plt.figure(); plt.plot(history["train_loss"],label="train loss"); plt.plot(history["val_loss"],label="validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig(out/"loss_curve.png",dpi=160); plt.close()
    elapsed=time.perf_counter()-start
    metrics["training_seconds"]=elapsed
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    print(f"Saved outputs to {out}")

if __name__=="__main__":
    main()
'''

path = Path("/mnt/data/train.py")
path.write_text(code, encoding="utf-8")
# Verify no accidental file-generation code is included and ending is correct.
lines = path.read_text(encoding="utf-8").splitlines()
assert lines[-2:] == ['if __name__=="__main__":', '    main()']
assert "path.write_text(code" not in path.read_text(encoding="utf-8")
print(f"Created {path} with {len(lines)} lines. Final lines: {lines[-2:]}")
