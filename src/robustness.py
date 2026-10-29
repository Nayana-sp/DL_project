"""Starter robustness evaluator utilities for already-preprocessed arrays."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def add_audio_noise(logmel, std):
    rng=np.random.default_rng(12345)
    return np.clip(logmel + rng.normal(0,std,size=logmel.shape).astype(np.float32), -2, 0)

def blur_video(frames, kernel=7):
    import cv2
    return np.stack([cv2.GaussianBlur(f,(kernel,kernel),0) for f in frames])

def main():
    ap=argparse.ArgumentParser(description="Create a fixed corruption manifest for controlled robustness experiments.")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output",default="results/robustness_plan.csv")
    args=ap.parse_args()
    df=pd.read_csv(args.manifest)
    df=df[df.preprocessing_status=="ok"].copy()
    rows=[]
    for _,r in df.iterrows():
        rows.extend([
            {"clip_id":r.clip_id,"condition":"clean","audio_scale":1.0,"video_scale":1.0},
            {"clip_id":r.clip_id,"condition":"audio_missing","audio_scale":0.0,"video_scale":1.0},
            {"clip_id":r.clip_id,"condition":"video_missing","audio_scale":1.0,"video_scale":0.0},
            {"clip_id":r.clip_id,"condition":"audio_degraded","audio_scale":0.5,"video_scale":1.0},
            {"clip_id":r.clip_id,"condition":"video_degraded","audio_scale":1.0,"video_scale":0.5},
        ])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(f"Wrote fixed corruption plan to {args.output}. Integrate these conditions into test-time inference before reporting robustness scores.")
if __name__=="__main__": main()
