# extract_pointmae_global.py
import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# plotting (optional)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# dataset
import sys
ROOT = osp.abspath(osp.join(os.getcwd(), "..", ".."))
sys.path.append(os.path.join(ROOT, 'utils'))
from shapenet_dataset import ShapenetDataset


# ------------------ main API ------------------
def _unit_sphere(x):  # normalize each cloud to unit sphere
    c = x.mean(1, keepdim=True)
    r = torch.quantile(torch.norm(x - c, dim=2), 0.95, dim=1, keepdim=True).unsqueeze(-1)
    return (x - c) / torch.clamp(r, 1e-6)

def analyse_pointMAE_backbone(model, device, npoints=10000, plots=True):
    torch.manual_seed(42)
    np.random.seed(42)

    # Dataset setup
    ROOT_dataset = osp.join(ROOT, "shapenet_like_out")
    ds = ShapenetDataset(ROOT_dataset, npoints=npoints, split="test", classification=True, normalize=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # Patch the pre-trained model (if required)
    model.eval()

    feats, labels = [], []
    with torch.no_grad():
        for pts, lbl in dl:
            x = _unit_sphere(pts.to(device).float())
            g = torch.nn.functional.normalize(model.forward_features(x), p=2, dim=1)
            feats.append(g.cpu().numpy())
            labels.append(lbl.squeeze(-1).cpu().numpy())

    X = np.vstack(feats)
    y = np.concatenate(labels).astype(int)
    print(f"[PointMAE-pretrain] features: {X.shape}, labels: {y.shape}")

    # Saving extracted features
    save_dir = osp.join(ROOT, "extracted_features", "PointMAE_pretrain")
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"feat_{i:04d}" for i in range(X.shape[1])])

    # Get dataset names and labels
    names = [osp.splitext(osp.basename(seg))[0] for *_, seg, _ in ds.datapath]
    n = min(len(df), len(names), len(y))
    df = df.iloc[:n].copy()
    df.insert(0, "foil_name", names[:n])
    df.insert(1, "label", y[:n])
    
    # Output paths
    out_csv  = osp.join(save_dir, "pointmae_features.csv")
    out_parq = osp.join(save_dir, "pointmae_features.parquet")
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        print(f"[{model}] Parquet save skipped ({e}).")

    # Plot features if required
    if plots:
        try:
            import sys
            sys.path.append(str(ROOT / 'utils'))
            from utils.plot_tsne import plot_extracted_features_tsne
            plot_extracted_features_tsne(X, ds, "PointMAE_pretrain", save_dir=save_dir)
        except Exception as e:
            print(f"[PointMAE-pretrain] plotting skipped: {e}")

    return {"model_name": "PointMAE_pretrain", "features": X, "labels": y}
