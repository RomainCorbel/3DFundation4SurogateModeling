
# Standard library
import os
import os.path as osp
import re
import time
from glob import glob

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassMatthewsCorrCoef

# Local modules
from open3d.web_visualizer import draw  # for non-Colab
from models.point_net.point_net import PointNetClassHead
from UTILS.shapenet_dataset import ShapenetDataset

NUM_TRAIN_POINTS = 2500
NUM_TEST_POINTS = 10000
NUM_CLASSES = 16
GLOBAL_FEATS = 1024
BATCH_SIZE = 32
CATEGORIES = {
    'Airplane': 0, 
    'Bag': 1, 
    'Cap': 2, 
    'Car': 3,
    'Chair': 4, 
    'Earphone': 5, 
    'Guitar': 6, 
    'Knife': 7, 
    'Lamp': 8, 
    'Laptop': 9,
    'Motorbike': 10, 
    'Mug': 11, 
    'Pistol': 12, 
    'Rocket': 13, 
    'Skateboard': 14, 
    'Table': 15}     



def analyze_pointnet_backbone(
    MODEL_PATH: str,
    DEVICE,
    plots: bool = True,
    *,
    eps: float = 3.0,
    min_samples: int = 10,
    random_state: int = 42,
    gallery_per_cluster: int = 10,
    cluster_cmap_name: str = "tab20",
    npoints: int = 10000
):
    """
    Load a PointNet classifier, extract backbone features on test_dataloader,
    (optionally visualize with t-SNE and DBSCAN clustering).
    If plots=False, only compute embeddings and clustering, no visual output.
    """

    
    ROOT = osp.abspath("shapenet_like_out")
    # test Dataset & DataLoader 
    test_dataset = ShapenetDataset(ROOT, npoints=npoints, split='test', classification=True, normalize=False) #already normalized in convert_to_shapenet_like, see later if we keep it like that or not
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    # ------------------------------------------------------------------------------- Build and load model
    classifier = PointNetClassHead(
        num_points=NUM_TEST_POINTS, num_global_feats=GLOBAL_FEATS, k=NUM_CLASSES
    ).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    classifier.load_state_dict(state)
    classifier.eval()

    print(f"[{model_name}] Model loaded from: {MODEL_PATH}")

    # --- Extract global features from the backbone ---
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for pts, labels in test_dataloader:  # pts: (B,N,3)
            x = pts.transpose(2, 1).to(DEVICE)  # -> (B,3,N)
            gfeat, crit_idxs, _ = classifier.backbone(x)  # gfeat: (B,1024) usually
            gfeat = torch.nn.functional.normalize(gfeat, p=2, dim=1)  # optional L2
            all_feats.append(gfeat.cpu().numpy())
            all_labels.append(labels.squeeze(1).cpu().numpy())

    X = np.vstack(all_feats)          # (num_samples, feat_dim)
    y = np.concatenate(all_labels)    # (num_samples,)
    n_samples = X.shape[0]
    print(f"[{model_name}] Collected features: X={X.shape}, labels={y.shape}")
    
    save_dir = osp.join(os.getcwd(), "extracted_features", model_name)
    os.makedirs(save_dir, exist_ok=True)
    out_csv  = osp.join(save_dir, f"{model_name}_features.csv")
    out_parq = osp.join(save_dir, f"{model_name}_features.parquet")

    # Collect foil names from dataset (assumes DataLoader shuffle=False)
    if hasattr(test_dataset, "datapath"):
        foil_names = [
            osp.splitext(osp.basename(seg_path))[0]  # remove .seg extension
            for *_, seg_path, _ in test_dataset.datapath
        ]
    else:
        foil_names = [f"sample_{i:04d}" for i in range(X.shape[0])]

    # Align everything safely
    nrows = min(len(foil_names), X.shape[0], y.shape[0])
    foil_names = foil_names[:nrows]
    X = X[:nrows]
    y = y[:nrows]

    # Build DataFrame: foil_name, label, feat_0000..feat_1023
    feat_cols = [f"feat_{i:04d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df.insert(0, "foil_name", foil_names)
    df.insert(1, "label", y.astype(int))

    # Save to disk
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        print(f"[{model_name}] Parquet save skipped ({e}).")

    print(f"[{model_name}] Saved extracted features to:\n  - {out_csv}\n  - {out_parq if osp.exists(out_parq) else '(parquet not written)'}")
    # 
    if plots:
        from UTILS.plot_tsne import plot_extracted_features_tsne
        plot_extracted_features_tsne(X, test_dataset, model_name, save_dir=save_dir)
 
    return {
        "model_name": model_name,
        "features": X,          # (n_samples, feat_dim)
        "labels": y,            # (n_samples,)
    }

def analyze_pointnet_backbone2(
    MODEL_PATH: str,
    DEVICE,
    class_choice: list[str],   # ← ICI !
    data_folder: str,
    plots: bool = True,
    *,
    npoints: int = 10000
):
    """
    Load a PointNet classifier, extract backbone features on test_dataloader,
    (optionally visualize with t-SNE and DBSCAN clustering).
    If plots=False, only compute embeddings and clustering, no visual output.
    """

    
    ROOT = osp.abspath(data_folder)
    # test Dataset & DataLoader 
    test_dataset = ShapenetDataset(ROOT, class_choice = class_choice, npoints=npoints, split='test', classification=True, normalize=False) #pre process of the foils already done in the creation of the categories
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    # ------------------------------------------------------------------------------- Build and load model
    classifier = PointNetClassHead(
        num_points=NUM_TEST_POINTS, num_global_feats=GLOBAL_FEATS, k=NUM_CLASSES
    ).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    classifier.load_state_dict(state)
    classifier.eval()

    print(f"[{model_name}] Model loaded from: {MODEL_PATH}")

    # --- Extract global features from the backbone ---
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for pts, labels in test_dataloader:  # pts: (B,N,3)
            x = pts.transpose(2, 1).to(DEVICE)  # -> (B,3,N)
            gfeat, crit_idxs, _ = classifier.backbone(x)  # gfeat: (B,1024) usually
            gfeat = torch.nn.functional.normalize(gfeat, p=2, dim=1)  # optional L2
            all_feats.append(gfeat.cpu().numpy())
            all_labels.append(labels.squeeze(1).cpu().numpy())

    X = np.vstack(all_feats)          # (num_samples, feat_dim)
    y = np.concatenate(all_labels)    # (num_samples,)
    n_samples = X.shape[0]
    print(f"[{model_name}] Collected features: X={X.shape}, labels={y.shape}")
    
    save_dir = osp.join(os.getcwd(), "extracted_features", model_name)
    os.makedirs(save_dir, exist_ok=True)
    out_csv  = osp.join(save_dir, f"{model_name}_features.csv")
    out_parq = osp.join(save_dir, f"{model_name}_features.parquet")

    # Collect foil names from dataset (assumes DataLoader shuffle=False)
    if hasattr(test_dataset, "datapath"):
        foil_names = [
            osp.splitext(osp.basename(seg_path))[0]  # remove .seg extension
            for *_, seg_path, _ in test_dataset.datapath
        ]
    else:
        foil_names = [f"sample_{i:04d}" for i in range(X.shape[0])]

    # Align everything safely
    nrows = min(len(foil_names), X.shape[0], y.shape[0])
    foil_names = foil_names[:nrows]
    X = X[:nrows]
    y = y[:nrows]

    # Build DataFrame: foil_name, label, feat_0000..feat_1023
    feat_cols = [f"feat_{i:04d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df.insert(0, "foil_name", foil_names)
    df.insert(1, "label", y.astype(int))

    # Save to disk
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        print(f"[{model_name}] Parquet save skipped ({e}).")

    print(f"[{model_name}] Saved extracted features to:\n  - {out_csv}\n  - {out_parq if osp.exists(out_parq) else '(parquet not written)'}")
    # 
    if plots:
        from UTILS.plot_tsne import plot_extracted_features_tsne
        plot_extracted_features_tsne(X, test_dataset, model_name, save_dir=save_dir)
 
    return {
        "model_name": model_name,
        "features": X,          # (n_samples, feat_dim)
        "labels": y,            # (n_samples,)
    }
