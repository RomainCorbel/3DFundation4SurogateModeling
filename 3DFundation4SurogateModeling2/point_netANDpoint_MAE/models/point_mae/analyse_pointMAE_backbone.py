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
from utils.shapenet_dataset import ShapenetDataset

# model builder + classes
from models.point_mae.models.build import build_model_from_cfg
from models.point_mae.models.Point_MAE import Point_MAE, PointTransformer

# ---------------------------------------------------------------
# PointMAE analyzer
# ---------------------------------------------------------------

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
# ------------------------------ helpers ------------------------------

# extract_pointmae_pretrain_features.py
import os
import os.path as osp
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.shapenet_dataset import ShapenetDataset

from models.point_mae.models.build import build_model_from_cfg
from models.point_mae.models.Point_MAE import Point_MAE, PointTransformer

# ------------------------------ utils ------------------------------
import os, os.path as osp, yaml
from types import SimpleNamespace
# fixed paths
CFG_PATH = "models/point_mae/cfgs/pretrain.yaml"
CKPT_PATH = "models/point_mae/models/checkpoints/pretrain.pth"


# ------------------ helpers ------------------
def _to_attr(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_attr(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_attr(v) for v in x]
    return x

def _unit_sphere(x):  # normalize each cloud to unit sphere
    c = x.mean(1, keepdim=True)
    r = torch.quantile(torch.norm(x - c, dim=2), 0.95, dim=1, keepdim=True).unsqueeze(-1)
    return (x - c) / torch.clamp(r, 1e-6)

def _fps(x, m):  # simple FPS
    B, N, _ = x.shape
    if m >= N: return x
    dev = x.device
    idx = torch.zeros(B, m, dtype=torch.long, device=dev)
    dist = torch.full((B, N), 1e10, device=dev)
    far = torch.norm(x - x.mean(1, keepdim=True), dim=2).argmax(1)
    idx[:, 0] = far
    for i in range(1, m):
        last = x[torch.arange(B), idx[:, i-1]].unsqueeze(1)
        dist = torch.minimum(dist, torch.cdist(last, x).squeeze(1))
        idx[:, i] = dist.argmax(1)
    return x[torch.arange(B).unsqueeze(1), idx]

# ------------------ model setup ------------------
def _build_pointmae_pretrain(device):
    with open(CFG_PATH, "r") as f:
        full = yaml.safe_load(f)
    cfg = _to_attr(full.get("model", full.get("MODEL", full)))

    model = Point_MAE(cfg).to(device)
    state = torch.load(CKPT_PATH, map_location="cpu")
    state = state.get("state_dict") or state.get("model") or state
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model

def _patch_pointmae_pretrain(model: Point_MAE):
    if not hasattr(model, "MAE_encoder") or not hasattr(model, "group_divider"):
        raise RuntimeError("Not a PointMAE pretrain encoder.")

    # --- force no masking (different repos store it in different places)
    for obj in (
        model,
        getattr(model, "MAE_encoder", None),
        getattr(model, "config", None),
        getattr(model, "args", None),
        getattr(getattr(model, "args", None), "transformer_config", None),
    ):
        if obj is not None and hasattr(obj, "mask_ratio"):
            try: setattr(obj, "mask_ratio", 0.0)
            except Exception: pass

    def forward_features(pts_bnc: torch.Tensor):  # (B,N,3)
        neigh, center = model.group_divider(pts_bnc)
        # some repos accept noaug, some don't—be permissive
        try:
            out = model.MAE_encoder(neigh, center, noaug=True)
        except TypeError:
            out = model.MAE_encoder(neigh, center)

        toks = out[0] if isinstance(out, (tuple, list)) else out  # (B,T,C)

        # prefer CLS if present, else mean over tokens
        if getattr(model, "cls_token", None) is not None and toks.size(1) >= 1:
            feat = toks[:, 0, :]
        else:
            feat = toks.mean(1)
        return feat

    model.forward_features = forward_features
    # optional sanity:
    mr = None
    for obj in (model, getattr(model, "MAE_encoder", None), getattr(model, "config", None)):
        if obj is not None and hasattr(obj, "mask_ratio"): mr = getattr(obj, "mask_ratio")
    print(f"[PointMAE] effective mask_ratio: {mr}")
    return model


# ------------------ main API ------------------
def analyse_pointMAE_backbone(device, npoints=10000, plots=True):
    torch.manual_seed(42)
    np.random.seed(42)

    ROOT = osp.abspath("shapenet_like_out")
    ds = ShapenetDataset(ROOT, npoints=npoints, split="test", classification=True, normalize=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = _build_pointmae_pretrain(device)
    _patch_pointmae_pretrain(model)
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

    save_dir = osp.join(os.getcwd(), "extracted_features", "PointMAE_pretrain")
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"feat_{i:04d}" for i in range(X.shape[1])])

    names = [osp.splitext(osp.basename(seg))[0] for *_, seg, _ in ds.datapath]
    n = min(len(df), len(names), len(y))
    df = df.iloc[:n].copy()
    df.insert(0, "foil_name", names[:n])
    df.insert(1, "label", y[:n])
    out_csv  = osp.join(save_dir, "pointmae_features.csv")
    out_parq = osp.join(save_dir, "pointmae_features.parquet")
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        print(f"[{model}] Parquet save skipped ({e}).")

    if plots:
        try:
            from utils.plot_tsne import plot_extracted_features_tsne
            plot_extracted_features_tsne(X, ds, "PointMAE_pretrain", save_dir=save_dir)
        except Exception as e:
            print(f"[PointMAE-pretrain] plotting skipped: {e}")

    return {"model_name": "PointMAE_pretrain", "features": X, "labels": y}

