import numpy as np
import pyvista as pv
from reorganize import reorganize
import os.path as osp

import torch
from torch_geometric.data import Data

from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm





# ---------------------------------------------------------------------
# --- Sampling utilities
# ---------------------------------------------------------------------
def _sample_surface_points(aerofoil, n_points):
    """
    Uniformly sample N points on the airfoil polyline (by edge length).
    Returns:
        surf_pos : (N, 2) XY coordinates
        idx_edges: (N,)  indices of sampled edges
        u        : (N, 1) linear interpolation factor
    """
    lines = aerofoil.lines.reshape(-1, 3)[:, 1:]  # (E, 2) node indices per edge
    pts = aerofoil.points                        # (P, 3)

    # Edge lengths & probabilities
    seg = pts[lines]
    lengths = np.linalg.norm(seg[:, 1, :2] - seg[:, 0, :2], axis=1) + 1e-12
    p = lengths / lengths.sum()

    idx_edges = np.random.choice(len(lines), size=n_points, p=p)
    u = np.random.uniform(size=(n_points, 1))

    a = pts[lines[idx_edges, 0]][:, :2]
    b = pts[lines[idx_edges, 1]][:, :2]
    surf_pos = u * a + (1.0 - u) * b
    return surf_pos, idx_edges, u


# ---------------------------------------------------------------------
# --- Surface-level feature construction
# ---------------------------------------------------------------------
def _compute_surface_io(aerofoil, internal, case_name, n_points):
    """
    Builds surface-only inputs X [N,7] and targets y [N,1] for a foil case.
    X = [x, y, U∞_x, U∞_y, 0, n_x, n_y]
    y = wall pressure (p)
    """
    parts = case_name.split('_')
    Uinf = float(parts[2])
    alpha = float(parts[3]) * np.pi / 180.0
    Uinf_vec = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float32) * Uinf

    # Sample points on airfoil
    surf_pos, idx_edges, u = _sample_surface_points(aerofoil, n_points)
    lines = aerofoil.lines.reshape(-1, 3)[:, 1:]

    # Interpolate normals & pressure
    n0 = -aerofoil.point_data['Normals'][lines[idx_edges, 0], :2]
    n1 = -aerofoil.point_data['Normals'][lines[idx_edges, 1], :2]
    nxny = u * n0 + (1.0 - u) * n1

    p0 = aerofoil.point_data['p'][lines[idx_edges, 0]]
    p1 = aerofoil.point_data['p'][lines[idx_edges, 1]]
    p = (u[:, 0] * p0 + (1.0 - u[:, 0]) * p1).astype(np.float32)

    # Build X
    N = surf_pos.shape[0]
    X = np.zeros((N, 7), dtype=np.float32)
    X[:, 0:2] = surf_pos
    X[:, 2:4] = Uinf_vec[None, :]
    X[:, 4] = 0.0  # distance to wall
    X[:, 5:7] = nxny

    # Target y = pressure
    y = p.reshape(-1, 1)

    # Torch tensors
    pos = torch.tensor(surf_pos, dtype=torch.float32)
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    surf_mask = torch.ones(N, dtype=torch.bool)

    return pos, x, y, surf_mask


# ---------------------------------------------------------------------
# --- Optional: load or compute 1024-d global descriptor
# ---------------------------------------------------------------------
def _load_global_parquet(parquet_path: str) -> dict[str, np.ndarray]:
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(parquet_path)
    name_col = df.columns[0]
    label_col = df.columns[1]
    # Colonnes de features = toutes les colonnes sauf celle du nom.
    feat_cols = [c for c in df.columns if c not in [name_col, label_col]]
    G = {}
    for _, row in df.iterrows():
        foil_name = str(row[name_col]).strip()
        feats = row[feat_cols].to_numpy(dtype=np.float32)
        G[foil_name] = feats
    return G

# ---------------------------------------------------------------------
# --- Main dataset builder
# ---------------------------------------------------------------------
'''
def Dataset(
    set,
    *,
    norm: bool = False,
    coef_norm=None,
    sample='uniform',
    surf_ratio=1,
    n_surface_points: int = 1000,
    # --- Global features (defaults to your cls_focal_clr model) ---
    global_features_parquet: str | None = "../point_net/extracted_features/cls_model_15/cls_model_15_features.parquet",
    use_global_features: bool = True,
):
    """
    Builds list of torch_geometric Data objects for training.

    Each Data includes:
        - x: [N,7] local inputs
        - y: [N,1] wall pressure
        - pos: [N,2] coordinates
        - surf: mask (all True)
        - g: [1024] optional global descriptor
    """
    if norm and coef_norm is not None:
        raise ValueError('Provide either norm=True or coef_norm, not both.')

    dataset = []

    # ---- Load global features (optional) ----
    G = None
    if use_global_features:
        if not osp.exists(global_features_parquet):
            raise FileNotFoundError(global_features_parquet)
        G = _load_global_parquet(global_features_parquet)
    # ----------------------------------------

    # Accumulators for normalization
    if norm and coef_norm is None:
        old_length = 0
        mean_in = None
        mean_out = None

    for k, s in enumerate(tqdm(set)):
        internal = pv.read(osp.join('..', 'Dataset', s, s + '_internal.vtu'))
        aerofoil = pv.read(osp.join('..', 'Dataset', s, s + '_aerofoil.vtp'))

        # Local surface data
        pos, x, y, surf = _compute_surface_io(aerofoil, internal, s, n_surface_points)
        data = Data(pos=pos, x=x, y=y, surf=surf)

        # --- Attach global feature vector if available ---
        if G is not None:
            foil_key = s.strip()
            if foil_key in G:
                data.g = torch.tensor(G[foil_key], dtype=torch.float32)  # (1024,)
        # -------------------------------------------------

        # --- Running mean for normalization ---
        if norm and coef_norm is None:
            xi = x.numpy()
            yi = y.numpy()
            if k == 0:
                old_length = xi.shape[0]
                mean_in = xi.mean(axis=0, dtype=np.double)
                mean_out = yi.mean(axis=0, dtype=np.double)
            else:
                new_length = old_length + xi.shape[0]
                mean_in += (xi.sum(axis=0, dtype=np.double) - xi.shape[0]*mean_in) / new_length
                mean_out += (yi.sum(axis=0, dtype=np.double) - xi.shape[0]*mean_out) / new_length
                old_length = new_length

        dataset.append(data)

    # --- Normalization ---
    if norm and coef_norm is None:
        mean_in = mean_in.astype(np.float32)
        mean_out = mean_out.astype(np.float32)
        old_length = 0
        std_in = None
        std_out = None

        for k, data in enumerate(dataset):
            xi = data.x.numpy()
            yi = data.y.numpy()
            if k == 0:
                old_length = xi.shape[0]
                std_in = ((xi - mean_in)**2).sum(axis=0, dtype=np.double) / old_length
                std_out = ((yi - mean_out)**2).sum(axis=0, dtype=np.double) / old_length
            else:
                new_length = old_length + xi.shape[0]
                std_in += (((xi - mean_in)**2).sum(axis=0, dtype=np.double) - xi.shape[0]*std_in) / new_length
                std_out += (((yi - mean_out)**2).sum(axis=0, dtype=np.double) - xi.shape[0]*std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in).astype(np.float32)
        std_out = np.sqrt(std_out).astype(np.float32)

        for data in dataset:
            data.x = (data.x - torch.tensor(mean_in)) / (torch.tensor(std_in) + 1e-8)
            data.y = (data.y - torch.tensor(mean_out)) / (torch.tensor(std_out) + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)
        return dataset, coef_norm

    elif coef_norm is not None:
        mi, si, mo, so = coef_norm
        mi, si, mo, so = map(torch.tensor, (mi, si, mo, so))
        for data in dataset:
            data.x = (data.x - mi) / (si + 1e-8)
            data.y = (data.y - mo) / (so + 1e-8)
    return dataset
'''
def Dataset(
    set,
    *,
    norm: bool = False,
    coef_norm=None,
    sample='uniform',
    surf_ratio=1,
    n_surface_points: int = 1000,
    global_features_parquet: str | None = "../point_net/extracted_features/cls_model_15/cls_model_15_features.parquet",
    use_global_features: bool = True,
):
    """
    Builds list of torch_geometric Data objects for training / eval.

    Returns:
      - (dataset, coef_norm) if norm=True and coef_norm is None  (first call)
      - dataset              if coef_norm is provided or norm=False

    coef_norm has 6 elements:
      (mean_x, std_x, mean_y, std_y, mean_g or None, std_g or None)

    g is a [1024] global feature vector per foil (optional).
    """
    # -----------------------------
    # Decide what we do this call
    # -----------------------------
    compute_norm = (norm and coef_norm is None)

    # -----------------------------
    # Load global features (optional)
    # -----------------------------
    G = None
    if use_global_features:
        if not osp.exists(global_features_parquet):
            raise FileNotFoundError(global_features_parquet)
        G = _load_global_parquet(global_features_parquet)  # dict: foil_id -> (1024,)

    dataset = []

    # Lists for computing normalization when needed
    xs, ys = [], []
    gs = []   # one g per foil

    # -----------------------------
    # FIRST PASS — build Data list
    # -----------------------------
    for s in tqdm(set):
        internal = pv.read(osp.join('..', 'Dataset', s, f"{s}_internal.vtu"))
        aerofoil = pv.read(osp.join('..', 'Dataset', s, f"{s}_aerofoil.vtp"))

        pos, x, y, surf = _compute_surface_io(aerofoil, internal, s, n_surface_points)
        data = Data(pos=pos, x=x, y=y, surf=surf)

        # Attach global features if available
        if G is not None:
            key = s.strip()
            if key in G:
                g_np = np.asarray(G[key], dtype=np.float32)  # (1024,)
                data.g = torch.tensor(g_np)
                if compute_norm:
                    gs.append(g_np)
            else:
                data.g = None
        else:
            data.g = None

        # Collect x,y for normalization on first call
        if compute_norm:
            xs.append(x.numpy())
            ys.append(y.numpy())

        dataset.append(data)

    # --------------------------------------------------
    # Case 1: no norm and no coef_norm → return raw data
    # --------------------------------------------------
    if not norm and coef_norm is None:
        return dataset

    # ==================================================
    # COMPUTE NORMALIZATION COEFS (first call only)
    # ==================================================
    if compute_norm:
        X = np.vstack(xs)  # [sum_points, 7]
        Y = np.vstack(ys)  # [sum_points, 1]

        mean_x = X.mean(axis=0).astype(np.float32)
        std_x  = X.std(axis=0).astype(np.float32)

        mean_y = Y.mean(axis=0).astype(np.float32)
        std_y  = Y.std(axis=0).astype(np.float32)

        if len(gs) > 0:
            Gmat = np.vstack(gs)  # [num_foils, 1024]
            mean_g = Gmat.mean(axis=0).astype(np.float32)
            std_g  = Gmat.std(axis=0).astype(np.float32)
        else:
            mean_g = None
            std_g  = None

        coef_norm = (mean_x, std_x, mean_y, std_y, mean_g, std_g)

    # At this point, if we reach here, we must have coef_norm
    mean_x, std_x, mean_y, std_y, mean_g, std_g = coef_norm

    mean_x = torch.tensor(mean_x)
    std_x  = torch.tensor(std_x)
    mean_y = torch.tensor(mean_y)
    std_y  = torch.tensor(std_y)

    mean_g = torch.tensor(mean_g) if mean_g is not None else None
    std_g  = torch.tensor(std_g) if std_g is not None else None

    # ==================================================
    # APPLY NORMALIZATION (train / val / test)
    # ==================================================
    for data in dataset:
        data.x = (data.x - mean_x) / (std_x + 1e-8)
        data.y = (data.y - mean_y) / (std_y + 1e-8)

        if data.g is not None and mean_g is not None:
            data.g = (data.g - mean_g) / (std_g + 1e-8)

    # ==================================================
    # RETURN FORMAT
    # ==================================================
    if compute_norm:
        # first call: training set
        return dataset, coef_norm
    else:
        # val / test: coef_norm was given
        return dataset