# foil_dataset.py
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from dataset import Dataset

import matplotlib

import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_PATH_IN = "../Dataset"  # folder containing manifest.json and the "Dataset/" dir


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_pts_xyz(path: str, xyz: np.ndarray) -> None:
    with open(path, "w") as f:
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_seg_labels(path: str, labels: np.ndarray) -> None:
    with open(path, "w") as f:
        for v in labels.astype(int):
            f.write(f"{int(v)}\n")


def make_png_scatter(path: str, xy: np.ndarray) -> None:
    fig = plt.figure(figsize=(3, 3), dpi=200)
    ax = plt.gca()
    ax.scatter(xy[:, 0], xy[:, 1], s=1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # Re-save via PIL to strip metadata and ensure PNG is clean
    Image.open(path).save(path)


def to_fixed_size(points_xy: np.ndarray, npoints: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    N = len(points_xy)
    if N == npoints:
        return points_xy
    if N > npoints:
        idx = rng.choice(N, size=npoints, replace=False)
    else:
        extra = rng.choice(N, size=npoints - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
    return points_xy[idx]


def tensor_to_numpy(t):
    if t is None:
        return None
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


'''def convert_to_shapenet_like(
    npoints: int = 100,
    path_in: str = DEFAULT_PATH_IN,
    root_out: str = "shapenet_like_out",
    category_name: str = "Airplane",
    category_id: str = "0",
    manifest_key: str = "full_test",   
):
    """
    Convert items listed in manifest[manifest_key] into a ShapeNet-like layout.
    Only the chosen manifest key (default: 'full_test') is used to build the *test* split.
    """
    # -------- inputs --------
    path_in = osp.abspath(path_in)                     # .../3D/Dataset
    if not osp.isdir(path_in):
        raise FileNotFoundError(f"Dossier Dataset introuvable: {path_in}")
    manifest_path = osp.join(path_in, "manifest.json")
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json introuvable: {manifest_path}")

    # parent containing "Dataset/"
    dataset_root = osp.dirname(path_in)                # .../3D

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if manifest_key not in manifest:
        raise KeyError(f"Clé '{manifest_key}' absente du manifest {manifest_path}")

    file_list = manifest[manifest_key]
    if not isinstance(file_list, (list, tuple)) or not file_list:
        raise ValueError(f"manifest['{manifest_key}'] est vide ou invalide.")

    # -------- outputs (ABSOLUTE paths) --------
    root_out_abs = osp.abspath(root_out)
    cat_dir     = ensure_dir(osp.join(root_out_abs, category_id))
    points_dir  = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))
    split_dir   = ensure_dir(osp.join(root_out_abs, "train_test_split"))

    with open(osp.join(root_out_abs, "synsetoffset2category.txt"), "w") as f:
        f.write(f"{category_name} {category_id}\n")

    test_tokens: list[str] = []

    # -------- conversion --------
    cwd_before = os.getcwd()
    try:
        # Change CWD so Dataset resolves "Dataset/<src>/..."
        os.chdir(dataset_root)

        for src in tqdm(file_list, desc=f"Converting '{manifest_key}' to ShapeNet-like"):
            data_list = Dataset([src], n_boot = npoints)
            if not (data_list and isinstance(data_list[0], Data)):
                raise RuntimeError(f"{src}: Dataset n'a pas retourné un Data valide.")
            data: Data = data_list[0]

            # Get (x, y)
            if hasattr(data, "pos") and data.pos is not None and data.pos.numel() > 0:
                xy = tensor_to_numpy(data.pos)
            else:
                x_tensor = getattr(data, "x", None)
                if x_tensor is None:
                    raise ValueError(f"{src}: Data ne contient pas 'pos' ni 'x'.")
                xy = tensor_to_numpy(x_tensor)[:, :2]

            xy  = to_fixed_size(xy, npoints)
            xyz = np.concatenate([xy, np.zeros((len(xy), 1), dtype=xy.dtype)], axis=1)
            seg = np.zeros((len(xy),), dtype=np.int64)

            raw_token = osp.splitext(osp.basename(src))[0]
            token = raw_token
            k = 1
            while osp.exists(osp.join(points_dir, f"{token}.pts")):
                token = f"{raw_token}_{k}"
                k += 1

            save_pts_xyz(osp.join(points_dir, f"{token}.pts"), xyz)
            save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)
            make_png_scatter(osp.join(segimg_dir, f"{token}.png"), xy)

            test_tokens.append(f"{category_id}/{token}")

    finally:
        os.chdir(cwd_before)

    # -------- splits --------
    with open(osp.join(split_dir, "shuffled_test_file_list.json"), "w") as f:
        json.dump(test_tokens, f, indent=2)
    # ensure empty train/val lists exist
    for empty in ("shuffled_train_file_list.json", "shuffled_val_file_list.json"):
        p = osp.join(split_dir, empty)
        if not osp.exists(p):
            with open(p, "w") as f:
                json.dump([], f)

    print(f"[OK] Test set prêt dans: {root_out_abs}")
    print(f"  Catégorie: {category_name} -> {category_id}")
    print(f"  Exemples (objets): {len(test_tokens)} | npoints/objet: {npoints}")
    print(f"  Manifest key utilisée: {manifest_key}")

    return {"root_out": root_out_abs, "tokens": test_tokens}
'''


def convert_to_shapenet_like(
    npoints: int = 100,
    path_in: str = DEFAULT_PATH_IN,
    root_out: str = "shapenet_like_out",
    category_name: str = "Airplane",
    category_id: str = "0",
    manifest_keys = ["full_test","full_train"],
):
    """
    Convert items listed in manifest[manifest_keys[1],manifest_keys[2],...] into a ShapeNet-like layout.
    Only the chosen manifest keys is used to build the *test* split.
    """
    # -------- inputs --------
    path_in = osp.abspath(path_in)                     # .../3D/Dataset
    if not osp.isdir(path_in):
        raise FileNotFoundError(f"Dossier Dataset introuvable: {path_in}")
    manifest_path = osp.join(path_in, "manifest.json")
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json introuvable: {manifest_path}")

    # parent containing "Dataset/"
    dataset_root = osp.dirname(path_in)                # .../3D

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    all_sources = []
    for key in manifest_keys:
        if key not in manifest:
            raise KeyError(f"Clé '{key}' absente du manifest {manifest_path}")
        file_list = manifest[key]
        if not isinstance(file_list, (list, tuple)) or not file_list:
            raise ValueError(f"manifest['{key}'] est vide ou invalide.")
        all_sources.extend(file_list)

    # -------- outputs (ABSOLUTE paths) --------
    root_out_abs = osp.abspath(root_out)
    cat_dir     = ensure_dir(osp.join(root_out_abs, category_id))
    points_dir  = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))
    split_dir   = ensure_dir(osp.join(root_out_abs, "train_test_split"))

    with open(osp.join(root_out_abs, "synsetoffset2category.txt"), "w") as f:
        f.write(f"{category_name} {category_id}\n")

    test_tokens: list[str] = []

    # -------- conversion --------
    cwd_before = os.getcwd()
    try:
        # Change CWD so Dataset resolves "Dataset/<src>/..."
        os.chdir(dataset_root)

        for src in tqdm(all_sources, desc=f"Converting to ShapeNet-like"):
            data_list = Dataset([src], n_boot = npoints)
            if not (data_list and isinstance(data_list[0], Data)):
                raise RuntimeError(f"{src}: Dataset n'a pas retourné un Data valide.")
            data: Data = data_list[0]

            # Get (x, y)
            if hasattr(data, "pos") and data.pos is not None and data.pos.numel() > 0:
                xy = tensor_to_numpy(data.pos)
            else:
                x_tensor = getattr(data, "x", None)
                if x_tensor is None:
                    raise ValueError(f"{src}: Data ne contient pas 'pos' ni 'x'.")
                xy = tensor_to_numpy(x_tensor)[:, :2]

            xy  = to_fixed_size(xy, npoints)
            xyz = np.concatenate([xy, np.zeros((len(xy), 1), dtype=xy.dtype)], axis=1)
            seg = np.zeros((len(xy),), dtype=np.int64)

            raw_token = osp.splitext(osp.basename(src))[0]
            token = raw_token
            k = 1
            while osp.exists(osp.join(points_dir, f"{token}.pts")):
                token = f"{raw_token}_{k}"
                k += 1

            save_pts_xyz(osp.join(points_dir, f"{token}.pts"), xyz)
            save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)
            make_png_scatter(osp.join(segimg_dir, f"{token}.png"), xy)

            test_tokens.append(f"{category_id}/{token}")

    finally:
        os.chdir(cwd_before)

    # -------- splits --------
    with open(osp.join(split_dir, "shuffled_test_file_list.json"), "w") as f:
        json.dump(test_tokens, f, indent=2)
    # ensure empty train/val lists exist
    for empty in ("shuffled_train_file_list.json", "shuffled_val_file_list.json"):
        p = osp.join(split_dir, empty)
        if not osp.exists(p):
            with open(p, "w") as f:
                json.dump([], f)

    print(f"[OK] Test set prêt dans: {root_out_abs}")
    print(f"  Catégorie: {category_name} -> {category_id}")
    print(f"  Exemples (objets): {len(test_tokens)} | npoints/objet: {npoints}")

    return {"root_out": root_out_abs, "tokens": test_tokens}