# ============================ common_plots.py =============================
import os, os.path as osp
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

def plot_extracted_features_tsne(
    X,
    dataset,
    model_name,
    *,
    eps: float = 3.0,
    min_samples: int = 10,
    random_state: int = 42,
    gallery_per_cluster: int = 10,
    cluster_cmap_name: str = "tab20",
    perplexity: float = 30.0,
    save_dir: str | None = None,
):
    """
    Make the standardized plots used in your analyzers:
      1) t-SNE on features (cosine metric, PCA init)
      2) DBSCAN clustering in t-SNE space + colored scatter + legend
      3) Cluster galleries (2x5) sampling raw xy point clouds
      4) t-SNE colored by polygon area (white→red)
      5) t-SNE colored by inferred NACA parameters (camber, thickness)

    Args:
        X: (N, D) L2-normalized feature matrix.
        dataset: the ShapenetDataset used to produce X (no shuffle; aligns with X).
        model_name: used in titles and default filenames.
        labels: optional int labels (N,) for gallery subtitles.
        eps, min_samples: DBSCAN params (applied in t-SNE space).
        random_state: for reproducible t-SNE and gallery sampling.
        gallery_per_cluster: up to this many samples per cluster (capped at 10).
        cluster_cmap_name: e.g., "tab20".
        perplexity: t-SNE perplexity.
        save_dir: if provided, save PNGs and NumPy arrays here.

    Returns:
        dict with {"X2": (N,2) array, "cluster_ids": (N,) int array}
    """

    # ----------------- helpers -----------------
    def _compute_area_from_xy(xy: np.ndarray) -> float:
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 3:
            return 0.0
        c = xy.mean(axis=0)
        ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
        xy_sorted = xy[np.argsort(ang)]
        return Polygon(xy_sorted).area

    def _robust_minmax(a):
        a = np.asarray(a, dtype=float)
        valid = np.isfinite(a)
        if not np.any(valid): return 0.0, 1.0
        lo = np.nanpercentile(a[valid], 2)
        hi = np.nanpercentile(a[valid], 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(a[valid]), np.nanmax(a[valid])
            if lo == hi: hi = lo + 1e-12
        return lo, hi

    def _parse_naca_from_filename(name: str):
        try:
            base = os.path.basename(name)
            parts = base.split('_')
            nums = []
            for p in parts:
                try: nums.append(float(p))
                except ValueError: pass
            if len(nums) < 5: return None
            core = nums[2:]  # drop U∞ and α
            if len(core) == 4:
                L, P, Q, XX = core
                return {"series": 5, "camber": 0.15 * L, "thickness": XX / 100.0}
            elif len(core) == 3:
                M, P, XX = core
                return {"series": 4, "camber": M / 100.0, "thickness": XX / 100.0}
        except Exception:
            pass
        return None

    def _maybe_save(fig_name: str, arr_name: str | None, arr: np.ndarray | None):
        if not save_dir: return
        os.makedirs(save_dir, exist_ok=True)
        if fig_name:
            path = osp.join(save_dir, fig_name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[plot] saved {path}")
        if arr_name and arr is not None:
            path = osp.join(save_dir, arr_name)
            np.save(path, arr)
            print(f"[save] {arr_name} → {path}")

    rng = np.random.default_rng(random_state)

    # ----------------- 1) t-SNE -----------------
    tsne = TSNE(
        n_components=2,
        metric='cosine',
        init='pca',
        learning_rate='auto',
        perplexity=perplexity,
        random_state=random_state,
        verbose=1
    )
    X2 = tsne.fit_transform(X)
    print(f"[{model_name}] t-SNE computed: X2={X2.shape}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=7, edgecolors='none')
    plt.title(f'Backbone features (t-SNE) — {model_name}')
    plt.tight_layout()
    _maybe_save(f"{model_name}_tsne_embeddings.png", f"{model_name}_tsne_embeddings.npy", X2)
    plt.show()
# 
#     # ----------------- 2) DBSCAN on t-SNE -----------------
#     db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X2)
#     cids = db.labels_
#     uniq = sorted(set(cids) - {-1})
#     k = len(uniq)
#     n_noise = int((cids == -1).sum())
#     print(f"[{model_name}] DBSCAN: clusters={k}, noise={n_noise} (eps={eps}, min_samples={min_samples})")

#     base_cmap = plt.cm.get_cmap(cluster_cmap_name, max(k, 1))
#     cluster_colors = np.array([base_cmap(i) for i in range(max(k, 1))])
#     cid_to_idx = {cid: i for i, cid in enumerate(uniq)}
#     color_idx = np.array([cid_to_idx.get(cid, -1) for cid in cids])

#     colors = np.empty((len(cids), 4), dtype=float)
#     colors[color_idx >= 0] = cluster_colors[color_idx[color_idx >= 0]]
#     colors[color_idx < 0]  = (0.7, 0.7, 0.7, 0.6)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(X2[:, 0], X2[:, 1], s=7, c=colors, edgecolors='none')
#     legend_handles = [plt.Line2D([], [], marker='o', linestyle='', color=cluster_colors[i], label=f"Cluster {cid}")
#                       for i, cid in enumerate(uniq)]
#     if -1 in set(cids):
#         legend_handles.append(plt.Line2D([], [], marker='o', linestyle='', color=(0.7,0.7,0.7,0.9), label="Noise (-1)"))
#     plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title='DBSCAN clusters')
#     plt.title(f'Backbone features (t-SNE by DBSCAN clusters) — {model_name}\n'
#               f'{k} clusters, {n_noise} noise • eps={eps}, min_samples={min_samples}')
#     plt.tight_layout()
#     _maybe_save(f"{model_name}_tsne_dbscan.png", None, None)
#     plt.show()

#     # ----------------- 3) Cluster galleries -----------------
#     def _plot_gallery(indices, title, ncols=5):
#         nrows = 2
#         fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
#         axes = axes.ravel()
#         for ax, i in zip(axes, indices):
#             pts, lbl = dataset[int(i)]
#             pts = np.asarray(pts)
#             xy = pts[:, :2] if pts.ndim == 2 and pts.shape[1] >= 2 else pts
#             ax.scatter(xy[:, 0], xy[:, 1], s=1)
#             ax.set_aspect('equal', 'box')
#             ax.set_xticks([]); ax.set_yticks([])
#             lbl_int = int(lbl) if np.ndim(lbl) == 0 or (hasattr(lbl, "shape") and getattr(lbl, "shape", ()) == ()) else int(np.array(lbl).squeeze())
#             ax.set_title(f"idx {int(i)} | cls {lbl_int}", fontsize=8)
#         for ax in axes[len(indices):]:
#             ax.axis('off')
#         fig.suptitle(f"{title} — {model_name}", y=0.98)
#         plt.tight_layout()
#         plt.show()

#     for cid in np.unique(cids):
#         if cid == -1:
#             continue
#         idxs = np.where(cids == cid)[0]
#         kshow = min(gallery_per_cluster, len(idxs), 10)
#         if kshow <= 0:
#             continue
#         chosen = rng.choice(idxs, kshow, replace=False)
#         _plot_gallery(chosen, f"Cluster {cid} — {len(idxs)} samples")

    # ----------------- 4) Area-colored t-SNE -----------------
    areas = []
    # compute only as many as embeddings available (alignment guard)
    n_for_emb = X2.shape[0]
    for i in range(min(len(dataset), n_for_emb)):
        pts, _ = dataset[i]
        pts = np.asarray(pts)
        xy = pts[:, :2] if (pts.ndim == 2 and pts.shape[1] >= 2) else pts
        areas.append(_compute_area_from_xy(xy))
    areas = np.asarray(areas, dtype=np.float64)
    if areas.shape[0] < n_for_emb:
        areas = np.pad(areas, (0, n_for_emb - areas.shape[0]), constant_values=np.nan)

    vmin = np.nanpercentile(areas[:n_for_emb], 2)
    vmax = np.nanpercentile(areas[:n_for_emb], 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(areas[:n_for_emb]), np.nanmax(areas[:n_for_emb])

    white_red = LinearSegmentedColormap.from_list("white_red", [(1.0,1.0,1.0), (1.0,0.0,0.0)], N=256)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], s=7, c=areas[:n_for_emb], cmap=white_red, norm=norm, edgecolors='none')
    cbar = plt.colorbar(sc, shrink=0.9); cbar.set_label("Airfoil area")
    plt.title(f'Backbone features (t-SNE colored by airfoil area) — {model_name}')
    plt.tight_layout()
    _maybe_save(f"{model_name}_tsne_area.png", None, None)
    plt.show()

    # ----------------- 5) NACA-parameter plots -----------------
    if not hasattr(dataset, "datapath"):
        warnings.warn(f"[{model_name}] ShapenetDataset has no 'datapath'; skipping parameter-colored plots.")
    else:
        names = [os.path.splitext(os.path.basename(seg_path))[0] for _, _, seg_path, _ in dataset.datapath]
        n_all = len(names)
        camber_vals    = np.full(n_all, np.nan, dtype=float)
        thickness_vals = np.full(n_all, np.nan, dtype=float)
        series_ids     = np.full(n_all, -1, dtype=int)

        for i, nm in enumerate(names):
            info = _parse_naca_from_filename(nm)
            if info is None: continue
            camber_vals[i]    = info["camber"]
            thickness_vals[i] = info["thickness"]
            series_ids[i]     = 0 if info["series"] == 4 else 1

        n4 = int(np.sum(series_ids == 0)); n5 = int(np.sum(series_ids == 1)); nU = int(np.sum(series_ids == -1))
        print(f"[{model_name}] NACA kinds — 4-digit: {n4}, 5-digit: {n5}, unknown: {nU}")

        if n_all != n_for_emb:
            warnings.warn(f"[{model_name}] Count mismatch: tokens={n_all} vs embedding={n_for_emb}. "
                          "Ensure DataLoader uses shuffle=False and drop_last=False.")

        # align lengths
        camber_vals    = camber_vals[:n_for_emb]
        thickness_vals = thickness_vals[:n_for_emb]

        blue_grad = LinearSegmentedColormap.from_list("blue_grad", [(0.78,0.88,1.00), (0.05,0.10,0.60)], N=256)
        red_grad  = LinearSegmentedColormap.from_list("red_grad",  [(1.00,0.78,0.78), (0.60,0.05,0.05)], N=256)

        # camber
        camber_vmin, camber_vmax = _robust_minmax(camber_vals)
        plt.figure(figsize=(8, 6))
        sc1 = plt.scatter(X2[:, 0], X2[:, 1], s=7, c=camber_vals, cmap=blue_grad,
                          norm=plt.Normalize(vmin=camber_vmin, vmax=camber_vmax), edgecolors='none')
        plt.colorbar(sc1, shrink=0.9).set_label("Camber (4-digit: M/100 | 5-digit proxy: 0.15·L)")
        plt.title(f'Backbone features (t-SNE colored by camber) — {model_name}')
        plt.tight_layout()
        _maybe_save(f"{model_name}_tsne_camber.png", None, None)
        plt.show()

        # thickness
        thick_vmin, thick_vmax = _robust_minmax(thickness_vals)
        plt.figure(figsize=(8, 6))
        sc2 = plt.scatter(X2[:, 0], X2[:, 1], s=7, c=thickness_vals, cmap=red_grad,
                          norm=plt.Normalize(vmin=thick_vmin, vmax=thick_vmax), edgecolors='none')
        plt.colorbar(sc2, shrink=0.9).set_label("Thickness (XX / 100 of chord)")
        plt.title(f'Backbone features (t-SNE colored by thickness) — {model_name}')
        plt.tight_layout()
        _maybe_save(f"{model_name}_tsne_thickness.png", None, None)
        plt.show()

    # return {"X2": X2, "cluster_ids": cids}
    return {"X2": X2}
# ========================== end common_plots.py ==========================
