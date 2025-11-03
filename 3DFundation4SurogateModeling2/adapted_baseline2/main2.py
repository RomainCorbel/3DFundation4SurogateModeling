import argparse, yaml, os, json, glob
import torch
import train
import metrics2 
from dataset import Dataset
import os.path as osp
import numpy as np

from models.MLP import MLP

def run_training_pipeline(
    model: str,
    task: str,
    dataset_save_dir: str,
    features_dir: str,
    scores_dir: str,
    metrics_dir: str,
    param: str,
):
    """
    Minimal refactor of the original script into a function with only these parameters:
      - model: 'MLP' | 'GraphSAGE' | 'PointNet' | 'GUNet' | and their *Global variants
      - task: 'full' | 'scarce' | 'reynolds' | 'aoa'
      - dataset_save_dir: directory to cache/save the built Dataset objects
      - features_dir: directory containing the extracted global features (.parquet)
      - scores_dir: directory where scores/metrics arrays for test will be saved
      - metrics_dir: directory where training logs/figures (per-task/per-model) are saved

    Notes:
      - Keeps other behavior as in the original script: nmodel=1, weight=1, MSE loss, etc.
      - Always computes scores (was gated by --score before).
      - Uses the first *.parquet found under `features_dir` for global features.
    """

    # ---------- Args (formerly argparse) ----------
    args_model = model
    args_task = task
    nmodel = 1          # was: --nmodel (default 1)
    weight = 1.0        # was: --weight (default 1)
    # score flag removed; we now always compute scores because scores_dir is provided

    # ---------- Manifest & dataset splits ----------
    with open('../Dataset/manifest.json', 'r') as f:
        manifest = json.load(f)

    manifest_train = manifest[args_task + '_train']
    test_dataset = manifest[args_task + '_test'] if args_task != 'scarce' else manifest['full_test']
    n = int(.1 * len(manifest_train))
    train_dataset = manifest_train[:-n]
    val_dataset = manifest_train[-n:]
    ''' 
    if os.path.exists(dataset_save_dir):
        train_cache = osp.join(dataset_save_dir, 'train_dataset')
        val_cache   = osp.join(dataset_save_dir, 'val_dataset')
        test_cache  = osp.join(dataset_save_dir, 'test_dataset')
        norm_cache  = osp.join(dataset_save_dir, 'normalization')
        print("loading train_dataset, val_dataset and test_dataset")
        train_dataset = torch.load(train_cache, map_location="cpu", weights_only=False)
        val_dataset   = torch.load(val_cache,   map_location="cpu", weights_only=False)
        test_dataset  = torch.load(test_cache,  map_location="cpu", weights_only=False)
        coef_norm     = torch.load(norm_cache,  map_location="cpu", weights_only=False)

    else:
        train_cache = osp.join(dataset_save_dir, 'train_dataset')
        val_cache   = osp.join(dataset_save_dir, 'val_dataset')
        test_cache  = osp.join(dataset_save_dir, 'test_dataset')
        norm_cache  = osp.join(dataset_save_dir, 'normalization')
        print("Building train_dataset, val_dataset and test_dataset")
        train_dataset, coef_norm = Dataset(train_dataset, norm = True, global_features_parquet = features_dir)
        val_dataset = Dataset(val_dataset, coef_norm=coef_norm, global_features_parquet = features_dir)
        test_dataset = Dataset(test_dataset, coef_norm=coef_norm, global_features_parquet = features_dir)
        os.makedirs(dataset_save_dir, exist_ok=True)
        torch.save(train_dataset, osp.join(dataset_save_dir, 'train_dataset'))
        torch.save(coef_norm,     osp.join(dataset_save_dir, 'normalization'))
        torch.save(val_dataset,  osp.join(dataset_save_dir, 'val_dataset'))
        torch.save(test_dataset,  osp.join(dataset_save_dir, 'test_dataset'))
        print(f"[SAVE] Train -> {osp.abspath(osp.join(dataset_save_dir,'train_dataset'))}")
        print(f"[SAVE] Val   -> {osp.abspath(osp.join(dataset_save_dir,'val_dataset'))}")
        print(f"[SAVE] Norm  -> {osp.abspath(osp.join(dataset_save_dir,'normalization'))}")
        '''
    train_cache = osp.join(dataset_save_dir, "train_dataset")
    val_cache   = osp.join(dataset_save_dir, "val_dataset")
    test_cache  = osp.join(dataset_save_dir, "test_dataset")
    norm_cache  = osp.join(dataset_save_dir, "normalization")

    # load if everything is present; otherwise build and save
    if all(osp.exists(p) for p in [train_cache, val_cache, test_cache, norm_cache]):
        print("[CACHE] loading train_dataset, val_dataset, test_dataset, and normalization")
        train_dataset = torch.load(train_cache, map_location="cpu", weights_only=False)
        val_dataset   = torch.load(val_cache,   map_location="cpu", weights_only=False)
        test_dataset  = torch.load(test_cache,  map_location="cpu", weights_only=False)
        coef_norm     = torch.load(norm_cache,  map_location="cpu", weights_only=False)
    else:
        print("[BUILD] Building train_dataset, val_dataset, test_dataset, and normalization")
        # build train with normalization; reuse coef_norm for val/test
        train_dataset, coef_norm = Dataset(train_dataset, norm=True,  global_features_parquet=features_dir)
        val_dataset              = Dataset(val_dataset,   coef_norm=coef_norm, global_features_parquet=features_dir)
        test_dataset             = Dataset(test_dataset,  coef_norm=coef_norm, global_features_parquet=features_dir)

        os.makedirs(dataset_save_dir, exist_ok=True)
        torch.save(train_dataset, train_cache)
        torch.save(val_dataset,   val_cache)
        torch.save(test_dataset,  test_cache)
        torch.save(coef_norm,     norm_cache)

        print(f"[SAVE] Train -> {osp.abspath(train_cache)}")
        print(f"[SAVE] Val   -> {osp.abspath(val_cache)}")
        print(f"[SAVE] Test  -> {osp.abspath(test_cache)}")
        print(f"[SAVE] Norm  -> {osp.abspath(norm_cache)}")
    # ---------- Device ----------
    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    print('Using GPU' if use_cuda else 'Using CPU')

    # ---------- Hyperparameters ----------
    with open(f'{param}.yaml', 'r') as f:  # hyperparameters of the model
        hparams = yaml.safe_load(f)[args_model]

    # ---------- Model construction ----------
    models = []
    for i in range(nmodel):
        encoder = MLP(hparams['encoder'], batch_norm=False)
        decoder = MLP(hparams['decoder'], batch_norm=False)

        if args_model == 'GraphSAGE':
            from models.GraphSAGE import GraphSAGE
            model_obj = GraphSAGE(hparams, encoder, decoder)

        elif args_model == 'PointNet':
            from models.PointNet import PointNet
            model_obj = PointNet(hparams, encoder, decoder)

        elif args_model == 'MLP':
            from models.NN import NN
            model_obj = NN(hparams, encoder, decoder)

        elif args_model == 'GUNet':
            from models.GUNet import GUNet
            model_obj = GUNet(hparams, encoder, decoder)

        # NEW global-fusion variants
        elif args_model == "GraphSAGEGlobal":
            from models.GraphSAGEGlobal import GraphSAGEGlobal
            model_obj = GraphSAGEGlobal(hparams, encoder, decoder)
        elif args_model == "GUNetGlobal":
            from models.GUNetGlobal import GUNetGlobal
            model_obj = GUNetGlobal(hparams, encoder, decoder)
        elif args_model == "MLPGlobal":
            from models.NNGlobal import NNGlobal
            model_obj = NNGlobal(hparams, encoder, decoder)
        elif args_model == "PointNetGlobal":
            from models.PointNetGlobal import PointNetGlobal
            model_obj = PointNetGlobal(hparams, encoder, decoder)
        else:
            raise ValueError(f"Unknown model: {args_model}")

        if "Global" in args_model:
            # quick sanity: first train sample must carry .g
            sample = train_dataset[0]
            if not hasattr(sample, "g"):
                raise RuntimeError(
                    "Global model selected but dataset samples have no `.g` global vector. "
                    "Enable use_global_features in Dataset() / provide the parquet."
                )

        # ---------- Training & logging path ----------
        log_path = osp.join(metrics_dir, args_task, args_model)
        os.makedirs(log_path, exist_ok=True)

        model_obj = train.main(
            device, train_dataset, val_dataset, model_obj, hparams, log_path,
            criterion='MSE', val_iter=10, reg=weight, name_mod=args_model, val_sample=True
        )
        models.append(model_obj)

    # save trained models
    torch.save(models, osp.join(log_path, args_model))

    # ---------- Always compute and save scores ----------
    s = args_task + '_test' if args_task != 'scarce' else 'full_test'
    true_coefs, pred_mean, pred_std = metrics2.Results_test(
        device, [models], [hparams], coef_norm,
        path_in='../Dataset', path_out=scores_dir,
        n_test=3, criterion='MSE', s=s, test_dataset = test_dataset
    )

    # Save into structured folder: scores/<task>/<model>
    score_dir = os.path.join(scores_dir, args_task, args_model)
    os.makedirs(score_dir, exist_ok=True)
    np.save(osp.join(score_dir, 'true_coefs'), true_coefs)
    np.save(osp.join(score_dir, 'pred_coefs_mean'), pred_mean)
    np.save(osp.join(score_dir, 'pred_coefs_std'), pred_std)
    print(f"Scores saved in: {score_dir}")

    return {
        "device": device,
        "metrics_path": log_path,
        "scores_path": score_dir,
        "models_path": osp.join(log_path, args_model),
        "dataset_cache": {
            "train": train_cache,
            "val": val_cache,
            "test": test_cache,
            "norm": norm_cache,
        }
    }
