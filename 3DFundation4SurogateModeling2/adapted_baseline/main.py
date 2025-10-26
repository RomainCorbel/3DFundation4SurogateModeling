import argparse, yaml, os, json, glob
import torch
import train, metrics
from dataset import Dataset
import os.path as osp

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-s', '--score', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = int)
args = parser.parse_args()

with open('../Dataset/manifest.json', 'r') as f:
    manifest = json.load(f)

manifest_train = manifest[args.task + '_train']
test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(.1*len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]

if os.path.exists('save_dataset/train_dataset'):
     print("loading train_dataset and val_dataset")
     train_dataset = torch.load('save_dataset/train_dataset', map_location="cpu", weights_only=False)
     val_dataset = torch.load('save_dataset/val_dataset', map_location="cpu", weights_only=False)
     coef_norm = torch.load('save_dataset/normalization', map_location="cpu", weights_only=False)
else:
    print("Building train_dataset and val_dataset")
    train_dataset, coef_norm = Dataset(train_dataset, norm = True)
    val_dataset = Dataset(val_dataset, coef_norm=coef_norm)

    save_dir = 'save_dataset' 
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_dataset, osp.join(save_dir, 'train_dataset'))
    torch.save(coef_norm,     osp.join(save_dir, 'normalization'))
    torch.save(val_dataset,  osp.join(save_dir, 'val_dataset'))
    print(f"[SAVE] Train -> {osp.abspath(osp.join(save_dir,'train_dataset'))}")
    print(f"[SAVE] Val   -> {osp.abspath(osp.join(save_dir,'val_dataset'))}")
    print(f"[SAVE] Norm  -> {osp.abspath(osp.join(save_dir,'normalization'))}")


# Cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
device = 'cuda:0' if use_cuda else 'cpu'

if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

with open('params.yaml', 'r') as f: # hyperparameters of the model
    hparams = yaml.safe_load(f)[args.model]

from models.MLP import MLP
models = []
for i in range(args.nmodel):
    encoder = MLP(hparams['encoder'], batch_norm = False)
    decoder = MLP(hparams['decoder'], batch_norm = False)

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model = GraphSAGE(hparams, encoder, decoder)
    
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model = PointNet(hparams, encoder, decoder)

    elif args.model == 'MLP':
        from models.NN import NN
        model = NN(hparams, encoder, decoder)

    elif args.model == 'GUNet':
        from models.GUNet import GUNet
        model = GUNet(hparams, encoder, decoder)    

    # NEW global-fusion variants
    elif args.model == "GraphSAGEGlobal":
        from models.GraphSAGEGlobal import GraphSAGEGlobal
        model = GraphSAGEGlobal(hparams, encoder, decoder) 
    elif args.model == "GUNetGlobal":
        from models.GUNetGlobal import GUNetGlobal
        model = GUNetGlobal(hparams, encoder, decoder) 
    elif args.model == "MLPGlobal":
        from models.NNGlobal import NNGlobal
        model = NNGlobal(hparams, encoder, decoder) 
    elif args.model == "PointNetGlobal":
        from models.PointNetGlobal import PointNetGlobal
        model = PointNetGlobal(hparams, encoder, decoder) 
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    if "Global" in args.model:
        # quick sanity: first train sample must carry .g
        sample = train_dataset[0]
        if not hasattr(sample, "g"):
            raise RuntimeError("Global model selected but dataset samples have no `.g` global vector. "
                            "Enable use_global_features in Dataset() / provide the parquet.")

    log_path = osp.join('metrics', args.task, args.model) # path where you want to save log and figures    
    model = train.main(device, train_dataset, val_dataset, model, hparams, log_path, 
                criterion = 'MSE', val_iter = 10, reg = args.weight, name_mod = args.model, val_sample = True)
    models.append(model)
torch.save(models, osp.join('metrics', args.task, args.model, args.model))

if bool(args.score):
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    true_coefs, pred_mean, pred_std = metrics.Results_test(
        device, [models], [hparams], coef_norm,
        path_in='../Dataset', path_out='scores',
        n_test=3, criterion='MSE', s=s
    )

   # Créer le dossier pour ce modèle spécifique
    score_dir = os.path.join('scores', args.task, args.model)
    os.makedirs(score_dir, exist_ok=True)
    
    # Sauvegarder dans le bon dossier
    np.save(osp.join(score_dir, 'true_coefs'), true_coefs)
    np.save(osp.join(score_dir, 'pred_coefs_mean'), pred_mean)
    np.save(osp.join(score_dir, 'pred_coefs_std'), pred_std)

    print(f"Scores saved in: {score_dir}")