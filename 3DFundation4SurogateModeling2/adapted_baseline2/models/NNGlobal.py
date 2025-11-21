# models/NN.py
import torch
import torch.nn as nn
from models.MLP import MLP
from models.GlobalFusion import GlobalFusion

class NNGlobal(nn.Module):
    """
    Baseline NN with an optional global fusion *after* the encoder output.
    When alpha=0 in GlobalFusion, this is exactly the baseline.
    """
    def __init__(self, hparams, encoder, decoder):
        super(NNGlobal, self).__init__()

        self.nb_hidden_layers   = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool            = hparams['bn_bool']

        # Keep your original modules untouched:
        self.encoder = encoder                     # baseline encoder (e.g., [7,64,64,8])
        self.decoder = decoder                     # baseline decoder (e.g., [8,64,64,1])

        # This matches the baseline feature width passed to self.nn:
        self.dim_enc = hparams['encoder'][-1]      # e.g., 8
        # Baseline mid-MLP (unchanged):
        self.nn = MLP([self.dim_enc] + [self.size_hidden_layers]*self.nb_hidden_layers + [self.dim_enc],
                      batch_norm=self.bn_bool)

        self.use_global_fusion = hparams.get('use_global_fusion', False)
        if self.use_global_fusion:
            self.fuse = GlobalFusion(
                W_global_in = hparams.get('global_in'),
                W_fuse      = self.dim_enc,                    # project g: 1024 -> 8
            )

    def forward(self, data):
        z = self.encoder(data.x)                               # (N, dim_enc) — baseline ## CHECK THIS CLAIM check shapes 
        if self.use_global_fusion and hasattr(data, 'g'):
            z = self.fuse(z, data.g, getattr(data, "batch", None))  # alpha=0 → no-op
        z = self.nn(z)                                         # (N, dim_enc) — baseline mid MLP
        z = self.decoder(z)                                    # (N, 1)       — baseline
        return z
