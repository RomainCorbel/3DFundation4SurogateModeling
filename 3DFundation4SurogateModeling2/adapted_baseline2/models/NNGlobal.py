# models/NN.py
import torch
import torch.nn as nn
from models.MLP import MLP
from models.GlobalFusion import GlobalFusion

class NNGlobal(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(NNGlobal, self).__init__()

        self.nb_hidden_layers   = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool            = hparams['bn_bool']

        self.encoder = encoder            
        self.decoder = decoder        

        self.dim_enc = hparams['encoder'][-1]
        self.nn = MLP([self.dim_enc] + [self.size_hidden_layers]*self.nb_hidden_layers + [self.dim_enc],
                      batch_norm=self.bn_bool)

        self.fuse = GlobalFusion(
                W_global_in = hparams.get('global_in'),
                W_fuse      = self.dim_enc,              
            )

    def forward(self, data):
        z = self.encoder(data.x)                           
        z = self.fuse(z, data.g, getattr(data, "batch", None)) 
        z = self.nn(z)                          
        z = self.decoder(z)                     
        return z
