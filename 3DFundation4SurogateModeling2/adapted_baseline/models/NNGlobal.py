# models/NN.py
import torch.nn as nn
from models.MLP import MLP
from models.GlobalFusion import GlobalFusion

class NNGlobal(nn.Module):
    """
    MLP + Global fusion :
      - encoder : [N,7] -> [N,64]
      - GlobalFusion : proj(g:1024->64), broadcast, addition résiduelle à 64
      - bloc MLP 'milieu' : [N,64] -> [N, decoder_in]
      - decoder : [N, decoder_in] -> [N,1]
    """
    def __init__(self, hparams, encoder, decoder):
        super(NNGlobal, self).__init__()

        self.nb_hidden_layers   = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool            = hparams['bn_bool']

        self.encoder = encoder
        self.decoder = decoder

        # largeur en sortie d'encodeur (doit être 64 pour la variante Global)
        self.dim_enc = hparams['encoder'][-1]            # attendu = 64 ici
        self.decoder_in = hparams['decoder'][0]          # ex: 64 (variante Global)

        # ---- Fusion globale juste après l'encodeur (merge à 64) ----
        self.use_global_fusion = hparams.get('use_global_fusion', False)
        if self.use_global_fusion:
            self.fuse = GlobalFusion(
                W_local     = self.dim_enc,                      # 64
                W_global_in = hparams.get('global_in', 1024),    # 1024
                W_fuse      = hparams.get('global_fuse', self.dim_enc)  # 64
            )

        # ---- MLP "milieu" (entre encoder/fusion et decoder) ----
        # Ex: [64] + [64]*nb_hidden + [decoder_in]
        channels = [self.dim_enc] + [self.size_hidden_layers]*self.nb_hidden_layers + [self.decoder_in]
        self.nn_mid = MLP(channels, batch_norm=self.bn_bool)

    def forward(self, data):
        # 1) encodeur local : [N,7] -> [N,64]
        z = self.encoder(data.x)  # (N, 64)

        # 2) fusion globale à 64 (projet(g), broadcast, addition)
        if self.use_global_fusion and hasattr(data, 'g'):
            z = self.fuse(z, data.g, getattr(data, "batch", None))  # (N, 64)

        # 3) MLP milieu
        z = self.nn_mid(z)    # (N, decoder_in)

        # 4) tête de sortie
        z = self.decoder(z)   # (N, 1)
        return z
