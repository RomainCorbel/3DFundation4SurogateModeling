# models/PointNet.py
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from models.MLP import MLP
from models.GlobalFusion import GlobalFusion

class PointNetGlobal(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(PointNetGlobal, self).__init__()

        self.base_nb  = hparams['base_nb']
        self.encoder  = encoder   # baseline encoder, e.g. [7,64,64,8]
        self.decoder  = decoder
        self.dim_enc  = hparams['encoder'][-1]   # e.g., 8 (IMPORTANT)

        # ---- Global fusion AFTER encoder output (fuse at dim_enc) ----
        self.use_global_fusion = hparams.get('use_global_fusion', False)
        if self.use_global_fusion:
            self.fuse = GlobalFusion(
                W_local     = self.dim_enc,                   # fuse at 8
                W_global_in = hparams.get('global_in', 1024),
                W_fuse      = self.dim_enc,                   # 1024 -> 8
            )
            # learn-from-zero: start at 0 (trainable)
            # HARD KO by default: ensure alpha = 0 and frozen
            #with torch.no_grad():
            #    self.fuse.alpha.fill_(0.0)
            # self.fuse.alpha.requires_grad_(False)
            self.fuse.alpha.data.fill_(0.0)

        # ---- Standard PointNet blocks (unchanged topology) ----
        # Local point MLP takes encoder output width (dim_enc)
        self.in_block  = MLP([self.dim_enc, self.base_nb, self.base_nb*2], batch_norm=False)

        # Global token extractor before pooling
        self.max_block = MLP([self.base_nb*2, self.base_nb*4, self.base_nb*8, self.base_nb*32], batch_norm=False)

        # Out block consumes concat(local bn*2, repeated global bn*32)
        self.out_block = MLP([self.base_nb*(32 + 2), self.base_nb*16, self.base_nb*8, self.base_nb*4], batch_norm=False)

        # Reduce back to encoder width before final decoder (== dim_enc)
        self.fcfinal = nn.Linear(self.base_nb*4, self.dim_enc)

    def forward(self, data):
        z = data.x.float()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = z.new_zeros(z.size(0), dtype=torch.long)

        # 1) Encoder → (N, dim_enc)
        z = self.encoder(z)  # (N, dim_enc)

        # 2) Global fusion at dim_enc (α=0 → no-op; KO == baseline)
        if self.use_global_fusion and hasattr(data, 'g'):
            z = self.fuse(z, data.g, batch)  # (N, dim_enc)

        # 3) Standard PointNet pipeline
        z_local = self.in_block(z)                                 # (N, bn*2)
        global_feat = self.max_block(z_local)                      # (N, bn*32)
        global_feat = nng.global_max_pool(global_feat, batch)      # (B, bn*32)

        # repeat pooled global to per-point
        nb_points = torch.zeros(global_feat.shape[0], device=z.device, dtype=torch.long)
        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        global_rep = torch.repeat_interleave(global_feat, nb_points, dim=0)  # (N, bn*32)

        # concat and head
        z = torch.cat([z_local, global_rep], dim=1)                # (N, bn*(32+2))
        z = self.out_block(z)                                      # (N, bn*4)
        z = self.fcfinal(z)                                        # (N, dim_enc)
        z = self.decoder(z)                                        # (N, 1)
        return z
