# models/PointNet.py
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from models.MLP import MLP
from models.GlobalFusion import GlobalFusion

class PointNetGlobal(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(PointNetGlobal, self).__init__()

        self.base_nb = hparams['base_nb']

        # Encoder/decoder (same interface as your baseline)
        self.encoder = encoder          # expects [N,7] -> [N,64]
        self.decoder = decoder          # expects [N, encoder[-1]] -> [N,1] (usually [N,64] -> [N,1])

        # ---- Global fusion right after encoder (merge at 64) ----
        self.use_global_fusion = hparams.get('use_global_fusion', False)
        if self.use_global_fusion:
            enc_out = hparams['encoder'][-1]  # 64
            self.fuse = GlobalFusion(
                W_local     = enc_out,                          # 64
                W_global_in = hparams.get('global_in', 1024),   # 1024
                W_fuse      = hparams.get('global_fuse', enc_out)  # 64
            )

        # ---- Standard PointNet blocks (unchanged) ----
        # Local point MLP
        self.in_block  = MLP([hparams['encoder'][-1], self.base_nb, self.base_nb*2], batch_norm=False)

        # Global token extractor (before pooling)
        self.max_block = MLP([self.base_nb*2, self.base_nb*4, self.base_nb*8, self.base_nb*32], batch_norm=False)

        # Out block consumes concatenation of local (bn*2) and repeated global (bn*32)
        self.out_block = MLP([self.base_nb*(32 + 2), self.base_nb*16, self.base_nb*8, self.base_nb*4], batch_norm=False)

        # Reduce back to encoder width before final decoder
        self.fcfinal = nn.Linear(self.base_nb*4, hparams['encoder'][-1])  # -> 64

    def forward(self, data):
        z = data.x.float()
        batch = getattr(data, "batch", None)
        if batch is None:
            # For safety; PointNet normally uses batched graphs
            batch = z.new_zeros(z.size(0), dtype=torch.long)

        # 1) Encoder: [N,7] -> [N,64]
        z = self.encoder(z)  # (N,64)

        # 2) Global fusion at 64: project g:1024->64, broadcast, residual add
        if self.use_global_fusion and hasattr(data, 'g'):
            z = self.fuse(z, data.g, batch)  # (N,64)

        # 3) Standard PointNet pipeline (unchanged)
        z_local = self.in_block(z)  # (N, base_nb*2)

        global_feat = self.max_block(z_local)                # (N, base_nb*32)
        global_feat = nng.global_max_pool(global_feat, batch=batch)  # (B, base_nb*32)

        # Repeat each graph's pooled global to match its number of points
        nb_points = torch.zeros(global_feat.shape[0], device=z.device, dtype=torch.long)
        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        global_rep = torch.repeat_interleave(global_feat, nb_points, dim=0)  # (N, base_nb*32)

        # Concatenate local and repeated global, then head
        z = torch.cat([z_local, global_rep], dim=1)          # (N, base_nb*(32+2))
        z = self.out_block(z)                                 # (N, base_nb*4)
        z = self.fcfinal(z)                                   # (N, 64)
        z = self.decoder(z)                                   # (N, 1)

        return z
