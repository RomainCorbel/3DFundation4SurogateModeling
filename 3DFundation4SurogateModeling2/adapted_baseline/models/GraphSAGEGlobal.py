import torch.nn as nn
import torch_geometric.nn as nng
from models.GlobalFusion import GlobalFusion

class GraphSAGEGlobal(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GraphSAGEGlobal, self).__init__()

        self.nb_hidden_layers   = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool            = hparams['bn_bool']
        self.activation         = nn.ReLU()

        # --- baseline modules unchanged ---
        self.encoder = encoder                         # e.g. [7,64,64,8]
        self.decoder = decoder
        self.dim_enc = hparams['encoder'][-1]          # e.g. 8  (IMPORTANT)

        # --- fuse AFTER encoder output (at dim_enc) ---
        self.use_global_fusion = hparams.get('use_global_fusion', False)
        if self.use_global_fusion:
            self.fuse = GlobalFusion(
                W_local     = self.dim_enc,               # fuse at 8
                W_global_in = hparams.get('global_in', 1024),
                W_fuse      = self.dim_enc,               # project g: 1024 -> 8
                bn_local    = False,
                dropout     = 0.0
            )
            # learn-from-zero: start at 0 but trainable
            # HARD KO by default: ensure alpha = 0 and frozen
            #with torch.no_grad():
            #    self.fuse.alpha.fill_(0.0)
            # self.fuse.alpha.requires_grad_(False)
            self.fuse.alpha.data.fill_(0.0)

        # --- GraphSAGE stack expects dim_enc in, same as baseline ---
        self.in_layer = nng.SAGEConv(
            in_channels  = self.dim_enc,                  # NOT 64
            out_channels = self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList([
            nng.SAGEConv(
                in_channels  = self.size_hidden_layers,
                out_channels = self.size_hidden_layers
            ) for _ in range(self.nb_hidden_layers - 1)
        ])

        self.out_layer = nng.SAGEConv(
            in_channels  = self.size_hidden_layers,
            out_channels = hparams['decoder'][0]          # should match baseline (e.g., 8)
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))
            for _ in range(self.nb_hidden_layers - 1):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, data):
        # Encoder output: (N, dim_enc) e.g., (N,8)
        z = self.encoder(data.x)

        # Fusion at dim_enc; with alpha=0 this is a no-op (baseline)
        if self.use_global_fusion and hasattr(data, 'g'):
            z = self.fuse(z, data.g, getattr(data, "batch", None))  # (N, dim_enc)

        # SAGE stack (unchanged)
        z = self.in_layer(z, data.edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for i, layer in enumerate(self.hidden_layers, start=1):
            z = layer(z, data.edge_index)
            if self.bn_bool:
                z = self.bn[i](z)
            z = self.activation(z)

        z = self.out_layer(z, data.edge_index)
        z = self.decoder(z)  # (N,1)
        return z
