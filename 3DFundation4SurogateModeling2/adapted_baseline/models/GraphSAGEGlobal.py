import torch.nn as nn
import torch_geometric.nn as nng
from models.GlobalFusion import GlobalFusion  # <-- use the class from your models module

class GraphSAGEGlobal(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GraphSAGEGlobal, self).__init__()

        self.nb_hidden_layers   = hparams['nb_hidden_layers']   # total conv layers = 1 (in_layer) + (nb_hidden_layers - 1)
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool            = hparams['bn_bool']
        self.activation         = nn.ReLU()

        self.encoder = encoder          # expects [N,7] -> [N,64]
        self.decoder = decoder          # expects input features == hparams['decoder'][0] (e.g., 8) -> [N,1]

        # ---- Global fusion right after encoder (fuse at 64) ----
        self.use_global_fusion = hparams.get('use_global_fusion', False)
        if self.use_global_fusion:
            enc_out = hparams['encoder'][-1]                  # 64
            self.fuse = GlobalFusion(
                W_local     = enc_out,
                W_global_in = hparams.get('global_in', 1024), # 1024 by default
                W_fuse      = hparams.get('global_fuse', enc_out),  # 64 → fuse at 64
                bn_local    = False,
                dropout     = 0.0
            )

        # ---- GraphSAGE stack ----
        # First SAGEConv takes 64-d features (post-encoder/fusion) to hidden width
        self.in_layer = nng.SAGEConv(
            in_channels  = hparams['encoder'][-1],            # 64
            out_channels = self.size_hidden_layers            # e.g., 64
        )

        # Hidden SAGE layers (each keeps hidden width)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(
                nng.SAGEConv(
                    in_channels  = self.size_hidden_layers,
                    out_channels = self.size_hidden_layers
                )
            )

        # Final SAGE layer maps hidden width -> decoder input channels (e.g., 8)
        self.out_layer = nng.SAGEConv(
            in_channels  = self.size_hidden_layers,
            out_channels = hparams['decoder'][0]
        )

        # Optional BatchNorm: one BN per conv output (in_layer + each hidden layer)
        if self.bn_bool:
            self.bn = nn.ModuleList()
            # BN after in_layer
            self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))
            # BN after each hidden layer
            for _ in range(self.nb_hidden_layers - 1):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, data):
        # 1) pointwise encoder: [N,7] → [N,64]
        z = self.encoder(data.x)  # (N, 64)

        # 2) optional global fusion at 64-dim
        if self.use_global_fusion and hasattr(data, 'g'):
            z = self.fuse(z, data.g, getattr(data, "batch", None))  # (N, 64)

        # 3) SAGE: in_layer
        z = self.in_layer(z, data.edge_index)  # (N, hidden)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        # 4) SAGE: hidden layers
        for i, layer in enumerate(self.hidden_layers, start=1):
            z = layer(z, data.edge_index)  # (N, hidden)
            if self.bn_bool:
                z = self.bn[i](z)          # align BN index with layer index
            z = self.activation(z)

        # 5) SAGE: out layer to decoder input channels
        z = self.out_layer(z, data.edge_index)  # (N, decoder_in)

        # 6) per-point decoder to pressure
        z = self.decoder(z)  # (N, 1)

        return z
