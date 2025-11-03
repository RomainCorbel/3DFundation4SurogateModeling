# models/GlobalFusion.py
import torch
import torch.nn as nn
from typing import Optional
from models.MLP import MLP

class GlobalFusion(nn.Module):
    def __init__(
        self,
        W_local: int,
        W_global_in: int = 1024,
        W_fuse: int = 64,
        use_local_proj: Optional[bool] = None,
        norm_global: bool = True,
        dropout: float = 0.0,
        bn_local: bool = False,
    ):
        super().__init__()

        if use_local_proj is None:
            use_local_proj = (W_local != W_fuse)

        # Local path
        self.local_proj = (
            MLP([W_local, W_fuse], batch_norm=bn_local)
            if use_local_proj else nn.Identity()
        )

        # Global path
        self.global_proj = MLP([W_global_in, W_fuse], batch_norm=False)
        self.global_norm = nn.LayerNorm(W_fuse) if norm_global else nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # --- KO gate: scalar α multiplying the global term ---
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, local: torch.Tensor, g: torch.Tensor, batch: Optional[torch.Tensor] = None):
        N, device = local.size(0), local.device
        P = local if isinstance(self.local_proj, nn.Identity) else self.local_proj(local)

        if g.dim() == 1:
            G = self.global_proj(g.unsqueeze(0).to(device))
            G = self.global_norm(G).expand(N, -1)
        elif g.dim() == 2 and batch is not None:
            G_graph = self.global_norm(self.global_proj(g.to(device)))
            G = G_graph[batch]
        else:
            raise ValueError("GlobalFusion: unexpected g shape or missing batch")

        # Residual fusion with α = 0 (hard KO)
        H = P + self.alpha * G
        return self.dropout(H)
