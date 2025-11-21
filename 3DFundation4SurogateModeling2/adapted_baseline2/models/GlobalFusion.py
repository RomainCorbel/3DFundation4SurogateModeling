# models/GlobalFusion.py
import torch
import torch.nn as nn
from typing import Optional
from models.MLP import MLP

class GlobalFusion(nn.Module):
    def __init__(
        self,
        W_global_in: int,
        W_fuse: int,
    ):
        super().__init__()
        self.global_proj = MLP([W_global_in, 64, 64, W_fuse], batch_norm=False)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, local: torch.Tensor, g: torch.Tensor, batch: Optional[torch.Tensor] = None):
        N, device = local.size(0), local.device
        P = local
        G = self.global_proj(g.unsqueeze(0).to(device))
        G = G.expand(N, -1)
        H = P + self.alpha * G
        return H
