# models/GlobalFusion.py
import torch
import torch.nn as nn
from typing import Optional
from models.MLP import MLP

class GlobalFusion(nn.Module):
    """
    GlobalFusion: fuse per-point local features with a single foil-level vector (g).

    Inputs
    ------
    local : Tensor, shape (N, W_local)
        Per-point features after your model's encoder (or any local block).
    g : Tensor, shape (1024,) or (B, 1024)
        Global foil descriptor(s). For batch_size=1, this is typically (1024,).
        For batches, provide (B, 1024) where B is the number of graphs in the batch.
    batch : Optional[Tensor], shape (N,)
        Graph index for each point (usual PyG `data.batch`). Required if g is batched.

    Behavior
    --------
    - Optionally projects `local` -> W_fuse (if W_local != W_fuse).
    - Projects `g` 1024 -> W_fuse once per graph.
    - Broadcasts the projected global vector across the graph's points (no memory copy).
    - Residual add: H = P_local + G_broadcast, returns (N, W_fuse).
    """

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
        # If not specified, project local only when dimensions differ.
        if use_local_proj is None:
            use_local_proj = (W_local != W_fuse)

        self.W_local = W_local
        self.W_global_in = W_global_in
        self.W_fuse = W_fuse
        self.use_local_proj = use_local_proj

        # Local path (pointwise)
        self.local_proj = (
            MLP([W_local, W_fuse], batch_norm=bn_local)
            if use_local_proj else nn.Identity()
        )

        # Global path (foil-level)
        self.global_proj = MLP([W_global_in, W_fuse], batch_norm=False)

        # Light normalization on the projected global feature helps stability
        self.global_norm = nn.LayerNorm(W_fuse) if norm_global else nn.Identity()

        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(
        self,
        local: torch.Tensor,           # (N, W_local)
        g: torch.Tensor,               # (1024,) or (B, 1024)
        batch: Optional[torch.Tensor] = None  # (N,) graph indices, required if g is batched
    ) -> torch.Tensor:
        N = local.size(0)
        device = local.device

        # Local projection to fusion width (if needed)
        if isinstance(self.local_proj, nn.Identity):
            P = local
        else:
            P = self.local_proj(local)  # (N, W_fuse)

        # Global projection once per graph
        if g.dim() == 1:
            # Single graph in the batch: (1024,) -> (1, 1024)
            g_in = g.unsqueeze(0).to(device)
            G = self.global_proj(g_in)              # (1, W_fuse)
            G = self.global_norm(G)                 # (1, W_fuse)
            G = G.expand(N, -1)                     # (N, W_fuse) (broadcasted view)
        elif g.dim() == 2:
            # Batched graphs: (B, 1024)
            if batch is None:
                # If there is no batch vector, assume g already matches per-point (rare),
                # otherwise we can't map graphs to points.
                if g.size(0) == N:
                    G = self.global_proj(g.to(device))  # (N, W_fuse)
                    G = self.global_norm(G)
                else:
                    raise ValueError(
                        "GlobalFusion: got batched g with shape (B, 1024) but no `batch` tensor to map points to graphs."
                    )
            else:
                g_in = g.to(device)                     # (B, 1024)
                G_graph = self.global_proj(g_in)        # (B, W_fuse)
                G_graph = self.global_norm(G_graph)     # (B, W_fuse)
                # Index-select per point via its graph id
                G = G_graph[batch]                      # (N, W_fuse)
        else:
            raise ValueError(f"GlobalFusion: unexpected g.dim()={g.dim()} (expected 1 or 2).")

        H = P + G                      # residual add
        H = self.dropout(H)            # optional small dropout to reduce over-reliance on G
        return H
