import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "perceiver-pytorch-main"))
import torch
from torch import nn
from perceiver_pytorch import PerceiverIO


class PerceiverOptimizer(nn.Module):
    def __init__(self, flat_dim: int, depth: int = 6, num_latents: int = 256, latent_dim: int = 256):
        super().__init__()
        self.flat_dim = flat_dim
        self.io = PerceiverIO(
            depth=depth,
            dim=1,
            queries_dim=1,
            logits_dim=1,
            num_latents=num_latents,
            latent_dim=latent_dim,
        )

    def forward(self, weights: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor | None = None) -> torch.Tensor:
        if t_end is None:
            times = t_start.unsqueeze(-1)
        else:
            times = torch.stack([t_start, t_end], dim=-1)
        data = torch.cat([weights.unsqueeze(-1), times.unsqueeze(-1)], dim=1)
        queries = torch.zeros(weights.size(0), self.flat_dim, 1, device=weights.device)
        out = self.io(data, queries=queries)
        return out.squeeze(-1)
