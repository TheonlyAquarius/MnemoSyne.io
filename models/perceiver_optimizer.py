import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "perceiver-pytorch-main"))
import torch
from torch import nn
from perceiver_pytorch.perceiver_io import PerceiverIO


class PerceiverOptimizer(nn.Module):
    def __init__(
        self,
        flat_dim: int,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 8,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        depth: int = 8,
        seq_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.flat_dim = flat_dim
        self.perceiver_io = PerceiverIO(
            dim=2,
            queries_dim=2,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            depth=depth,
            logits_dim=1,
            seq_dropout_prob=seq_dropout_prob,
        )

    def forward(
        self,
        weights_flat: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len = weights_flat.shape

        if t_end is None:
            time_tokens = t_start.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            time_tokens = torch.stack([t_start, t_end], dim=-1)
            time_tokens = time_tokens.unsqueeze(1).expand(-1, seq_len, -1)

        weights_expanded = weights_flat.unsqueeze(-1)
        input_seq = torch.cat((weights_expanded, time_tokens), dim=-1)
        output = self.perceiver_io(input_seq, queries=input_seq)
        return output.squeeze(-1)