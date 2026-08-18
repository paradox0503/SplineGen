import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import MLP


def monotonic_parameters(logits, mask, epsilon=1e-6):
    """Convert per-point logits into strictly increasing parameters on [0, 1]."""
    if logits.ndim != 2 or logits.shape != mask.shape:
        raise ValueError("logits and mask must both have shape [batch, sequence]")

    mask = mask.bool()
    lengths = mask.sum(dim=-1)
    if torch.any(lengths < 2):
        raise ValueError("every open curve must contain at least two valid points")
    expected = torch.arange(mask.size(1), device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError("valid ordered points must form a prefix of the padded sequence")

    interval_mask = mask[:, 1:] & mask[:, :-1]
    increments = (F.softplus(logits[:, :-1]) + epsilon) * interval_mask.to(logits.dtype)
    cumulative = torch.cat(
        [torch.zeros_like(increments[:, :1]), torch.cumsum(increments, dim=-1)], dim=-1
    )
    last_index = (lengths - 1).unsqueeze(-1)
    totals = cumulative.gather(1, last_index).clamp_min(epsilon)
    params = cumulative / totals
    return params.masked_fill(~mask, 0.0)


class MonotonicParameterHead(nn.Module):
    """Parallel ordered parameter decoder conditioned on generated knot states."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.knot_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Linear(d_model, 1)

    def forward(self, point_embeddings, point_mask, knot_embeddings=None, knot_mask=None):
        features = point_embeddings
        if knot_embeddings is not None:
            if knot_mask is None:
                knot_mask = torch.ones(
                    knot_embeddings.shape[:2], dtype=torch.bool, device=knot_embeddings.device
                )
            attended, _ = self.knot_attention(
                query=features,
                key=knot_embeddings,
                value=knot_embeddings,
                key_padding_mask=~knot_mask.bool(),
                need_weights=False,
            )
            features = self.attention_norm(features + attended)

        features = self.sequence_encoder(
            features, src_key_padding_mask=~point_mask.bool()
        )
        logits = self.output(features).squeeze(-1)
        return monotonic_parameters(logits, point_mask)


class paramDecoderMLP(nn.Module):
    def __init__(self, c_input:int,hidden_dim:int,dropout=0.1,output_dim=1) -> None:
        super().__init__()
        self.c_input = c_input
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.mlp=MLP(c_input,hidden_channels=[hidden_dim,hidden_dim,output_dim],norm_layer=nn.LayerNorm,dropout=dropout,activation_layer=nn.ReLU)

    def forward(self,x,mask):
        return self.mlp(x)

class paramDecoderTransformer(nn.Module):
    def __init__(self, c_input:int,hidden_dim:int,dropout=0.1) -> None:
        super().__init__()
        self.c_input = c_input
        self.hidden_dim = hidden_dim
        self.mlp=nn.TransformerEncoder(nn.TransformerEncoderLayer(c_input, 4, hidden_dim, dropout), 1)
    def forward(self,x,mask):
        return self.mlp(x)