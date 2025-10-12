
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    """
    Improved prior-association for Anomaly Transformer.
    - Prior: learnable Mixture of Gaussians over relative temporal distance |i-j|
    - Series: scaled dot-product attention with adaptive temperature (per token & head)
    Returns (V, series, prior, sigma_effective) to keep compatibility with the solver.
    """
    def __init__(
        self,
        win_size: int,
        mask_flag: bool = True,
        scale: float | None = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
        mixture_components: int = 3,         # K >= 1; if 1, reduces to single Gaussian
        adaptive_temperature: bool = True,   # enable learnable tau
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.K = max(1, int(mixture_components))
        self.adaptive_temperature = bool(adaptive_temperature)
        self.eps = eps

        # Precompute |i-j| matrix for the window, kept as a buffer (moved to device during forward)
        dist = torch.arange(win_size).unsqueeze(0) - torch.arange(win_size).unsqueeze(1)
        dist = dist.abs().float()  # [L, L]
        self.register_buffer("distances", dist, persistent=False)

    def _row_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize last dimension so each row sums to 1 (avoid all-zero by eps)
        denom = x.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return x / denom

    def forward(
        self,
        queries: torch.Tensor,   # [B, L, H, E]
        keys: torch.Tensor,      # [B, L, H, E]
        values: torch.Tensor,    # [B, L, H, E]
        sigma_raw: torch.Tensor, # [B, L, H] or [B, L, H, K]
        attn_mask=None,
        mix_logits: torch.Tensor | None = None,  # [B, L, H, K] or None
        tau_raw: torch.Tensor | None = None,     # [B, L, H]   or None
    ):
        B, L, H, E = queries.shape
        device = queries.device

        # ---- 1) Scores ----
        # [B,H,L,L]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)

        # ---- 2) Adaptive temperature for series-attention ----
        # tau >= 0 via softplus; if not given, default tau=1
        if self.adaptive_temperature and tau_raw is not None:
            # tau_raw: [B,L,H] -> [B,H,L,1] for broadcasting over 'j'
            tau = F.softplus(tau_raw).transpose(1, 2).unsqueeze(-1) + self.eps
        else:
            tau = 1.0

        scale = self.scale or (1.0 / sqrt(E))
        logits = (scale * scores) / tau
        series = torch.softmax(logits, dim=-1)
        series = self.dropout(series)  # [B,H,L,L]

        # ---- 3) Mixture-of-Gaussians Prior over |i-j| ----
        # sigma_raw can be [B,L,H] or [B,L,H,K]; enforce 4D with K
        if sigma_raw.dim() == 3:
            sigma_raw = sigma_raw.unsqueeze(-1)  # [B,L,H,1]
        # sigma > 0 via softplus
        sigma = F.softplus(sigma_raw) + self.eps         # [B,L,H,K]
        sigma = sigma.transpose(1, 2)                    # [B,H,L,K]

        # mixture weights
        if self.K > 1 and mix_logits is not None:
            mix_w = torch.softmax(mix_logits, dim=-1).transpose(1, 2)  # [B,H,L,K]
        else:
            mix_w = torch.ones_like(sigma) / sigma.shape[-1]           # uniform over K

        # distances for broadcasting: [1,1,L,L,1]
        dist = self.distances.to(device).view(1, 1, L, L, 1)
        sigma_e = sigma.unsqueeze(3)                                    # [B,H,L,1,K]

        prior_k = (1.0 / (math.sqrt(2 * math.pi) * sigma_e)) * torch.exp(-(dist ** 2) / (2.0 * (sigma_e ** 2)))
        mix_e = mix_w.unsqueeze(3)                                      # [B,H,L,1,K]
        prior = (prior_k * mix_e).sum(dim=-1)                           # [B,H,L,L]
        prior = self._row_normalize(prior)

        # effective sigma (for logging/compat) -> weighted RMS per row: sqrt(sum_k w*sigma^2)
        sigma_eff = torch.sqrt((mix_w * (sigma ** 2)).sum(dim=-1)).contiguous()  # [B,H,L]

        # ---- 4) Apply series to values ----
        V = torch.einsum("bhls,bshd->blhd", series, values).contiguous()  # [B,L,H,E]

        if self.output_attention:
            return V, series, prior, sigma_eff
        else:
            return V, None


class AttentionLayer(nn.Module):
    """
    Multi-head attention wrapper (compatible with existing calls):
      - forward(queries, keys, values, attn_mask=None)
      - Projections for Gaussian mixture prior (sigma + mixture weights)
      - Projection for adaptive temperature
    Keeps return signature: (out, series, prior, sigma).
    """
    def __init__(
        self,
        attention: AnomalyAttention,
        d_model: int,
        n_heads: int,
        mixture_components: int = 3,         # default on: K=3
        adaptive_temperature: bool = True,   # default on
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.K = max(1, int(mixture_components))
        self.adaptive_temperature = bool(adaptive_temperature)

        self.inner_attention = attention
        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # Prior parameters per token (from queries before projection)
        self.sigma_projection = nn.Linear(d_model, n_heads * self.K)
        self.mix_projection   = nn.Linear(d_model, n_heads * self.K) if self.K > 1 else None

        # Adaptive temperature per token & head
        self.tau_projection   = nn.Linear(d_model, n_heads) if self.adaptive_temperature else None

        self.out_projection   = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask=None):
        """
        queries/keys/values: [B, L, d_model]
        returns (out, series, prior, sigma)
        """
        B, L, D = queries.shape
        H = self.n_heads

        # In the original code all three are the same tensor (self-attention). We keep compatibility.
        # Apply a pre-norm on queries; reuse the same normed tensor for K,V to match original.
        qn = self.norm(queries)
        kn = self.norm(keys)
        vn = self.norm(values)

        # Project Q,K,V and reshape to [B,L,H,E]
        Q = self.query_projection(qn).view(B, L, H, -1)
        K = self.key_projection(kn).view(B, L, H, -1)
        V = self.value_projection(vn).view(B, L, H, -1)

        # Prior params from queries
        sigma_raw = self.sigma_projection(qn).view(B, L, H, self.K)  # [B,L,H,K]
        mix_logits = self.mix_projection(qn).view(B, L, H, self.K) if self.mix_projection is not None else None

        # Temperature
        tau_raw = self.tau_projection(qn).view(B, L, H) if self.tau_projection is not None else None

        # Call inner attention (AnomalyAttention)
        out, series, prior, sigma = self.inner_attention(
            Q, K, V,
            sigma_raw, attn_mask,
            mix_logits, tau_raw
        )

        out = out.view(B, L, -1)  # concat heads
        return self.out_projection(out), series, prior, sigma
