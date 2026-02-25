import torch
import torch.nn as nn


class CARDModel(nn.Module):
    """
    CARD: Channel Aligned Robust Blend Transformer.
    Reference: Wang et al., "CARD: Channel Aligned Robust Blend Transformer
               for Time Series Forecasting" (ICLR 2024)

    Key idea: blend two complementary branches:
        Branch A — local branch (DLinear-style: series decomposition + linear)
                   captures trend and seasonal structure efficiently
        Branch B — global branch (patch self-attention)
                   captures long-range temporal dependencies

    A learned sigmoid gate α weights: output = α·A + (1−α)·B

    For univariate series (enc_in=1), both branches share the same channel,
    and the channel-alignment mechanism reduces to the gate alone.

    Compatible with TorchOneStepForecaster — expects:
        configs.seq_len   : lookback window length
        configs.pred_len  : forecast horizon (1 for the one-step forecaster)
        configs.enc_in    : number of channels (1 = univariate)
    """

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ── Branch A: Decomposition-Linear (local) ────────────────────────
        # Moving-average kernel for trend extraction
        kernel_size = min(25, max(3, configs.seq_len // 4))
        if kernel_size % 2 == 0:
            kernel_size += 1   # keep odd for symmetric boundary padding

        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=0
        )
        self.linear_trend    = nn.Linear(configs.seq_len, configs.pred_len)
        self.linear_seasonal = nn.Linear(configs.seq_len, configs.pred_len)

        # ── Branch B: Patch attention (global) ────────────────────────────
        self.patch_len = min(12, max(4, configs.seq_len // 8))
        self.stride_b  = max(1, self.patch_len // 2)
        num_patches = (configs.seq_len - self.patch_len) // self.stride_b + 1

        d_model = 32
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.patch_embed_b = nn.Linear(self.patch_len, d_model)
        self.attn_b        = nn.TransformerEncoder(attn_layer, num_layers=1)
        self.norm_b        = nn.LayerNorm(d_model)
        self.head_b        = nn.Linear(d_model * num_patches, configs.pred_len)

        # ── Learnable blend gate ──────────────────────────────────────────
        # Initialised at 0.0 so sigmoid(gate) = 0.5 → equal blend at start
        self.gate = nn.Parameter(torch.zeros(1))

    def _decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (seasonal, trend) using symmetric-padded moving average."""
        B, L = x.shape
        pad = (self.kernel_size - 1) // 2
        x_padded = torch.cat(
            [x[:, :1].expand(B, pad), x, x[:, -1:].expand(B, pad)], dim=1
        )
        trend = self.moving_avg(x_padded.unsqueeze(1)).squeeze(1)   # [B, L]
        seasonal = x - trend
        return seasonal, trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Channel-independent: processes channel 0 (close log-return) only.
        Args:
            x: [B, seq_len, enc_in]
        Returns:
            [B, pred_len, 1]
        """
        B, L, C = x.shape
        x = x[:, :, 0]           # [B, seq_len] — close channel

        # Instance normalisation
        x_last = x[:, -1:].detach()
        x_norm = x - x_last

        # ── Branch A ──────────────────────────────────────────────────────
        seasonal, trend = self._decompose(x_norm)
        out_a = (
            self.linear_seasonal(seasonal)
            + self.linear_trend(trend)
            + x_last
        )                                                             # [B, pred_len]

        # ── Branch B ──────────────────────────────────────────────────────
        patches = x_norm.unfold(1, self.patch_len, self.stride_b)    # [B, P, patch_len]
        tokens  = self.patch_embed_b(patches)                        # [B, P, d_model]
        tokens  = self.attn_b(tokens)
        tokens  = self.norm_b(tokens)
        out_b   = self.head_b(tokens.reshape(B, -1)) + x_last       # [B, pred_len]

        # ── Blend ─────────────────────────────────────────────────────────
        alpha = torch.sigmoid(self.gate)                              # scalar in (0, 1)
        out = alpha * out_a + (1.0 - alpha) * out_b

        return out.unsqueeze(-1)   # [B, pred_len, 1]

