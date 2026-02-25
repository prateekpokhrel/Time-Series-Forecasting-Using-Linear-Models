import torch
import torch.nn as nn


class PatchTSTModel(nn.Module):
    """
    PatchTST: Time Series Transformer with patch tokenization.
    Reference: Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)

    Channel-independent multivariate: each feature channel is processed through
    the same Transformer independently. Only channel 0 (close log-return) is
    returned as the prediction output, but all channels share model weights
    during training, enabling cross-channel knowledge transfer.

    Compatible with TorchOneStepForecaster — expects:
        configs.seq_len   : lookback window length
        configs.pred_len  : forecast horizon (1 for the one-step forecaster)
        configs.enc_in    : number of channels (N_FEATURES)
    """

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in   = getattr(configs, "enc_in", 1)

        self.patch_len   = min(16, max(4, configs.seq_len // 8))
        self.stride      = max(1, self.patch_len // 2)
        self.num_patches = (configs.seq_len - self.patch_len) // self.stride + 1

        d_model = 64
        nhead   = 4

        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm        = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model * self.num_patches, configs.pred_len)

    def _forward_channel(self, x_c: torch.Tensor) -> torch.Tensor:
        """[B, seq_len] → [B, pred_len]"""
        B    = x_c.size(0)
        last = x_c[:, -1:].detach()
        xn   = x_c - last
        patches = xn.unfold(1, self.patch_len, self.stride)
        tokens  = self.patch_embed(patches) + self.pos_embed[:, :xn.unfold(1, self.patch_len, self.stride).size(1), :]
        out     = self.norm(self.transformer(tokens))
        return self.head(out.reshape(B, -1)) + last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, enc_in]  →  [B, pred_len, 1]"""
        return self._forward_channel(x[:, :, 0]).unsqueeze(-1)
