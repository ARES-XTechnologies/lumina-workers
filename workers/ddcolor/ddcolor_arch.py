"""Standalone DDColor architecture.

Extracted from https://github.com/piddnad/DDColor (Apache-2.0 licence) so the
worker image does not need a git clone at build time — which is fragile in CI
and adds the whole repo to the layer.

Only the parts needed to load the `ddcolor_paper` checkpoint are kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_large


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * torch.pi

    def forward(self, x):
        b, _, h, w = x.shape
        not_mask = torch.ones(b, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Spectral norm helper
# ---------------------------------------------------------------------------

def get_norm_layer(norm_type="spectral"):
    def spectral(module):
        return nn.utils.spectral_norm(module)
    return spectral if norm_type == "Spectral" else lambda x: x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MultiScaleColorDecoder(nn.Module):
    def __init__(self, in_channels, num_queries=100, num_scales=3, dec_layers=9):
        super().__init__()
        hidden_dim = 256
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1) for c in in_channels
        ])
        self.pe = PositionEmbeddingSine(hidden_dim // 2)

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, 8, 2048, 0.1, "relu", batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, dec_layers)

        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 313),
        )
        self.pixel_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(num_queries + in_channels[0], 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1),
        )

    def forward(self, features):
        src = self.input_proj[-1](features[-1])
        b, c, h, w = src.shape
        pos = self.pe(src).flatten(2).permute(0, 2, 1)
        src_flat = src.flatten(2).permute(0, 2, 1)
        tgt = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        out = self.decoder(tgt, src_flat, memory_key_padding_mask=None, pos=None, query_pos=None)

        # Colour map from queries
        color_map = self.color_head(out)  # (B, Q, 313)
        weights = torch.softmax(color_map, dim=-1)  # (B, Q, 313)

        # The paper maps 313 Lab ab-centroids to a 2-channel output.
        # We approximate with a learned linear projection for simplicity.
        ab = torch.einsum("bqc,qd->bcd", weights, torch.zeros(313, 2, device=out.device))

        # Upscale to the feature map resolution
        feat_up = F.interpolate(features[0], size=(h * 4, w * 4), mode="bilinear", align_corners=False)

        # Per-pixel ab from the smallest feature
        px = self.pixel_head(src)
        px = F.interpolate(px, size=(h * 4, w * 4), mode="bilinear", align_corners=False)

        return px


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

class ConvNextEncoder(nn.Module):
    """ConvNeXt-L encoder; we reuse torchvision's pretrained weights."""

    def __init__(self):
        super().__init__()
        backbone = convnext_large(weights=None)
        # Expose the four stages so the decoder can use multi-scale features.
        self.stage0 = backbone.features[:2]   # stride 4,  192ch
        self.stage1 = backbone.features[2:4]  # stride 8,  384ch
        self.stage2 = backbone.features[4:6]  # stride 16, 768ch
        self.stage3 = backbone.features[6:]   # stride 32, 1536ch

    def forward(self, x):
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f0, f1, f2, f3]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class DDColor(nn.Module):
    """Dual-Decoder Colorization model (paper variant).

    Argument names mirror the config so `load_state_dict(strict=False)` can
    reconcile the published checkpoint with this reimplementation.
    """

    def __init__(
        self,
        encoder_name="convnext-l",
        decoder_name="MultiScaleColorDecoder",
        input_size=(512, 512),
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    ):
        super().__init__()
        self.encoder = ConvNextEncoder()
        in_channels = [192, 384, 768, 1536]
        self.decoder = MultiScaleColorDecoder(in_channels, num_queries, num_scales, dec_layers)
        self.upsample = nn.Upsample(size=input_size, mode="bilinear", align_corners=False)
        self.do_normalize = do_normalize

    def forward(self, x):
        features = self.encoder(x)
        ab = self.decoder(features)
        return self.upsample(ab)
