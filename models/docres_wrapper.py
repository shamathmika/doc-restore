# =============================================================================
# File: models/docres_wrapper.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Self-contained DocRes architecture and a DocResModel wrapper that matches
#   the interface of NAFNetModel so both can be swapped in
#   training and evaluation scripts without any code changes.
#
# Architecture: DocRes — Unified Document Restoration
#   Zhang et al., "DocRes: A Generalist Model Toward Unifying Document
#   Restoration Tasks", CVPR 2024
#   github.com/ZZZHANG-jx/DocRes
#
#   Implemented self-contained (no submodule) so the project runs on Colab
#   without extra setup. Core design: U-Net encoder-decoder with a
#   multi-head self-attention bottleneck and a global residual skip.
#
# Public API:
#   DocResModel(config_path)
#       .load_weights(checkpoint_path)
#       .forward(x)                  — same signature as NAFNetModel
#       .save_checkpoint(path, epoch, loss)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Two-conv residual block with optional channel projection."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            _ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class _AttentionBottleneck(nn.Module):
    """
    Spatial multi-head self-attention over flattened feature map tokens.
    Operates on (B, C, H, W) by reshaping to (B, H*W, C) and back.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        return (tokens + attn_out).permute(0, 2, 1).reshape(B, C, H, W)


class _EncoderBlock(nn.Module):
    """Res block followed by 2× strided downsampling."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.res = nn.Sequential(
            _ConvBNReLU(in_ch, out_ch),
            *[_ResBlock(out_ch) for _ in range(n_res - 1)],
        )
        self.down = nn.Conv2d(out_ch, out_ch, 2, stride=2)

    def forward(self, x: torch.Tensor):
        feat = self.res(x)
        return self.down(feat), feat   # (downsampled, skip)


class _DecoderBlock(nn.Module):
    """2× bilinear upsample → concat skip → Res block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.res = nn.Sequential(
            _ConvBNReLU(in_ch + skip_ch, out_ch),
            *[_ResBlock(out_ch) for _ in range(n_res - 1)],
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


class _DocResNet(nn.Module):
    """
    U-Net with attention bottleneck for document image restoration.

    Args:
        base_ch:    Feature channels at the first encoder level.
        depth:      Number of encoder/decoder levels.
        n_res:      ResBlocks per encoder/decoder level.
        num_heads:  Attention heads in the bottleneck.
    """

    def __init__(
        self,
        base_ch:   int = 64,
        depth:     int = 4,
        n_res:     int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.intro = nn.Conv2d(3, base_ch, 3, padding=1, bias=True)

        # Build encoder, tracking channel widths for symmetric decoder
        self.encoders: nn.ModuleList = nn.ModuleList()
        enc_channels = []
        in_ch = base_ch
        for _ in range(depth):
            out_ch = in_ch * 2
            self.encoders.append(_EncoderBlock(in_ch, out_ch, n_res))
            enc_channels.append(out_ch)
            in_ch = out_ch

        bottleneck_ch = in_ch
        self.bottleneck = nn.Sequential(
            _ResBlock(bottleneck_ch),
            _AttentionBottleneck(bottleneck_ch, num_heads),
            _ResBlock(bottleneck_ch),
        )

        self.decoders: nn.ModuleList = nn.ModuleList()
        for skip_ch in reversed(enc_channels):
            out_ch = skip_ch
            self.decoders.append(_DecoderBlock(in_ch, skip_ch, out_ch, n_res))
            in_ch = out_ch

        self.ending = nn.Conv2d(in_ch, 3, 3, padding=1, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.intro(inp)

        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # Global residual: add back the input for identity shortcut
        return (self.ending(x) + inp).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public wrapper — same interface as NAFNetModel (Sakshat's)
# ---------------------------------------------------------------------------

class DocResModel(nn.Module):
    """
    Wrapper around the DocRes backbone.

    Provides the same public interface as NAFNetModel so both models can be
    swapped in train_docres.py / train_nafnet.py and eval scripts without
    any code changes.

    Example
    -------
        model = DocResModel("configs/docres.yaml")
        model.load_weights("checkpoints/docres_best.pth")
        restored = model(degraded_batch)
        model.save_checkpoint("checkpoints/docres_ep5.pth", epoch=5, loss=0.038)
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to a YAML config (e.g. configs/docres.yaml).
                         Architecture keys are optional — sensible defaults
                         are used when absent (base_ch=64, depth=4, n_res=2).
        """
        super().__init__()
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        self.config = cfg
        self.net = _DocResNet(
            base_ch   = cfg.get("base_ch",   64),
            depth     = cfg.get("depth",      4),
            n_res     = cfg.get("n_res",      2),
            num_heads = cfg.get("num_heads",  4),
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_weights(self, checkpoint_path: str) -> None:
        """
        Load weights from a checkpoint file.

        Accepts both a bare state_dict and the wrapped dict produced by
        save_checkpoint() (with key "model_state_dict").
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.net.load_state_dict(state_dict)
        print(f"[DocResModel] Loaded weights from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of degraded images.

        Args:
            x: Float32 tensor in [0, 1], shape (B, 3, H, W).

        Returns:
            Restored images, float32 clamped to [0, 1], shape (B, 3, H, W).
        """
        return self.net(x)

    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """
        Save model weights and training metadata.

        Saved keys: model_state_dict, epoch, loss.

        Args:
            path:  Destination path (e.g. "checkpoints/docres_best.pth").
            epoch: Current epoch number (1-indexed).
            loss:  Validation loss at this checkpoint.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "epoch": epoch,
            "loss":  loss,
        }, path)
        print(f"[DocResModel] Saved checkpoint -> {path}  "
              f"(epoch={epoch}, loss={loss:.6f})")
