# =============================================================================
# File: models/nafnet_wrapper.py
# Owner: Sakshat
#
# Purpose:
#   Self-contained NAFNet architecture and a NAFNetModel wrapper class that
#   matches the interface of DocResModel (Apoorva's) so both can be swapped
#   in training and evaluation scripts without any changes.
#
# Architecture: NAFNet — Nonlinear Activation Free Network
#   Chen et al., "Simple Baselines for Image Restoration", ECCV 2022
#   github.com/megvii-research/NAFNet
#   Implemented directly to keep the project self-contained on Colab.
#
# Public API:
#   NAFNetModel(config_path)
#       .load_weights(checkpoint_path)
#       .forward(x)                  — same signature as DocResModel
#       .save_checkpoint(path, epoch, loss)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# NAFNet building blocks
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.Module):
    """Per-channel LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias   = nn.Parameter(torch.zeros(channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _SimpleGate(nn.Module):
    """Splits channels in half and multiplies element-wise. Replaces all nonlinear activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class _NAFBlock(nn.Module):
    """
    One NAFNet residual block.
    Branch 1: LayerNorm → 1×1 conv → depth-wise 3×3 → SimpleGate → SCA → 1×1 conv
    Branch 2: LayerNorm → 1×1 conv → SimpleGate → 1×1 conv (FFN)
    Learnable per-block scale parameters beta and gamma.
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand

        self.norm1 = _LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.sg1   = _SimpleGate()
        self.sca   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1, bias=True)

        self.norm2 = _LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.sg2   = _SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, 1, bias=True)

        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Branch 1
        x = self.sg1(self.conv2(self.conv1(self.norm1(inp))))
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        # Branch 2 (FFN)
        x = self.conv5(self.sg2(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class _NAFNet(nn.Module):
    """
    NAFNet encoder-decoder backbone.

    Args:
        img_channel:    Input/output channels (3 for RGB).
        width:          Base feature width (number of channels after intro conv).
        middle_blk_num: NAFBlocks in the bottleneck.
        enc_blks:       NAFBlocks per encoder level.
        dec_blks:       NAFBlocks per decoder level.
    """

    def __init__(
        self,
        img_channel:    int        = 3,
        width:          int        = 32,
        middle_blk_num: int        = 12,
        enc_blks:       list[int]  = None,
        dec_blks:       list[int]  = None,
    ):
        super().__init__()
        enc_blks = enc_blks or [2, 2, 4, 8]
        dec_blks = dec_blks or [2, 2, 2, 2]

        self.intro  = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()

        chan = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[_NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[_NAFBlock(chan) for _ in range(middle_blk_num)])

        for n in dec_blks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(chan) for _ in range(n)]))

        # Input must be divisible by 2^depth for skip connections to align
        self._pad_to = 2 ** len(enc_blks)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, H, W = inp.shape
        inp_padded = self._pad(inp)

        x = self.intro(inp_padded)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x) + skip
            x = dec(x)

        # Residual skip over the whole network + crop back to original size
        return (self.ending(x) + inp_padded)[:, :, :H, :W]

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = (self._pad_to - h % self._pad_to) % self._pad_to
        pw = (self._pad_to - w % self._pad_to) % self._pad_to
        return F.pad(x, (0, pw, 0, ph))


# ---------------------------------------------------------------------------
# Public wrapper — same interface as DocResModel (Apoorva's)
# ---------------------------------------------------------------------------

class NAFNetModel(nn.Module):
    """
    Wrapper around the NAFNet backbone.

    Provides the same public interface as DocResModel so both models can be
    swapped in train_nafnet.py / train_docres.py and eval scripts without
    any code changes.

    Example
    -------
        model = NAFNetModel("configs/nafnet.yaml")
        model.load_weights("checkpoints/nafnet_best.pth")
        restored = model(degraded_batch)
        model.save_checkpoint("checkpoints/nafnet_ep5.pth", epoch=5, loss=0.042)
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to a YAML config (e.g. configs/nafnet.yaml).
                         Architecture keys are optional — sensible defaults
                         are used when absent (width=32, middle_blks=12,
                         enc_blks=[2,2,4,8], dec_blks=[2,2,2,2]).
        """
        super().__init__()
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        self.config = cfg
        self.net = _NAFNet(
            img_channel    = 3,
            width          = cfg.get("width",       32),
            middle_blk_num = cfg.get("middle_blks", 12),
            enc_blks       = cfg.get("enc_blks",    [2, 2, 4, 8]),
            dec_blks       = cfg.get("dec_blks",    [2, 2, 2, 2]),
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
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.net.load_state_dict(state_dict)
        print(f"[NAFNetModel] Loaded weights from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of degraded images.

        Args:
            x: Float32 tensor in [0, 1], shape (B, 3, H, W).

        Returns:
            Restored images, float32 clamped to [0, 1], shape (B, 3, H, W).
        """
        return self.net(x).clamp(0.0, 1.0)

    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """
        Save model weights and training metadata.

        Saved keys: model_state_dict, epoch, loss.

        Args:
            path:  Destination path (e.g. "checkpoints/nafnet_best.pth").
            epoch: Current epoch number (1-indexed).
            loss:  Validation loss at this checkpoint.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "epoch": epoch,
            "loss":  loss,
        }, path)
        print(f"[NAFNetModel] Saved checkpoint → {path}  "
              f"(epoch={epoch}, loss={loss:.6f})")
