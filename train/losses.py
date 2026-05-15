# =============================================================================
# File: train/losses.py
# Owner: Shamathmika
#
# Purpose:
#   Loss functions for document image restoration training.
#   Imported by both train/train_docres.py and train/train_nafnet.py.
#
# Exports:
#   - L1Loss
#   - PerceptualLoss
#   - TextAwareLoss
#   - CombinedLoss              (L1 + Perceptual)
#   - TextAwareCombinedLoss     (L1 + Perceptual + TextAware)
#   - get_loss(loss_type: str) -> nn.Module
#
# Dependencies:
#   - torch, torchvision (VGG16)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

# ImageNet stats used to normalise inputs before passing to VGG16
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Laplacian kernel used by TextAwareLoss to detect text edges
_LAPLACIAN = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
).view(1, 1, 3, 3)


class L1Loss(nn.Module):
    """Pixel-wise L1 loss between restored and clean images."""

    def __init__(self):
        super().__init__()
        self._loss = nn.L1Loss()

    def forward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            restored: Model output, float32 in [0, 1], shape (B, 3, H, W).
            clean:    Ground-truth clean image, same shape.

        Returns:
            Scalar loss tensor.
        """
        return self._loss(restored, clean)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 relu2_2 feature maps.

    Computes MSE between VGG16 features of the restored and clean images.
    Inputs are normalised with ImageNet mean/std before being passed to VGG16.
    VGG16 weights are frozen; only used for feature extraction.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # relu2_2 is the output of the 9th layer (0-indexed)
        self._features = nn.Sequential(*list(vgg.features.children())[:9])
        for param in self._features.parameters():
            param.requires_grad = False
        self._mse = nn.MSELoss()

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        mean = _IMAGENET_MEAN.to(x.device)
        std  = _IMAGENET_STD.to(x.device)
        return (x - mean) / std

    def forward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            restored: Model output, float32 in [0, 1], shape (B, 3, H, W).
            clean:    Ground-truth clean image, same shape.

        Returns:
            Scalar MSE loss between VGG16 feature maps.
        """
        restored_feat = self._features(self._normalise(restored))
        clean_feat    = self._features(self._normalise(clean))
        return self._mse(restored_feat, clean_feat)


class TextAwareLoss(nn.Module):
    """
    Text-region-aware L1 loss.

    Uses a Laplacian filter on the clean image to detect high-contrast edges
    (i.e. text stroke boundaries).  Those pixels are up-weighted so the model
    is penalised more for blurring or missing fine text detail.

    Weight map:  w(x) = 1 + lambda_edge * |Lap(gray(clean))|_normalised
    Loss:        mean( w * |restored - clean| )

    With lambda_edge=2.0 the loss ranges from 1x at smooth background pixels
    to 3x at sharp text-edge pixels.  This acts as a lightweight, fully
    differentiable proxy for OCR readability without requiring a text detector
    or a differentiable OCR engine.
    """

    def __init__(self, lambda_edge: float = 2.0):
        super().__init__()
        self.lambda_edge = lambda_edge

    def forward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            restored: Model output, float32 in [0, 1], shape (B, 3, H, W).
            clean:    Ground-truth clean image, same shape.

        Returns:
            Scalar weighted-L1 loss tensor.
        """
        # Convert clean to grayscale — text edges are luminance edges
        gray   = 0.299 * clean[:, :1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:]
        kernel = _LAPLACIAN.to(clean.device)
        edges  = F.conv2d(gray, kernel, padding=1).abs()

        # Normalise per-image so weight map is in [0, 1]
        edges  = edges / (edges.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

        # Weight map: background = 1x, text edges = up to (1 + lambda_edge)x
        weight = 1.0 + self.lambda_edge * edges

        return (weight * (restored - clean).abs()).mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of L1Loss and PerceptualLoss.

        loss = L1(restored, clean) + lambda_perceptual * Perceptual(restored, clean)
    """

    def __init__(self, lambda_perceptual: float = 0.1):
        """
        Args:
            lambda_perceptual: Weight applied to the perceptual loss term.
        """
        super().__init__()
        self._l1          = L1Loss()
        self._perceptual  = PerceptualLoss()
        self.lambda_perceptual = lambda_perceptual

    def forward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            restored: Model output, float32 in [0, 1], shape (B, 3, H, W).
            clean:    Ground-truth clean image, same shape.

        Returns:
            Scalar combined loss tensor.
        """
        return self._l1(restored, clean) + self.lambda_perceptual * self._perceptual(restored, clean)


class TextAwareCombinedLoss(nn.Module):
    """
    Three-term loss: L1 + Perceptual + TextAware.

        loss = L1(r, c)
             + lambda_perceptual * Perceptual(r, c)
             + lambda_text       * TextAware(r, c)

    L1 drives global pixel fidelity.
    Perceptual pulls high-level structure into alignment via VGG features.
    TextAware specifically penalises blurring of text-edge regions, acting as
    a differentiable proxy for OCR readability.
    """

    def __init__(self, lambda_perceptual: float = 0.1, lambda_text: float = 0.5):
        super().__init__()
        self._l1          = L1Loss()
        self._perceptual  = PerceptualLoss()
        self._text        = TextAwareLoss()
        self.lambda_perceptual = lambda_perceptual
        self.lambda_text       = lambda_text

    def forward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            restored: Model output, float32 in [0, 1], shape (B, 3, H, W).
            clean:    Ground-truth clean image, same shape.

        Returns:
            Scalar three-term loss tensor.
        """
        l1_term   = self._l1(restored, clean)
        perc_term = self._perceptual(restored, clean)
        text_term = self._text(restored, clean)
        return l1_term + self.lambda_perceptual * perc_term + self.lambda_text * text_term


def get_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory that returns the requested loss module.

    Args:
        loss_type: One of "l1", "perceptual", "combined", "text_aware", "text_aware_combined".
        **kwargs:  Passed to the loss constructor (e.g. lambda_perceptual, lambda_text).

    Returns:
        Instantiated nn.Module loss.

    Raises:
        ValueError: If loss_type is not recognised.
    """
    loss_type = loss_type.lower()
    if loss_type == "l1":
        return L1Loss()
    if loss_type == "perceptual":
        return PerceptualLoss()
    if loss_type == "text_aware":
        return TextAwareLoss(**kwargs)
    if loss_type == "combined":
        return CombinedLoss(**{k: v for k, v in kwargs.items() if k == "lambda_perceptual"})
    if loss_type == "text_aware_combined":
        return TextAwareCombinedLoss(**kwargs)
    raise ValueError(
        f"Unknown loss type '{loss_type}'. "
        "Choose from: l1, perceptual, combined, text_aware, text_aware_combined."
    )
