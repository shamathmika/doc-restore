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
#   - CombinedLoss
#   - get_loss(loss_type: str) -> nn.Module
#
# Dependencies:
#   - torch, torchvision (VGG16)
# =============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

# ImageNet stats used to normalise inputs before passing to VGG16
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    VGG16 weights are frozen — only used for feature extraction.
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


def get_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory that returns the requested loss module.

    Args:
        loss_type: One of "l1", "perceptual", "combined".
        **kwargs:  Passed to the loss constructor (e.g. lambda_perceptual).

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
    if loss_type == "combined":
        return CombinedLoss(**kwargs)
    raise ValueError(f"Unknown loss type '{loss_type}'. Choose from: l1, perceptual, combined.")
