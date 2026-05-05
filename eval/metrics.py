# =============================================================================
# File: eval/metrics.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Image quality and OCR evaluation metrics for DocRestore.
#   Imported by eval/run_eval.py and demo/inference.py (Sakshat's file).
#
# Exports:
#   compute_psnr(clean_img, restored_img) -> float
#   compute_ssim(clean_img, restored_img) -> float
#   compute_ocr_cer(clean_img, restored_img) -> float
#
# All functions accept numpy arrays (H x W x C, uint8).
# =============================================================================

import numpy as np
import pytesseract
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(clean_img: np.ndarray, restored_img: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio between a clean and a restored image.

    Args:
        clean_img:    Ground-truth image, uint8 HxWxC.
        restored_img: Model output image, uint8 HxWxC.

    Returns:
        PSNR in dB. Higher is better.
    """
    return float(peak_signal_noise_ratio(clean_img, restored_img, data_range=255))


def compute_ssim(clean_img: np.ndarray, restored_img: np.ndarray) -> float:
    """
    Structural Similarity Index between a clean and a restored image.

    Args:
        clean_img:    Ground-truth image, uint8 HxWxC.
        restored_img: Model output image, uint8 HxWxC.

    Returns:
        SSIM in [-1, 1]. Higher is better (1.0 = identical).
    """
    return float(
        structural_similarity(clean_img, restored_img, channel_axis=2, data_range=255)
    )


def _edit_distance(a: str, b: str) -> int:
    """Standard DP Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    # Use two rows to keep memory O(n)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_ocr_cer(clean_img: np.ndarray, restored_img: np.ndarray) -> float:
    """
    Character Error Rate between OCR outputs of the clean and restored images.

    CER = edit_distance(ocr_clean, ocr_restored) / max(len(ocr_clean), 1)

    A CER of 0.0 means the OCR output is identical; 1.0 means completely wrong.
    Values can slightly exceed 1.0 when the restored text is longer than clean.

    Args:
        clean_img:    Ground-truth image, uint8 HxWxC.
        restored_img: Model output image, uint8 HxWxC.

    Returns:
        CER as a float >= 0. Lower is better.
    """
    ocr_clean    = pytesseract.image_to_string(clean_img)
    ocr_restored = pytesseract.image_to_string(restored_img)
    denom = max(len(ocr_clean), 1)
    return _edit_distance(ocr_clean, ocr_restored) / denom
