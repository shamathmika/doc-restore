# =============================================================================
# File: demo/components/image_panel.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Image display and metric wiring utilities for the DocRestore Gradio demo.
#   Imported by demo/app.py (Shamathmika's).
#
# Exports:
#   build_image_panel(degraded, restored) -> PIL.Image
#       Returns a side-by-side composite PIL image with column labels.
#       Compatible with gr.Image(type="pil") outputs in app.py.
#
#   compute_demo_metrics(degraded, restored) -> tuple[float, float, float]
#       Calls eval/metrics.py and returns (psnr, ssim, cer).
#       Designed to be called in app.py immediately after run_inference().
# =============================================================================

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Make eval/metrics importable when this file is loaded from demo/
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import compute_psnr, compute_ssim, compute_ocr_cer

_LABEL_BAR_H = 24   # height of the label strip above each image
_LABEL_COLOR  = (30, 30, 30)
_TEXT_COLOR   = (255, 255, 255)
_DIVIDER_W    = 4   # pixel gap between the two columns
_DIVIDER_COLOR = (200, 200, 200)


def _draw_label(img: Image.Image, label: str) -> Image.Image:
    """Return a new PIL image with a dark label bar prepended at the top."""
    labeled = Image.new("RGB", (img.width, img.height + _LABEL_BAR_H), _LABEL_COLOR)
    labeled.paste(img, (0, _LABEL_BAR_H))
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 4), label, fill=_TEXT_COLOR, font=font)
    return labeled


def build_image_panel(degraded: Image.Image, restored: Image.Image) -> Image.Image:
    """
    Compose a side-by-side comparison image for the Gradio demo output.

    Both images are resized to the same height before compositing so the
    panel looks consistent regardless of input dimensions.

    Args:
        degraded: Input (degraded) document image as a PIL Image (RGB).
        restored: Model output (restored) image as a PIL Image (RGB).

    Returns:
        A single PIL Image containing:
            Left  — degraded image labelled "Input (Degraded)"
            Right — restored image labelled "Output (Restored)"
        Suitable for a gr.Image(type="pil") output component.
    """
    degraded = degraded.convert("RGB")
    restored = restored.convert("RGB")

    # Resize restored to match degraded height (preserving aspect ratio)
    target_h = degraded.height
    if restored.height != target_h:
        scale = target_h / restored.height
        restored = restored.resize(
            (int(restored.width * scale), target_h), Image.LANCZOS
        )

    left  = _draw_label(degraded, "Input (Degraded)")
    right = _draw_label(restored, "Output (Restored)")

    total_w = left.width + _DIVIDER_W + right.width
    total_h = max(left.height, right.height)

    canvas = Image.new("RGB", (total_w, total_h), _DIVIDER_COLOR)
    canvas.paste(left,  (0, 0))
    canvas.paste(right, (left.width + _DIVIDER_W, 0))

    return canvas


def compute_demo_metrics(
    degraded: Image.Image,
    restored: Image.Image,
) -> tuple[float, float, float]:
    """
    Compute image quality metrics between the degraded input and the restored
    output for display in the demo.

    Note: in the absence of a ground-truth clean image, this measures how
    different the model's output is from the input — useful as a quick
    signal that the model is doing something, but not a substitute for
    eval/run_eval.py which uses the true clean reference.

    Args:
        degraded: Input (degraded) document image as a PIL Image.
        restored: Model output (restored) image as a PIL Image.

    Returns:
        Tuple of (psnr, ssim, cer) as floats.
        On Tesseract errors, cer is returned as float("nan").
    """
    size = degraded.size   # keep original resolution for metrics
    deg_arr  = np.array(degraded.convert("RGB").resize(size, Image.LANCZOS), dtype=np.uint8)
    rest_arr = np.array(restored.convert("RGB").resize(size, Image.LANCZOS), dtype=np.uint8)

    psnr = compute_psnr(deg_arr, rest_arr)
    ssim = compute_ssim(deg_arr, rest_arr)
    try:
        cer = compute_ocr_cer(deg_arr, rest_arr)
    except Exception:
        cer = float("nan")

    return psnr, ssim, cer
