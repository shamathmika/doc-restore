# =============================================================================
# File: data/augment.py
# Owner: Shamathmika
#
# Purpose:
#   Builds an Augraphy-based degradation pipeline to synthetically degrade
#   clean document images into realistic degraded versions for training.
#
# Exports:
#   - build_augmentation_pipeline() -> AugraphyPipeline
#       Constructs and returns the full Augraphy pipeline with stages:
#       InkBleed, LowInkPeriodicLines, DirtyDrum, Folding, Brightness,
#       and low-contrast effects.
#
#   - augment_image(clean_img_path: str, output_path: str) -> None
#       Loads a clean image, applies the pipeline, saves the degraded result.
#
#   - augment_dataset(clean_dir: str, output_dir: str, n_per_image: int = 2)
#           -> int
#       Iterates over all images in clean_dir, generates n_per_image degraded
#       versions per image, saves them to output_dir with naming convention:
#           <original_stem>_aug0.png, <original_stem>_aug1.png, ...
#       Returns the total count of generated pairs.
#
# Dependencies:
#   - augraphy
#   - Pillow (PIL)
#   - numpy
#
# Notes:
#   - All random ops use seed=42 for reproducibility
#   - Output images are saved as PNG
# =============================================================================

import random as _random
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Python 3.12 compatibility patch
# ---------------------------------------------------------------------------
# augraphy passes numpy.float64 values to random.randint() in several internal
# modules. Python 3.12 made random.randint strict about integer types, raising
# TypeError at runtime. Patching once here fixes all affected augmentations
# globally. Using a named function (not a lambda) keeps Numba JIT unaffected.

_orig_randint = _random.randint


def _safe_randint(a, b):
    return _orig_randint(int(a), int(b))


_random.randint = _safe_randint

# ---------------------------------------------------------------------------

from augraphy import AugraphyPipeline
from augraphy.augmentations import (
    BleedThrough,
    Brightness,
    ColorPaper,
    DirtyDrum,
    Folding,
    InkBleed,
    Jpeg,
    LowInkPeriodicLines,
    Markup,
    SubtleNoise,
)

SEED = 42


def build_augmentation_pipeline() -> AugraphyPipeline:
    """
    Constructs and returns the Augraphy degradation pipeline.

    This is the single shared pipeline definition for the project.
    All code that generates degraded training data should use this function
    rather than defining its own pipeline.

    Stages (in order):
        Ink phase:   InkBleed, BleedThrough, LowInkPeriodicLines
        Paper phase: ColorPaper, Brightness
        Post phase:  DirtyDrum, SubtleNoise, Folding, Jpeg, Markup

    Returns:
        AugraphyPipeline that operates on BGR uint8 numpy arrays.
    """
    ink_phase = [
        InkBleed(intensity_range=(0.4, 0.7), kernel_size=(5, 5), severity=(0.3, 0.5), p=0.8),
        BleedThrough(p=0.4),
        LowInkPeriodicLines(count_range=(2, 5), period_range=(8, 16), noise_probability=0.3, p=0.5),
    ]

    paper_phase = [
        ColorPaper(p=0.4),
        Brightness(brightness_range=(0.6, 0.95), p=0.8),
    ]

    post_phase = [
        DirtyDrum(line_width_range=(1, 4), line_concentration=0.05, direction=2,
                  noise_intensity=0.5, noise_value=(64, 224), ksize=(3, 3), sigmaX=0, p=0.5),
        SubtleNoise(p=0.5),
        Folding(fold_count=2, fold_noise=0.02, fold_angle_range=(-10, 10),
                gradient_width=(0.1, 0.2), gradient_height=(0.01, 0.02), p=0.4),
        Jpeg(p=0.3),
        Markup(p=0.2),
    ]

    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
    )


def apply_degradation(img_bgr: np.ndarray, pipeline: AugraphyPipeline = None) -> np.ndarray:
    """
    Applies the degradation pipeline to a single BGR image array.

    This is the core function used by both augment_image() and
    download_shabby.py. Call build_augmentation_pipeline() once and pass the
    result here to avoid rebuilding the pipeline for every image.

    Args:
        img_bgr:  BGR uint8 numpy array (H x W x 3).
        pipeline: An AugraphyPipeline instance. If None, a new one is built.

    Returns:
        Degraded BGR uint8 numpy array.
    """
    if pipeline is None:
        pipeline = build_augmentation_pipeline()

    result = pipeline(img_bgr)

    # Some augraphy versions return a dict instead of a plain array.
    if isinstance(result, dict):
        result = result.get("output", result.get("augmented", img_bgr))

    return result


def augment_image(clean_img_path: str, output_path: str, pipeline: AugraphyPipeline = None) -> None:
    """
    Applies the degradation pipeline to a single image file and saves the result.

    Handles RGB ↔ BGR conversion so callers work entirely with standard image
    files and do not need to manage colour channel order.

    Args:
        clean_img_path: Path to the clean source image.
        output_path:    Path where the degraded image will be saved (PNG).
        pipeline:       Optional pre-built pipeline. Pass one when calling in a
                        loop to avoid rebuilding per image.
    """
    img_rgb = np.array(Image.open(clean_img_path).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()
    degraded_bgr = apply_degradation(img_bgr, pipeline)
    degraded_rgb = degraded_bgr[:, :, ::-1].copy()
    Image.fromarray(degraded_rgb).save(output_path)


def augment_dataset(clean_dir: str, output_dir: str, n_per_image: int = 2) -> int:
    """
    Generates multiple degraded versions of every image in a directory.

    For each source image, produces n_per_image augmented outputs saved as:
        <original_stem>_aug0.png, <original_stem>_aug1.png, ...

    Builds the pipeline once and reuses it across all images for efficiency.

    Args:
        clean_dir:    Directory containing clean source images.
        output_dir:   Directory where degraded outputs will be written.
                      Created automatically if it does not exist.
        n_per_image:  Number of degraded variants to generate per source image.

    Returns:
        Total number of degraded images generated.
    """
    _random.seed(SEED)
    np.random.seed(SEED)

    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_augmentation_pipeline()

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in extensions)

    count = 0
    for img_path in image_paths:
        for i in range(n_per_image):
            out_name = f"{img_path.stem}_aug{i}.png"
            augment_image(str(img_path), str(output_dir / out_name), pipeline=pipeline)
            count += 1

    return count
