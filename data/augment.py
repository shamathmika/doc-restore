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

import random
from pathlib import Path

import numpy as np
from PIL import Image

from augraphy import (
    AugraphyPipeline,
    Brightness,
    DirtyDrum,
    Folding,
    InkBleed,
    LowInkPeriodicLines,
)

SEED = 42


def build_augmentation_pipeline() -> AugraphyPipeline:
    """
    Constructs and returns the Augraphy degradation pipeline.

    Stages (in order):
        Ink phase:   InkBleed, LowInkPeriodicLines
        Paper phase: DirtyDrum, Folding
        Post phase:  Brightness (reduced to simulate low contrast)

    Returns:
        AugraphyPipeline ready to be called on a numpy RGB image array.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    ink_phase = [
        InkBleed(
            intensity_range=(0.4, 0.7),
            kernel_size=(5, 5),
            severity=(0.3, 0.5),
        ),
        LowInkPeriodicLines(
            count_range=(2, 5),
            period_range=(8, 16),
            noise_probability=0.3,
        ),
    ]

    paper_phase = [
        DirtyDrum(
            line_width_range=(1, 4),
            line_concentration=0.05,
            direction=2,
            noise_intensity=0.5,
            noise_value=(64, 224),
            ksize=(3, 3),
            sigmaX=0,
        ),
        Folding(
            fold_count=2,
            fold_noise=0.02,
            fold_angle_range=(-10, 10),
            gradient_width=(0.1, 0.2),
            gradient_height=(0.01, 0.02),
        ),
    ]

    post_phase = [
        Brightness(
            brightness_range=(0.7, 0.95),
        ),
    ]

    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
    )


def augment_image(clean_img_path: str, output_path: str) -> None:
    """
    Applies the degradation pipeline to a single image and saves the result.

    Args:
        clean_img_path: Path to the clean source image.
        output_path:    Path where the degraded image will be saved (PNG).
    """
    pipeline = build_augmentation_pipeline()
    img = np.array(Image.open(clean_img_path).convert("RGB"))
    degraded = pipeline(img)
    Image.fromarray(degraded).save(output_path)


def augment_dataset(clean_dir: str, output_dir: str, n_per_image: int = 2) -> int:
    """
    Generates multiple degraded versions of every image in a directory.

    For each source image, produces n_per_image augmented outputs saved as:
        <original_stem>_aug0.png, <original_stem>_aug1.png, ...

    Args:
        clean_dir:    Directory containing clean source images.
        output_dir:   Directory where degraded outputs will be written.
                      Created automatically if it does not exist.
        n_per_image:  Number of degraded variants to generate per source image.

    Returns:
        Total number of degraded images generated.
    """
    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in extensions)

    count = 0
    for img_path in image_paths:
        for i in range(n_per_image):
            out_name = f"{img_path.stem}_aug{i}.png"
            augment_image(str(img_path), str(output_dir / out_name))
            count += 1

    return count
