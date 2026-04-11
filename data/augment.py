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
