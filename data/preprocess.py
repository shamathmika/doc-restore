# =============================================================================
# File: data/preprocess.py
# Owner: Sakshat Patil
#
# Purpose:
#   Preprocessing utilities applied to raw images before training or eval.
#   Used by both the DataLoader and the evaluation pipeline.
#
# Exports:
#   - preprocess_image(img: PIL.Image) -> PIL.Image
#       Applies normalisation, resizing, or any other preprocessing step to
#       a single image. Returns the processed PIL image.
#
#   - preprocess_dataset(src_dir: str, dst_dir: str) -> None
#       Applies preprocess_image() to every image in src_dir and saves the
#       results to dst_dir, preserving filenames.
#
# Dependencies:
#   - Pillow (PIL)
#   - numpy
# =============================================================================
