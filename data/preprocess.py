"""
preprocess.py

Image preprocessing utilities for DocRestore.

Public API
----------
preprocess_image(img_path, output_size=(256, 256)) -> np.ndarray
    Load → resize → convert to RGB → normalise [0, 1] as float32.

preprocess_dataset(input_dir, output_dir, output_size=(256, 256)) -> None
    Apply preprocess_image to every image in input_dir and save to
    output_dir as PNG files with the same base-names.

Usage
-----
    # Single image
    from data.preprocess import preprocess_image
    arr = preprocess_image("data/noisy/degraded/img_01.png")

    # Whole folder (call separately for clean/ and degraded/)
    from data.preprocess import preprocess_dataset
    preprocess_dataset("data/noisy/clean",    "data/noisy_pp/clean")
    preprocess_dataset("data/noisy/degraded", "data/noisy_pp/degraded")

    # CLI - process a folder in one shot
    python data/preprocess.py --input data/noisy/clean \
                               --output data/noisy_pp/clean \
                               --size 256 256
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Supported source extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def preprocess_image(
    img_path: str | Path,
    output_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Load and preprocess a single image.

    Steps:
      1. Open with Pillow (handles all common formats).
      2. Convert to RGB (drops alpha, converts grayscale to 3-channel).
      3. Resize to *output_size* (width, height) using LANCZOS resampling.
      4. Cast to float32 and normalise pixel values to [0.0, 1.0].

    Parameters
    ----------
    img_path : str | Path
        Path to the source image file.
    output_size : tuple[int, int]
        (width, height) of the output array.  Default (256, 256).

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype float32, values in [0.0, 1.0].
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path)

    # Ensure RGB (3 channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    img = img.resize(output_size, Image.LANCZOS)

    # Normalise
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    output_size: tuple[int, int] = (256, 256),
) -> None:
    """Preprocess all images in *input_dir* and save to *output_dir*.

    - Skips non-image files silently.
    - Preserves base-names (changes extension to .png).
    - Creates *output_dir* if it does not exist.
    - Prints a per-file progress line and a final summary.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing source images (non-recursive).
    output_dir : str | Path
        Directory where processed PNGs will be written.
    output_size : tuple[int, int]
        (width, height) passed to preprocess_image.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )

    if not image_paths:
        print(f"[preprocess] No images found in {input_dir}")
        return

    processed = 0
    errors = 0

    for img_path in image_paths:
        out_path = output_dir / (img_path.stem + ".png")
        try:
            arr = preprocess_image(img_path, output_size)
            # Convert float32 [0,1] → uint8 [0,255] for PNG storage
            pil_out = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
            pil_out.save(out_path)
            processed += 1
            print(f"  [OK] {img_path.name} -> {out_path.name}")
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [ERR] {img_path.name}: {exc}")

    print(
        f"\n[preprocess] Done - {processed} saved, {errors} errors "
        f"(output_size={output_size}, dir={output_dir})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a folder of images for DocRestore."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Source image directory",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Destination directory for preprocessed PNGs",
    )
    parser.add_argument(
        "--size", nargs=2, type=int, default=[256, 256],
        metavar=("W", "H"),
        help="Output width and height (default: 256 256)",
    )
    args = parser.parse_args()
    preprocess_dataset(args.input, args.output, output_size=tuple(args.size))
