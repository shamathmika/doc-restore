# =============================================================================
# File: data/split.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Splits the full ShabbyPages dataset into train / val / test CSV files
#   used by the DataLoader (data/dataloader.py).
#
# Exports:
#   - split_dataset(shabby_dir: str, output_dir: str,
#                   train_ratio=0.8, val_ratio=0.1) -> None
#       Scans shabby_dir for clean/degraded pairs, shuffles with seed=42,
#       and writes three CSV files to output_dir:
#           train.csv, val.csv, test.csv
#       Each CSV has columns: clean_path, degraded_path
#
# Dependencies:
#   - pandas
#   - pathlib
#
# Notes:
#   - Output CSVs are consumed by data/dataloader.py
#   - Paths in CSVs are relative to project root
# =============================================================================

import random
from pathlib import Path

import pandas as pd

SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def split_dataset(
    shabby_dir: str = "data/shabby",
    output_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """
    Reads matched clean/degraded pairs from shabby_dir, shuffles them with a
    fixed seed, and writes train.csv, val.csv, test.csv to output_dir.

    The test split receives the remainder after train and val are allocated
    (i.e. test_ratio = 1 - train_ratio - val_ratio).

    Args:
        shabby_dir:  Root of the ShabbyPages dataset. Must contain clean/ and
                     degraded/ subdirectories produced by download_shabby.py.
        output_dir:  Directory where the three CSV files are written.
        train_ratio: Fraction of pairs used for training (default: 0.8).
        val_ratio:   Fraction of pairs used for validation (default: 0.1).

    Raises:
        FileNotFoundError: If clean/ or degraded/ subdirectory is missing.
        ValueError:        If train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0"
        )

    shabby = Path(shabby_dir)
    clean_dir = shabby / "clean"
    degraded_dir = shabby / "degraded"

    if not clean_dir.is_dir():
        raise FileNotFoundError(
            f"clean/ directory not found at {clean_dir}. "
            "Run data/download_shabby.py first."
        )
    if not degraded_dir.is_dir():
        raise FileNotFoundError(
            f"degraded/ directory not found at {degraded_dir}. "
            "Run data/download_shabby.py first."
        )

    # --- Collect matched pairs ---
    clean_stems = {
        f.stem: f
        for f in clean_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    }
    degraded_stems = {
        f.stem: f
        for f in degraded_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    }

    matched_stems = sorted(clean_stems.keys() & degraded_stems.keys())

    unmatched_clean = sorted(clean_stems.keys() - degraded_stems.keys())
    unmatched_degraded = sorted(degraded_stems.keys() - clean_stems.keys())

    if unmatched_clean or unmatched_degraded:
        print(
            f"[split] WARNING — {len(unmatched_clean)} clean-only and "
            f"{len(unmatched_degraded)} degraded-only images will be skipped."
        )

    if not matched_stems:
        raise ValueError(
            f"No matched pairs found in {shabby_dir}. "
            "Ensure clean/ and degraded/ contain files with matching names."
        )

    print(f"[split] Found {len(matched_stems)} matched pairs.")

    # --- Shuffle reproducibly ---
    rng = random.Random(SEED)
    rng.shuffle(matched_stems)

    n = len(matched_stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_stems = matched_stems[:n_train]
    val_stems = matched_stems[n_train: n_train + n_val]
    test_stems = matched_stems[n_train + n_val:]

    print(
        f"[split] Split → train: {len(train_stems)}, "
        f"val: {len(val_stems)}, test: {len(test_stems)}"
    )

    # --- Write CSVs ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems,
    }

    for split_name, stems in splits.items():
        rows = [
            {
                "clean_path": clean_stems[stem].as_posix(),
                "degraded_path": degraded_stems[stem].as_posix(),
            }
            for stem in stems
        ]
        df = pd.DataFrame(rows, columns=["clean_path", "degraded_path"])
        csv_path = out / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[split] Wrote {len(df)} rows → {csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split ShabbyPages into train/val/test CSV files."
    )
    parser.add_argument(
        "--shabby-dir",
        default="data/shabby",
        help="Path to the ShabbyPages root directory (default: data/shabby)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where train.csv, val.csv, test.csv are written (default: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation (default: 0.1)",
    )
    args = parser.parse_args()

    split_dataset(
        shabby_dir=args.shabby_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
