"""
download_noisy.py

Downloads the NoisyOffice dataset from Kaggle (mariajosecastrobleda/noisyoffice),
organizes clean/degraded pairs into:
    data/noisy/clean/
    data/noisy/degraded/
and prints a summary of matched pairs.

Requires the Kaggle CLI to be configured (~/.kaggle/kaggle.json).

Usage:
    python data/download_noisy.py
    python data/download_noisy.py --dest data/noisy
"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "mariajosecastrobleda/noisyoffice"
DEFAULT_DEST = Path("data/noisy")
DOWNLOAD_TMP = Path("data/_noisy_tmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_kaggle(dataset: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset zip into *dest_dir* using the Kaggle CLI."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset: {dataset} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle CLI failed (exit {result.returncode}). "
            "Ensure ~/.kaggle/kaggle.json exists and is valid."
        )
    # Kaggle names the file after the dataset slug, e.g. noisyoffice.zip
    slug = dataset.split("/")[-1]
    zip_path = dest_dir / f"{slug}.zip"
    if not zip_path.exists():
        # Fallback: find any zip in the dir
        zips = list(dest_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(
                f"No zip file found in {dest_dir} after Kaggle download."
            )
        zip_path = zips[0]
    print(f"Downloaded to: {zip_path}")
    return zip_path


def _extract(zip_path: Path, extract_to: Path) -> None:
    """Extract *zip_path* into *extract_to*."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _find_image_files(root: Path) -> list[Path]:
    """Return all image files under *root* (recursive)."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in extensions
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _classify_and_copy(extracted_root: Path, dest: Path) -> tuple[int, list[str]]:
    """
    Walk the extracted archive, classify images by filename, copy to
    dest/clean/ and dest/degraded/, and return
    (number_of_matched_pairs, list_of_mismatch_warnings).

    NoisyOffice naming convention (Kaggle release):
      Font<X>_Clean_<split>.png   - clean originals
      Font<X>_Noise<type>_<split>.png  - degraded versions

    Matching key: (<font_id>, <split>) e.g. ("Fontfre", "TE")
    Each clean image may have multiple degraded partners (one per noise
    type: c, f, p, w).  We count unique (font_id, split) pairs.
    """
    clean_dest = dest / "clean"
    degraded_dest = dest / "degraded"
    clean_dest.mkdir(parents=True, exist_ok=True)
    degraded_dest.mkdir(parents=True, exist_ok=True)

    all_images = _find_image_files(extracted_root)

    clean_files: list[Path] = []
    noisy_files: list[Path] = []

    for p in all_images:
        stem_lower = p.stem.lower()
        if "_clean_" in stem_lower:
            clean_files.append(p)
        elif "_noise" in stem_lower:
            noisy_files.append(p)
        # skip anything that matches neither (thumbnails, icons, etc.)

    for p in clean_files:
        shutil.copy2(p, clean_dest / p.name)
    for p in noisy_files:
        shutil.copy2(p, degraded_dest / p.name)

    # Build matching keys: strip the middle token (_Clean_ / _Noise<x>_)
    # Filename pattern: <font_id>_<type>_<split>.<ext>
    def _key(p: Path) -> tuple[str, str]:
        parts = p.stem.split("_")
        # parts[0] = font id, parts[-1] = split (TR/VA/TE/RE)
        return (parts[0], parts[-1])

    clean_keys = {_key(p) for p in clean_files}
    noisy_keys = {_key(p) for p in noisy_files}

    matched = clean_keys & noisy_keys
    mismatches: list[str] = []

    for key in sorted(clean_keys - noisy_keys):
        mismatches.append(f"  Clean-only (no degraded pair): font={key[0]} split={key[1]}")
    for key in sorted(noisy_keys - clean_keys):
        mismatches.append(f"  Degraded-only (no clean pair): font={key[0]} split={key[1]}")

    return len(matched), mismatches


def download_noisyoffice(dest: Path = DEFAULT_DEST) -> None:
    """Main entry point: download, extract, organise, summarise."""
    extract_dir = DOWNLOAD_TMP / "extracted"

    try:
        # 1. Download via Kaggle CLI
        existing_zips = list(DOWNLOAD_TMP.glob("*.zip")) if DOWNLOAD_TMP.exists() else []
        if existing_zips:
            zip_path = existing_zips[0]
            print(f"Archive already exists at {zip_path}, skipping download.")
        else:
            zip_path = _download_kaggle(KAGGLE_DATASET, DOWNLOAD_TMP)

        # 2. Extract
        if extract_dir.exists():
            print(f"Already extracted at {extract_dir}, skipping extraction.")
        else:
            _extract(zip_path, extract_dir)

        # 3. Organise
        print("Organising clean / degraded pairs ...")
        n_pairs, mismatches = _classify_and_copy(extract_dir, dest)

        # 4. Summary
        clean_count = len(list((dest / "clean").iterdir()))
        degraded_count = len(list((dest / "degraded").iterdir()))
        print("\n--- NoisyOffice Download Summary ---")
        print(f"  Destination      : {dest.resolve()}")
        print(f"  Clean images     : {clean_count}")
        print(f"  Degraded images  : {degraded_count}")
        print(f"  Matched pairs    : {n_pairs}")
        if mismatches:
            print(f"  Mismatches ({len(mismatches)}):")
            for m in mismatches:
                print(m)
        else:
            print("  Mismatches       : none")
        print("------------------------------------\n")

    finally:
        # Clean up temp directory
        if DOWNLOAD_TMP.exists():
            shutil.rmtree(DOWNLOAD_TMP)
            print(f"Cleaned up temp dir: {DOWNLOAD_TMP}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and organise the NoisyOffice dataset."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Root output directory (default: data/noisy)",
    )
    args = parser.parse_args()
    download_noisyoffice(dest=args.dest)
