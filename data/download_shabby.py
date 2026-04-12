# =============================================================================
# File: data/download_shabby.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Builds a ShabbyPages-style training dataset without any external accounts.
#   Downloads arXiv PDFs listed in data/arxiv_papers.json, converts every page
#   to a clean document image, then runs the Augraphy pipeline to synthesise a
#   matching degraded counterpart — producing paired clean/degraded images.
#
#   Output structure:
#       data/shabby/clean/      — original PDF pages (reference)
#       data/shabby/degraded/   — Augraphy-degraded counterparts
#
#   Filenames follow the pattern:  {arxiv_id}_p{page:03d}.jpg
#   e.g.  1706.03762_p001.jpg  (page 1 of the Transformer paper)
#
# Usage:
#   # Full run — all papers in arxiv_papers.json (~600+ pairs):
#   python data/download_shabby.py
#
#   # Test mode — exactly 5 pages, then stop:
#   python data/download_shabby.py --test
#
#   # Custom papers file or output dir:
#   python data/download_shabby.py --papers data/arxiv_papers.json --dest data/shabby
#
# Dependencies:
#   augraphy, pymupdf (fitz), opencv-python (cv2), numpy
#   pip install augraphy pymupdf opencv-python
# =============================================================================

import json
import random as _random
import time
import urllib.request
from pathlib import Path

import cv2
import fitz          # PyMuPDF
import numpy as np

# ---------------------------------------------------------------------------
# Compatibility patch — augraphy 8.2.6 + Python 3.12
# ---------------------------------------------------------------------------
# augraphy passes numpy.float64 values to random.randint() throughout several
# modules (PaperFactory, inkgenerator, etc.). Python 3.12 made random.randint
# strict about integer types, so this raises TypeError at runtime.
# Patching once here at import time fixes all affected augmentations globally.
# Note: using a named function (not a lambda) so Numba JIT compilation is
# unaffected if any augmentation uses Numba internally.

_orig_randint = _random.randint


def _safe_randint(a, b):
    return _orig_randint(int(a), int(b))


_random.randint = _safe_randint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARXIV_PDF_URL  = "https://arxiv.org/pdf/{arxiv_id}"
DPI_SCALE      = 2.0    # 2× zoom ≈ 150 dpi — legible without being huge
TEST_N         = 5      # pages to generate in test mode
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5      # seconds between retries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_shabby(
    papers_json: str = "data/arxiv_papers.json",
    dest_dir:    str = "data/shabby",
    test_mode:   bool = False,
) -> None:
    """
    Main entry point. Reads the paper list from papers_json, downloads each
    PDF, converts every page to an image, and runs Augraphy to produce a
    degraded counterpart. Writes all pairs to dest_dir/clean/ and
    dest_dir/degraded/.

    Already-generated pairs are skipped, so the function is safe to re-run
    after an interrupted session.

    Args:
        papers_json: Path to the JSON file listing arXiv papers.
        dest_dir:    Root directory for output pairs.
        test_mode:   If True, stop after generating TEST_N pages total.
    """
    papers_path = Path(papers_json)
    if not papers_path.exists():
        raise FileNotFoundError(
            f"Papers list not found: {papers_path}\n"
            "Expected data/arxiv_papers.json — make sure it exists."
        )

    with papers_path.open() as f:
        data = json.load(f)
    papers = data["papers"]

    dest     = Path(dest_dir)
    clean_dir    = dest / "clean"
    degraded_dir = dest / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"TEST (first {TEST_N} pages)" if test_mode else f"FULL ({len(papers)} papers)"
    print(f"\n[download_shabby] Mode : {mode_label}")
    print(f"[download_shabby] Output: {dest.resolve()}\n")

    total_generated = 0
    total_skipped   = 0
    failed_papers   = []

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        title    = paper.get("title", arxiv_id)

        if test_mode and total_generated >= TEST_N:
            break

        print(f"-> [{arxiv_id}] {title}")

        # Download PDF
        try:
            pdf_bytes = _fetch_pdf(arxiv_id)
        except Exception as exc:
            print(f"  SKIP — download failed: {exc}")
            failed_papers.append(arxiv_id)
            continue

        # Extract pages
        try:
            pages = _pdf_to_images(pdf_bytes, scale=DPI_SCALE)
        except Exception as exc:
            print(f"  SKIP — PDF render failed: {exc}")
            failed_papers.append(arxiv_id)
            continue

        print(f"  {len(pages)} pages extracted", end="")

        for page_idx, img_bgr in enumerate(pages, start=1):
            if test_mode and total_generated >= TEST_N:
                break

            filename     = f"{arxiv_id}_p{page_idx:03d}.jpg"
            clean_out    = clean_dir    / filename
            degraded_out = degraded_dir / filename

            if clean_out.exists() and degraded_out.exists():
                total_skipped += 1
                continue

            # Save clean page
            cv2.imwrite(str(clean_out), img_bgr)

            # Generate and save degraded version
            degraded = _apply_degradation(img_bgr)
            cv2.imwrite(str(degraded_out), degraded)

            total_generated += 1

        print(f" | done")

    _print_summary(clean_dir, degraded_dir, total_generated, total_skipped, failed_papers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_pdf(arxiv_id: str) -> bytes:
    """
    Downloads an arXiv PDF and returns its raw bytes.
    Retries up to RETRY_ATTEMPTS times on transient errors.

    Raises:
        urllib.error.URLError: if all retries fail.
    """
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    headers = {"User-Agent": "DocRestore/1.0 (research project)"}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"  retry {attempt}/{RETRY_ATTEMPTS} after error: {exc}")
            time.sleep(RETRY_DELAY)


def _pdf_to_images(pdf_bytes: bytes, scale: float = 2.0) -> list:
    """
    Renders every page of a PDF to a BGR numpy array.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        scale:     Zoom factor applied to each page (2.0 ≈ 150 dpi).

    Returns:
        List of BGR uint8 numpy arrays, one per page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(scale, scale)
    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Ensure 3-channel (Augraphy requirement)
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        images.append(img_bgr)

    doc.close()
    return images


def _apply_degradation(img: np.ndarray) -> np.ndarray:
    """
    Applies the Augraphy default pipeline to synthesise realistic document
    degradation (ink bleed, stains, folds, photocopy artifacts, etc.).

    Requires numpy < 2 (numpy 1.26.x). Install with:
        pip install "numpy<2"

    Args:
        img: BGR uint8 numpy array (H x W x 3).

    Returns:
        Degraded BGR uint8 numpy array.
    """
    from augraphy import AugraphyPipeline
    from augraphy.augmentations import (
        BleedThrough, Brightness, ColorPaper, DirtyDrum,
        Folding, InkBleed, Jpeg, LowInkPeriodicLines,
        Markup, SubtleNoise,
    )

    # Custom pipeline — excludes PaperFactory (numpy.float64 → randint bug,
    # Python 3.12) and BadPhotoCopy (Numba JIT incompatibility).
    pipeline = AugraphyPipeline(
        ink_phase   = [InkBleed(p=0.5), BleedThrough(p=0.4), LowInkPeriodicLines(p=0.3)],
        paper_phase = [ColorPaper(p=0.4), Brightness(p=0.6)],
        post_phase  = [DirtyDrum(p=0.4), SubtleNoise(p=0.5), Folding(p=0.3), Jpeg(p=0.3), Markup(p=0.2)],
    )
    result = pipeline(img)
    if isinstance(result, dict):
        result = result.get("output", result.get("augmented", img))
    return result


def _print_summary(
    clean_dir:       Path,
    degraded_dir:    Path,
    generated:       int,
    skipped:         int,
    failed_papers:   list,
) -> None:
    """Prints a final summary of what was produced."""
    matched = sum(1 for _ in clean_dir.glob("*.jpg"))

    print(f"\n{'='*55}")
    print(f"[download_shabby] Summary")
    print(f"  Pairs generated this run : {generated}")
    print(f"  Pairs skipped (existed)  : {skipped}")
    print(f"  Total pairs on disk      : {matched}")
    print(f"  clean/    : {clean_dir.resolve()}")
    print(f"  degraded/ : {degraded_dir.resolve()}")

    if failed_papers:
        print(f"\n  Failed papers ({len(failed_papers)}):")
        for pid in failed_papers:
            print(f"    {pid}")
    else:
        print(f"\n  All papers processed successfully.")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build a ShabbyPages-style dataset from arXiv PDFs + Augraphy. "
            "Paper list is read from data/arxiv_papers.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--papers",
        default="data/arxiv_papers.json",
        metavar="JSON",
        help="Path to the arXiv papers JSON file",
    )
    parser.add_argument(
        "--dest",
        default="data/shabby",
        metavar="DIR",
        help="Output directory (will contain clean/ and degraded/ subdirs)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: generate only {TEST_N} pages then stop",
    )
    args = parser.parse_args()

    download_shabby(
        papers_json=args.papers,
        dest_dir=args.dest,
        test_mode=args.test,
    )
