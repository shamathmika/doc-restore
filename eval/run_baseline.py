# =============================================================================
# File: eval/run_baseline.py
# Owner: Shamathmika
#
# Purpose:
#   Compute baseline PSNR / SSIM / CER by comparing the degraded input
#   directly against the clean reference (no model). Used to establish
#   a lower bound for comparison with DocRes and NAFNet results.
#
# Usage:
#   python eval/run_baseline.py
#   python eval/run_baseline.py --test-csv data/test.csv --out-dir eval/outputs/baseline
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import compute_psnr, compute_ssim, compute_ocr_cer

_INFER_SIZE = (512, 512)


def run_baseline(test_csv: str = "data/test.csv", out_dir: str = "eval/outputs/baseline") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    print(f"[baseline] Test pairs={len(df)}")

    records = []

    for idx, row in df.iterrows():
        clean_pil    = Image.open(row["clean_path"]).convert("RGB")
        degraded_pil = Image.open(row["degraded_path"]).convert("RGB")

        # PSNR / SSIM at 256×256 (matches model inference resolution)
        clean_arr    = np.array(clean_pil.resize(_INFER_SIZE, Image.LANCZOS))
        degraded_arr = np.array(degraded_pil.resize(_INFER_SIZE, Image.LANCZOS))

        psnr = compute_psnr(clean_arr, degraded_arr)
        ssim = compute_ssim(clean_arr, degraded_arr)

        # CER at full original resolution — Tesseract fails on 256×256 text
        clean_full    = np.array(clean_pil)
        degraded_full = np.array(degraded_pil)
        cer           = compute_ocr_cer(clean_full, degraded_full)

        records.append({
            "degraded_path": row["degraded_path"],
            "clean_path":    row["clean_path"],
            "psnr":          round(psnr, 4),
            "ssim":          round(ssim, 4),
            "cer":           round(cer,  4),
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
            print(f"  [{idx + 1}/{len(df)}]  PSNR={psnr:.2f}  SSIM={ssim:.4f}  CER={cer:.4f}")

    results_csv = out_path / "results.csv"
    pd.DataFrame(records).to_csv(results_csv, index=False)

    mean_psnr = np.mean([r["psnr"] for r in records])
    mean_ssim = np.mean([r["ssim"] for r in records])
    mean_cer  = np.mean([r["cer"]  for r in records])
    print(f"\n[baseline] Mean PSNR={mean_psnr:.2f} dB  SSIM={mean_ssim:.4f}  CER={mean_cer:.4f}")
    print(f"[baseline] Results saved -> {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation (no model)")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--out-dir",  default="eval/outputs/baseline")
    args = parser.parse_args()
    run_baseline(args.test_csv, args.out_dir)
