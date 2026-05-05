# =============================================================================
# File: eval/run_eval.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Evaluate a trained DocRes or NAFNet checkpoint on the test split.
#   For each test pair: runs inference, computes PSNR / SSIM / CER.
#   Writes a per-image results CSV and a visual grid PNG.
#
# Usage:
#   # From project root:
#   python eval/run_eval.py --model docres --checkpoint checkpoints/docres_best.pth
#   python eval/run_eval.py --model nafnet --checkpoint checkpoints/nafnet_best.pth
#
#   # Custom test CSV or output dir:
#   python eval/run_eval.py --model docres \
#       --checkpoint checkpoints/docres_best.pth \
#       --test-csv data/test.csv \
#       --out-dir  eval/outputs
#
# Outputs:
#   eval/outputs/results.csv        — per-image PSNR, SSIM, CER
#   eval/outputs/visual_grid.png    — 4-col grid for first 10 test images
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import compute_psnr, compute_ssim, compute_ocr_cer

_DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INFER_SIZE = (256, 256)   # must match training config input_size
_GRID_ROWS  = 10           # number of test images in the visual grid


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_type: str, checkpoint: str) -> torch.nn.Module:
    model_type = model_type.lower()
    if model_type == "docres":
        from models.docres_wrapper import DocResModel
        model = DocResModel(str(_ROOT / "configs" / "docres.yaml"))
    elif model_type == "nafnet":
        from models.nafnet_wrapper import NAFNetModel
        model = NAFNetModel(str(_ROOT / "configs" / "nafnet.yaml"))
    else:
        raise ValueError(f"Unknown model '{model_type}'. Choose 'docres' or 'nafnet'.")
    model.load_weights(checkpoint)
    model.eval()
    return model.to(_DEVICE)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL (RGB) -> float32 tensor (1, 3, H, W) in [0, 1]."""
    img = pil_img.convert("RGB").resize(_INFER_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Float32 tensor (1, 3, H, W) -> uint8 HxWxC numpy array."""
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def _run_inference(model: torch.nn.Module, degraded_pil: Image.Image) -> np.ndarray:
    inp = _pil_to_tensor(degraded_pil).to(_DEVICE)
    out = model(inp)
    return _tensor_to_uint8(out)


# ---------------------------------------------------------------------------
# Visual grid
# ---------------------------------------------------------------------------

def _add_label(img_arr: np.ndarray, label: str) -> np.ndarray:
    """Draw a label string at the top of an image array."""
    pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 255), font=font)
    return np.array(pil)


def _make_diff(clean: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """Absolute pixel difference, scaled to [0, 255] for visibility."""
    diff = np.abs(clean.astype(np.int16) - restored.astype(np.int16)).astype(np.uint8)
    # Boost contrast so small errors are visible
    diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
    return diff


def _save_visual_grid(rows: list[dict], out_path: Path) -> None:
    """
    Builds and saves a PNG grid with columns:
        degraded | restored | clean | diff
    for each dict in rows (each dict has keys: degraded, restored, clean).
    """
    cell_h, cell_w = _INFER_SIZE
    n_cols = 4
    n_rows = len(rows)
    canvas = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    for r, entry in enumerate(rows):
        degraded = entry["degraded"]
        restored = entry["restored"]
        clean    = entry["clean"]
        diff     = _make_diff(clean, restored)

        cells = [
            _add_label(degraded, "Degraded"),
            _add_label(restored, "Restored"),
            _add_label(clean,    "Clean"),
            _add_label(diff,     "Diff (x4)"),
        ]
        for c, cell in enumerate(cells):
            canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print(f"[run_eval] Visual grid saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model_type: str,
    checkpoint:  str,
    test_csv:    str = "data/test.csv",
    out_dir:     str = "eval/outputs",
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_type, checkpoint)
    df    = pd.read_csv(test_csv)
    print(f"[run_eval] Model={model_type}  Device={_DEVICE}  Test pairs={len(df)}")

    records    = []
    grid_rows  = []

    for idx, row in df.iterrows():
        degraded_pil = Image.open(row["degraded_path"]).convert("RGB")
        clean_pil    = Image.open(row["clean_path"]).convert("RGB")

        restored_arr = _run_inference(model, degraded_pil)
        clean_arr    = np.array(clean_pil.resize(_INFER_SIZE, Image.LANCZOS))
        degraded_arr = np.array(degraded_pil.resize(_INFER_SIZE, Image.LANCZOS))

        psnr = compute_psnr(clean_arr, restored_arr)
        ssim = compute_ssim(clean_arr, restored_arr)
        cer  = compute_ocr_cer(clean_arr, restored_arr)

        records.append({
            "degraded_path": row["degraded_path"],
            "clean_path":    row["clean_path"],
            "psnr":          round(psnr, 4),
            "ssim":          round(ssim, 4),
            "cer":           round(cer,  4),
        })

        if len(grid_rows) < _GRID_ROWS:
            grid_rows.append({
                "degraded": degraded_arr,
                "restored": restored_arr,
                "clean":    clean_arr,
            })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
            print(f"  [{idx + 1}/{len(df)}]  "
                  f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  CER={cer:.4f}")

    # -- Save results CSV ----------------------------------------------------
    results_csv = out_path / "results.csv"
    pd.DataFrame(records).to_csv(results_csv, index=False)
    print(f"[run_eval] Results saved -> {results_csv}")

    # -- Save visual grid ----------------------------------------------------
    _save_visual_grid(grid_rows, out_path / "visual_grid.png")

    # -- Print summary -------------------------------------------------------
    mean_psnr = np.mean([r["psnr"] for r in records])
    mean_ssim = np.mean([r["ssim"] for r in records])
    mean_cer  = np.mean([r["cer"]  for r in records])
    print(f"\n[run_eval] Mean PSNR={mean_psnr:.2f} dB  "
          f"SSIM={mean_ssim:.4f}  CER={mean_cer:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DocRestore checkpoint")
    parser.add_argument("--model",      required=True,
                        help="Model type: 'docres' or 'nafnet'")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--test-csv",   default="data/test.csv",
                        help="Path to test split CSV (default: data/test.csv)")
    parser.add_argument("--out-dir",    default="eval/outputs",
                        help="Output directory for results (default: eval/outputs)")
    args = parser.parse_args()

    evaluate(
        model_type = args.model,
        checkpoint  = args.checkpoint,
        test_csv    = args.test_csv,
        out_dir     = args.out_dir,
    )
