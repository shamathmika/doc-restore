# =============================================================================
# File: demo/inference.py
# Owner: Sakshat
#
# Purpose:
#   Model loading and inference utilities for the DocRestore Gradio demo.
#   Imported by demo/app.py.
#
# Exports:
#   run_inference(model, image_input) -> PIL.Image
#       Preprocess a PIL image, run it through the model, return restored PIL.
#
#   load_best_model(model_type, checkpoints_dir) -> nn.Module
#       Scan checkpoints/ for the best checkpoint, load and return the model.
#
# Usage (standalone sanity check):
#   python demo/inference.py \
#       --model nafnet \
#       --checkpoint checkpoints/nafnet_best.pth \
#       --image path/to/degraded.png \
#       --out   path/to/restored.png
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make project root importable when this file is executed directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocess import preprocess_image

# ---------------------------------------------------------------------------
# Lazy imports for model wrappers
# ---------------------------------------------------------------------------
try:
    from models.nafnet_wrapper import NAFNetModel
    _NAFNET_AVAILABLE = True
except ImportError:
    _NAFNET_AVAILABLE = False

try:
    from models.docres_wrapper import DocResModel
    _DOCRES_AVAILABLE = True
except ImportError:
    _DOCRES_AVAILABLE = False

_INPUT_SIZE = (512, 512)   # must match training config input_size


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL (RGB) → float32 tensor (1, 3, H, W) in [0, 1]."""
    arr = preprocess_image(pil_img if isinstance(pil_img, (str, Path)) else pil_img,
                           output_size=_INPUT_SIZE)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)


def _pil_to_tensor_from_pil(pil_img: Image.Image) -> torch.Tensor:
    """PIL image → float32 tensor (1, 3, H, W) without touching the filesystem."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(_INPUT_SIZE, Image.LANCZOS)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Float32 tensor (1, 3, H, W) in [0, 1] → PIL image (RGB)."""
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(model: torch.nn.Module, image_input: Image.Image) -> Image.Image:
    """
    Resize to _INPUT_SIZE, run model, upscale output back to original resolution.
    Upscaling before OCR matches how the eval script computes CER.
    """
    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    orig_size = image_input.size  # (W, H)
    device = next(model.parameters()).device

    inp = _pil_to_tensor_from_pil(image_input).to(device)
    model.eval()
    with torch.no_grad():
        out = model(inp)

    restored_small = _tensor_to_pil(out)
    return restored_small.resize(orig_size, Image.LANCZOS)


def load_best_model(
    model_type: str,
    checkpoints_dir: str,
) -> torch.nn.Module:
    """
    Scan *checkpoints_dir* for the best checkpoint for *model_type* and
    return a model ready for inference.

    Checkpoint lookup order (first match wins):
        1. <checkpoints_dir>/<model_type>_best.pth
        2. Any *.pth file whose name starts with <model_type>
           with the lowest loss stored in the checkpoint dict.

    Args:
        model_type:      "nafnet" or "docres" (case-insensitive).
        checkpoints_dir: Directory that contains *.pth checkpoint files.

    Returns:
        nn.Module on CPU, in eval mode, weights loaded.

    Raises:
        ValueError:  If model_type is not recognised.
        FileNotFoundError: If no checkpoint is found.
        ImportError: If the required model wrapper is not available.
    """
    model_type = model_type.lower()
    ckpt_dir   = Path(checkpoints_dir)

    # -- Locate checkpoint ---------------------------------------------------
    best_path = ckpt_dir / f"{model_type}_best.pth"
    if not best_path.exists():
        candidates = sorted(ckpt_dir.glob(f"{model_type}*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found for model_type='{model_type}' "
                f"in {ckpt_dir}. Run training first."
            )
        # Pick candidate with lowest stored loss (fallback: first alphabetically)
        def _loss(p: Path) -> float:
            try:
                ckpt = torch.load(p, map_location="cpu")
                return float(ckpt.get("loss", float("inf"))) if isinstance(ckpt, dict) else float("inf")
            except Exception:
                return float("inf")

        best_path = min(candidates, key=_loss)

    print(f"[inference] Loading checkpoint: {best_path}")

    # -- Instantiate model ---------------------------------------------------
    config_path = str(_PROJECT_ROOT / "configs" / f"{model_type}.yaml")

    if model_type == "nafnet":
        if not _NAFNET_AVAILABLE:
            raise ImportError("models/nafnet_wrapper.py is not available.")
        model = NAFNetModel(config_path)

    elif model_type == "docres":
        if not _DOCRES_AVAILABLE:
            raise ImportError(
                "models/docres_wrapper.py is not available yet."
            )
        model = DocResModel(config_path)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'nafnet' or 'docres'."
        )

    model.load_weights(str(best_path))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI for quick sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DocRestore inference on one image")
    parser.add_argument("--model",      required=True,  help="nafnet or docres")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--image",      required=True,  help="Path to degraded input image")
    parser.add_argument("--out",        required=True,  help="Where to save the restored image")
    args = parser.parse_args()

    _model = load_best_model(args.model, str(Path(args.checkpoint).parent))
    _input = Image.open(args.image).convert("RGB")
    _restored = run_inference(_model, _input)
    _restored.save(args.out)
    print(f"Saved restored image → {args.out}")
