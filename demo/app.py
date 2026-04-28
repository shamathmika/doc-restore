# =============================================================================
# File: demo/app.py
# Owner: Shamathmika
#
# Purpose:
#   Gradio web demo for DocRestore. Accepts a degraded document image,
#   runs inference with a selected model (DocRes or NAFNet), displays the
#   restored result alongside the input, and reports PSNR / SSIM / CER.
#
# Dependencies:
#   - demo/inference.py                  — load_best_model(), run_inference()
#   - demo/components/image_panel.py     — build_image_panel()
#   - demo/components/metrics_panel.py   — build_metrics_panel()
#   - eval/metrics.py                    — compute_psnr(), compute_ssim(), compute_ocr_cer()
#   - gradio, Pillow, numpy
#
# Usage:
#   python demo/app.py           # opens local URL
#   python demo/app.py --share   # creates public Gradio share link (required on Colab)
# =============================================================================

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths and patch sys.path before guarded imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEMO_DIR     = Path(__file__).resolve().parent
for _p in [str(_PROJECT_ROOT), str(_DEMO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHECKPOINT_DIR = str(_PROJECT_ROOT / "checkpoints")

# ---------------------------------------------------------------------------
# Guarded imports — teammate modules may not be pushed yet
# ---------------------------------------------------------------------------
try:
    from inference import load_best_model, run_inference
    _INFERENCE_AVAILABLE = True
except ImportError:
    _INFERENCE_AVAILABLE = False

try:
    from components.image_panel import build_image_panel
    _IMAGE_PANEL_AVAILABLE = True
except ImportError:
    _IMAGE_PANEL_AVAILABLE = False

try:
    from components.metrics_panel import build_metrics_panel
    _METRICS_PANEL_AVAILABLE = True
except ImportError:
    _METRICS_PANEL_AVAILABLE = False

try:
    from eval.metrics import compute_psnr, compute_ssim, compute_ocr_cer
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_models: dict = {}


def _get_model(model_name: str):
    """Load and cache model weights by name."""
    if not _INFERENCE_AVAILABLE:
        raise RuntimeError(
            "demo/inference.py is not available yet. "
            "Merge Sakshat's code before running inference."
        )
    if model_name not in _models:
        _models[model_name] = load_best_model(model_name, CHECKPOINT_DIR)
    return _models[model_name]


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------
def restore(degraded_pil, model_name: str) -> tuple:
    """
    Gradio callback: receives a PIL image, returns (panel_or_restored, metrics_text).

    Args:
        degraded_pil: Uploaded image from Gradio Image component (PIL).
        model_name:   "DocRes" or "NAFNet".

    Returns:
        Tuple of (PIL image, metrics str).
    """
    if degraded_pil is None:
        return None, "Upload an image first."

    model = _get_model(model_name)
    restored_pil = run_inference(model, degraded_pil)

    # Build image panel (side-by-side degraded | restored)
    if _IMAGE_PANEL_AVAILABLE:
        panel = build_image_panel(degraded_pil, restored_pil)
    else:
        panel = restored_pil

    # Compute metrics
    if _METRICS_AVAILABLE:
        deg_arr  = np.array(degraded_pil)
        rest_arr = np.array(restored_pil)
        psnr = compute_psnr(deg_arr, rest_arr)
        ssim = compute_ssim(deg_arr, rest_arr)
        cer  = compute_ocr_cer(deg_arr, rest_arr)
    else:
        psnr = ssim = cer = float("nan")

    if _METRICS_PANEL_AVAILABLE:
        metrics_text = build_metrics_panel(psnr, ssim, cer)
    else:
        def _fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "N/A"
        metrics_text = f"PSNR : {_fmt(psnr)}\nSSIM : {_fmt(ssim)}\nCER  : {_fmt(cer)}"

    return panel, metrics_text


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="DocRestore Demo") as app:
        gr.Markdown("## DocRestore — Document Image Restoration")
        gr.Markdown(
            "Upload a degraded document image. "
            "Select a model and click **Restore** to see the cleaned result."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Upload Degraded Document",
                    type="pil",
                    image_mode="RGB",
                )
                model_selector = gr.Radio(
                    choices=["DocRes", "NAFNet"],
                    value="DocRes",
                    label="Model",
                )
                run_btn = gr.Button("Restore", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="Restored Output",
                    type="pil",
                )
                metrics_box = gr.Textbox(
                    label="Metrics",
                    lines=4,
                    interactive=False,
                )

        run_btn.click(
            fn=restore,
            inputs=[input_img, model_selector],
            outputs=[output_img, metrics_box],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocRestore Gradio demo")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link (required on Colab)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Local port to serve on (default: 7860)")
    args = parser.parse_args()

    app = build_app()
    app.launch(share=args.share, server_port=args.port)
