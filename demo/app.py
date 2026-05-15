# =============================================================================
# File: demo/app.py
# Owner: Shamathmika
#
# Purpose:
#   Gradio web demo for DocRestore. Accepts a degraded document image (crop
#   at native resolution for best results), runs inference with a selected
#   model, and shows the restored output alongside OCR text comparison.
#
# Dependencies:
#   - demo/inference.py   - load_best_model(), run_inference()
#   - gradio, Pillow, pytesseract
#
# Usage:
#   python demo/app.py           # opens local URL
#   python demo/app.py --share   # creates public Gradio share link (required on Colab)
# =============================================================================

import argparse
import sys
from pathlib import Path

import gradio as gr

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
# Guarded imports
# ---------------------------------------------------------------------------
try:
    from inference import load_best_model, run_inference
    _INFERENCE_AVAILABLE = True
except ImportError:
    _INFERENCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_models: dict = {}


def _get_model(model_name: str):
    """Load and cache model weights by name."""
    if not _INFERENCE_AVAILABLE:
        raise RuntimeError(
            "demo/inference.py is not available yet. "
        )
    if model_name not in _models:
        _models[model_name] = load_best_model(model_name, CHECKPOINT_DIR)
    return _models[model_name]


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------
def restore(degraded_pil, model_name: str) -> tuple:
    if degraded_pil is None:
        return None, "", ""

    model = _get_model(model_name)
    restored_pil = run_inference(model, degraded_pil)

    try:
        import pytesseract
        deg_ocr  = pytesseract.image_to_string(degraded_pil)
        rest_ocr = pytesseract.image_to_string(restored_pil)
    except Exception as e:
        deg_ocr  = f"OCR unavailable: {e}"
        rest_ocr = ""

    return restored_pil, deg_ocr, rest_ocr


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="DocRestore Demo") as app:
        gr.Markdown("## DocRestore: Document Image Restoration")
        gr.Markdown(
            "Upload a degraded document image. "
            "Select a model and click **Restore** to see the cleaned result."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Degraded Input",
                    type="pil",
                    image_mode="RGB",
                )
                model_selector = gr.Radio(
                    choices=["DocRes", "NAFNet"],
                    value="NAFNet",
                    label="Model",
                )
                run_btn = gr.Button("Restore", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="Restored Output",
                    type="pil",
                )

        with gr.Row():
            deg_ocr_box  = gr.Textbox(label="OCR: Degraded Input",  lines=12, interactive=False)
            rest_ocr_box = gr.Textbox(label="OCR: Restored Output", lines=12, interactive=False)

        run_btn.click(
            fn=restore,
            inputs=[input_img, model_selector],
            outputs=[output_img, deg_ocr_box, rest_ocr_box],
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
