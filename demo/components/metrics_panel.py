# =============================================================================
# File: demo/components/metrics_panel.py
# Owner: Sakshat
#
# Purpose:
#   Formats PSNR / SSIM / CER values for display in the DocRestore Gradio demo.
#   Imported by demo/app.py.
#
# Exports:
#   build_metrics_panel(psnr, ssim, cer) -> str
#       Returns a formatted multi-line string suitable for a gr.Textbox output.
#       - PSNR in dB, 2 decimal places
#       - SSIM as 0.XXX, 3 decimal places
#       - CER as percentage, 1 decimal place
# =============================================================================

import math


def build_metrics_panel(psnr: float, ssim: float, cer: float) -> str:
    """
    Format PSNR / SSIM / CER values for display in a Gradio Textbox.

    Args:
        psnr: Peak Signal-to-Noise Ratio in dB.
        ssim: Structural Similarity Index (0–1).
        cer:  Character Error Rate (0–1, where 0 is perfect).

    Returns:
        A multi-line string with labelled, formatted values.
        NaN / inf values are shown as "N/A".

    Example output:
        PSNR :  28.45 dB
        SSIM :   0.873
        CER  :  12.3 %
    """
    def _fmt_psnr(v: float) -> str:
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:6.2f} dB"

    def _fmt_ssim(v: float) -> str:
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:6.3f}"

    def _fmt_cer(v: float) -> str:
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v * 100:5.1f} %"

    lines = [
        f"PSNR :  {_fmt_psnr(psnr)}",
        f"SSIM :  {_fmt_ssim(ssim)}",
        f"CER  :  {_fmt_cer(cer)}",
    ]
    return "\n".join(lines)
