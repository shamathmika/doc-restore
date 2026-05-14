# DocRestore: Document Image Restoration for OCR Readability

An end-to-end deep learning system that restores degraded scanned and printed document images into clean, high-contrast, machine-readable versions. Degradations handled include coffee stains, ink bleed, wrinkles, fold marks, photocopy artifacts, and uneven contrast.

---

## Problem Statement

Scanned and photographed document images frequently suffer from degradations that reduce OCR accuracy and make documents difficult to read or archive. DocRestore trains document-specific restoration models on synthetically degraded data, measurably improving both visual quality (PSNR, SSIM) and machine readability (OCR character error rate).

---

## Team Members

| Name | Email |
|------|-------|
| Apoorva Adimulam | apoorva.adimulam@sjsu.edu |
| Sakshat Patil | sakshat.patil@sjsu.edu |
| FNU Shamathmika | shamathmika.shamathmika@sjsu.edu |

---

## Models

| Model | Architecture | Loss |
|-------|-------------|------|
| NAFNet-TextAware | Nonlinear Activation Free Network (Chen et al., ECCV 2022) | L1 + 0.1 x Perceptual + 1.0 x TextAware |
| DocRes-TextAware | U-Net with multi-head attention bottleneck (Zhang et al., CVPR 2024) | L1 + 0.1 x Perceptual + 1.0 x TextAware |

TextAwareLoss upweights the pixel loss on text-stroke edges (detected via Laplacian) as a differentiable proxy for OCR readability.

---

## Datasets

| Dataset | Pairs | Degradation Types | Use |
|---------|-------|-------------------|-----|
| Synthetic (arXiv + Augraphy) | ~900 | Ink bleed, bleed-through, stains, folds, noise, JPEG artifacts | Train / Val / Test (70/15/15) |
| NoisyOffice | 72 | Noise, blur, low contrast | Domain gap evaluation only |

---

## Results

| Model | PSNR (dB) | SSIM | CER |
|-------|-----------|------|-----|
| Baseline (no model) | TBD | TBD | TBD |
| NAFNet-TextAware | TBD | TBD | TBD |
| DocRes-TextAware | TBD | TBD | TBD |

Evaluated on the 15% held-out test split at 512x512 resolution.

---

## Primary Notebook: Kaggle

**`notebooks/train_kaggle.ipynb`** is the single entry point for training and evaluation. It runs on Kaggle (T4 x2 GPU, free tier).

| Cell | What it does |
|------|-------------|
| 1 | GPU check |
| 2 | Install dependencies (augraphy, pymupdf, pytesseract, scikit-image) |
| 3 | Load repo from attached Kaggle dataset |
| 4 | Write training configs (kaggle_nafnet_textaware.yaml, kaggle_docres.yaml) |
| 5 | Generate data from arXiv PDFs via Augraphy, build 70/15/15 split |
| 6 | Train NAFNet-TextAware (50 epochs, 512x512, batch=2) |
| 7 | Train DocRes-TextAware (50 epochs, 512x512, batch=2) |
| 8 | Evaluate both models + baseline on test split |
| 9 | Print results table (PSNR / SSIM / CER) |
| 10 | Domain gap test on NoisyOffice real scans |
| 11 | Generate graphs (loss curves, metrics bar, domain gap) |
| 12 | Zip and download all results |

### Setup on Kaggle

1. Upload this repo as a Kaggle dataset named `doc-restore-code`
2. Create a new notebook, attach the dataset, set Accelerator to **GPU T4 x2**
3. Copy in `notebooks/train_kaggle.ipynb` (or use the notebook directly from the dataset)
4. Run all cells top to bottom

### Running locally (eval / demo only)

Training requires a GPU. Use the Kaggle notebook above for training. Steps below are for local evaluation of downloaded checkpoints or running the demo.

```bash
pip install -r requirements.txt

# Download data
python data/download_shabby.py    # generates data/shabby/clean and data/shabby/degraded
python data/download_noisy.py     # downloads NoisyOffice for domain gap eval

# Evaluate a checkpoint
python eval/run_eval.py \
    --model nafnet \
    --checkpoint /path/to/nafnet_textaware_best.pth \
    --test-csv data/test.csv \
    --out-dir eval/outputs/nafnet_textaware

# Baseline (no model)
python eval/run_baseline.py --test-csv data/test.csv --out-dir eval/outputs/baseline

# Gradio demo
python demo/app.py          # local
python demo/app.py --share  # Colab / remote (generates public link)
```

---

## Project Structure

```
doc-restore/
├── configs/
│   ├── docres.yaml                  # DocRes architecture config (used by run_eval.py)
│   └── nafnet.yaml                  # NAFNet architecture config (used by run_eval.py)
├── data/
│   ├── augment.py                   # Augraphy pipeline (single source of truth)
│   ├── dataloader.py                # PyTorch Dataset and DataLoader factory
│   ├── download_shabby.py           # arXiv PDF -> clean/degraded pairs
│   ├── download_noisy.py            # NoisyOffice dataset download
│   ├── preprocess.py                # Resize and normalize utilities
│   ├── split.py                     # Train / val / test CSV split
│   └── arxiv_papers.json            # 99 arXiv papers used for synthetic data
├── demo/
│   ├── app.py                       # Gradio web app
│   ├── inference.py                 # Model loading and inference
│   └── components/
│       ├── image_panel.py
│       └── metrics_panel.py
├── eval/
│   ├── metrics.py                   # compute_psnr, compute_ssim, compute_ocr_cer
│   ├── run_eval.py                  # Batch evaluation (PSNR / SSIM / CER + visual grid)
│   └── run_baseline.py              # Baseline evaluation (no model)
├── models/
│   ├── docres_wrapper.py            # DocRes U-Net architecture
│   └── nafnet_wrapper.py            # NAFNet architecture
├── notebooks/
│   └── train_kaggle.ipynb           # Primary training + evaluation notebook (Kaggle)
├── train/
│   ├── losses.py                    # L1, Perceptual, TextAware, Combined losses
│   ├── scheduler.py                 # LR scheduler factory
│   ├── train_docres.py              # DocRes training loop
│   └── train_nafnet.py              # NAFNet training loop
└── requirements.txt
```

---

## Key Design Decisions

- **TextAwareLoss**: Laplacian edge detection on the clean image produces a spatial weight map that penalizes errors on text strokes 1x to 3x more than background. Acts as a differentiable OCR proxy without requiring a differentiable OCR engine.
- **Shared model interface**: Both DocRes and NAFNet implement identical `forward()`, `save_checkpoint()`, and `load_weights()` APIs, allowing drop-in comparison across all scripts.
- **512x512 training**: Larger crop size than a 256x256 baseline retains more document context per batch and improves text legibility in outputs.
- **70/15/15 split**: Larger val/test sets give more stable metric estimates across approximately 135 test images.
