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
| NoisyOffice | 216 pairs | Noise, blur, low contrast | Domain gap evaluation only |

---

## Results

### Synthetic test set (90 images, 512x512)

| Model | PSNR (dB) | SSIM | CER |
|-------|-----------|------|-----|
| Baseline (no model) | 14.19 | 0.7738 | 0.3425 |
| NAFNet | 20.70 | 0.8763 | 0.6790 |
| DocRes | **24.11** | **0.8993** | **0.4191** |

DocRes achieves +9.9 dB PSNR and +0.126 SSIM over the unrestored baseline.

### Domain gap — NAFNet on NoisyOffice real scans (216 pairs)

| Domain | PSNR (dB) | SSIM | CER |
|--------|-----------|------|-----|
| Synthetic test (train domain) | 20.70 | 0.8763 | 0.6790 |
| NoisyOffice (real scans) | 21.74 | 0.9430 | 0.0619 |

No domain gap penalty — NAFNet generalises well to real-world scan artifacts.

---

## Checkpoints

Model weights are too large for GitHub (DocRes: 504 MB, NAFNet: 111 MB).

Download from Google Drive: https://drive.google.com/drive/folders/1i_cHhIiLGNhBLl227sZiNqfY-x3Oae47?usp=sharing

Place downloaded `.pth` files in the `checkpoints/` directory before running eval or demo.

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
    --checkpoint checkpoints/nafnet_best.pth \
    --test-csv data/test.csv \
    --out-dir eval/outputs/nafnet

python eval/run_eval.py \
    --model docres \
    --checkpoint checkpoints/docres_best.pth \
    --test-csv data/test.csv \
    --out-dir eval/outputs/docres

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
│   ├── preprocess.py                # Image preprocessing for demo inference
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

- **TextAwareLoss**: Laplacian edge detection on the clean image produces a spatial weight map that penalises errors on text strokes 1x to 3x more than background. Acts as a differentiable OCR proxy without requiring a differentiable OCR engine.
- **Shared model interface**: Both DocRes and NAFNet implement identical `forward()`, `save_checkpoint()`, and `load_weights()` APIs, allowing drop-in comparison across all scripts.
- **512x512 training**: Larger crop size retains more document context per batch and improves text legibility in outputs.
- **70/15/15 split**: Larger val/test sets give more stable metric estimates across approximately 135 test images.
