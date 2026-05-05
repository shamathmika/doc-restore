# DocRestore: Document Image Restoration for OCR Readability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/notebooks/data_exploration.ipynb)

An end-to-end deep learning system that restores degraded scanned and printed document images into clean, high-contrast, machine-readable versions. Degradations handled include coffee stains, ink bleed, wrinkles, fold marks, photocopy artifacts, and uneven contrast.

---

## Problem Statement

Scanned and photographed document images frequently suffer from degradations - ink bleed, paper fold marks, photocopy drum artifacts, uneven lighting, and JPEG compression - that reduce OCR accuracy and make documents difficult to read or archive. Existing general-purpose image restoration models are not optimized for the specific characteristics of text documents. DocRestore addresses this by training document-specific restoration models on synthetically degraded data, measurably improving both visual quality (PSNR, SSIM) and machine readability (OCR character error rate).

---

## Team Members

| Name | Email |
|------|-------|
| Apoorva Adimulam | apoorva.adimulam@sjsu.edu |
| Sakshat Patil | sakshat.patil@sjsu.edu |
| FNU Shamathmika | shamathmika.shamathmika@sjsu.edu |

---

## Datasets

| Dataset | Pairs | Degradation Types | Split |
|---------|-------|-------------------|-------|
| Synthetic (arXiv + Augraphy) | ~900 | Ink bleed, bleed-through, stains, folds, noise, JPEG artifacts | Train / Val / Test |
| NoisyOffice | 72 | Noise, blur, low contrast | Evaluation only |

---

## Approach

- **Models:** DocRes (fine-tuned) and NAFNet (baseline)
- **Augmentation:** Augraphy pipeline - InkBleed, BleedThrough, LowInkPeriodicLines, DirtyDrum, SubtleNoise, Folding, Brightness, ColorPaper, Jpeg, Markup
- **Loss:** L1 + perceptual loss (VGG16 features)
- **Evaluation:** PSNR, SSIM, OCR character error rate (CER) via Tesseract
- **Demo:** Gradio web app

---

## Results

| Model | PSNR | SSIM | CER |
|-------|------|------|-----|
| DocRes | - | - | - |
| NAFNet | - | - | - |

*To be filled in after evaluation.*

---

## How to Run

```bash
# 1. Install dependencies
#    numpy<2 is required for augraphy compatibility with Python 3.12
#    pymupdf is required for PDF-to-image conversion in download_shabby.py
pip install -r requirements.txt

# 2. Generate training data (downloads 50 arXiv PDFs, applies Augraphy degradations)
#    Produces ~900 clean/degraded pairs in data/shabby/clean/ and data/shabby/degraded/
#    Run with --test to generate only 5 pages for a quick sanity check
python data/download_shabby.py           # full run (~10-20 min)
python data/download_shabby.py --test    # 5 pages only

# 3. Split into train / val / test CSVs (80/10/10)
#    Writes data/train.csv, data/val.csv, data/test.csv
python data/split.py

# 4. Download evaluation dataset
python data/download_noisy.py

# 5. Preprocess
python data/preprocess.py

# 6. Train
python train/train_docres.py --config configs/docres.yaml
python train/train_nafnet.py --config configs/nafnet.yaml

# 7. Evaluate
python eval/run_eval.py --model docres --checkpoint checkpoints/docres_best.pth
python eval/run_eval.py --model nafnet --checkpoint checkpoints/nafnet_best.pth

# 8. Run demo
python demo/app.py          # local
python demo/app.py --share  # Colab (generates public link)
```

> **Training requires a GPU.** 50 epochs on ~900 pairs takes several hours on CPU. Use Google Colab (free T4 GPU) for training. Steps 1-5 and the demo can run locally.

### Running on Google Colab

1. Open the badge link at the top of this README
2. In Colab, run:
```python
!git clone https://github.com/shamathmika/doc-restore.git
%cd doc-restore
!pip install -r requirements.txt
!python train/train_docres.py --config configs/docres.yaml
```
3. For the demo on Colab, use `python demo/app.py --share` to get a public Gradio link.

### Notebooks

| Notebook | Description | Open |
|----------|-------------|------|
| `notebooks/augmentation_preview.ipynb` | Visualize Augraphy degradation stages | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/notebooks/augmentation_preview.ipynb) |
| `notebooks/training_curves.ipynb` | Loss curves and LR schedule | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/notebooks/training_curves.ipynb) |
| `notebooks/data_exploration.ipynb` | Dataset statistics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/notebooks/data_exploration.ipynb) |
| `eval/error_analysis.ipynb` | Failure case analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/eval/error_analysis.ipynb) |
| `eval/model_comparison.ipynb` | DocRes vs NAFNet comparison | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/eval/model_comparison.ipynb) |
| `eval/ablation.ipynb` | Loss and augmentation ablation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/doc-restore/blob/main/eval/ablation.ipynb) |

### Dataset Generation Notes

Training data is generated synthetically from arXiv PDFs using [Augraphy](https://github.com/sparkfish/augraphy).
The paper list is in `data/arxiv_papers.json` - add more entries to grow the dataset.

Augraphy degradations applied: ink bleed, bleed-through, periodic ink lines,
brightness variation, dirty drum stains, subtle noise, folding, JPEG artifacts, and markup.

---

## Next Steps

- Train on larger arXiv PDF datasets to improve generalization across document styles
- Explore diffusion-based restoration models (e.g., DocDiff) as an alternative backbone
- Add domain adaptation to handle real-world scan artifacts not covered by Augraphy
- Extend OCR evaluation to full-document word error rate (WER) in addition to CER
- Investigate self-supervised pretraining on unlabeled scanned documents

---

## Project Structure

```
doc-restore/
├── configs/          # Training configs (docres.yaml, nafnet.yaml)
├── data/             # Data pipeline
│   ├── augment.py          # Augraphy degradation - single source of truth
│   ├── dataloader.py       # PyTorch Dataset and DataLoader factory
│   ├── download_shabby.py  # arXiv PDF -> clean/degraded pairs
│   ├── download_noisy.py   # NoisyOffice dataset download
│   ├── preprocess.py       # Resize and normalize images
│   └── split.py            # Train / val / test CSV split
├── demo/             # Gradio web demo
│   ├── app.py              # Main Gradio app
│   ├── inference.py        # Model loading and inference
│   └── components/
│       ├── image_panel.py    # Side-by-side image display + metrics wiring
│       └── metrics_panel.py  # PSNR / SSIM / CER formatted display
├── eval/             # Evaluation
│   ├── metrics.py          # compute_psnr, compute_ssim, compute_ocr_cer
│   ├── run_eval.py         # Batch evaluation script - writes results CSV
│   ├── error_analysis.ipynb    # Failure case analysis
│   ├── model_comparison.ipynb  # DocRes vs NAFNet comparison
│   └── ablation.ipynb          # Loss and augmentation ablation
├── models/           # Model definitions
│   ├── docres_wrapper.py   # DocRes U-Net architecture
│   └── nafnet_wrapper.py   # NAFNet architecture
├── notebooks/        # Exploratory notebooks
│   ├── augmentation_preview.ipynb  # Visualize Augraphy pipeline stages
│   ├── training_curves.ipynb       # Loss curves and LR schedule
│   └── data_exploration.ipynb      # Dataset statistics
├── train/            # Training scripts
│   ├── losses.py         # L1, perceptual, and combined loss
│   ├── scheduler.py      # LR scheduler factory
│   ├── train_docres.py   # DocRes training loop
│   └── train_nafnet.py   # NAFNet training loop
└── checkpoints/      # Saved model weights (gitignored)
```
