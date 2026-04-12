# DocRestore – Document Image Restoration for OCR Readability

An end-to-end deep learning system that restores degraded scanned and printed document images into clean, high-contrast, machine-readable versions. Degradations handled include coffee stains, ink bleed, wrinkles, fold marks, photocopy artifacts, and uneven contrast.

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
- **Augmentation:** Augraphy pipeline — InkBleed, BleedThrough, LowInkPeriodicLines, DirtyDrum, SubtleNoise, Folding, Brightness, ColorPaper, Jpeg, Markup
- **Loss:** L1 + perceptual loss (VGG16 features)
- **Evaluation:** PSNR, SSIM, OCR character error rate (CER) via Tesseract
- **Demo:** Gradio web app

---

## Results

| Model | PSNR | SSIM | CER |
|-------|------|------|-----|
| DocRes | — | — | — |
| NAFNet | — | — | — |

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
python eval/run_eval.py

# 8. Run demo
python demo/app.py
```

### Dataset Generation Notes

Training data is generated synthetically from arXiv PDFs using [Augraphy](https://github.com/sparkfish/augraphy).
The paper list is in `data/arxiv_papers.json` — add more entries to grow the dataset.

Augraphy degradations applied: ink bleed, bleed-through, periodic ink lines,
brightness variation, dirty drum stains, subtle noise, folding, JPEG artifacts, and markup.
