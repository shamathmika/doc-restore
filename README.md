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
| ShabbyPages | ~6,000 | Ink bleed, stains, folds, photocopy artifacts | Train / Val / Test |
| NoisyOffice | 72 | Noise, blur, low contrast | Evaluation only |

---

## Approach

- **Models:** DocRes (fine-tuned) and NAFNet (baseline)
- **Augmentation:** Augraphy pipeline — InkBleed, LowInkPeriodicLines, DirtyDrum, Folding, Brightness
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
pip install -r requirements.txt

# 2. Download datasets
python data/download_shabby.py
python data/download_noisy.py

# 3. Preprocess and split
python data/preprocess.py
python data/split.py

# 4. Train
python train/train_docres.py --config configs/docres.yaml
python train/train_nafnet.py --config configs/nafnet.yaml

# 5. Evaluate
python eval/run_eval.py

# 6. Run demo
python demo/app.py
```
