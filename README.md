# Micro-LED-based PUF Pipeline

End-to-end pipeline for **micro-LED near-field image recognition** and **PUF
(Physical Unclonable Function) bit generation**.

## Overview

1. **Detection & cropping (YOLOv8):** Detect the key region in raw micro-LED
   images and crop to **80×80** for classification.
2. **Classification & 256-bit output (ResNet):** Train a ResNet-based classifier
   and generate **256-bit binary embeddings**.
3. **Stabilization (Fuzzy Extractor):** Normalize noisy bit outputs so samples
   from the same class converge to a consistent bitstring.

---

## Repository Contents

| File / Package | Purpose |
| --- | --- |
| `YOLOv8_train.py` | Train YOLOv8 using labeled detection data. |
| `YOLOv8_image.py` | Crop raw images using the trained YOLOv8 model. |
| `frames_train_YOLO.rar` | YOLOv8 training dataset + labels. |
| `TestImageSet_SelectedbyYOLO.rar` | Cropped 6-class dataset for classification/PUF. |
| `ResNet_highPrivacy_final - Eng.py` | Final classifier training + 256-bit output. |
| `FuzzyExtractor.py` | Stabilize noisy 256-bit outputs into consistent bits. |

---

## Data Packages

- **`frames_train_YOLO.rar`**
  - Detection dataset for YOLOv8 training (images + labels).
- **`TestImageSet_SelectedbyYOLO.rar`**
  - Cropped dataset with 6 micro-LED classes.
  - Used by `ResNet_highPrivacy_final - Eng.py`.

> Extract the `.rar` files to local directories before running the scripts.

---

## Requirements

- Python 3.8+
- ultralytics (YOLOv8)
- PyTorch + torchvision
- numpy, opencv-python, scikit-learn, scipy, matplotlib, seaborn, tqdm

Install dependencies:

```bash
pip install ultralytics torch torchvision numpy opencv-python scikit-learn scipy matplotlib seaborn tqdm
```

---

## Quick Start

### 1) Train YOLOv8 (Detection Model)

Update the dataset path in `YOLOv8_train.py` if needed:

```python
# dataset_yaml = r'D:/Research/PUF_micro-LED/videos3/frames_YOLO/dataset.yaml'
```

Run:

```bash
python YOLOv8_train.py
```

The model is saved to:

```
runs/detect/train/weights/best.pt
```

### 2) Crop Raw Images with YOLOv8

Edit `YOLOv8_image.py`:

```python
model_path = r"runs/detect/train/weights/best.pt"
input_root = r"<path-to-raw-images>"
output_root = r"<path-to-save-crops>"
```

Run:

```bash
python YOLOv8_image.py
```

Cropped outputs are resized to a uniform size for classification.

### 3) Train the ResNet Classifier + Generate 256-bit Output

Update the dataset path inside `ResNet_highPrivacy_final - Eng.py` and run:

```bash
python "ResNet_highPrivacy_final - Eng.py"
```

### 4) Stabilize Bits with the Fuzzy Extractor

Update paths inside `FuzzyExtractor.py` as needed, then run:

```bash
python FuzzyExtractor.py
```

---

## Notes

- The 256-bit outputs are **noisy** across samples of the same class; the fuzzy
  extractor normalizes them to a consistent bitstring.
- Ensure the YOLO crops preserve the most informative micro-LED region for best
  classification/PUF stability.

---

## License

MIT License. See `LICENSE`.
