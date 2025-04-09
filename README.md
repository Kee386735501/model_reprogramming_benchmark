

---

# Model Reprogramming Benchmark

> A benchmark suite for evaluating model reprogramming methods across multiple datasets, backbones, and label mapping strategies.

## 📌 Overview

This repository provides a modular framework for training and evaluating **model reprogramming** techniques, with a focus on:

- Visual prompts (e.g., MR, SMM, FFT-based prompts)
- Label mapping strategies (e.g., ILM, FLM, RLM)
- Diverse target datasets (e.g., CIFAR-10/100, SVHN, GTSRB, QuickDraw, DomainNet)
- Backbone models such as **ResNet** and **ViT**

The framework supports:
- Mixed-precision training (AMP)
- Multi-GPU distributed training (DDP)
- Training logs and metrics via TensorBoard and CSV
- Easy extension to new datasets, prompts, and mapping methods

## 🧩 Directory Structure

```bash
.
├── assets/                 # Visualization assets (prompt masks, examples, etc.)
├── dataset_tools/         # Dataset preprocessing, label mapping, and dataloaders
├── model/                 # Core model definitions and reprogramming modules
├── logs/                  # Training logs (tensorboard + csv)
├── *.ipynb                # Interactive notebooks for analysis and testing
├── 3月27日.md             # Experiment notes (Chinese)
└── ...
```

## 🧪 Supported Techniques

### Prompting Methods
- **MR (Mask Reprogramming)**
- **SMM (Soft Masked Modulation)**
- **FFT-based Prompting** (Fourier reprogramming)

### Label Mapping
- ILM: Independent Label Mapping
- FLM: Frequency-based Label Mapping
- RLM: Random Label Mapping

### Backbone Networks
- ResNet18, ResNet50
- Vision Transformer (ViT-B/32)

## 📚 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Datasets will be automatically downloaded to `./data/` the first time they are used.
Alternatively, manually download and place them under:
```
./data/<dataset_name>/
```

### 3. Train Model

Example (MR + ResNet18 on QuickDraw):
```bash
python train.py --dataset quickdraw --model resnet18 --prompt mr --mapping ilm
```

### 4. Visualize Results

Launch TensorBoard:
```bash
tensorboard --logdir logs/
```

Or check CSV logs in the `logs/` folder.

## 📊 Notebooks

Explore experiments and visualizations:
- `mr.ipynb` – Visual Prompt experiments
- `fft.ipynb` – Fourier-based prompt
- `smm.ipynb` – SMM method

## 📈 Results (Example)

| Prompt Type | Dataset   | Backbone | Accuracy |
|-------------|-----------|----------|----------|
| MR          | CIFAR-10  | ResNet18 | 84.3%    |
| SMM         | SVHN      | ResNet18 | 91.2%    |
| FFT         | QuickDraw | ViT-B/32 | 87.5%    |

*(Detailed benchmarks in progress...)*

## 🧠 Citation

> This project is in development. Citation info will be provided once the paper is available.

## 🙌 Acknowledgements

Inspired by research in model reprogramming and visual prompt tuning, including:
- [Model Reprogramming](https://arxiv.org/abs/2002.11944)
- [VPT: Visual Prompt Tuning](https://arxiv.org/abs/2203.12152)

---
