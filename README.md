# FAL-AD: Federated and Augmented Learning Framework for Alzheimer's Disease Detection via Speech

## 📄 Paper

> **Breaking Data Efficiency Dilemma: A Federated and Augmented Learning Framework for Alzheimer's Disease Detection Via Speech**  
> ICASSP 2026 · DOI: [10.1109/ICASSP55912.2026.11463930](https://doi.org/10.1109/ICASSP55912.2026.11463930)

---

## 🔍 Overview

This repository implements **FAL-AD**, a privacy-preserving federated learning framework augmented by voice conversion for Alzheimer's disease (AD) detection from speech. The framework addresses two critical challenges in clinical AI:

1. **Data efficiency** — Leveraging voice conversion based data augmentation to compensate for the limited availability of labeled AD speech data
2. **Data privacy** — Employing federated learning to train models across distributed institutions without sharing raw patient data

The framework supports **three learning paradigms**:

| Mode | Description |
|------|-------------|
| `cl` | Centralized Learning — all data pooled on a single node |
| `ll` | Local Learning — each client trains independently without collaboration |
| `fl` | Federated Learning — collaborative training via Flower framework |

---

## 📊 Results

The table below summarizes the reported performance on ADReSSo. It includes previous centralized methods and our implementations under Centralized Learning (CL), Local Learning (LL), and Federated Learning (FL), with and without augmentation (Aug). CL is the strict reproduction version of CogniAlign using its source code.

| Modality | Metric | C-Attn | Ying | Bang | CogniAlign | CL | CL+Aug | LL | LL+Aug | FL | FL+Aug |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Audio | Acc | 75.30 | 71.20 | 69.01 | <u>80.12</u> | 74.55 | <u>79.39</u> | <u>68.69</u> | 68.08 | 83.84 | <b><u>85.05</u></b> |
| Audio | F1 | 76.00 | 73.10 | 70.39 | <u>79.46</u> | 73.39 | <u>79.14</u> | 65.68 | <u>67.10</u> | 83.67 | <b><u>84.64</u></b> |
| Text | Acc | 73.50 | 78.90 | 83.10 | <u>86.77</u> | 84.85 | <u>86.67</u> | 78.39 | <u>79.80</u> | 87.68 | <b><u>90.30</u></b> |
| Text | F1 | 73.50 | 79.00 | 83.10 | <u>86.59</u> | 84.69 | <u>86.63</u> | 77.48 | <u>79.55</u> | 87.64 | <b><u>90.28</u></b> |
| Both | Acc | 77.20 | 83.70 | 87.32 | <u>90.36</u> | 86.06 | <u>86.67</u> | 78.59 | <u>80.61</u> | 89.70 | <b><u>91.52</u></b> |
| Both | F1 | 77.60 | 83.30 | 87.25 | <u>90.11</u> | 85.89 | <u>86.64</u> | 77.16 | <u>80.35</u> | 89.65 | <b><u>91.45</u></b> |

The underline marks the best value within each comparison group, and bold marks the global best value.

---

## 🏗️ Architecture

### Multimodal Encoder

The paper uses a dual-encoder design that keeps speech and text as sequences until the fusion stage:

| Modality | Backbone | Output |
|----------|----------|--------|
| **Text** | DistilBERT | Contextual token embeddings from `distilbert-base-uncased` |
| **Audio** | wav2vec 2.0 | Contextual frame-level embeddings from `facebook/wav2vec2-base` |

The implementation also keeps a Mel / EGEMAPS front-end branch for some ablations, but the paper configuration centers on sequence-level fusion between audio and text features rather than early concatenation.

### Cross-Attention Fusion Module

The core fusion block is **`GatedCrossAttentionFusion`**, stacked inside `CrossAttentionTransformerEncoder` for multiple layers:

1. **Pre-norm cross-attention** — Both source and memory are normalized before attention, which stabilizes optimization in deeper stacks.
2. **Unidirectional fusion** — Audio features query the text memory bank (`audio → text`), allowing speech representations to selectively attend to linguistically relevant tokens.
3. **Gated residual connection** — The attention output is concatenated with the original source representation and passed through a sigmoid gate, controlling how much cross-modal signal enters the residual path.
4. **Feed-forward refinement** — A position-wise MLP with ReLU and dropout further transforms the fused sequence.


### Pooling and Classifier

After fusion, the sequence is reduced by one of three strategies: `mean`, `cls`, or `attn` / `gatedattn`. The final classifier is a LayerNorm + Dropout + MLP head that predicts the two classes (AD / HC). In the shipped configs, CL/LL use `mean` pooling, while FL uses `attn` pooling.

### Federated Learning
- Built on **Flower** (`flwr`) for robust federated orchestration
- Supports multiple aggregation strategies: FedAvg, FedAdam, FedAdagrad, FedYogi, FedProx
- Client-server architecture with configurable round count and local training epochs

---

## 📁 Project Structure

```
fal-ad/
├── main.py         # Entry point — supports cl / ll / fl modes
├── model.py        # Model definitions (GatedCrossAttentionFusion, CrossAttentionTransformerEncoder, pooling)
├── server.py       # Flower server — strategy selection & metrics logging
├── client.py       # Flower client — local training & evaluation
├── dataset.py      # ADReSSo21 dataset loader with CV and federated splits
├── utils.py        # Training utilities: train(), evaluation(), config management
├── stats.py        # Lightweight script for model statistics
└── configs/        # YAML configuration files (one per mode)
```

---

## ⚙️ Requirements

The environment is installed with a mixed **Conda + pip** workflow. PyTorch/CUDA packages must be installed through conda to ensure CUDA builds correctly; the remaining packages are installed with pip (audio libraries such as librosa and soundfile tend to be more stable via pip).

### Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate fl
```

> All versions are pinned in `environment.yml`, so the runtime environment can be reproduced exactly.

### Supplemental pip Packages (3 Total)

```bash
pip install transformers==4.48.0 librosa==0.10.0 soundfile==0.13.1
```

> ⚠️ **Do not install torch with pip outside the conda environment**. The pip build of torch does not include CUDA drivers, which will cause `torch.cuda.is_available()` to return `False`.

### Complete Dependency Version Table

**Installed with conda:**

| Package | Version | Description |
|----|------|------|
| python | 3.9 | |
| pytorch-cuda | 12.1 | PyTorch build compiled with CUDA 12.1 |
| torchaudio | 2.1.1 | |
| torchvision | 0.16.1 | |
| numpy | 1.23.5 | |
| pandas | 2.2.3 | |
| flwr | 1.7.0 | Flower federated learning framework |
| wandb | 0.21.0 | Weights & Biases logging |
| scipy | 1.13.1 | |
| scikit-learn | 1.6.1 | |
| pyyaml | 6.0.3 | |

**Installed with pip (after activating the conda environment):**

| Package | Version | Description |
|----|------|------|
| transformers | 4.48.0 | DistilBERT / wav2vec2 models |
| librosa | 0.10.0 | Audio processing |
| soundfile | 0.13.1 | Audio file I/O |

---

## 📊 Dataset: ADReSSo21

The project uses the **ADReSSo21** (Alzheimer's Dementia Recognition through Spontaneous Speech) dataset. You must obtain the dataset separately from the official source and place it in the configured path.

> ⚠️ **ADReSSo21 requires a data access agreement.** Please visit the official ADReSSo challenge page to request access.

## Dataset Structure Expected

This project expects the following two directory structures.

### Official ADReSSo21 dataset
```text
ADReSSo21/diagnosis/train/
├── audio/
│   ├── ad/        # Alzheimer's patients audio
│   └── cn/        # Healthy controls audio
```

### Locally augmented dataset
```text
ADReSSo21/diagnosis/train_aug/
├── audio/
│   ├── ad/        # Augmented Alzheimer's patients audio
│   └── cn/        # Augmented healthy controls audio
```


## 🚀 Quick Start

### 1. Preprocessing

If the Whisper transcripts and `.pt` feature files are already provided, this step can be skipped. To regenerate them, first run ASR transcription and then extract DistilBERT / wav2vec2 embeddings:

```bash
python preprocess/preprocesswhisper.py
python preprocess/preprocessembeddings.py
```

For the augmented training set, use the corresponding augmented preprocessing scripts:

```bash
python preprocess/preprocesswhisper_aug.py
python preprocess/preprocessembeddings_aug.py
```

### 2. Main Experiments

The paper reports Centralized Learning (CL), Local Learning (LL), and Federated Learning (FL) under audio-only, text-only, and fusion settings. The main fusion experiments can be launched with:

```bash
python main.py --config configs/experiments/cl_fusion_aug.yaml
python main.py --config configs/experiments/ll_fusion_aug.yaml
python main.py --config configs/experiments/fl_fusion_aug.yaml
```

All experiment configs are under `configs/experiments/` and follow the naming pattern:

```text
<paradigm>_<modality>[_aug].yaml
```

Examples include `cl_audio.yaml`, `ll_text_aug.yaml`, and `fl_fusion_aug.yaml`.

### 3. Ablation Examples

Run audio-only, text-only, or no-augmentation variants by switching the config path:

```bash
python main.py --config configs/experiments/fl_audio_aug.yaml
python main.py --config configs/experiments/fl_text_aug.yaml
python main.py --config configs/experiments/fl_fusion.yaml
```

---

## ⚙️ Configuration

Key configuration parameters in YAML config files:

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `seed` | Random seed for reproducibility | `42` |
| `epochs` | Local training epochs per round | `5` |
| `lr` | Learning rate | `1e-3` |
| `batch_size` | Batch size | `16` |
| `weight_decay` | L2 regularization | `1e-4` |
| `num_classes` | Classification categories | `2` (AD / HC) |
| `checkpoint_dir` | Path to save model checkpoints | `./checkpoints/` |
| `strategy` | FL aggregation strategy | `"FedAvg"` |
| `num_rounds` | Total FL rounds | `100` |
| `num_clients` | Number of federated clients | `3` |
| `fl_fraction` | Fraction of clients per round | `1.0` |

---

## 🧠 Model Variants

The code also keeps several ablation-ready encoders, but the **paper setting** is `CrossAttentionTransformerEncoder` with `GatedCrossAttentionFusion`:

| Model Class | Role | Notes |
|-------------|------|-------|
| `CrossAttentionTransformerEncoder` | ✅ **Paper** | Main sequence fusion encoder used for AD detection |
| `GatedCrossAttentionFusion` | ✅ **Paper** | Pre-norm audio→text cross-attention with gated residual update |
| `AttnPooling` / `GatedAttnPooling` | ✅ **Paper** | Sequence pooling heads; FL uses `attn`, CL/LL use `mean` |
| `MyTransformerEncoder` | ❌ Experimental | Single-modal baseline for audio or text only |

---

## 📈 Federated Learning Workflow

Federated training follows a standard server-client loop:

1. The server initializes the global model and broadcasts its parameters to the selected clients.
2. Each client trains locally on its own private data.
3. Clients return updated parameters together with local metrics such as loss, accuracy, and F1.
4. The server aggregates the updates using the configured strategy (`FedAvg`, `FedAdam`, `FedAdagrad`, `FedYogi`, or `FedProx`).
5. The updated global model is sent back to the clients and the next round begins.

> In this repository's `main.py` runner, the Flower server is started first, then the client processes are launched after a short delay for each cross-validation fold.


---

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{Wei2026,
  author    = {Wei X and Wen B and Lin Y and Li K and Gu M and Wang X and Wang L and Dang J},
  title     = {{Breaking Data Efficiency Dilemma: A Federated and Augmented Learning Framework for Alzheimer's Disease Detection Via Speech}},
  booktitle = {ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {19147--19151},
  year      = {2026},
  doi       = {10.1109/ICASSP55912.2026.11463930}
}
```

---

## 📄 License

This project is for research purposes. Please refer to the LICENSE file and ensure compliance with ADReSSo21's data usage agreement.
