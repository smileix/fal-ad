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

环境采用 **Conda + pip 混合**方式安装。PyTorch/CUDA 类包必须通过 conda，以确保 CUDA 编译正确；其余包用 pip 安装（librosa/soundfile 等音频库 pip 版本更稳定）。

### Conda 环境（推荐）

```bash
conda env create -f environment.yml
conda activate fl
```

> `environment.yml` 中已固定全部版本号，可完全复现本项目的运行环境。

### pip 补充包（仅 3 个）

```bash
pip install transformers==4.48.0 librosa==0.10.0 soundfile==0.13.1
```

> ⚠️ **不要跳过 conda 环境单独用 pip 安装 torch**，pip 版本的 torch 不含 CUDA 驱动，会导致 `torch.cuda.is_available()` 返回 `False`。

### 完整依赖版本对照表

**通过 conda 安装：**

| 包 | 版本 | 说明 |
|----|------|------|
| python | 3.9 | |
| pytorch-cuda | 12.1 | 含 CUDA 12.1 编译版 PyTorch |
| torchaudio | 2.1.1 | |
| torchvision | 0.16.1 | |
| numpy | 1.23.5 | |
| pandas | 2.2.3 | |
| flwr | 1.7.0 | Flower 联邦学习框架 |
| wandb | 0.21.0 | Weights & Biases 日志 |
| scipy | 1.13.1 | |
| scikit-learn | 1.6.1 | |
| pyyaml | 6.0.3 | |

**通过 pip 安装（conda 环境激活后执行）：**

| 包 | 版本 | 说明 |
|----|------|------|
| transformers | 4.48.0 | DistilBERT / wav2vec2 模型 |
| librosa | 0.10.0 | 音频处理 |
| soundfile | 0.13.1 | 音频文件读写 |

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

### 1. Centralized Learning (`cl`)

Train with all data pooled on a single node:

```bash
python main.py --mode cl --config configs/cl_config.yaml
```

### 2. Local Learning (`ll`)

Each client trains independently on its local data only:

```bash
python main.py --mode ll --config configs/ll_config.yaml
```

### 3. Federated Learning (`fl`)

Start the Flower server and client(s):

```bash
python main.py --mode fl --config configs/fl_server.yaml
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