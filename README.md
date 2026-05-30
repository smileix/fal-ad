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

The framework jointly encodes speech and text through a dual-encoder architecture:

| Modality | Backbone | Details |
|----------|----------|---------|
| **Text** | DistilBERT | Pre-trained `distilbert-base-uncased` Transformer encoder |
| **Audio** | wav2vec 2.0 | Pre-trained `facebook/wav2vec2-base` — raw waveform → contextualized 768-dim embeddings |

The Mel spectrogram (with pause features when `pauses: True`) is fed directly into wav2vec 2.0 as the frontend feature extractor; no hand-crafted feature engineering is applied.

### Cross-Attention Fusion Module

The core fusion module is **`GatedCrossAttentionFusion`**, integrated inside `CrossAttentionTransformerEncoder`:

1. **Pre-Norm Cross-Attention** — Audio queries the text key-value bank (unidirectional, audio → text), enabling the audio representation to selectively attend to linguistically relevant regions
2. **Gated Residual Connection** — A learnable gate (`σ(W·[x; attn(x)]))`) modulates how much cross-modal information flows into the residual, preventing attention noise from dominating
3. **Feed-Forward MLP** — Standard post-norm FFN with ReLU activation

```
Audio (query) ──► Cross-Attention ◄── Text (key/value)
                          │
                    gated residual ← GatedCrossAttentionFusion
                          │
                    Feed-Forward ML
```

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

```bash
torch>=2.0
transformers
flwr>=1.0
wandb
scikit-learn
pandas
numpy
librosa
soundfile
pyroomacoustics
resampy
```

Install all dependencies:

```bash
pip install torch transformers flwr wandb scikit-learn pandas numpy librosa soundfile pyroomacoustics resamply
```

---

## 📊 Dataset: ADReSSo21

The project uses the **ADReSSo21** (Alzheimer's Dementia Recognition through Spontaneous Speech) dataset. You must obtain the dataset separately from the official source and place it in the configured path.

> ⚠️ **ADReSSo21 requires a data access agreement.** Please visit the official ADReSSo challenge page to request access.

### Dataset Structure Expected

```
adresso21/
├── train/
│   ├── AD/          # Alzheimer's patients — training set
│   └── HC/          # Healthy controls — training set
├── test/
│   ├── AD/
│   └── HC/
└── adresso21_train.csv   # Metadata with audio paths and labels
```

### CSV Format

The `adresso21_train.csv` should contain at minimum:

| Column | Description |
|--------|-------------|
| `file_path` | Relative path to audio file |
| `label` | `AD` or `HC` |
| `client_id` | Integer client ID for FL partitioning |

---

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
# Terminal 1 — Start server
python main.py --mode fl --config configs/fl_server.yaml

# Terminal 2+ — Start clients (one per terminal)
python main.py --mode fl --config configs/fl_client.yaml --client_id 0
python main.py --mode fl --config configs/fl_client.yaml --client_id 1
python main.py --mode fl --config configs/fl_client.yaml --client_id 2
```

> Use `--save_model True` to save the global model after FL training.

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

The code contains several experimental fusion modules, but the **paper uses `CrossAttentionTransformerEncoder`** with `GatedCrossAttentionFusion` throughout:

| Model Class | Status | Description |
|-------------|--------|-------------|
| `CrossAttentionTransformerEncoder` | ✅ **Paper** | Unidirectional cross-attention + gated residual (audio → text) |
| `GatedCrossAttentionFusion` | ✅ **Paper** | Gated cross-attention layer used inside above |
| `AttnPooling` / `GatedAttnPooling` | ✅ **Paper** | Optional pooling strategies (FL uses `attn`; CL/LL use `mean`) |
| `BidirectionalCrossAttentionTransformerEncoder` | ❌ Experimental | Bidirectional dual-layer cross-attention (in code but not used) |
| `ElementWiseFusionEncoder` | ❌ Experimental | Element-wise fusion variants (in code but not used) |
| `MyTransformerEncoder` | ❌ Experimental | Single-modal transformer (for ablation, not used in final experiments) |

---

## 📈 Federated Learning Workflow

```
┌─────────┐    parameters     ┌─────────┐    parameters     ┌─────────┐
│ Client 0│◄─────────────────►│ Server  │◄─────────────────►│ Client 1│
└─────────┘    flwr (gRPC)    │ (Flower)│    flwr (gRPC)    └─────────┘
     │              ▲              │              ▲
     │              │              │              │
     ▼              │              ▼              │
 local train        │         aggregate           │
     │              │              │              │
     ▼              │              ▼              ▼
 parameters ───────┘         global model    local train
                                               │
                                      parameters ───────┘
```

Each FL round:
1. Server sends global model parameters to selected clients
2. Each client performs local training on their private data
3. Clients send updated parameters back to server
4. Server aggregates parameters (FedAvg by default)
5. Repeat for `num_rounds` iterations

---

## 📊 Results

After training, the framework logs:

- **Loss curves** via W&B (wandb)
- **Accuracy, Precision, Recall, F1, AUC** per round
- **Confusion matrix** on test set
- **Per-client metrics** in federated mode

Example output during FL training:

```
Round 10 | Accuracy: 0.823 | F1: 0.815 | AUC: 0.891
Round 20 | Accuracy: 0.847 | F1: 0.839 | AUC: 0.912
Round 30 | Accuracy: 0.861 | F1: 0.854 | AUC: 0.928
```

---

## 🔧 Utilities

### Model Statistics

```bash
python stats.py --model_path ./checkpoints/best_model.pt
```

Outputs total parameters, trainable parameters, and model size in MB.

### Cross-Validation

```python
from dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir="adresso21",
    csv_path="adresso21_train.csv",
    batch_size=16,
    cv_folds=5,
    fold=0  # Use fold 0 as validation
)
```

### Federated Local Dataloaders

```python
from dataset import get_local_dataloaders_cv

local_loaders = get_local_dataloaders_cv(
    data_dir="adresso21",
    csv_path="adresso21_train.csv",
    batch_size=16,
    n_clients=3,
    cv_folds=5,
    fold=0
)
# local_loaders[client_id] → DataLoader for that client's data
```

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