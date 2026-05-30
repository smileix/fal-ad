# FAL-AD：基于语音的阿尔茨海默病联邦增强学习检测框架

## 📄 论文信息

> **Breaking Data Efficiency Dilemma: A Federated and Augmented Learning Framework for Alzheimer's Disease Detection Via Speech**  
> ICASSP 2026 · DOI: [10.1109/ICASSP55912.2026.11463930](https://doi.org/10.1109/ICASSP55912.2026.11463930)

---

## 🔍 项目概述

本仓库实现了 **FAL-AD**，一个基于差分隐私联邦学习和语音转换数据增强的阿尔茨海默病（AD）语音检测框架。该框架针对临床 AI 中的两个核心挑战：

1. **数据效率** — 利用语音转换（Voice Conversion）数据增强技术，缓解有标注 AD 语音数据稀缺的困境
2. **数据隐私** — 采用联邦学习（Federated Learning）在多机构间协同训练模型，无需共享原始患者数据

框架支持**三种学习范式**：

| 模式 | 说明 |
|------|------|
| `cl` | 集中学习（Centralized Learning）— 所有数据汇聚于单一节点 |
| `ll` | 本地学习（Local Learning）— 各客户端独立训练，不进行协作 |
| `fl` | 联邦学习（Federated Learning）— 通过 Flower 框架进行协作训练 |

---

## 🏗️ 模型架构

### 多模态编码器

框架通过双编码器架构联合编码语音和文本：

| 模态 | Backbone | 说明 |
|------|----------|------|
| **文本** | DistilBERT | 预训练 `distilbert-base-uncased` Transformer 编码器 |
| **音频** | wav2vec 2.0 | 预训练 `facebook/wav2vec2-base` —— 原始波形 → 768 维上下文嵌入 |

Mel 频谱图（开启 `pauses: True` 时附加停顿时长特征）直接送入 wav2vec 2.0 作为前端特征提取器，无需手工设计声学特征。

### 交叉注意力融合模块

核心融合模块是 **`GatedCrossAttentionFusion`**，集成在 `CrossAttentionTransformerEncoder` 中：

1. **Pre-Norm 交叉注意力** — 音频作为 query，查询文本的 key/value 池（单向，audio → text），使音频表征有选择地关注语言相关区域
2. **门控残差连接** — 可学习门控（`σ(W·[x; attn(x)]))`) 调制跨模态信息流量，防止注意力噪声主导
3. **前馈 MLP** — 标准 FFN + ReLU 激活

```
Audio (query) ──► Cross-Attention ◄── Text (key/value)
                          │
                    gated residual ← GatedCrossAttentionFusion
                          │
                    Feed-Forward MLP
```

### 联邦学习
- 基于 **Flower**（`flwr`）构建联邦编排
- 支持多种聚合策略：FedAvg、FedAdam、FedAdagrad、FedYogi、FedProx
- 客户端-服务器架构，可配置轮数和本地训练轮次

---

## 📁 项目结构

```
fal-ad/
├── main.py         # 主入口 — 支持 cl / ll / fl 三种模式
├── model.py        # 模型定义（GatedCrossAttentionFusion、CrossAttentionTransformerEncoder、池化层）
├── server.py       # Flower 服务器 — 策略选择与指标记录
├── client.py       # Flower 客户端 — 本地训练与评估
├── dataset.py      # ADReSSo21 数据集加载，支持 CV 和联邦划分
├── utils.py        # 训练工具函数：train()、evaluation()、配置管理等
├── stats.py        # 模型统计轻量脚本
└── configs/        # YAML 配置文件（每种模式对应一个）
```

---

## ⚙️ 环境依赖

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

安装所有依赖：

```bash
pip install torch transformers flwr wandb scikit-learn pandas numpy librosa soundfile pyroomacoustics resamply
```

---

## 📊 数据集：ADReSSo21

本项目使用 **ADReSSo21**（Alzheimer's Dementia Recognition through Spontaneous Speech）数据集。您需自行从官方渠道获取数据集并放置于配置路径下。

> ⚠️ **ADReSSo21 需要签署数据使用协议。** 请访问 ADReSSo 官方挑战赛页面申请访问权限。

### 期望的数据集目录结构

```
adresso21/
├── train/
│   ├── AD/          # 阿尔茨海默病患者 — 训练集
│   └── HC/          # 健康对照 — 训练集
├── test/
│   ├── AD/
│   └── HC/
└── adresso21_train.csv   # 元数据，包含音频路径与标签
```

### CSV 格式说明

`adresso21_train.csv` 应至少包含以下列：

| 列名 | 说明 |
|------|------|
| `file_path` | 音频文件的相对路径 |
| `label` | `AD` 或 `HC` |
| `client_id` | 用于 FL 划分的客户端 ID（整数） |

---

## 🚀 快速上手

### 1. 集中学习（`cl`）

在单一节点上使用所有数据训练：

```bash
python main.py --mode cl --config configs/cl_config.yaml
```

### 2. 本地学习（`ll`）

各客户端仅使用本地数据独立训练：

```bash
python main.py --mode ll --config configs/ll_config.yaml
```

### 3. 联邦学习（`fl`）

启动 Flower 服务器和一个或多个客户端：

```bash
# 终端 1 — 启动服务器
python main.py --mode fl --config configs/fl_server.yaml

# 终端 2+ — 启动客户端（每终端一个）
python main.py --mode fl --config configs/fl_client.yaml --client_id 0
python main.py --mode fl --config configs/fl_client.yaml --client_id 1
python main.py --mode fl --config configs/fl_client.yaml --client_id 2
```

> 使用 `--save_model True` 可以在 FL 训练结束后保存全局模型。

---

## ⚙️ 配置参数

YAML 配置文件中关键参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `seed` | 随机种子，保证可复现性 | `42` |
| `epochs` | 每轮本地训练轮次 | `5` |
| `lr` | 学习率 | `1e-3` |
| `batch_size` | 批大小 | `16` |
| `weight_decay` | L2 正则化系数 | `1e-4` |
| `num_classes` | 分类类别数 | `2`（AD / HC） |
| `checkpoint_dir` | 模型检查点保存路径 | `./checkpoints/` |
| `strategy` | FL 聚合策略 | `"FedAvg"` |
| `num_rounds` | FL 总轮数 | `100` |
| `num_clients` | 联邦客户端数量 | `3` |
| `fl_fraction` | 每轮参与客户端比例 | `1.0` |

---

## 🧠 模型变体

代码中包含多个实验性融合模块，但**论文实际采用的是 `CrossAttentionTransformerEncoder`（含 `GatedCrossAttentionFusion`）**：

| 模型类 | 状态 | 说明 |
|--------|------|------|
| `CrossAttentionTransformerEncoder` | ✅ **论文使用** | 单向交叉注意力 + 门控残差（audio → text） |
| `GatedCrossAttentionFusion` | ✅ **论文使用** | 论文中使用的门控交叉注意力层 |
| `AttnPooling` / `GatedAttnPooling` | ✅ **论文使用** | 可选池化策略（FL 用 `attn`，CL/LL 用 `mean`） |
| `BidirectionalCrossAttentionTransformerEncoder` | ❌ 实验性代码 | 双向双层交叉注意力（代码中有但未使用） |
| `ElementWiseFusionEncoder` | ❌ 实验性代码 | 逐元素融合变体（代码中有但未使用） |
| `MyTransformerEncoder` | ❌ 实验性代码 | 单模态 Transformer（用于消融，论文最终方案未采用） |

---

## 📈 联邦学习工作流程

```
┌─────────┐   parameters    ┌─────────┐   parameters    ┌─────────┐
│ Client 0│◄───────────────►│ Server  │◄───────────────►│ Client 1│
└─────────┘   flwr (gRPC)   │ (Flower)│   flwr (gRPC)   └─────────┘
     │            ▲               │            ▲
     │            │               │            │
     ▼            │               ▼            ▼
  local train     │          aggregate      local train
     │            │               │              │
     ▼            │               ▼              ▼
 parameters ──────┘         global model   parameters ──────┘
```

每轮 FL 的执行步骤：
1. 服务器将全局模型参数发送给被选中的客户端
2. 各客户端使用私有数据执行本地训练
3. 客户端将更新后的参数发回服务器
4. 服务器执行参数聚合（默认使用 FedAvg）
5. 重复 `num_rounds` 轮迭代

---

## 📊 训练结果

训练完成后，框架会记录以下指标：

- 通过 **W&B**（wandb）可视化损失曲线
- 每轮记录 **准确率、精确率、召回率、F1、AUC**
- 测试集上的 **混淆矩阵**
- 联邦模式下各 **客户端独立指标**

训练期间输出示例：

```
Round 10 | Accuracy: 0.823 | F1: 0.815 | AUC: 0.891
Round 20 | Accuracy: 0.847 | F1: 0.839 | AUC: 0.912
Round 30 | Accuracy: 0.861 | F1: 0.854 | AUC: 0.928
```

---

## 🔧 工具函数

### 模型统计

```bash
python stats.py --model_path ./checkpoints/best_model.pt
```

输出模型总参数量、可训练参数量及模型大小（MB）。

### 交叉验证

```python
from dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir="adresso21",
    csv_path="adresso21_train.csv",
    batch_size=16,
    cv_folds=5,
    fold=0  # 使用第 0 折作为验证集
)
```

### 联邦场景的本地 DataLoader

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
# local_loaders[client_id] → 该客户端数据的 DataLoader
```

---

## 📝 引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@inproceedings{Wei2026FALAD,
  title     = {{Breaking Data Efficiency Dilemma: A Federated and Augmented Learning Framework for Alzheimer's Disease Detection Via Speech}},
  author    = {Xiao Wei and Bin Wen and Yuqin Lin and Kai Li and Mingyang Gu and Xiaobao Wang and Longbiao Wang and Jianwu Dang},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  doi       = {10.1109/ICASSP55912.2026.11463930}
}
```

---

## 📄 许可说明

本项目仅供研究使用。请参考 LICENSE 文件，并确保遵守 ADReSSo21 的数据使用协议。