# FAL-AD：基于语音的阿尔茨海默病联邦增强学习检测框架

## 📊 实验结果

下表汇总了 ADReSSo 数据集上的性能表现，包含已有的集中式方法，以及我们在 Centralized Learning (CL)、Local Learning (LL) 和 Federated Learning (FL) 三种范式下、带/不带数据增强（Aug）的实现结果。这里的 CL 是使用其源代码严格复现的 CogniAlign 版本。

| 模态 | 指标 | C-Attn | Ying | Bang | CogniAlign | CL | CL+Aug | LL | LL+Aug | FL | FL+Aug |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 音频 | Acc | 75.30 | 71.20 | 69.01 | <u>80.12</u> | 74.55 | <u>79.39</u> | <u>68.69</u> | 68.08 | 83.84 | <b><u>85.05</u></b> |
| 音频 | F1 | 76.00 | 73.10 | 70.39 | <u>79.46</u> | 73.39 | <u>79.14</u> | 65.68 | <u>67.10</u> | 83.67 | <b><u>84.64</u></b> |
| 文本 | Acc | 73.50 | 78.90 | 83.10 | <u>86.77</u> | 84.85 | <u>86.67</u> | 78.39 | <u>79.80</u> | 87.68 | <b><u>90.30</u></b> |
| 文本 | F1 | 73.50 | 79.00 | 83.10 | <u>86.59</u> | 84.69 | <u>86.63</u> | 77.48 | <u>79.55</u> | 87.64 | <b><u>90.28</u></b> |
| 融合 | Acc | 77.20 | 83.70 | 87.32 | <u>90.36</u> | 86.06 | <u>86.67</u> | 78.59 | <u>80.61</u> | 89.70 | <b><u>91.52</u></b> |
| 融合 | F1 | 77.60 | 83.30 | 87.25 | <u>90.11</u> | 85.89 | <u>86.64</u> | 77.16 | <u>80.35</u> | 89.65 | <b><u>91.45</u></b> |

下划线表示该组中的最优值，粗体表示全局最优值。

### 多模态编码器

论文采用双编码器结构，在融合阶段之前一直保留语音和文本的序列表示：

| 模态 | 骨干模型 | 输出 |
|------|----------|------|
| **文本** | DistilBERT | 来自 `distilbert-base-uncased` 的上下文 token 表示 |
| **音频** | wav2vec 2.0 | 来自 `facebook/wav2vec2-base` 的帧级上下文表示 |

实现中也保留了 Mel / EGEMAPS 前端分支，用于部分消融实验；但论文主配置强调的是音频与文本序列之间的交叉注意力融合，而不是提前拼接。

### 交叉注意力融合模块

核心融合模块是 **`GatedCrossAttentionFusion`**，它被堆叠在 `CrossAttentionTransformerEncoder` 中形成多层融合堆栈：

1. **Pre-Norm 交叉注意力** — 先对 source 和 memory 做 LayerNorm，提高深层堆叠的训练稳定性。
2. **单向融合** — 以音频特征作为 query、文本特征作为 key/value（audio → text），让语音表示有选择地关注与语言相关的 token。
3. **门控残差连接** — 将注意力输出与原始 source 表示拼接后送入 sigmoid gate，控制跨模态信息写入残差分支的强度。
4. **前馈细化** — 使用带 ReLU 和 dropout 的位置前馈 MLP 进一步变换融合后的序列表示。


### 联邦学习
- 基于 **Flower**（`flwr`）构建联邦编排
- 支持多种聚合策略：FedAvg、FedAdam、FedAdagrad、FedYogi、FedProx
- 客户端-服务器架构，可配置轮数和本地训练轮次

## 🧠 模型变体

代码中保留了若干可用于消融的编码器，但**论文主方案**是 `CrossAttentionTransformerEncoder` + `GatedCrossAttentionFusion`：

| 模型类 | 作用 | 说明 |
|-------------|------|-------|
| `CrossAttentionTransformerEncoder` | ✅ **论文** | AD 检测的主序列融合编码器 |
| `GatedCrossAttentionFusion` | ✅ **论文** | 带门控残差更新的 pre-norm 音频→文本交叉注意力 |
| `AttnPooling` / `GatedAttnPooling` | ✅ **论文** | 序列池化头；FL 使用 `attn`，CL/LL 使用 `mean` |
| `BidirectionalCrossAttentionTransformerEncoder` | ❌ 实验性 | 双向两次交叉注意力，用于消融 |
| `ElementWiseFusionEncoder` | ❌ 实验性 | 通过 concat / sum / mean / product 等方式做替代融合 |
| `MyTransformerEncoder` | ❌ 实验性 | 单模态基线，可用于仅音频或仅文本 |

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

## 📊 数据集：ADReSSo21

本项目使用 **ADReSSo21**（Alzheimer's Dementia Recognition through Spontaneous Speech）数据集。您需自行从官方渠道获取数据集并放置于配置路径下。

> ⚠️ **ADReSSo21 需要签署数据使用协议。** 请访问 ADReSSo 官方挑战赛页面申请访问权限。

### 期望的数据集目录结构

本项目包含以下两种目录结构，其中 `train_aug` 为本地扩增数据：

### 官方 ADReSSo21 数据集

```text
ADReSSo21/diagnosis/train/
├── audio/
│   ├── ad/        # 阿尔茨海默症患者音频
│   └── cn/        # 健康对照组音频
```

### 本地扩增数据集

```text
ADReSSo21/diagnosis/train_aug/
├── audio/
│   ├── ad/        # 扩增后的阿尔茨海默症患者音频
│   └── cn/        # 扩增后的健康对照组音频
```

## 📝 引用

如果您在研究中使用了本工作，请引用：

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

## 📄 许可说明

本项目仅供研究使用。请参考 LICENSE 文件，并确保遵守 ADReSSo21 的数据使用协议。