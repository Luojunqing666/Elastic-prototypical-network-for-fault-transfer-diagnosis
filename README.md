# EProtoNet

**Meta-learning with Elastic Prototypical Network for Fault Transfer Diagnosis of Bearings under Unstable Speeds**

[![Paper](https://img.shields.io/badge/Paper-Reliability%20Engineering%20%26%20System%20Safety-blue)](https://doi.org/10.1016/j.ress.2024.110001)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Jingjie Luo, Haidong Shao*, Jian Lin, Bin Liu
>
> *Reliability Engineering & System Safety*, Volume 245, 2024, 110001

---

## Overview

EProtoNet introduces an **elastic distance metric** into the prototypical network framework for few-shot fault diagnosis under cross-domain conditions (e.g., varying rotational speeds). The key innovation is a learnable scale factor that modulates the Euclidean distance between query samples and class prototypes, enabling more robust classification when source and target domains exhibit significant distribution shift.

<div align="center">
<img src="Framework diagram of EProtoNet.png" width="700" />
<p><em>Framework diagram of EProtoNet</em></p>
</div>

<div align="center">
<img src="The overall procedure of this method.png" width="700" />
<p><em>The overall procedure of the proposed method</em></p>
</div>

## Key Features

- **Elastic Distance Metric**: Scaled Euclidean distance with a tunable elasticity factor for adaptive prototype matching
- **SE-Attention Backbone**: CNN4 backbone with Squeeze-and-Excitation (SE) channel attention on the final layer
- **Meta-Learning Framework**: Episode-based training with learn2learn for few-shot generalization
- **Multiple Baselines**: Includes MAML and Relation Network implementations for comparison
- **Cross-Domain Transfer**: Designed for fault diagnosis under speed variation (unstable operating conditions)

## Project Structure

```
EProtoNet/
├── main.py                  # Unified CLI entry point
├── train_protonet.py        # EProtoNet training & testing (core method)
├── train_maml.py            # MAML baseline
├── train_relation.py        # Relation Network baseline
├── requirements.txt         # Dependencies
├── models/
│   ├── __init__.py
│   ├── backbone.py          # CNN4 backbone with SE attention
│   ├── protonet.py          # Elastic Prototypical Network
│   ├── maml_net.py          # MAML-compatible CNN
│   └── relation_net.py      # Relation Network (encoder + relation module)
├── datasets/
│   └── __init__.py          # CWRU dataset loader & meta-learning dataset
├── utils/
│   ├── __init__.py          # Helpers (seed, accuracy, normalization, weight init)
│   └── mmd.py              # MMD loss for domain adaptation
├── results/                 # Saved checkpoints (gitignored)
└── figures/                 # Paper figures
```

## Installation

```bash
git clone https://github.com/Luojunqing666/Elastic-prototypical-network-for-fault-transfer-diagnosis.git
cd Elastic-prototypical-network-for-fault-transfer-diagnosis/EProtoNet
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- learn2learn >= 0.1.7
- NumPy >= 1.22
- SciPy >= 1.7
- scikit-learn >= 1.0

## Dataset Preparation

This code uses the **CWRU Bearing Dataset** (Case Western Reserve University). Organize your data as follows:

```
data/
├── source/          # Source domain CSV files (e.g., speed condition A)
│   ├── normal.csv
│   ├── inner_fault.csv
│   ├── outer_fault.csv
│   ├── roller_fault.csv
│   └── compound_fault.csv
└── target/          # Target domain CSV files (e.g., speed condition B)
    ├── normal.csv
    ├── inner_fault.csv
    ├── outer_fault.csv
    ├── roller_fault.csv
    └── compound_fault.csv
```

Each CSV file contains single-column vibration signal data. The code automatically segments signals into fixed-length samples (default: 1024 points).

## Usage

### Train EProtoNet (Proposed Method)

```bash
# 5-way 5-shot with elastic distance (recommended)
python main.py --method protonet --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100 --episodes 30

# Without elastic distance (standard ProtoNet)
python main.py --method protonet --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100 --no_elastic
```

### Train Baselines

```bash
# MAML
python main.py --method maml --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100

# Relation Network
python main.py --method relation --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100
```

### Testing

```bash
# Test EProtoNet
python main.py --method protonet --test --data_dir ./data \
    --load_path ./results/eprotonet_best.pt --ways 5 --shots 5

# Test MAML
python main.py --method maml --test --data_dir ./data \
    --load_path ./results/maml_best.pt --ways 5 --shots 5

# Test Relation Network
python main.py --method relation --test --data_dir ./data \
    --load_path ./results/relation_best.pt --ways 5 --shots 5
```

### Specifying Custom Data Files

```bash
python main.py --method protonet --train \
    --data_dir ./data \
    --source_files ./data/N900.csv ./data/I900.csv ./data/Out900.csv ./data/R900.csv ./data/C900.csv \
    --target_files ./data/Nf.csv ./data/If.csv ./data/Outf.csv ./data/Rf.csv ./data/Cf.csv \
    --ways 5 --shots 5
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `protonet` | Method: `protonet`, `maml`, `relation` |
| `--ways` | `5` | Number of classes per task (N-way) |
| `--shots` | `5` | Support samples per class (K-shot) |
| `--epochs` | `100` | Training epochs |
| `--episodes` | `30` | Episodes per epoch |
| `--lr` | `0.001` | Learning rate |
| `--no_elastic` | `False` | Disable elastic distance (use standard Euclidean) |
| `--scale_factor` | `0.01` | Elasticity factor for distance scaling |
| `--data_dir` | - | Root directory of dataset |
| `--seed` | `2023` | Random seed for reproducibility |

## Citation

If you find this work useful, please cite:

```bibtex
@article{luo2024eprotonet,
  title     = {Meta-learning with elastic prototypical network for fault transfer diagnosis of bearings under unstable speeds},
  author    = {Luo, Jingjie and Shao, Haidong and Lin, Jian and Liu, Bin},
  journal   = {Reliability Engineering \& System Safety},
  volume    = {245},
  pages     = {110001},
  year      = {2024},
  doi       = {10.1016/j.ress.2024.110001},
  publisher = {Elsevier}
}
```

## Contact

- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn

---

# EProtoNet

**基于弹性原型网络的元学习方法用于不稳定转速下轴承故障迁移诊断**

本科期间的成果了，现在回看，还有很多能进一步改进和优化的，请多见谅！

[![论文](https://img.shields.io/badge/论文-Reliability%20Engineering%20%26%20System%20Safety-blue)](https://doi.org/10.1016/j.ress.2024.110001)
[![许可证](https://img.shields.io/badge/许可证-MIT-green.svg)](LICENSE)

> 罗靖捷, 邵海东*, 林健, 刘斌
>
> *Reliability Engineering & System Safety*, 第245卷, 2024, 110001

---

## 概述

EProtoNet 在原型网络框架中引入了**弹性距离度量**，用于跨域条件下（如不同转速）的小样本故障诊断。核心创新是一个可调节的缩放因子，用于调制查询样本与类原型之间的欧氏距离，使得在源域和目标域存在显著分布偏移时仍能实现鲁棒分类。

<div align="center">
<img src="Framework diagram of EProtoNet.png" width="700" />
<p><em>EProtoNet 框架图</em></p>
</div>

<div align="center">
<img src="The overall procedure of this method.png" width="700" />
<p><em>所提方法的整体流程</em></p>
</div>

## 主要特点

- **弹性距离度量**：带可调弹性因子的缩放欧氏距离，实现自适应原型匹配
- **SE注意力骨干网络**：CNN4骨干网络在最后一层集成通道注意力（Squeeze-and-Excitation）
- **元学习框架**：基于 episode 的训练方式，使用 learn2learn 实现小样本泛化
- **多种基线方法**：包含 MAML 和关系网络实现，便于对比实验
- **跨域迁移**：专为转速变化（不稳定工况）下的故障诊断设计

## 项目结构

```
EProtoNet/
├── main.py                  # 统一命令行入口
├── train_protonet.py        # EProtoNet 训练与测试（核心方法）
├── train_maml.py            # MAML 基线
├── train_relation.py        # 关系网络基线
├── requirements.txt         # 依赖包
├── models/
│   ├── __init__.py
│   ├── backbone.py          # CNN4 骨干网络 + SE 注意力
│   ├── protonet.py          # 弹性原型网络
│   ├── maml_net.py          # MAML 兼容 CNN
│   └── relation_net.py      # 关系网络（编码器 + 关系模块）
├── datasets/
│   └── __init__.py          # CWRU 数据集加载 & 元学习数据集
├── utils/
│   ├── __init__.py          # 工具函数（随机种子、准确率、归一化、权重初始化）
│   └── mmd.py              # MMD 损失（域自适应）
├── results/                 # 模型检查点（已忽略）
└── figures/                 # 论文图片
```

## 安装

```bash
git clone https://github.com/Luojunqing666/Elastic-prototypical-network-for-fault-transfer-diagnosis.git
cd Elastic-prototypical-network-for-fault-transfer-diagnosis/EProtoNet
pip install -r requirements.txt
```

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- learn2learn >= 0.1.7
- NumPy >= 1.22
- SciPy >= 1.7
- scikit-learn >= 1.0

## 数据准备

本代码默认使用 **CWRU 轴承数据集**（凯斯西储大学）。请按如下方式组织数据：

```
data/
├── source/          # 源域 CSV 文件（如工况 A 的转速条件）
│   ├── normal.csv
│   ├── inner_fault.csv
│   ├── outer_fault.csv
│   ├── roller_fault.csv
│   └── compound_fault.csv
└── target/          # 目标域 CSV 文件（如工况 B 的转速条件）
    ├── normal.csv
    ├── inner_fault.csv
    ├── outer_fault.csv
    ├── roller_fault.csv
    └── compound_fault.csv
```

每个 CSV 文件包含单列振动信号数据。代码会自动将信号分割为固定长度的样本（默认：1024 个数据点）。

## 使用方法

### 训练 EProtoNet（所提方法）

```bash
# 5-way 5-shot，使用弹性距离（推荐）
python main.py --method protonet --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100 --episodes 30

# 不使用弹性距离（标准 ProtoNet）
python main.py --method protonet --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100 --no_elastic
```

### 训练基线方法

```bash
# MAML
python main.py --method maml --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100

# 关系网络
python main.py --method relation --train --data_dir ./data \
    --ways 5 --shots 5 --epochs 100
```

### 测试

```bash
# 测试 EProtoNet
python main.py --method protonet --test --data_dir ./data \
    --load_path ./results/eprotonet_best.pt --ways 5 --shots 5

# 测试 MAML
python main.py --method maml --test --data_dir ./data \
    --load_path ./results/maml_best.pt --ways 5 --shots 5

# 测试关系网络
python main.py --method relation --test --data_dir ./data \
    --load_path ./results/relation_best.pt --ways 5 --shots 5
```

### 指定自定义数据文件

```bash
python main.py --method protonet --train \
    --data_dir ./data \
    --source_files ./data/N900.csv ./data/I900.csv ./data/Out900.csv ./data/R900.csv ./data/C900.csv \
    --target_files ./data/Nf.csv ./data/If.csv ./data/Outf.csv ./data/Rf.csv ./data/Cf.csv \
    --ways 5 --shots 5
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `protonet` | 方法：`protonet`、`maml`、`relation` |
| `--ways` | `5` | 每个任务的类别数（N-way） |
| `--shots` | `5` | 每类支持样本数（K-shot） |
| `--epochs` | `100` | 训练轮数 |
| `--episodes` | `30` | 每轮 episode 数 |
| `--lr` | `0.001` | 学习率 |
| `--no_elastic` | `False` | 禁用弹性距离（使用标准欧氏距离） |
| `--scale_factor` | `0.01` | 距离缩放的弹性因子 |
| `--data_dir` | - | 数据集根目录 |
| `--seed` | `2023` | 随机种子 |

## 引用

如果本工作对您有帮助，请引用以下论文：

```bibtex
@article{luo2024eprotonet,
  title     = {Meta-learning with elastic prototypical network for fault transfer diagnosis of bearings under unstable speeds},
  author    = {Luo, Jingjie and Shao, Haidong and Lin, Jian and Liu, Bin},
  journal   = {Reliability Engineering \& System Safety},
  volume    = {245},
  pages     = {110001},
  year      = {2024},
  doi       = {10.1016/j.ress.2024.110001},
  publisher = {Elsevier}
}
```

## 联系方式

- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn
