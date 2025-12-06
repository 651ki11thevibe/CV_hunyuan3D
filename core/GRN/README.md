# GRN: Geometry Refinement Network

三阶段3D生成框架的阶段2：几何精炼网络（Geometry Refinement Network）

## 概述

GRN旨在修复粗糙3D模型中的噪声、孔洞和法线不一致问题。采用自监督学习策略，通过合成退化几何进行训练。

## 功能特性

- **轻量级UNet架构**：参数量约0.8M-1.3M
- **自监督学习**：无需成对的粗糙-精细数据
- **合成退化数据**：从Objaverse/ShapeNet等高质量数据集生成训练数据
- **多通道输入/输出**：输入5通道（Depth=1, Normal=3, Mask=1），输出4通道（Refined Depth=1, Refined Normal=3）

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 测试模型

首先测试模型是否能正常工作：

```bash
python test_model.py
```

### 2. 准备数据

准备训练数据（从Objaverse/ShapeNet）：

```bash
# 使用Objaverse
python setup_data.py --source objaverse --num_samples 100

# 使用ShapeNet
python setup_data.py --source shapenet --shapenet_path /path/to/shapenet

# 或两者都使用
python setup_data.py --source both
```

数据目录结构：
```
data/
├── train/          # 训练集3D模型
└── val/            # 验证集3D模型
```

### 3. 训练

开始训练：

```bash
python train.py --config configs/default.yaml
```

训练过程会：
- 自动从高质量3D模型生成退化几何
- 使用自监督学习训练GRN模型
- 保存检查点到 `checkpoints/`
- 记录训练日志到 `logs/`（可用TensorBoard查看）

### 4. 推理

使用训练好的模型精炼粗糙3D模型：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input path/to/coarse_mesh.obj \
    --output ./output \
    --visualize
```

## 项目结构

```
GRN/
├── models/              # 模型定义
│   ├── __init__.py
│   └── grn.py          # GRN UNet架构
├── data/                # 数据加载和预处理
│   ├── __init__.py
│   └── dataset.py       # 数据集和退化操作
├── losses/              # 损失函数
│   ├── __init__.py
│   └── grn_loss.py     # GRN损失函数
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── checkpoint.py   # 检查点管理
│   └── visualization.py # 可视化工具
├── configs/             # 配置文件
│   └── default.yaml    # 默认训练配置
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── test_model.py        # 模型测试脚本
├── setup_data.py        # 数据准备脚本
└── requirements.txt     # 依赖包
```

## 模型架构

GRN使用轻量级UNet架构：

- **编码器**：32 → 64 → 128 通道
- **瓶颈层**：128 通道
- **解码器**：128 → 64 → 32 通道
- **输入**：5通道（深度1 + 法线3 + 掩码1）
- **输出**：4通道（精炼深度1 + 精炼法线3）

## 训练策略

### 自监督学习

1. **数据生成**：从高质量3D数据集（Objaverse/ShapeNet）加载干净几何
2. **退化模拟**：添加随机噪声、孔洞和畸变，模拟Hunyuan3D的粗糙输出
3. **训练目标**：使用退化几何作为输入，恢复原始干净几何

### 损失函数

GRN使用以下损失函数组合：

**Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap**

其中：

1. **Lrec (重建损失)**：L1损失，用于重建深度图
   - 计算预测深度和目标深度之间的L1距离

2. **Lsmooth (平滑度约束)**：Total Variation损失，约束局部平滑度
   - 使用梯度约束保证几何光滑

3. **Lnormal (法线一致性损失)**：余弦相似度损失，保证法线一致性
   - 确保预测法线与目标法线方向一致

4. **Llap (拉普拉斯平滑项)**：拉普拉斯平滑损失
   - 公式：Llap = ||∆v − Lap(∆v)||²₂
   - 用于网格优化，保证几何的平滑性

默认权重：
- λrec = 1.0
- λsm = 0.1
- λnc = 2.0
- λlap = 0.05

## 配置说明

主要配置项（`configs/default.yaml`）：

- `data`: 数据路径、图像尺寸、视角数量、退化参数
- `model`: 输入/输出通道数
- `training`: 批次大小、学习率、训练轮数
- `loss`: 各项损失的权重

## 注意事项

1. **数据准备**：需要准备Objaverse或ShapeNet数据，或使用自己的3D模型数据集
2. **渲染依赖**：需要Open3D进行3D渲染，确保正确安装
3. **GPU推荐**：训练过程建议使用GPU加速
4. **内存需求**：根据批次大小和图像尺寸调整，默认配置需要约4GB显存


