# 纹理一致性增强 (Texture Consistency Enhancement, TCE) 模块

## 概述

TCE 模块用于解决纹理生成中的"多头问题"及纹理接缝问题。该模块通过三个核心策略来增强纹理的一致性：

1. **CLIP 语义一致性**：约束从任意视角渲染的图像与文本提示在 CLIP 特征空间的一致性
2. **Warp Consistency**：利用光流法约束相邻视角的纹理像素对应关系，强迫纹理在空间上连续
3. **显式稀疏优化**：对生成的纹理贴图进行 Patch 级别的显式优化，而非仅依赖隐式扩散先验

## 核心公式

### CLIP 语义一致性损失

```
L_sem = 1 - sim(CLIP(I_render), CLIP(T_prompt))
```

其中：
- `I_render` 是从任意视角渲染的图像
- `T_prompt` 是文本提示
- `sim()` 是 CLIP 特征空间的相似度函数

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
import torch
from tce_module import TCEModule

# 初始化 TCE 模块
device = "cuda" if torch.cuda.is_available() else "cpu"
tce = TCEModule(
    clip_model_name="ViT-B/32",
    lambda_sem=1.0,      # CLIP 语义一致性权重
    lambda_warp=1.0,     # Warp Consistency 权重
    lambda_sparse=0.5,   # 稀疏优化权重
    patch_size=16,
    device=device
)

# 准备数据
rendered_images = torch.rand(4, 3, 256, 256).to(device)  # 渲染图像
texture_map = torch.rand(3, 256, 256).to(device)          # 纹理贴图
text_prompt = "a beautiful landscape"                      # 文本提示

# 计算损失
losses = tce(
    rendered_images=rendered_images,
    text_prompt=text_prompt,
    texture_map=texture_map
)

print(f"总损失: {losses['total'].item():.4f}")
```

### 纹理优化

```python
# 优化纹理贴图
optimized_texture = tce.optimize_texture(
    texture_map=initial_texture,
    rendered_images=rendered_images,
    text_prompt=text_prompt,
    num_iterations=100,
    lr=0.01
)
```

## 模块结构

### 主要模块

- **`tce_module.py`**: TCE 主模块，整合三个核心策略
- **`clip_consistency.py`**: CLIP 语义一致性损失实现
- **`warp_consistency.py`**: Warp Consistency 损失实现
- **`sparse_optimization.py`**: 显式稀疏优化实现

### 核心类

#### `TCEModule`

主要的 TCE 模块类，整合所有功能。

**参数：**
- `clip_model_name` (str): CLIP 模型名称，默认 "ViT-B/32"
- `lambda_sem` (float): CLIP 语义一致性损失权重
- `lambda_warp` (float): Warp Consistency 损失权重
- `lambda_sparse` (float): 稀疏优化损失权重
- `patch_size` (int): Patch 大小（用于稀疏优化）
- `device` (str): 计算设备

**方法：**
- `forward()`: 计算所有损失
- `optimize_texture()`: 对纹理贴图进行显式优化

#### `CLIPConsistencyLoss`

CLIP 语义一致性损失模块。

#### `WarpConsistencyLoss`

Warp Consistency 损失模块，使用光流法约束纹理连续性。

#### `SparsePatchOptimizer`

显式稀疏优化器，进行 Patch 级别的优化。

## 运行示例

```bash
python example_usage.py
```

## 依赖项

- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- numpy >= 1.21.0
- clip-by-openai >= 1.0
- Pillow >= 9.0.0

## 注意事项

1. **CLIP 模型下载**：首次运行时会自动下载 CLIP 模型，需要网络连接
2. **光流计算**：当前实现使用简化的光流计算方法，对于更精确的结果，建议使用 RAFT 或 FlowNet 等专业光流估计网络
3. **内存使用**：处理大尺寸纹理时注意 GPU 内存使用情况
4. **设备选择**：建议使用 GPU 以获得更好的性能

## 扩展建议

1. **更精确的光流估计**：集成 RAFT 或 FlowNet 等专业光流网络
2. **多尺度优化**：在不同尺度上进行纹理优化
3. **自适应权重**：根据训练进度动态调整各损失项的权重
4. **批量处理**：优化批量处理多个纹理的效率

## 许可证

本项目遵循 MIT 许可证。

