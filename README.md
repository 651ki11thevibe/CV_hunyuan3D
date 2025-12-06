# CV_3D - 基于 Hunyuan3D-2 的 3D 资产生成

基于 Hunyuan3D-2 的文本/图像到 3D 资产生成系统，支持纹理渲染。

## 功能特性

- **文本 → 3D**：通过 FLUX.1-schnell 文生图 + Hunyuan3D-2 图像到 3D
- **图像 → 3D**：直接使用图像生成 3D 资产
- **纹理渲染**：支持高质量纹理生成
- **输出格式**：`.obj`、`.glb`（支持纹理）

## 服务器部署

### 1. 环境准备

在 autodl 上可直接使用最新版本的 PyTorch 与 CUDA 环境，无需手动安装。

部署硬件 5090*1 ，显存>=32GB

如需要创建独立环境：
```bash
conda create -n cv3d python=3.10 -y
conda activate cv3d
```

### 2. 安装 Hunyuan3D-2 官方依赖

```bash
# 克隆官方仓库
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2
pip install -r requirements.txt
pip install -e .

# 编译纹理渲染 CUDA 扩展（必需）
# 先确认 GPU 架构版本
python -c "import torch; print(torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)"
# 输出示例：12 0，则使用 12.0

export TORCH_CUDA_ARCH_LIST="12.0"  # 根据实际输出修改
cd hy3dgen/texgen/custom_rasterizer && python setup.py install && cd ../..
cd hy3dgen/texgen/differentiable_renderer && python setup.py install && cd ../..
cd ../..
```

### 3. 安装项目依赖

```bash
# 返回项目根目录
cd CV_3D

# 安装基础依赖
pip install -r requirements.txt

# 安装文生图依赖（必需）
pip install diffusers transformers accelerate sentencepiece modelscope
```

### 4. 下载模型

所有模型从 ModelScope 下载：

#### 4.1 Hunyuan3D-2 模型

```bash
pip install modelscope

# 下载形状生成模型,仓库中有注明纹理、形状、光照模型
python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/Hunyuan3D-2', 
                  cache_dir='/root/autodl-tmp/models/Hunyuan3D-2-local')
"

# 下载纹理生成模型（已在上述下载中包含，确认路径为）
# /root/autodl-tmp/models/Hunyuan3D-2-local/hunyuan3d-paint-v2-0
```

**ModelScope 链接**：
- 形状模型：https://modelscope.cn/models/AI-ModelScope/Hunyuan3D-2
- 纹理模型：包含在上述仓库中

#### 4.2 FLUX.1-schnell 文生图模型

```bash
python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/FLUX.1-schnell',
                  cache_dir='/root/autodl-tmp/models/FLUX.1-schnell')
"
```

**ModelScope 链接**：https://modelscope.cn/models/AI-ModelScope/FLUX.1-schnell

### 5. 配置

编辑 `configs/hunyuan3d_default.yaml`，确认模型路径：

```yaml
# 单视角模型配置（默认）
model_path: "/root/autodl-tmp/models/Hunyuan3D-2-local"
model_subfolder: "hunyuan3d-dit-v2-0"

# 多视角模型配置（自动切换）
# 当输入为多视角图像时，系统会自动使用 multi-view 模型
multi_view:
  enabled: true  # 是否启用多视角自动切换
  model_path: "/root/autodl-tmp/models/Hunyuan3D-2-local"
  model_subfolder: "hunyuan3d-dit-v2-mv"
  variant: "fp16"  # 可选，指定模型变体

texture_model_path: "/root/autodl-tmp/models/Hunyuan3D-2-local"
texture_subfolder: "hunyuan3d-paint-v2-0"
text_to_image:
  local_model_path: "/root/autodl-tmp/models/FLUX.1-schnell"
```

**自动模型切换说明：**
- 当输入为**单张图像**时，系统自动使用单视角模型（`hunyuan3d-dit-v2-0`）
- 当输入为**多视角图像**（front, left, back）时，系统自动切换到多视角模型（`hunyuan3d-dit-v2-mv`）
- 无需手动修改配置文件，系统会根据输入自动选择对应的模型

## 使用方法

### 文本 → 3D

```bash
python scripts/run_text2asset.py --prompt "a wooden chair" --name chair
```

### 图像 → 3D

**单张图像：**
```bash
python scripts/run_image2asset.py --image path/to/image.png --name output
```

**多视角图像（三个视角）：**
```bash
python scripts/run_image2asset.py \
    --front path/to/front.png \
    --left path/to/left.png \
    --back path/to/back.png \
    --name output_mv
```

**自动模型切换：**
- 系统会自动检测输入类型（单张图像或多视角图像）
- 多视角输入时，自动切换到 multi-view 模型（需在配置文件中启用 `multi_view.enabled: true`）
- 无需手动切换配置，首次使用时会自动加载对应的模型

### 输出文件

- `outputs/{name}.glb` - 3D 模型文件（支持纹理）
- `outputs/{name}_albedo.png` - 纹理贴图（如启用纹理）
- `outputs/{name}_metadata.json` - 元数据
- `outputs/{name}_intermediate.png` - 文生图中间产物（仅文本输入）

## 配置说明

- `enable_texture: true/false` - 是否启用纹理生成（默认开启，耗时约 20 分钟）
- `low_vram_mode: true` - 低显存模式（推荐开启）
-  `enable_background_removal: true` 启用背景移除（使用 Hunyuan3D 的 BackgroundRemover）,会占用显存
## 注意事项

- **显存要求**：建议 24GB+ GPU（5090 等）
- **纹理生成**：使用 CPU 渲染，GPU 利用率低但显存占用高，耗时约 20 分钟
- **模型路径**：确保配置文件中的路径与实际下载路径一致

## 项目结构

```
CV_3D/
├── configs/              # 配置文件
├── core/                 # 核心模块
│   ├── models/          # 模型封装
│   ├── pipeline/        # 生成流水线
│   ├── io/              # 输入输出
│   └── preprocess/      # 预处理
├── scripts/              # 命令行脚本
└── outputs/              # 输出目录
```

## 参考

- [Hunyuan3D-2 官方仓库](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [ModelScope 模型库](https://modelscope.cn)
