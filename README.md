# CV_3D - 基于 Hunyuan3D-2 的 3D 资产生成

基于 Hunyuan3D-2 的文本/图像到 3D 资产生成系统，支持纹理渲染。

## 功能特性

- **多视图 → 3D** (New!)：支持使用 Hunyuan3D-2mv 模型，通过前/后/左/右多视图生成高精度 3D 资产
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

# 安装文生图及下载依赖
pip install diffusers transformers accelerate sentencepiece huggingface_hub
```

### 4. 下载模型

我们提供了一键下载脚本，支持从 Hugging Face 官方源或国内镜像下载。

```bash
# 下载 Hunyuan3D-2mv 和 Texture 模型到 weights/ 目录
# 默认使用 Hugging Face 官方源（香港/海外服务器）
python scripts/download_models.py

# 如果在内地服务器，可使用国内镜像加速
python scripts/download_models.py --mirror
```

模型将保存在 `weights/` 目录下，结构如下：
- `weights/hunyuan3d-dit-v2-mv` (形状模型)
- `weights/hunyuan3d-paint-v2-0` (纹理模型)

### 5. 配置

编辑 `configs/hunyuan3d_mv_local.yaml` (推荐) 或 `configs/hunyuan3d_default.yaml`。

`configs/hunyuan3d_mv_local.yaml` 默认配置指向 `weights/` 目录：

```yaml
model_path: "weights"
model_subfolder: "hunyuan3d-dit-v2-mv"
texture_model_path: "weights"
texture_subfolder: "hunyuan3d-paint-v2-0"
```

## 使用方法

### 多视图 → 3D (New!)

使用 `Hunyuan3D-2mv` 模型，支持提供多张视角图片以获得更精准的几何形状。

```bash
python scripts/run_mv_image2asset.py \
    --config configs/hunyuan3d_mv_local.yaml \
    --front path/to/front.png \
    --back path/to/back.png \
    --left path/to/left.png \
    --right path/to/right.png \
    --name my_mv_asset
```
*注意：至少需要提供 `--front` 视图。*

### 文本 → 3D

```bash
python scripts/run_text2asset.py --prompt "a wooden chair" --name chair
```

### 单图像 → 3D

```bash
python scripts/run_image2asset.py --image path/to/image.png --name output
```

### 输出文件

- `outputs/{name}.glb` - 3D 模型文件（支持纹理）
- `outputs/{name}_albedo.png` - 纹理贴图（如启用纹理）
- `outputs/{name}_metadata.json` - 元数据
- `outputs/{name}_intermediate.png` - 文生图中间产物（仅文本输入）

## 配置说明

- `enable_texture: true/false` - 是否启用纹理生成（默认开启，耗时约 20 分钟）
- `low_vram_mode: true` - 低显存模式（推荐开启）
- `enable_background_removal: true` 启用背景移除（使用 Hunyuan3D 的 BackgroundRemover）,会占用显存

## 注意事项

- **显存要求**：建议 24GB+ GPU（5090 等）
- **纹理生成**：使用 CPU 渲染，GPU 利用率低但显存占用高，耗时约 20 分钟
- **模型路径**：确保配置文件中的路径与实际下载路径一致

## 项目结构

```
CV_3D/
├── configs/              # 配置文件 (含 hunyuan3d_mv_local.yaml)
├── core/                 # 核心模块
│   ├── models/          # 模型封装
│   ├── pipeline/        # 生成流水线 (含 mv_pipeline.py)
│   ├── io/              # 输入输出
│   └── preprocess/      # 预处理
├── scripts/              # 命令行脚本
│   ├── download_models.py   # 模型下载脚本
│   ├── run_mv_image2asset.py # 多视图生成脚本
│   └── ...
└── outputs/              # 输出目录
└── weights/              # 模型权重目录
```

## 参考

- [Hunyuan3D-2 官方仓库](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [Hunyuan3D-2mv HuggingFace](https://huggingface.co/tencent/Hunyuan3D-2mv)
