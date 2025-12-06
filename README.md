# CV_3D - 基于三阶段优化框架的单视图 3D 资产生成

基于 Hunyuan3D-2 的文本/图像到 3D 资产生成系统，采用**三阶段优化框架**，通过几何修复网络（GRN）和纹理一致性增强（TCE）模块，显著提升单视图 3D 生成的鲁棒性与可用性。

## 功能特性

### 核心功能
- **文本 → 3D**：通过 FLUX.1-schnell 文生图 + Hunyuan3D-2 图像到 3D
- **图像 → 3D**：支持单视图和多视图输入生成 3D 资产
- **纹理渲染**：支持高质量纹理生成
- **输出格式**：`.obj`、`.glb`（支持纹理）

### 三阶段优化框架
- **阶段一：基础生成** - 使用 Hunyuan3D-2 快速构建粗糙几何和纹理
- **阶段二：几何修复（GRN）** - 轻量级几何修复网络，修复孔洞、噪声和非流形结构
- **阶段三：纹理一致性增强（TCE）** - 基于 CLIP 和光流的纹理一致性优化，解决跨视角纹理不一致问题

### 后处理与优化
- **Blender 后处理**：自动修复网格缺陷（非流形边、孔洞、法线等）
- **多视角优化**：支持多视角输入，自动切换单视图/多视图模型
- **评估系统**：完整的评估指标（Chamfer Distance、F1 Score、Consistency Score、CLIPScore）

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

# 安装 GRN 模块依赖（可选，用于几何修复）
cd core/GRN && pip install -r requirements.txt && cd ../..

# 安装 TCE 模块依赖（可选，用于纹理一致性增强）
cd core/TCE && pip install -r requirements.txt && cd ../..
```

**注意**：GRN 和 TCE 模块为可选功能，如果不需要使用几何修复和纹理一致性增强，可以跳过这两个步骤。

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

### 基础配置
- `enable_texture: true/false` - 是否启用纹理生成（默认开启，耗时约 20 分钟）
- `low_vram_mode: true` - 低显存模式（推荐开启）
- `enable_background_removal: true` - 启用背景移除（使用 Hunyuan3D 的 BackgroundRemover），会占用显存

### 三阶段优化配置
- `enable_grn: true/false` - 是否启用几何修复网络（GRN），需要训练好的模型
- `enable_tce: true/false` - 是否启用纹理一致性增强（TCE）
- `enable_postprocess: true/false` - 是否启用 Blender 后处理（需要安装 Blender）

### GRN 配置
在 `core/GRN/configs/default.yaml` 中配置：
- 模型检查点路径
- 输入/输出通道数
- 损失函数权重

### TCE 配置
在代码中或配置文件中设置：
- `lambda_sem`: CLIP 语义一致性权重（默认 1.0）
- `lambda_warp`: Warp Consistency 权重（默认 1.0）
- `lambda_sparse`: 稀疏优化权重（默认 0.5）

## 使用方法

### 基础生成

#### 文本 → 3D
```bash
python scripts/run_text2asset.py --prompt "a wooden chair" --name chair
```

#### 图像 → 3D（单视图）
```bash
python scripts/run_image2asset.py --image path/to/image.png --name output
```

#### 图像 → 3D（多视图）
```bash
python scripts/run_image2asset.py \
    --front path/to/front.png \
    --left path/to/left.png \
    --back path/to/back.png \
    --name output_mv
```

### 三阶段优化流程

#### 1. 基础生成 + GRN 几何修复
```python
from core.pipeline.generation_pipeline import GenerationPipeline
from core.GRN.grn_wrapper import refine_mesh_with_grn

# 基础生成
pipeline = GenerationPipeline.from_config_file("configs/hunyuan3d_default.yaml")
result = pipeline.generate_from_image(image, output_name="output")

# GRN 几何修复
refined_mesh = refine_mesh_with_grn(
    input_mesh_path="outputs/output.glb",
    checkpoint_path="core/GRN/checkpoints/best_model.pth",
    output_path="outputs/output_refined.glb"
)
```

#### 2. 基础生成 + TCE 纹理一致性增强
```python
from core.TCE.tce_wrapper import apply_tce_optimization

# 基础生成后应用 TCE
optimized_texture = apply_tce_optimization(
    texture_map=initial_texture,
    rendered_images=rendered_views,
    text_prompt="a wooden chair",
    num_iterations=100
)
```

#### 3. 完整三阶段流程
```python
# 阶段一：基础生成
pipeline = GenerationPipeline.from_config_file("configs/hunyuan3d_default.yaml")
asset = pipeline.generate_from_image(image, output_name="output")

# 阶段二：GRN 几何修复（如果启用）
if config.get("enable_grn", False):
    asset = refine_mesh_with_grn(asset, config)

# 阶段三：TCE 纹理一致性增强（如果启用）
if config.get("enable_tce", False):
    asset = apply_tce_optimization(asset, config)

# 后处理（可选）
if config.get("enable_postprocess", False):
    from core.postprocess.blender_postprocess import postprocess_with_blender
    postprocess_with_blender(
        input_mesh_path="outputs/output.glb",
        output_mesh_path="outputs/output_postprocessed.glb"
    )
```

### 后处理

#### Blender 后处理
```bash
# 需要先安装 Blender
python -c "
from core.postprocess.blender_postprocess import postprocess_with_blender
postprocess_with_blender(
    'outputs/asset.glb',
    'outputs/asset_cleaned.glb'
)
"
```

后处理功能包括：
- 移除重复顶点
- 删除孤立顶点和边
- 删除退化面
- 修复法线方向
- 修复非流形边
- 自动填充孔洞
- 模型简化和平滑

### 评估

使用评估脚本评估生成的 3D 资产：

```bash
# 完整评估（包含所有指标）
python scripts/evaluate_assets.py \
    --metadata outputs/asset_metadata.json \
    --ref_points reference/points.npy \
    --rendered_images outputs/rendered_*.png \
    --prompt "a wooden chair" \
    --use_real_clip \
    --output evaluation_results.json
```

**评估指标**：
- **Chamfer Distance (CD) ↓** - 衡量几何结构的准确性
- **F1 Score ↑** - 衡量点云覆盖率与精度的调和平均
- **Consistency Score ↑** - 基于 CLIP 和光流计算的纹理一致性得分
- **CLIPScore ↑** - 文本-图像语义一致性（文生图任务）

详细评估说明请参考 `core/eval/` 目录下的模块文档。

## 注意事项

- **显存要求**：建议 24GB+ GPU（5090 等）
- **纹理生成**：使用 CPU 渲染，GPU 利用率低但显存占用高，耗时约 20 分钟
- **模型路径**：确保配置文件中的路径与实际下载路径一致
- **GRN 模型**：需要预先训练或下载 GRN 模型检查点
- **Blender 后处理**：需要安装 Blender 并配置路径
- **CLIP 评估**：使用 `--use_real_clip` 需要安装 `open-clip-torch`

## 项目结构

```
CV_3D/
├── configs/                  # 配置文件
│   ├── hunyuan3d_default.yaml
│   └── hunyuan3d_mv.yaml
├── core/                     # 核心模块
│   ├── models/              # 模型封装
│   │   └── hunyuan3d_wrapper.py
│   ├── pipeline/            # 生成流水线
│   │   ├── generation_pipeline.py    # 基础生成流水线
│   │   ├── mv_pipeline.py            # 多视图流水线
│   │   └── qwen_edit_mv_pipeline.py  # Qwen 编辑多视图流水线
│   ├── preprocess/          # 预处理
│   │   ├── image_preprocess.py
│   │   └── text_preprocess.py
│   ├── GRN/                 # 几何修复网络（阶段二）
│   │   ├── models/          # GRN 模型定义
│   │   ├── data/           # 数据加载
│   │   ├── losses/         # 损失函数
│   │   ├── checkpoints/    # 模型检查点
│   │   ├── train.py        # 训练脚本
│   │   ├── inference.py    # 推理脚本
│   │   └── grn_wrapper.py  # GRN 封装接口
│   ├── TCE/                 # 纹理一致性增强（阶段三）
│   │   ├── tce_module.py   # TCE 主模块
│   │   ├── clip_consistency.py
│   │   ├── warp_consistency.py
│   │   ├── sparse_optimization.py
│   │   └── tce_wrapper.py  # TCE 封装接口
│   ├── postprocess/         # 后处理模块
│   │   ├── blender_postprocess.py
│   │   └── blender_postprocess_script.py
│   ├── eval/               # 评估模块
│   │   ├── geometry.py     # 几何评估（CD, F1 Score）
│   │   ├── texture.py     # 纹理评估（Consistency Score）
│   │   └── semantic.py    # 语义评估（CLIPScore）
│   └── io/                 # 输入输出
│       ├── inputs.py
│       └── outputs.py
├── scripts/                 # 命令行脚本
│   ├── run_text2asset.py   # 文本到 3D
│   ├── run_image2asset.py  # 图像到 3D
│   ├── run_mv_image2asset.py  # 多视图图像到 3D
│   ├── evaluate_assets.py  # 评估脚本
│   └── generate_*.py       # 其他生成脚本
└── outputs/                 # 输出目录
```

## 三阶段优化框架详解

### 阶段一：基础生成
使用 Hunyuan3D-2 进行快速的基础 3D 生成，支持：
- 单视图输入（`hunyuan3d-dit-v2-0`）
- 多视图输入（`hunyuan3d-dit-v2-mv`）
- 自动模型切换

### 阶段二：几何修复网络（GRN）
**功能**：修复单视图生成中的几何缺陷（孔洞、噪声、非流形结构）

**特点**：
- 轻量级 UNet 架构（0.8M-1.3M 参数）
- 自监督学习，无需成对数据
- 多通道输入/输出（深度 + 法线）

**使用**：
```python
from core.GRN.grn_wrapper import refine_mesh_with_grn

refined_mesh = refine_mesh_with_grn(
    input_mesh_path="coarse_mesh.glb",
    checkpoint_path="core/GRN/checkpoints/best_model.pth",
    output_path="refined_mesh.glb"
)
```

详细文档：`core/GRN/README.md`

### 阶段三：纹理一致性增强（TCE）
**功能**：解决纹理跨视角不一致问题（"多头问题"、接缝等）

**核心策略**：
1. **CLIP 语义一致性**：约束渲染图像与文本提示的语义一致性
2. **Warp Consistency**：利用光流约束相邻视角的纹理连续性
3. **显式稀疏优化**：Patch 级别的纹理优化

**使用**：
```python
from core.TCE.tce_wrapper import apply_tce_optimization

optimized_texture = apply_tce_optimization(
    texture_map=texture,
    rendered_images=views,
    text_prompt="prompt",
    num_iterations=100
)
```

详细文档：`core/TCE/README.md`

## 高级功能：多视角增强生成

项目提供了多种多视角生成策略，通过智能的多视角生成显著提升单视图 3D 生成质量。

### 1. Qwen 增强管道（HQE-Pipeline）⭐ 推荐

**核心思想**：使用大语言模型（LLM/VLM）驱动的图像编辑模型"脑补"出物体在不同视角下的样子，再将这些高质量的视角图喂给 3D 生成模型。

#### 架构特点
- **阶段 1：语义理解与描述** - 使用 `Qwen2.5-VL-3B-Instruct` 分析图像，生成详细描述
- **阶段 2：智能多视角生成** - 使用 `Qwen-Image-Edit-2509` 生成多视角图像
- **阶段 3：3D 资产生成** - 使用 `Hunyuan3D-2` 将多视图融合为 3D 模型

#### 使用方式

**快速运行（Simple 45° 模式 - 推荐）**：
```bash
bash run_full_pipeline.sh examples/demo.png my_test_job
```

**全视角运行（Full AR 模式）**：
```bash
bash run_full_pipeline.sh examples/demo.png my_test_job full
```

**Python 脚本直接调用**：
```bash
# 使用 Qwen AR 带描述生成
python scripts/generate_qwen_ar_with_caption.py \
    --input_image examples/demo.png \
    --output_dir outputs/views \
    --mode simple_45  # 或 full

# 然后使用生成的多视图进行 3D 重建
python scripts/run_mv_image2asset.py \
    --config configs/hunyuan3d_mv.yaml \
    --front outputs/views/front.png \
    --right outputs/views/right_45.png \
    --name result
```

#### 优势
- ✅ **语义一致性**：通过 VLM 提取语义，解决模糊指令导致的风格漂移
- ✅ **几何准确性**：引入 45度/多视角显式输入，解决"纸片人"问题
- ✅ **自动化**：双环境自动切换，一键运行

详细文档：`README_HQE.md`

### 2. Legacy 多视角管道（生成-筛选范式）

**核心思想**：为每个目标视角生成多个候选图，通过一致性过滤选择最佳候选。

#### 架构特点
- **Stage 0：输入与分析** - 使用 Qwen2.5-VL 提取语义属性
- **Stage 1：候选生成** - 为每个视角生成 K 个候选（默认 K=2）
- **Stage 2：一致性过滤** - 使用 CLIP 选择最符合原图身份的候选
- **Stage 3：3D 重建** - 将筛选后的视图输入 Hunyuan3D-2

#### 使用方式

```bash
bash run_mv_legacy.sh examples/demo.png my_legacy_test 4
```

**Python 脚本直接调用**：
```bash
python scripts/generate_mv_legacy.py \
    --input_image examples/demo.png \
    --output_dir outputs/views \
    --candidates 4  # 每个视角生成 4 个候选
```

详细文档：`README_LEGACY.md`

### 3. Legacy Full 管道（深度+感知度量）

**核心思想**：使用复合指标（CLIP + LPIPS + Depth-Aware Warp）进行严格的一致性验证。

#### 架构特点
- **语义分析**：Qwen2.5-VL 提取主体描述和材质风格
- **候选生成**：为每个视角生成 K 个候选（默认 K=4）
- **一致性验证**：使用复合评分函数
  ```
  Score = w1·S_CLIP + w2·S_LPIPS + w3·S_Warp
  ```

#### 使用方式

```bash
# 需要先安装额外依赖
pip install lpips imageio-ffmpeg

# 运行完整管道
bash run_mv_legacy_full.sh examples/demo.png test_legacy_full 4
```

**Python 脚本直接调用**：
```bash
python scripts/generate_mv_legacy_full.py \
    --input_image examples/demo.png \
    --output_dir outputs/views \
    --candidates 4
```

详细文档：`README_LEGACY_FULL.md`

### 4. 多视图流水线（直接使用多视图输入）

如果已有多个视角的图像，可以直接使用多视图流水线：

```bash
python scripts/run_mv_image2asset.py \
    --config configs/hunyuan3d_mv.yaml \
    --front path/to/front.png \
    --left path/to/left.png \
    --right path/to/right.png \
    --back path/to/back.png \
    --name output_mv
```

**Python API**：
```python
from core.pipeline.mv_pipeline import MultiViewPipeline

pipeline = MultiViewPipeline.from_config_file("configs/hunyuan3d_mv.yaml")
result = pipeline.generate_from_multiview({
    "front": "front.png",
    "left": "left.png",
    "right": "right.png",
    "back": "back.png"
})
```

### 管道对比

| 特性 | **HQE-Pipeline (Qwen AR)** | **Legacy Full** | **Legacy** |
|------|---------------------------|----------------|------------|
| **方法** | 自回归（视角到视角） | 并行生成 + 筛选 | 并行生成 + CLIP 筛选 |
| **上下文** | "看到"之前生成的视角 | 仅"看到"正面视图 | 仅"看到"正面视图 |
| **一致性** | **高**（继承结构） | 不定（依赖筛选） | 不定（依赖筛选） |
| **速度** | 快（每视角生成一次） | 慢（K 次生成 + 过滤） | 中等 |
| **指标** | 隐式（注意力机制） | 显式（CLIP/LPIPS/Depth） | 显式（CLIP） |
| **推荐场景** | 生产环境/追求质量 | 学术对比/严格几何指标 | 快速原型/简单场景 |

**建议**：
- **生产环境/追求质量**：使用 **HQE-Pipeline (Qwen AR)**
- **学术研究/严格指标**：使用 **Legacy Full**
- **快速原型**：使用 **Legacy** 或直接多视图输入

## 环境配置（双环境策略）

由于 Qwen 和 Hunyuan3D 的依赖存在部分冲突，项目采用**双环境切换**策略。

### 环境 1: `qwen_edit`（用于多视角生成）

负责运行 Qwen-VL 和 Qwen-Image-Edit。

```bash
conda create -n qwen_edit python=3.10 -y
conda activate qwen_edit

# 安装 Qwen 相关依赖
pip install torch>=2.0.0 torchvision
pip install transformers>=4.51.3
pip install git+https://github.com/huggingface/diffusers
pip install qwen-vl-utils accelerate
pip install lpips imageio-ffmpeg  # Legacy Full 需要
```

### 环境 2: `cv3d`（用于 3D 生成）

负责运行 Hunyuan3D-2（见上方"服务器部署"章节）。

### 自动环境切换

项目提供的 Shell 脚本（`run_full_pipeline.sh`、`run_mv_legacy.sh` 等）会自动处理环境切换，无需手动操作。

## 脚本说明

### 多视角生成脚本

- `scripts/generate_qwen_ar_with_caption.py` - Qwen AR 带描述生成（推荐）
- `scripts/generate_qwen_ar_multiview.py` - Qwen AR 多视角生成
- `scripts/generate_qwen_multiview.py` - Qwen 基础多视角生成
- `scripts/generate_mv_legacy.py` - Legacy 多视角生成（CLIP 筛选）
- `scripts/generate_mv_legacy_full.py` - Legacy Full 多视角生成（复合指标）

### 3D 生成脚本

- `scripts/run_text2asset.py` - 文本到 3D
- `scripts/run_image2asset.py` - 单视图图像到 3D
- `scripts/run_mv_image2asset.py` - 多视图图像到 3D

### 测试与评估脚本

- `scripts/test_qwen_image_only.py` - Qwen 单图测试
- `scripts/test_qwen_mv.py` - Qwen 多视图测试
- `scripts/evaluate_assets.py` - 3D 资产评估
- `scripts/download_models.py` - 模型下载工具

## 实验与评估

### 定量结果
基于三阶段优化框架的实验结果显示：
- **Chamfer Distance** 降低约 34%
- **F1 Score** 提升约 33%
- **Consistency Score** 提升约 31%

### 评估指标
项目提供完整的评估系统，支持：
- 几何质量评估（CD, F1 Score）
- 纹理一致性评估（Consistency Score）
- 语义一致性评估（CLIPScore）

详细评估方法请参考 `scripts/evaluate_assets.py` 和 `core/eval/` 目录。

## 相关文档

- **`README_HQE.md`** - Qwen 增强管道（HQE-Pipeline）详细文档
- **`README_LEGACY.md`** - Legacy 多视角管道文档
- **`README_LEGACY_FULL.md`** - Legacy Full 管道文档（深度+感知度量）
- **`core/GRN/README.md`** - 几何修复网络（GRN）文档
- **`core/TCE/README.md`** - 纹理一致性增强（TCE）文档

## 参考

- [Hunyuan3D-2 官方仓库](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [ModelScope 模型库](https://modelscope.cn)
- [FLUX.1-schnell](https://modelscope.cn/models/AI-ModelScope/FLUX.1-schnell)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) - 视觉语言模型
- [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) - 图像编辑模型
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP 模型实现
- [Blender](https://www.blender.org/) - 3D 后处理工具
