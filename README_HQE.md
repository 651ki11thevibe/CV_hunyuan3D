# Hunyuan3D-2-Qwen-Enhanced Pipeline (HQE-Pipeline)

本项目构建了一个结合 **Qwen-Image-Edit (20B+)** 强大的图像编辑能力与 **Hunyuan3D-2** 优秀的 3D 生成能力的增强型 Pipeline。

传统的 Image-to-3D 模型（如 Hunyuan3D）在仅有一张输入图像时，往往需要“猜测”物体的背面和侧面，导致生成的几何结构（尤其是背面）模糊、塌陷或纹理拉伸。

本 Pipeline 的核心思想是：**先用大语言模型（LLM/VLM）驱动的图像编辑模型“脑补”出物体在不同视角下的样子，再将这些高质量的视角图喂给 3D 生成模型。**

---

## 核心架构与设计理念

整个 Pipeline 分为三个智能阶段：

### 1. 语义理解与描述 (Semantic Understanding)
*   **模型**: `Qwen2.5-VL-3B-Instruct` (Vision-Language Model)
*   **作用**: “看懂”输入图片。
*   **流程**:
    *   输入一张原始图片。
    *   VLM 分析图片内容，生成一段极其详尽的描述（Caption）。
    *   **关键点**: 我们特意编写了 Prompt，要求模型关注物体的**类型、颜色、材质、姿态**，如果检测到动物，必须明确**腿的数量**和**站姿**。
*   **理由**: 相比于硬编码的 "Rotate the object"，基于内容的 Prompt (如 "Rotate the vintage red telephone...") 能极大地提高编辑模型的一致性，防止物体在旋转过程中变形或丢失特征。

### 2. 智能多视角生成 (Smart Multi-View Generation)
*   **模型**: `Qwen-Image-Edit-2509` (基于 Qwen2.5-VL 的图像编辑模型)
*   **策略**: 提供两种模式以适应不同需求。
    *   **Full AR Mode (全视角自回归模式)**:
        *   利用 Qwen 的**多图输入能力**，采用自回归策略 (Auto-Regressive)。
        *   `Front` -> 生成 `Right`。
        *   `[Front, Right]` -> 生成 `Left` (利用右视图约束左视图，保证对称性)。
        *   `[Front, Right, Left]` -> 生成 `Back` (汇聚所有信息生成背面)。
    *   **Simple 45° Mode (轻量增强模式 - 默认)**:
        *   仅生成一张 **45度角侧视图**。
        *   **理由**: 全视角生成虽然信息全，但如果生成的背面图质量不高（比如崩了），反而会误导 3D 模型。45度角是 Qwen 最擅长生成的视角，既提供了立体感信息，又保持了极高的成功率和一致性。

### 3. 3D 资产生成 (3D Asset Reconstruction)
*   **模型**: `Hunyuan3D-2` (Multi-View Diffusion + Sparse View Reconstruction)
*   **作用**: 将多张 2D 图片融合为 3D 模型。
*   **流程**:
    *   接收 Step 2 生成的 `Front` + `Right (45°)` (或全视角图)。
    *   利用 Hunyuan3D 的多视图注意力机制，重建高精度的 Mesh 和 Texture。
*   **输出**: `.glb` 格式的 3D 模型。

---

## 环境配置 (Environment Setup)

由于 Qwen 和 Hunyuan3D 的依赖存在部分冲突（主要是 `transformers` 版本和一些 CUDA 库），我们采用了**双环境切换**策略。

### 环境 1: `qwen_edit` (用于 Step 1 & 2)
负责运行 Qwen-VL 和 Qwen-Image-Edit。
*   Python 3.10+
*   PyTorch 2.0+ (CUDA 11.8/12.1)
*   Transformers >= 4.51.3
*   Diffusers (最新版 git)
*   qwen-vl-utils
*   accelerate

### 环境 2: `cv3d` (用于 Step 3)
负责运行 Hunyuan3D-2。
*   Python 3.9/3.10
*   PyTorch (匹配 Hunyuan3D 要求)
*   `hy3dgen` (Hunyuan3D 核心库)
*   pymeshlab, trimesh 等 3D 处理库

---

## 文件结构

```text
CV_hunyuan3D/
├── scripts/
│   ├── generate_qwen_ar_with_caption.py  # [核心] VLM 描述 + Qwen 编辑脚本
│   ├── run_mv_image2asset.py             # Hunyuan3D 生成入口
│   └── ...
├── run_full_pipeline.sh                  # [总控] 自动化 Shell 脚本，负责环境切换和串联
├── submit_job.slurm                      # SLURM 集群提交脚本
└── configs/
    └── hunyuan3d_mv.yaml                 # Hunyuan3D 配置文件
```

---

## 使用指南

### 1. 快速运行 (Simple 45° Mode)
这是默认推荐模式，速度快，效果稳。

```bash
# 在交互式终端或提交脚本中
bash run_full_pipeline.sh <image_path> <job_name>
# 例如
bash run_full_pipeline.sh examples/demo.png my_test_job
```

### 2. 全视角运行 (Full Mode)
如果你对背面细节有强需求，且原图简单（如简单的卡通角色），可以使用此模式。

```bash
bash run_full_pipeline.sh <image_path> <job_name> full
```

### 3. 集群提交 (SLURM)
直接修改 `submit_job.slurm` 中的图片路径，然后：

```bash
sbatch submit_job.slurm
```

---

## 优势总结

1.  **语义一致性**: 通过 VLM 提取语义，彻底解决了 "Rotate the object" 这种模糊指令导致的风格漂移问题。
2.  **几何准确性**: 引入 45度/多视角 显式输入，解决了单图生成 3D 时常见的“纸片人”或“扁平化”问题。
3.  **解耦与自动化**: 双环境自动切换脚本，让你无需操心复杂的依赖冲突，一键运行到底。
