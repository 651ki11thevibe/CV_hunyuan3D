# Legacy Multi-View Pipeline (MVP) | 经典多视角管线 (MVP版)

> **"Quality through Selection"** | **"通过筛选获得质量"**

This project implements a robust, production-ready pipeline for automatic multi-view image generation from a single input image. Unlike simple scripts that generate one image per view, this pipeline treats generation as a **selection process**, ensuring consistency and quality through attribute extraction and metric-based filtering.

本项目实现了一个鲁棒的、面向生产的单图转多视角生成管线。与简单生成单张图的脚本不同，本管线将生成视为一个 **筛选过程**，通过属性提取和基于指标的过滤来确保一致性和质量。

---

## 🏗 System Architecture | 系统架构

The pipeline consists of four distinct modules executed sequentially:
该管线包含四个按顺序执行的模块：

### Stage 0: Input & Analysis | 输入与分析
*   **Objective**: Canonicalize input and understand semantic attributes. (标准化输入并理解语义属性)
*   **Model**: `Qwen2.5-VL-3B-Instruct`
*   **Process**: 
    *   Input image is analyzed to extract **Subject** (e.g., "red vintage car"), **Style**, **Material**, and **Viewpoint**. (分析输入图像以提取 **主体**、**风格**、**材质** 和 **视角**)
    *   These attributes are used to construct deterministic prompts for the next stage. (这些属性用于构建下一阶段的确定性提示词)

### Stage 1: Candidate Generation (The "Legacy" Core) | 候选生成
*   **Objective**: Generate multiple diverse candidates for each target view. (为每个目标视角生成多个多样化的候选图)
*   **Model**: `Qwen-Image-Edit-2509`
*   **Process**:
    *   For each target view (Right, Left, Back), generate $K$ candidates (default $K=2$). (对于右/左/后视角，生成 $K$ 个候选，默认 $K=2$)
    *   Use different random seeds to ensure diversity in the candidates. (使用不同的随机种子确保多样性)

### Stage 2: Consistency Filtering | 一致性过滤
*   **Objective**: Select the best candidate that matches the original image's identity. (选择最符合原图身份的最佳候选)
*   **Model**: `CLIP-ViT-Large`
*   **Process**:
    *   Calculate **Semantic Similarity Score** between the Input Image (Reference) and each generated Candidate. (计算输入图与每个候选图之间的 **语义相似度分数**)
    *   $Score = \text{CosineSim}(E_{ref}, E_{cand})$
    *   Select the candidate with the highest score. (选择分数最高的候选)

### Stage 3: 3D Reconstruction | 3D 重建
*   **Objective**: Lift the consistent 2D views into 3D. (将一致的 2D 视角提升为 3D)
*   **Model**: `Hunyuan3D-2`
*   **Process**: Feed the filtered Front, Left, Right, and Back views into the reconstruction model. (将筛选后的前/左/右/后视图输入重建模型)

---

## 🚀 Quick Start | 快速开始

### Prerequisites | 前置条件
Ensure you have the `qwen_edit` and `cv3d` conda environments set up as per the main project README.
确保你已按照主项目 README 配置好 `qwen_edit` 和 `cv3d` 环境。

### Run Command | 运行命令
```bash
bash run_mv_legacy.sh <input_image_path> [job_name] [candidates_num]
```

**Example**:
```bash
bash run_mv_legacy.sh examples/demo.png my_legacy_test 4
```
*This will generate 4 candidates per view, select the best one, and build a 3D model.*
*这将为每个视角生成 4 个候选图，选出最好的一个，并构建 3D 模型。*

---

## 📂 Output Structure | 输出结构

```text
outputs/my_legacy_test_TIMESTAMP/
├── views/
│   ├── front.png   # Original (原图)
│   ├── right.png   # Generated (Best Selection) (生成的最佳右视图)
│   ├── left.png    # Generated (Best Selection) (生成的最佳左视图)
│   └── back.png    # Generated (Best Selection) (生成的最佳后视图)
├── model.glb       # Final 3D Model (最终 3D 模型)
└── ...
```

## 🛠 Key Engineering Features | 关键工程特性

1.  **VRAM Optimization**: Models are loaded and unloaded dynamically. VL model runs first, then unloads to make room for the Edit model. (显存优化：模型动态加载卸载。VL 模型先运行，然后卸载以腾出空间给编辑模型)
2.  **Attribute Persistence**: The prompt explicitly carries over attributes (color, material) extracted in Stage 0. (属性持久化：提示词显式继承了 Stage 0 提取的颜色、材质等属性)
3.  **Fail-safe Scoring**: If CLIP scoring fails for any reason, it gracefully falls back to the first candidate. (故障安全评分：如果 CLIP 评分失败，会自动降级选择第一个候选图)

---

## 📈 Future Improvements (Roadmap) | 未来改进路线

*   **Phase 2**: Add **LPIPS** (Perceptual Similarity) to the filtering stage for better texture matching. (加入 LPIPS 感知相似度以优化纹理匹配 - *已在 Full Pipeline 中实现*)
*   **Phase 3**: Implement **ControlNet** (Depth/Canny) guidance during generation to enforce geometric consistency. (实现 ControlNet 引导以强制几何一致性)
*   **Phase 4**: Close the loop by using the generated views as references for subsequent views (Auto-Regressive + Selection). (闭环：将生成的视图作为后续生成的参考)

