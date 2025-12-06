# Legacy Full Pipeline (LPIPS + Depth) | 经典全流程管线 (深度+感知度量)

This is the "Full" implementation of the classic **Generation-Selection** pipeline for single-image-to-3D tasks. Unlike modern Autoregressive approaches (like our `Smart-AR` pipeline), this method relies on rigorous **Consistency Filtering** to ensure quality.

这是单图转 3D 任务中经典 **“生成-筛选”** 范式的“完全体”实现。与现代的自回归方法（如我们的 `Smart-AR` 管线）不同，该方法依赖严格的 **一致性过滤** 来保证质量。

Although experimentation suggests the **Qwen AR** method yields superior results for consistency, this pipeline serves as a robust baseline and academic implementation of standard Multi-View Synthesis techniques involving geometric verification.

尽管实验表明 **Qwen AR** 方法在一致性上效果更好，但本管线作为一个包含几何验证的标准多视图合成技术的鲁棒基准（Baseline）和学术实现，依然具有重要价值。

---

## 🧠 Architecture | 架构

The pipeline follows a strict **Analysis -> Generation -> Verification** flow:
管线遵循严格的 **分析 -> 生成 -> 验证** 流程：

### 1. Semantic Analysis (Qwen2.5-VL) | 语义分析
*   **Input**: Single RGB Image. (单张 RGB 图像)
*   **Process**: The Vision-Language Model analyzes the image to extract: (视觉语言模型分析图像以提取：)
    *   Subject Description (e.g., "a wooden chair with velvet cushion") (主体描述)
    *   Material & Style (材质与风格)
*   **Output**: Context-aware text prompts. (上下文感知的文本提示词)

### 2. Candidate Generation (Qwen-Image-Edit) | 候选生成
*   **Process**: For each target view (Right, Left, Back), the model generates **$K$ candidates** (default $K=4$) using different random seeds.
*   **流程**: 对于每个目标视角（右、左、后），模型使用不同的随机种子生成 **$K$ 个候选图**（默认 $K=4$）。
*   **Why Candidates?**: T2I models are stochastic. Generating multiple candidates increases the probability of getting a geometrically consistent result.
*   **为什么需要候选？**: 文生图模型具有随机性。生成多个候选图可以增加获得几何一致结果的概率。

### 3. Consistency Verification (The Core) | 一致性验证 (核心)
We employ a composite metric to select the best candidate:
我们采用复合指标来选择最佳候选：

$$ Score = w_1 \cdot S_{CLIP} + w_2 \cdot S_{LPIPS} + w_3 \cdot S_{Warp} $$

*   **CLIP Score**: Ensures semantic consistency (is it still a "chair"?).
    *   **CLIP 分数**: 确保语义一致性（它还是一把“椅子”吗？）。
*   **Global LPIPS**: Ensures perceptual style consistency (texture, lighting).
    *   **全局 LPIPS**: 确保感知风格一致性（纹理、光照）。
*   **Depth-Aware Warp**:
    *   Use **Depth Anything V2** to estimate the depth of the Input View. (使用 Depth Anything V2 估计输入视图的深度)
    *   Project pixels to 3D space. (将像素投影到 3D 空间)
    *   (Conceptual) Warp the view to check geometric plausibility. (概念上：通过 Warp 变换检查几何合理性)

---

## ⚙️ Usage | 使用方法

### Prerequisites | 前置要求
Install additional dependencies in `qwen_edit` environment:
在 `qwen_edit` 环境中安装额外依赖：
```bash
pip install lpips imageio-ffmpeg
```

### Run Command | 运行命令
```bash
bash run_mv_legacy_full.sh <image_path> <job_name> [candidates_num]
```

**Example**:
```bash
bash run_mv_legacy_full.sh examples/demo.png test_legacy_full 4
```

---

## 📊 Comparison: Legacy vs. Smart-AR | 对比

| Feature | **Smart-AR (Qwen)** | **Legacy Full (This)** |
| :--- | :--- | :--- |
| **Method** | Autoregressive (View-to-View) <br> 自回归 (视角到视角) | Parallel Generation + Selection <br> 并行生成 + 筛选 |
| **Context** | "Sees" previous generated views <br> “看到”之前生成的视角 | "Sees" only Front view <br> 仅“看到”正面视图 |
| **Consistency** | **High** (Inherits structure) <br> **高** (继承结构) | Variable (Depends on selection) <br> 不定 (依赖筛选) |
| **Speed** | Fast (1 pass per view) <br> 快 (每视角生成一次) | Slow (K passes + Filtering) <br> 慢 (K 次生成 + 过滤) |
| **Metric** | Implicit (Attention) <br> 隐式 (注意力机制) | Explicit (CLIP/LPIPS) <br> 显式 (CLIP/LPIPS 指标) |

**Conclusion**: Use **Smart-AR** for production/quality. Use **Legacy Full** for research comparisons or when strictly enforcing geometric metrics.
**结论**: 生产环境/追求质量请使用 **Smart-AR**。用于学术对比或需要严格几何指标时使用 **Legacy Full**。
