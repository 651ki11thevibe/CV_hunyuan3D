#!/bin/bash
set -e # 遇到错误立即停止

# ================= 配置区域 =================
# 提示词 (多视图生成现在由 python 脚本内部控制，这里主要控制输入图像)
INPUT_IMAGE=$1
JOB_NAME=${2:-"mv_job"}
MODE=${3:-"simple_45"} # 默认使用 simple_45 模式 (节省时间，效果也不错)

if [ -z "$INPUT_IMAGE" ]; then
    echo "Usage: $0 <input_image_path> [job_name] [mode: simple_45|full]"
    exit 1
fi

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 创建带有时间戳的独立输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/${JOB_NAME}_${TIMESTAMP}"
MV_DIR="$OUTPUT_DIR/views" # 专门存放多视图图片的子目录
mkdir -p "$MV_DIR"

echo "========================================================="
echo "Job Started: $JOB_NAME"
echo "Mode: $MODE"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================="

# ================= 第一步：Qwen 多视图生成 =================
echo ">>> Step 1: Generating Multi-View Images with Qwen (Env: qwen_edit)"

# 尝试加载 conda 环境
if [ -f "/cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh" ]; then
    source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 激活编辑环境
conda activate qwen_edit

# 运行带有 Caption 的 AR 多视图生成
echo "   Using Smart Strategy: Caption -> AR Generation (Mode: $MODE)"

python scripts/generate_qwen_ar_with_caption.py \
  --input_image "$INPUT_IMAGE" \
  --output_dir "$MV_DIR" \
  --vl_model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --edit_model_id "Qwen/Qwen-Image-Edit-2509" \
  --mode "$MODE"

# 检查生成结果
if [ "$MODE" == "simple_45" ]; then
    if [ ! -f "$MV_DIR/front.png" ] || [ ! -f "$MV_DIR/right_45.png" ]; then
        echo "Error: Simple 45-deg generation failed."
        exit 1
    fi
    echo "   [√] Generated views (front, right_45) in: $MV_DIR"
else
    # Full mode check
    if [ ! -f "$MV_DIR/front.png" ] || [ ! -f "$MV_DIR/left.png" ]; then
        echo "Error: Full multi-view generation failed."
        exit 1
    fi
    echo "   [√] Generated full views in: $MV_DIR"
fi


# ================= 第二步：Hunyuan3D 多视图生成 =================
echo ">>> Step 2: Running Hunyuan3D Multi-View Generation (Env: cv3d)"

# 切换环境
conda deactivate
conda activate cv3d

if [ "$MODE" == "simple_45" ]; then
    # Simple Mode: 只传 Front 和 Right (这里的 Right 其实是 45 度图，作为辅助视角)
    # 注意：Hunyuan3D 的 MV 模型通常把侧视图作为 'right' 或 'left' 输入
    python scripts/run_mv_image2asset.py \
        --config configs/hunyuan3d_mv.yaml \
        --front "$MV_DIR/front.png" \
        --right "$MV_DIR/right_45.png" \
        --name "result" \
        --format glb
else
    # Full Mode: 传所有 4 张图
    python scripts/run_mv_image2asset.py \
        --config configs/hunyuan3d_mv.yaml \
        --front "$MV_DIR/front.png" \
        --left "$MV_DIR/left.png" \
        --right "$MV_DIR/right.png" \
        --back "$MV_DIR/back.png" \
        --name "result" \
        --format glb
fi

# 移动生成结果到统一文件夹
if [ -f "outputs/mv_assets/result.glb" ]; then
    mv "outputs/mv_assets/result.glb" "$OUTPUT_DIR/final_model.glb"
    mv "outputs/mv_assets/result_metadata.json" "$OUTPUT_DIR/metadata.json" 2>/dev/null || true
    echo "   [√] Final 3D model moved to: $OUTPUT_DIR/final_model.glb"
else
    echo "Warning: Could not find generated GLB in expected path. Check logs above."
fi

echo "========================================================="
echo "All Done! Results are in: $OUTPUT_DIR"
echo "========================================================="
