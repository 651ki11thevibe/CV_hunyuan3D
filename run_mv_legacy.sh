#!/bin/bash
set -e

INPUT_IMAGE=$1
JOB_NAME=${2:-"legacy_job"}
CANDIDATES=${3:-2} # 默认每个视角生成 2 张图进行筛选

if [ -z "$INPUT_IMAGE" ]; then
    echo "Usage: $0 <input_image> [job_name] [candidates_num]"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/${JOB_NAME}_${TIMESTAMP}"

echo "========================================="
echo "Starting Legacy Multi-View Pipeline"
echo "Input: $INPUT_IMAGE"
echo "Output: $OUTPUT_DIR"
echo "Candidates per view: $CANDIDATES"
echo "========================================="

# 1. 激活环境 (Qwen Edit Environment)
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate qwen_edit

# 2. 运行生成
python scripts/generate_mv_legacy.py \
    --input_image "$INPUT_IMAGE" \
    --output_dir "$OUTPUT_DIR/views" \
    --candidates "$CANDIDATES"

# 3. (可选) 调用 Hunyuan3D 进行重建
# 这一步可以选择开启或关闭，Legacy Pipeline 重点在图的生成
echo ">> Generating 3D Model..."
conda deactivate
conda activate cv3d

python scripts/run_mv_image2asset.py \
    --config configs/hunyuan3d_mv.yaml \
    --front "$OUTPUT_DIR/views/front.png" \
    --right "$OUTPUT_DIR/views/right.png" \
    --left "$OUTPUT_DIR/views/left.png" \
    --back "$OUTPUT_DIR/views/back.png" \
    --name "result" \
    --format glb

# Move result
mv "outputs/mv_assets/result.glb" "$OUTPUT_DIR/model.glb" 2>/dev/null || true

echo "All Done."

