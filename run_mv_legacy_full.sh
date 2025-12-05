#!/bin/bash
set -e

INPUT_IMAGE=$1
JOB_NAME=${2:-"legacy_full_job"}
CANDIDATES=${3:-4}

if [ -z "$INPUT_IMAGE" ]; then
    echo "Usage: $0 <input_image> [job_name] [candidates_num]"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/${JOB_NAME}_${TIMESTAMP}"

echo "=================================================="
echo "Legacy FULL Pipeline (Analysis + Depth + Warp)"
echo "Input: $INPUT_IMAGE"
echo "Output: $OUTPUT_DIR"
echo "=================================================="

# 1. 激活 Qwen 环境 (含 lpips/transformers/diffusers)
if [ -f "/cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh" ]; then
    source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate qwen_edit

# 2. 运行 Python 完整管线
python scripts/generate_mv_legacy_full.py \
    --input_image "$INPUT_IMAGE" \
    --output_dir "$OUTPUT_DIR/views" \
    --candidates "$CANDIDATES"

# 3. 重建 3D
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

mv "outputs/mv_assets/result.glb" "$OUTPUT_DIR/final_model.glb" 2>/dev/null || true

echo "Done. Results: $OUTPUT_DIR"

