#!/bin/bash
#SBATCH --job-name=qwen_hy3d_pipeline
#SBATCH --output=logs/%j_out.log
#SBATCH --error=logs/%j_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaiesuperpod

# ================= 用户配置区域 =================
# 输入图片路径
INPUT_IMAGE="examples/demo.png"
# 任务名称（用于输出文件夹命名）
JOB_NAME="batch_job_01"
# ===========================================

# 加载 conda 环境配置
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
# 或者如果你的 conda 在其他地方，请取消注释并修改下面这行：
# source ~/miniconda3/etc/profile.d/conda.sh

# 确保日志目录存在
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"

# 进入脚本目录
cd /home/ywangqy/img23d/CV_hunyuan3D

# 运行完整的 Pipeline 脚本
# 注意：run_full_pipeline.sh 内部会自动处理 conda 环境的切换 (qwen_edit <-> cv3d)
# 我们只需要直接调用它即可
bash run_full_pipeline.sh "$INPUT_IMAGE" "$JOB_NAME"

echo "End Time: $(date)"


