#!/bin/bash
#SBATCH --job-name=minicrit-7b
#SBATCH --output=logs/minicrit_%j.out
#SBATCH --error=logs/minicrit_%j.err
#SBATCH --partition=gh200
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================
# MiniCrit-7B Training Script for TACC Vista GH200
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
#
# Vista GH200 Specs (per node):
#   - 1x NVIDIA GH200 Grace Hopper Superchip
#   - 96GB HBM3 GPU memory
#   - 72 ARM Neoverse V2 CPU cores
#   - 480GB LPDDR5X memory
#   - 900 GB/s NVLink-C2C bandwidth
#
# Usage:
#   sbatch scripts/train_vista.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/minicrit_<jobid>.out
# ============================================================

echo "============================================================"
echo "MiniCrit-7B Training Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 1 (GH200)"
echo "Start time: $(date)"
echo "============================================================"

# ---- Environment Setup ----
module purge
module load gcc/13.1.0
module load cuda/12.3
module load python3/3.11
module load conda

# Activate conda environment (create if needed)
ENV_NAME="minicrit"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.11 -y
fi
conda activate $ENV_NAME

# Install dependencies (first run only)
if [ ! -f ".deps_installed" ]; then
    echo "Installing dependencies..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets peft accelerate wandb
    pip install deepspeed
    pip install rouge-score bert-score
    touch .deps_installed
fi

# ---- Distributed Training Setup ----
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NNODES
export RANK=$SLURM_PROCID
export LOCAL_RANK=0

# NCCL optimizations for GH200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Weights & Biases
export WANDB_PROJECT="minicrit-training"
export WANDB_RUN_ID="vista-${SLURM_JOB_ID}"
export WANDB_RESUME="allow"

# ---- Paths ----
export MINICRIT_DATA_FILE="/scratch/$USER/minicrit/minicrit_11.7M_CLEAN.csv"
export MINICRIT_CACHE_DIR="/scratch/$USER/minicrit/tokenized_cache"
export MINICRIT_OUTPUT_DIR="/scratch/$USER/minicrit/output_${SLURM_JOB_ID}"
export MINICRIT_CHECKPOINT_DIR="$MINICRIT_OUTPUT_DIR"

# Create directories
mkdir -p $MINICRIT_CACHE_DIR
mkdir -p $MINICRIT_OUTPUT_DIR
mkdir -p logs

# ---- Launch Training ----
echo ""
echo "Launching distributed training..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo ""

# Use torchrun for multi-node training
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_minicrit_7b.py \
    --config configs/7b_lora.yaml \
    --resume-latest

# ---- Post-Training ----
TRAIN_EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Copy final model to $WORK for persistence
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Copying final model to \$WORK..."
    FINAL_DIR="$WORK/minicrit/models/minicrit-7b-$(date +%Y%m%d)"
    mkdir -p $FINAL_DIR
    cp -r $MINICRIT_OUTPUT_DIR/minicrit-7b-final $FINAL_DIR/
    echo "Model saved to: $FINAL_DIR"
fi

exit $TRAIN_EXIT_CODE
