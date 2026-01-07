#!/bin/bash
# ============================================================
# TACC Vista Setup Script for MiniCrit Training
# Antagon Inc. | CAGE: 17E75
#
# Run this once to set up the environment:
#   bash scripts/vista_setup.sh
# ============================================================

set -e

echo "Setting up MiniCrit environment on TACC Vista..."

# Load modules
module purge
module load gcc/13.1.0
module load cuda/12.3
module load python3/3.11
module load conda

# Create conda environment
ENV_NAME="minicrit"
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
echo "Installing training dependencies..."
pip install \
    transformers>=4.36.0 \
    datasets>=2.16.0 \
    peft>=0.7.0 \
    accelerate>=0.25.0 \
    wandb>=0.16.0 \
    pandas>=2.0.0 \
    pyarrow>=14.0.0

# Install distributed training tools
echo "Installing distributed training tools..."
pip install deepspeed

# Install evaluation metrics
echo "Installing evaluation dependencies..."
pip install rouge-score bert-score

# Create directories
echo "Creating directories..."
mkdir -p /scratch/$USER/minicrit/{data,tokenized_cache,output}
mkdir -p logs

# Copy data if available
if [ -f "minicrit_11.7M_CLEAN.csv" ]; then
    echo "Copying dataset to scratch..."
    cp minicrit_11.7M_CLEAN.csv /scratch/$USER/minicrit/data/
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload your dataset to /scratch/$USER/minicrit/data/"
echo "  2. Update YOUR_ALLOCATION in scripts/train_vista.slurm"
echo "  3. Submit job: sbatch scripts/train_vista.slurm"
echo "============================================================"
