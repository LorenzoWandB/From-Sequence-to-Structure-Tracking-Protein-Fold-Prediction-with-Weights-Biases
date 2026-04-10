#!/bin/bash
# ── ESMFold v2 training — larger dataset, better curves ──
# Adjust partition, GPU type, and paths for your cluster.

#SBATCH --job-name=proteus-v2
#SBATCH --output=logs/train-v2-%j.out
#SBATCH --error=logs/train-v2-%j.err
#SBATCH --partition=gpu             # ← your GPU partition name (e.g. h100, a100, gpu)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e
mkdir -p logs

echo "=== ProteusAI v2 Training (Large Dataset) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd ~/proteus-fold  # ← path to your cloned repo on the cluster

# Step 1: Download large dataset if not already present
if [ ! -f data_large/manifest.json ]; then
    echo "Downloading large dataset..."
    uv run python download_data_large.py
fi

# Phase 1 (inference) — uncomment to run baseline predictions first
# uv run python train_esm.py --mode inference --max-seq-len 128 --data-dir data_large

# Phase 2: Fine-tune with frozen backbone (compare learning rates)
for LR in 5e-5 1e-4; do
    echo ""
    echo "=== Phase 2: Frozen backbone lr=${LR} ==="
    uv run python train_esm.py --mode finetune \
        --run-name "v2-frozen-lr${LR}-${SLURM_JOB_ID}" \
        --max-seq-len 128 \
        --epochs 30 \
        --lr $LR \
        --num-recycles 2 \
        --data-dir data_large
done

# Phase 3: Unfreeze last 4 ESM-2 layers
echo ""
echo "=== Phase 3: Unfreeze 4 layers ==="
uv run python train_esm.py --mode finetune \
    --run-name "v2-unfreeze4-${SLURM_JOB_ID}" \
    --max-seq-len 128 \
    --epochs 30 \
    --lr 3e-5 \
    --unfreeze-backbone --unfreeze-layers 4 \
    --num-recycles 2 \
    --data-dir data_large

# Phase 4: Production model with more recycling
echo ""
echo "=== Phase 4: Production ==="
uv run python train_esm.py --mode finetune \
    --run-name "v2-production-${SLURM_JOB_ID}" \
    --max-seq-len 128 \
    --epochs 40 \
    --lr 2e-5 \
    --unfreeze-backbone --unfreeze-layers 4 \
    --num-recycles 3 \
    --data-dir data_large

echo ""
echo "Done — check your W&B project for results"
