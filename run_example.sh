#!/bin/bash

# Example script to run federated learning with SAM+LoRA on Fed-KITS
# This script demonstrates the complete workflow

set -e

echo "=========================================="
echo "Federated Learning with SAM+LoRA"
echo "=========================================="
echo

# Configuration
SAM_CHECKPOINT="sam_vit_b_01ec64.pth"
RAW_DATA_DIR="path/to/kits_raw"  # Change this to your KiTS dataset path
PREPROCESSED_DIR="./data/kits_preprocessed"
NUM_CLIENTS=5
NUM_ROUNDS=10

# Step 1: Download SAM checkpoint (if not exists)
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Step 1: Downloading SAM checkpoint..."
    echo "----------------------------------------"
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo "✓ Downloaded SAM checkpoint"
    echo
else
    echo "✓ SAM checkpoint already exists"
    echo
fi

# Step 2: Preprocess dataset (if not already done)
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "Step 2: Preprocessing KiTS dataset..."
    echo "----------------------------------------"
    python preprocess_data.py \
        --raw_root "$RAW_DATA_DIR" \
        --output_root "$PREPROCESSED_DIR" \
        --num_clients $NUM_CLIENTS \
        --train_frac 0.8 \
        --seed 13
    echo
else
    echo "✓ Preprocessed data already exists"
    echo
fi

# Step 3: Run federated learning
echo "Step 3: Running federated learning..."
echo "----------------------------------------"
python flower_sim.py \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --data_root "$PREPROCESSED_DIR" \
    --model_type vit_b \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --lora_alpha 16.0 \
    --output_dir ./fl_output

echo
echo "=========================================="
echo "Federated learning completed!"
echo "=========================================="
