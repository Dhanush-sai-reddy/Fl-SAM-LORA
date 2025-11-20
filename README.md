# FL-SAM-LoRA: Simple Federated Learning with SAM+LoRA on KiTS

**Just FL + SAM + LoRA on FedKits. Nothing complex.**

Federated LoRA-tuned Segment Anything Model for medical image segmentation.

## What This Does

- ✅ **Federated Learning** using Flower framework
- ✅ **SAM (Segment Anything Model)** with LoRA fine-tuning
- ✅ **KiTS dataset** (kidney tumor segmentation) split across 5 clients
- ✅ **Privacy-preserving** - data stays on each client

## Quick Start (3 Steps)

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Download SAM checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. Run FL
```bash
# Preprocess KiTS dataset first
python preprocess_data.py \
    --raw_root /path/to/kits_raw \
    --output_root ./data/kits_preprocessed \
    --num_clients 5

# Need the KiTS dataset?
# (Colab/Kaggle quick helper)
pip install kagglehub
python - <<'PY'
import kagglehub

path = kagglehub.dataset_download("orvile/kits19-png-zipped")
print("Path to dataset files:", path)
PY

# Run federated learning
python flower_sim.py \
    --sam_checkpoint sam_vit_b_01ec64.pth \
    --data_root ./data/kits_preprocessed \
    --num_clients 5 \
    --num_rounds 10
```

## Files (Simple Structure)

- `flower_sim.py` - Main FL script (uses Flower)
- `lora_sam.py` - LoRA injection into SAM
- `dataset_kits.py` - KiTS dataset handling
- `train_utils.py` - Training loop & metrics
- `preprocess_data.py` - Dataset preprocessing helper

## How It Works

1. **LoRA** adds trainable adapters to SAM (only 1-2% of params)
2. **Flower** simulates 5 clients doing local training
3. **FedAvg** aggregates LoRA weights from all clients
4. Repeat for N rounds

## Configuration

```bash
python flower_sim.py \
    --sam_checkpoint <path>        # SAM weights
    --data_root <path>             # Preprocessed KiTS data
    --num_clients 5                # Number of clients
    --num_rounds 10                # FL rounds
    --batch_size 2                 # Batch size per client
    --learning_rate 1e-4           # Learning rate
    --lora_rank 4                  # LoRA rank (smaller = fewer params)
```

## Troubleshooting

**Out of memory?** → Use `--batch_size 1`  
**Too slow?** → Use `--max_steps_per_epoch 5` for testing  
**No GPU?** → Works on CPU (slower)

## That's It!

No heavy infrastructure, just straightforward FL with SAM+LoRA on medical images.
