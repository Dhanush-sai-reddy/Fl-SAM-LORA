# Quick Start Guide

Get up and running with Federated Learning + SAM + LoRA on Fed-KITS in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- 16GB+ RAM
- KiTS dataset downloaded

## Step-by-Step Setup

### 1. Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

### 2. Download SAM Checkpoint (2 minutes)

Choose one:

**ViT-B (Recommended for testing - 375MB)**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**ViT-L (Better performance - 1.2GB)**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

**ViT-H (Best performance - 2.4GB)**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 3. Prepare Dataset (5-10 minutes)

```bash
python preprocess_data.py \
    --raw_root /path/to/kits_raw \
    --output_root ./data/kits_preprocessed \
    --num_clients 5
```

### 4. Run Federated Learning (30+ minutes)

**Quick Test (3 rounds, ~5 minutes)**
```bash
python flower_sim.py \
    --sam_checkpoint sam_vit_b_01ec64.pth \
    --data_root ./data/kits_preprocessed \
    --num_clients 5 \
    --num_rounds 3 \
    --batch_size 2 \
    --max_steps_per_epoch 5
```

**Full Training (10 rounds, ~30 minutes)**
```bash
python flower_sim.py \
    --sam_checkpoint sam_vit_b_01ec64.pth \
    --data_root ./data/kits_preprocessed \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-4
```

## Expected Output

```
============================================================
Federated Learning Configuration
============================================================
Model: vit_b
Clients: 5
Rounds: 10
Local epochs: 1
Batch size: 2
Learning rate: 0.0001
LoRA rank: 4, alpha: 16.0
============================================================

[Client 0] Initializing SAM with LoRA...
[Client 0] Loaded 120 train, 30 val samples
[Client 0] Training for 1 epochs...
[Client 0] Epoch 1: loss=0.4523, dice=0.6234
[Client 0] Eval: loss=0.4312, dice=0.6456, iou=0.5234

...

============================================================
Federated learning completed!
============================================================
```

## Common Issues

### Out of Memory
```bash
# Reduce batch size
--batch_size 1

# Or use smaller model
--model_type vit_b
```

### Slow Training
```bash
# Limit steps for testing
--max_steps_per_epoch 10 --max_eval_steps 5

# Or reduce local epochs
--local_epochs 1
```

### No GPU
The code automatically uses CPU if GPU is unavailable. Training will be slower but functional.

## What's Happening?

1. **Round 1-N**: Each client trains locally on their data partition
2. **Aggregation**: Server averages LoRA parameters using FedAvg
3. **Distribution**: Updated model sent back to all clients
4. **Evaluation**: Clients evaluate on validation data
5. **Repeat**: Process continues for specified rounds

## Next Steps

- Adjust hyperparameters for better performance
- Increase number of rounds (10-50)
- Experiment with LoRA rank (4, 8, 16)
- Try different learning rates (1e-5 to 1e-3)
- Use validation metrics to save best model

## Performance Tips

1. **GPU Memory**: Reduce batch size if OOM occurs
2. **Training Speed**: Use ViT-B for faster experiments
3. **Convergence**: Monitor Dice coefficient (target: >0.7)
4. **Data Quality**: Ensure preprocessing completed successfully

## Help & Support

For issues or questions:
1. Check the full [README.md](README.md)
2. Verify dataset preprocessing completed
3. Check GPU memory usage
4. Try with smaller batch size first

Happy federated learning! 🚀
