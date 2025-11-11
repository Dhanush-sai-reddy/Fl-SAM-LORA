<<<<<<< HEAD
# Federated Learning with SAM+LoRA on Fed-KITS Dataset

This project implements federated learning using the Segment Anything Model (SAM) with Low-Rank Adaptation (LoRA) on the KiTS (Kidney Tumor Segmentation) dataset.

## Overview

- **Model**: SAM (Segment Anything Model) with LoRA adaptation
- **Framework**: Flower (federated learning)
- **Dataset**: Fed-KITS (Kidney Tumor Segmentation)
- **Clients**: 5 (configurable)
- **Task**: Medical image segmentation

## Features

- ✅ SAM with LoRA for parameter-efficient fine-tuning
- ✅ Federated learning with 5+ clients
- ✅ Fed-KITS dataset preprocessing and loading
- ✅ Dice loss and IoU metrics
- ✅ Flexible configuration options

## Project Structure

```
.
├── lora_sam.py           # LoRA implementation for SAM
├── train_utils.py        # Training utilities and loss functions
├── dataset_kits.py       # Fed-KITS dataset preprocessing
├── flower_sim.py         # Main federated learning script
├── sam_init.py           # SAM initialization utilities
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Clone the repository

```bash
cd "attempt fromscratch"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download SAM checkpoint

Download a SAM checkpoint (ViT-B, ViT-L, or ViT-H) from the official repository:

```bash
# ViT-B (smallest, fastest)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L (medium)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-H (largest, best performance)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Dataset Preparation

### 1. Download KiTS Dataset

Download the KiTS19 or KiTS21 dataset from:
- KiTS19: https://kits19.grand-challenge.org/
- KiTS21: https://kits21.kits-challenge.org/

### 2. Preprocess the dataset

```python
from dataset_kits import preprocess_kits, make_federated_splits

# Preprocess raw NIfTI files
raw_root = "path/to/kits_raw"  # Contains case_00000, case_00001, etc.
preprocessed_root = "path/to/kits_preprocessed"

preprocess_kits(
    raw_root=raw_root,
    out_root=preprocessed_root,
    target_spacing=(1.5, 1.5, 3.0),
    intensity_clip=(-200.0, 300.0),
)

# Create federated splits
make_federated_splits(
    preprocessed_root=preprocessed_root,
    num_clients=5,
    train_frac=0.8,
    seed=13
)
```

This will create:
- `index.json`: Index of all preprocessed cases
- `splits.json`: Federated data splits for each client

## Usage

### Basic Training

```bash
python flower_sim.py \
    --sam_checkpoint sam_vit_b_01ec64.pth \
    --data_root path/to/kits_preprocessed \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-4
```

### Advanced Configuration

```bash
python flower_sim.py \
    --sam_checkpoint sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --data_root path/to/kits_preprocessed \
    --num_clients 5 \
    --num_rounds 20 \
    --local_epochs 2 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_dropout 0.1 \
    --slice_axis 2 \
    --fraction_fit 1.0 \
    --fraction_evaluate 1.0 \
    --output_dir ./fl_output
```

### Quick Testing (with limited steps)

```bash
python flower_sim.py \
    --sam_checkpoint sam_vit_b_01ec64.pth \
    --data_root path/to/kits_preprocessed \
    --num_clients 5 \
    --num_rounds 3 \
    --local_epochs 1 \
    --batch_size 2 \
    --max_steps_per_epoch 5 \
    --max_eval_steps 3
```

## Configuration Options

### Model Arguments
- `--sam_checkpoint`: Path to SAM checkpoint file
- `--model_type`: SAM model type (vit_b, vit_l, vit_h)

### LoRA Arguments
- `--lora_rank`: LoRA rank (default: 4)
- `--lora_alpha`: LoRA alpha parameter (default: 16.0)
- `--lora_dropout`: LoRA dropout rate (default: 0.1)

### Data Arguments
- `--data_root`: Path to preprocessed dataset
- `--slice_axis`: Axis for slicing 3D volumes (default: 2)

### Federated Learning Arguments
- `--num_clients`: Number of clients (default: 5)
- `--num_rounds`: Number of federated rounds (default: 10)
- `--fraction_fit`: Fraction of clients for training (default: 1.0)
- `--fraction_evaluate`: Fraction of clients for evaluation (default: 1.0)

### Training Arguments
- `--local_epochs`: Local epochs per round (default: 1)
- `--batch_size`: Batch size (default: 2)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)

## How It Works

### 1. LoRA Adaptation

The LoRA (Low-Rank Adaptation) technique adapts the SAM model by adding trainable low-rank matrices to the attention layers:

```
h = W₀x + BAx × scaling
```

Where:
- `W₀` is the frozen pre-trained weight
- `B` and `A` are trainable low-rank matrices
- Only `B` and `A` are trained, making it parameter-efficient

### 2. Federated Learning

The Flower framework orchestrates federated learning:

1. **Initialization**: Each client initializes SAM with LoRA
2. **Local Training**: Clients train on local data
3. **Aggregation**: Server aggregates client updates using FedAvg
4. **Distribution**: Updated model sent back to clients
5. **Repeat**: Process repeats for multiple rounds

### 3. Data Distribution

The Fed-KITS dataset is split across clients:
- Each client gets a subset of cases
- Data is non-IID (naturally heterogeneous)
- Each case contains 3D CT volumes with kidney/tumor masks

## Evaluation Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Measures segmentation accuracy
- **Loss**: Combined BCE + Dice loss

## Output

The training produces:
- Console logs with per-round metrics
- Client-level training and evaluation metrics
- Aggregated global model performance

Example output:
```
[Client 0] Training for 1 epochs...
[Client 0] Epoch 1: loss=0.4523, dice=0.6234
[Client 0] Eval: loss=0.4312, dice=0.6456, iou=0.5234

Round 1: Global metrics aggregated
...
```

## Troubleshooting

### Out of Memory

Reduce batch size or use smaller model:
```bash
--batch_size 1 --model_type vit_b
```

### Slow Training

Use fewer steps for testing:
```bash
--max_steps_per_epoch 10 --max_eval_steps 5
```

### No GPU Available

The code automatically uses CPU if GPU is not available. Training will be slower.

## Citation

If you use this code, please cite:

```bibtex
@misc{kirillov2023segment,
    title={Segment Anything},
    author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Dollar, Piotr and Girshick, Ross},
    year={2023},
    journal={arXiv:2304.02643}
}

@inproceedings{beutel2020flower,
    title={Flower: A Friendly Federated Learning Research Framework},
    author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D},
    year={2020}
}
```

## License

This project is for research purposes. Please check the licenses of:
- SAM: Apache 2.0
- Flower: Apache 2.0
- KiTS Dataset: CC BY-NC-SA 4.0

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open a GitHub issue.
=======
# Fl-SAM-LORA
Federated LoRA-tuned Segment Anything Model for privacy-preserving visual learning across distributed clients
>>>>>>> 19fd3e49d4e42ad677d804f78bd4812ed80d6ad9
