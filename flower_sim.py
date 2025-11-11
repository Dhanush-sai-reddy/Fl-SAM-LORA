import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from lora_sam import inject_lora_into_sam, get_lora_parameters, save_lora_weights, load_lora_weights
from dataset_kits import load_client_dataset
from train_utils import KITSDataset, train_epoch, evaluate


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_lora_params_as_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract only LoRA parameters as numpy arrays"""
    if hasattr(model, "lora_layers"):
        return [p.detach().cpu().numpy() for p in model.lora_layers.parameters()]
    return []


def set_lora_params_from_numpy(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Set LoRA parameters from numpy arrays"""
    if hasattr(model, "lora_layers"):
        lora_params = list(model.lora_layers.parameters())
        for param, new_value in zip(lora_params, parameters):
            param.data = torch.tensor(new_value, dtype=param.dtype, device=param.device)


class SamClient(fl.client.NumPyClient):
    def __init__(self, cid: str, args: argparse.Namespace):
        self.cid = cid
        self.args = args
        self.device = get_device()
        
        print(f"[Client {cid}] Initializing SAM with LoRA...")
        
        # Initialize SAM model
        sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
        sam.to(device=self.device)
        
        # Inject LoRA layers
        sam = inject_lora_into_sam(
            sam,
            target_modules=["qkv"],  # Add LoRA to attention qkv projections
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        
        # Freeze all non-LoRA parameters
        for name, param in sam.named_parameters():
            param.requires_grad = False
        
        # Only LoRA parameters are trainable
        lora_params = get_lora_parameters(sam)
        for param in lora_params:
            param.requires_grad = True
        
        self.model = sam
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            get_lora_parameters(self.model),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # Load local dataset
        self.train_items = []
        self.val_items = []
        if args.data_root and os.path.isdir(args.data_root):
            try:
                self.train_items = load_client_dataset(
                    args.data_root, 
                    client_id=self.cid, 
                    mode="train", 
                    slice_axis=args.slice_axis
                )
                self.val_items = load_client_dataset(
                    args.data_root, 
                    client_id=self.cid, 
                    mode="val", 
                    slice_axis=args.slice_axis
                )
                print(f"[Client {cid}] Loaded {len(self.train_items)} train, {len(self.val_items)} val samples")
            except Exception as e:
                print(f"[Client {cid}] Could not load data: {e}")
        
        # Create data loaders
        self.train_loader = None
        self.val_loader = None
        if len(self.train_items) > 0:
            train_dataset = KITSDataset(self.train_items)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
        if len(self.val_items) > 0:
            val_dataset = KITSDataset(self.val_items)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

    # Flower API
    def get_parameters(self, config: Dict[str, str]):
        return get_lora_params_as_numpy(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]):
        # Set parameters from server
        set_lora_params_from_numpy(self.model, parameters)
        
        # Train locally
        if self.train_loader is not None and len(self.train_items) > 0:
            print(f"[Client {self.cid}] Training for {self.args.local_epochs} epochs...")
            for epoch in range(self.args.local_epochs):
                metrics = train_epoch(
                    self.model,
                    self.train_loader,
                    self.optimizer,
                    self.device,
                    max_steps=self.args.max_steps_per_epoch,
                )
                print(f"[Client {self.cid}] Epoch {epoch+1}: loss={metrics['loss']:.4f}, dice={metrics['dice']:.4f}")
        else:
            print(f"[Client {self.cid}] No training data available")
        
        # Return updated parameters
        return get_lora_params_as_numpy(self.model), len(self.train_items), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]):
        # Set parameters from server
        set_lora_params_from_numpy(self.model, parameters)
        
        # Evaluate
        if self.val_loader is not None and len(self.val_items) > 0:
            metrics = evaluate(
                self.model,
                self.val_loader,
                self.device,
                max_steps=self.args.max_eval_steps,
            )
            print(f"[Client {self.cid}] Eval: loss={metrics['loss']:.4f}, dice={metrics['dice']:.4f}, iou={metrics['iou']:.4f}")
            return metrics["loss"], len(self.val_items), {"dice": metrics["dice"], "iou": metrics["iou"]}
        else:
            print(f"[Client {self.cid}] No validation data available")
            return 0.0, 0, {"dice": 0.0, "iou": 0.0}


def client_fn(cid: str):
    # Access the global args captured at parse time
    return SamClient(cid=cid, args=GLOBAL_ARGS)


def main():
    parser = argparse.ArgumentParser(description="Federated Learning with SAM+LoRA on Fed-KITS")
    
    # Model arguments
    parser.add_argument("--sam_checkpoint", type=str, required=True, 
                        help="Path to SAM checkpoint file")
    parser.add_argument("--model_type", type=str, default="vit_b", 
                        choices=["vit_b", "vit_l", "vit_h"], 
                        help="SAM model type")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=4, 
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, 
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, 
                        help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Preprocessed data root with index.json and splits.json")
    parser.add_argument("--slice_axis", type=int, default=2, 
                        help="Slice axis for 3D volumes (default: 2)")
    
    # Federated learning arguments
    parser.add_argument("--num_clients", type=int, default=5, 
                        help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=10, 
                        help="Number of federated rounds")
    parser.add_argument("--fraction_fit", type=float, default=1.0,
                        help="Fraction of clients to sample for training")
    parser.add_argument("--fraction_evaluate", type=float, default=1.0,
                        help="Fraction of clients to sample for evaluation")
    
    # Training arguments
    parser.add_argument("--local_epochs", type=int, default=1, 
                        help="Number of local epochs per round")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="Maximum steps per epoch (for quick testing)")
    parser.add_argument("--max_eval_steps", type=int, default=None,
                        help="Maximum evaluation steps (for quick testing)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./fl_output",
                        help="Output directory for checkpoints")
    
    args = parser.parse_args()

    # Validate input paths
    if not os.path.isfile(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    global GLOBAL_ARGS
    GLOBAL_ARGS = args
    
    print(f"\n{'='*60}")
    print("Federated Learning Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_type}")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"{'='*60}\n")

    # Define federated learning strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=int(args.num_clients * args.fraction_fit),
        min_evaluate_clients=int(args.num_clients * args.fraction_evaluate),
        min_available_clients=args.num_clients,
    )

    # Start federated learning simulation
    print("Starting Flower simulation...\n")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.2 if torch.cuda.is_available() else 0,
        },
    )
    
    print("\n" + "="*60)
    print("Federated learning completed!")
    print("="*60)


if __name__ == "__main__":
    main()
