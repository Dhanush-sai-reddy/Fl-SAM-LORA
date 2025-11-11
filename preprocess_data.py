"""
Helper script to preprocess KiTS dataset and create federated splits
"""
import argparse
import os
from dataset_kits import preprocess_kits, make_federated_splits


def main():
    parser = argparse.ArgumentParser(description="Preprocess KiTS dataset for federated learning")
    
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Path to raw KiTS dataset (contains case_* directories)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to output directory for preprocessed data")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Number of federated clients")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of data for training (rest for validation)")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.5, 1.5, 3.0],
                        help="Target spacing for resampling (x, y, z)")
    parser.add_argument("--intensity_clip", type=float, nargs=2, default=[-200.0, 300.0],
                        help="Intensity clipping range (min, max)")
    parser.add_argument("--seed", type=int, default=13,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.raw_root):
        raise FileNotFoundError(f"Raw data directory not found: {args.raw_root}")
    
    print("="*60)
    print("KiTS Dataset Preprocessing")
    print("="*60)
    print(f"Raw data: {args.raw_root}")
    print(f"Output: {args.output_root}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Train fraction: {args.train_frac}")
    print(f"Target spacing: {args.target_spacing}")
    print(f"Intensity clip: {args.intensity_clip}")
    print("="*60)
    print()
    
    # Step 1: Preprocess raw data
    print("Step 1: Preprocessing raw NIfTI files...")
    print("-"*60)
    index = preprocess_kits(
        raw_root=args.raw_root,
        out_root=args.output_root,
        target_spacing=tuple(args.target_spacing),
        intensity_clip=tuple(args.intensity_clip),
        seed=args.seed,
    )
    print(f"✓ Preprocessed {len(index['cases'])} cases")
    print(f"✓ Saved to: {args.output_root}")
    print()
    
    # Step 2: Create federated splits
    print("Step 2: Creating federated splits...")
    print("-"*60)
    splits = make_federated_splits(
        preprocessed_root=args.output_root,
        num_clients=args.num_clients,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    
    # Print split statistics
    print(f"✓ Created splits for {splits['num_clients']} clients")
    print()
    print("Client statistics:")
    print("-"*60)
    for cid, data in splits["clients"].items():
        n_train = len(data["train"])
        n_val = len(data["val"])
        print(f"  Client {cid}: {n_train} train, {n_val} val samples")
    
    print()
    print("="*60)
    print("Preprocessing completed successfully!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Download a SAM checkpoint:")
    print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    print()
    print("2. Run federated learning:")
    print(f"   python flower_sim.py \\")
    print(f"       --sam_checkpoint sam_vit_b_01ec64.pth \\")
    print(f"       --data_root {args.output_root} \\")
    print(f"       --num_clients {args.num_clients} \\")
    print(f"       --num_rounds 10")
    print()


if __name__ == "__main__":
    main()
