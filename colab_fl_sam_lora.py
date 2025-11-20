import argparse

from fedavg_numpy import run_fedavg

try:
    import fed_kits  # noqa: F401
    HAS_FED_KITS = True
except ImportError:
    HAS_FED_KITS = False


def main():
    parser = argparse.ArgumentParser(
        description="Colab entrypoint for running FedAvg on Fed-KITS with FL-SAM-LoRA utilities",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Path to Fed-KITS NPZ data (client_*.npz). "
            "If omitted, fedavg_numpy will fall back to synthetic data."
        ),
    )
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--fraction_clients", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1)

    args = parser.parse_args()

    if HAS_FED_KITS:
        print("[Colab] Successfully imported fed_kits. Make sure your Fed-KITS NPZ root is passed via --root.")
    else:
        print("[Colab] Warning: fed_kits is not installed. Install it in Colab with `!pip install fed-kits`.")

    final_metrics = run_fedavg(
        root=args.root,
        num_rounds=args.num_rounds,
        fraction_clients=args.fraction_clients,
        lr=args.lr,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        l2=args.l2,
        seed=args.seed,
        eval_every=args.eval_every,
    )

    print("[Colab] Training finished. Final metrics:")
    print(final_metrics)


if __name__ == "__main__":
    main()
