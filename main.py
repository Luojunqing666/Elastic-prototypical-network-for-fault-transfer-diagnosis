"""
Main entry point for EProtoNet: Elastic Prototypical Network for fault transfer diagnosis.

Usage:
    # Train EProtoNet (with elastic distance)
    python main.py --method protonet --train --data_dir /path/to/data \
        --ways 5 --shots 5 --epochs 100

    # Test EProtoNet
    python main.py --method protonet --test --data_dir /path/to/data \
        --load_path ./results/eprotonet_best.pt --ways 5 --shots 5

    # Train MAML baseline
    python main.py --method maml --train --data_dir /path/to/data \
        --ways 5 --shots 5 --epochs 100

    # Train RelationNet baseline
    python main.py --method relation --train --data_dir /path/to/data \
        --ways 5 --shots 5 --epochs 100
"""

import argparse
import os

from datasets import CWRUConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="EProtoNet: Meta-learning with elastic prototypical network for fault transfer diagnosis"
    )

    # Method selection
    parser.add_argument("--method", type=str, default="protonet",
                        choices=["protonet", "maml", "relation"],
                        help="Meta-learning method (default: protonet).")

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the model.")
    mode_group.add_argument("--test", action="store_true", help="Test the model.")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the dataset (containing CSV files).")
    parser.add_argument("--source_files", type=str, nargs='+', default=None,
                        help="List of source domain CSV file paths (for training).")
    parser.add_argument("--target_files", type=str, nargs='+', default=None,
                        help="List of target domain CSV file paths (for testing).")

    # Meta-learning hyperparameters
    parser.add_argument("--ways", type=int, default=5, help="Number of classes per task.")
    parser.add_argument("--shots", type=int, default=5, help="Number of support samples per class.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per epoch.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed.")

    # EProtoNet specific
    parser.add_argument("--no_elastic", action="store_true",
                        help="Disable elastic distance (use standard Euclidean).")
    parser.add_argument("--scale_factor", type=float, default=0.01,
                        help="Elasticity scale factor for distance metric.")

    # Model I/O
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save checkpoints.")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load model for testing.")

    return parser.parse_args()


def build_config(args):
    """Build dataset configuration from arguments."""
    if args.source_files and args.target_files:
        return CWRUConfig(
            data_dir=args.data_dir,
            source_files=args.source_files,
            target_files=args.target_files,
        )
    else:
        # Default: look for CSV files in data_dir/source/ and data_dir/target/
        source_dir = os.path.join(args.data_dir, "source")
        target_dir = os.path.join(args.data_dir, "target")

        if os.path.isdir(source_dir) and os.path.isdir(target_dir):
            source_files = sorted([
                os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.csv')
            ])
            target_files = sorted([
                os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.csv')
            ])
        else:
            # Fallback: all CSV files in data_dir, split by naming convention
            all_files = sorted([
                os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.csv')
            ])
            # Use all files as both source and target (user should specify explicitly)
            source_files = all_files
            target_files = all_files

        assert len(source_files) > 0, f"No CSV files found in {source_dir}"
        assert len(target_files) > 0, f"No CSV files found in {target_dir}"

        return CWRUConfig(
            data_dir=args.data_dir,
            source_files=source_files,
            target_files=target_files,
        )


def main():
    args = parse_args()
    config = build_config(args)

    print(f"Method: {args.method}")
    print(f"Source classes: {config.num_source_classes}, Target classes: {config.num_target_classes}")

    if args.method == "protonet":
        from train_protonet import train, test

        if args.train:
            train(config, save_dir=args.save_dir, ways=args.ways, shots=args.shots,
                  epochs=args.epochs, episodes=args.episodes, lr=args.lr,
                  use_elastic=not args.no_elastic, seed=args.seed)
        elif args.test:
            assert args.load_path, "--load_path is required for testing."
            test(config, load_path=args.load_path, ways=args.ways, shots=args.shots,
                 use_elastic=not args.no_elastic)

    elif args.method == "maml":
        from train_maml import train, test

        if args.train:
            train(config, save_dir=args.save_dir, ways=args.ways, shots=args.shots,
                  epochs=args.epochs, lr=args.lr, seed=args.seed)
        elif args.test:
            assert args.load_path, "--load_path is required for testing."
            test(config, load_path=args.load_path, ways=args.ways, shots=args.shots)

    elif args.method == "relation":
        from train_relation import train, test

        if args.train:
            train(config, save_dir=args.save_dir, ways=args.ways, shots=args.shots,
                  epochs=args.epochs, episodes=args.episodes, lr=args.lr, seed=args.seed)
        elif args.test:
            assert args.load_path, "--load_path is required for testing."
            test(config, load_path=args.load_path, ways=args.ways, shots=args.shots)


if __name__ == "__main__":
    main()
