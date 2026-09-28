# generate_dataset.py - CLI: builds the synthetic dataset + splits
from __future__ import annotations
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.simulator.lightcurve_generator import generate_dataset as gen_fn
from src.simulator.config import SEED
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic light-curve dataset for space debris classification"
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=2000,
        help="Number of light curves to generate per class (default: 2000)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Standard deviation of Gaussian photon noise (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/lightcurves.npz",
        help="Output path for the .npz dataset (default: data/synthetic/lightcurves.npz)",
    )
    args = parser.parse_args()

    print(f"Generating synthetic dataset: {args.n_per_class} curves per class, {5} classes")
    print(f"Noise std: {args.noise_std}, Seed: {args.seed}")

    dataset = gen_fn(
        n_per_class=args.n_per_class,
        noise_std=args.noise_std,
        seed=args.seed,
        save_path=args.output,
    )

    curves = dataset["curves"]
    labels = dataset["labels"]
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"\nDataset summary:")
    print(f"  Total curves: {len(curves)}")
    print(f"  Curve length: {curves.shape[1]}")
    print(f"  Classes: {dict(zip(unique_labels, counts))}")
    print(f"  Shape: {curves.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        curves,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=args.seed,
    )

    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)} curves, test: {len(X_test)} curves")

    np.savez_compressed(
        "data/synthetic/lightcurves_train.npz",
        curves=X_train,
        labels=y_train,
    )
    np.savez_compressed(
        "data/synthetic/lightcurves_test.npz",
        curves=X_test,
        labels=y_test,
    )

    print(f"  Saved train split: data/synthetic/lightcurves_train.npz")
    print(f"  Saved test split: data/synthetic/lightcurves_test.npz")


if __name__ == "__main__":
    main()