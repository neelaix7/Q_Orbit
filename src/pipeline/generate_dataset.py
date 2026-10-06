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
        default=5000,
        help="Number of light curves to generate per class (default: 5000 for 25k)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Fixed Gaussian photon noise std; omit/None => diverse per-sample U[0.01,0.05]",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        default=True,
        help="Per-sample noise/dropout/exposure diversity + class-2/4 overlap (default: True)",
    )
    parser.add_argument(
        "--no-diverse",
        dest="diverse",
        action="store_false",
        help="Disable diversity mode (fixed noise/dropout)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean regime: fixed noise 0.02, fixed 5%% dropout, no exposure jitter, no class-2/4 overlap",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: 123 for 25k regime)",
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
        diverse=args.diverse,
        clean=args.clean,
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

    # Rebuild frozen 70/15/15 splits used for all fair comparisons
    from src.data.splits import create_frozen_splits
    import json as _json
    create_frozen_splits(args.output, "data/splits", args.seed,
                         train=0.70, val=0.15, test=0.15, force=True)
    _man_path = "data/splits/manifest.json"
    _man = _json.load(open(_man_path))
    _man["regime"] = "clean" if args.clean else "diverse"
    _man["note"] = (
        "frozen 70/15/15 seed123 clean regime (fixed noise 0.02, "
        "fixed 5% dropout, no jitter, no class overlap)"
        if args.clean else
        "frozen exact 70/15/15 seed123, 3500/750/750 per class"
    )
    _json.dump(_man, open(_man_path, "w"), indent=2)
    print(f"  Rebuilt frozen splits with regime={_man['regime']}")


if __name__ == "__main__":
    main()