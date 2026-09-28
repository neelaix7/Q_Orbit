# train_quantum.py - CLI: train hybrid model, save to models/
from __future__ import annotations
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classical.feature_engineering import extract_all_features
from src.simulator.lightcurve_generator import generate_dataset
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.utils.metrics import compute_metrics, save_metrics_report


def main():
    parser = argparse.ArgumentParser(
        description="Train the hybrid quantum-classifier on synthetic light curves"
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=2000,
        help="Number of training curves per class (default: 2000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=8,
        help="Number of qubits for quantum layer (default: 8)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Variational ansatz layers (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model (default: models)",
    )
    args = parser.parse_args()

    SEED = args.seed
    RNG = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Training hybrid quantum model")
    print(f"  Parameters: n_qubits={args.n_qubits}, n_layers={args.n_layers}")
    print(f"  Epochs: {args.epochs}, batch_size={args.batch_size}")
    print(f"  LR: {args.lr}, n_per_class: {args.n_per_class}")
    print()

    # Use frozen splits (fair comparison) — fit scaler on TRAIN only, no leakage
    frozen_train, frozen_val, frozen_test = "data/splits/train.npz", "data/splits/val.npz", "data/splits/test.npz"
    if os.path.exists(frozen_train):
        print(f"Using frozen splits from data/splits/ (seed {SEED})")
        tr = np.load(frozen_train, allow_pickle=True); te = np.load(frozen_test, allow_pickle=True); va = np.load(frozen_val, allow_pickle=True)
        curves_train, y_train = tr["curves"], tr["labels"].astype(int)
        curves_test, y_test = te["curves"], te["labels"].astype(int)
        curves_val, y_val = va["curves"], va["labels"].astype(int)
        # features
        X_train_raw = np.array([extract_all_features(c) for c in curves_train], dtype=np.float32)
        X_test_raw = np.array([extract_all_features(c) for c in curves_test], dtype=np.float32)
        X_val_raw = np.array([extract_all_features(c) for c in curves_val], dtype=np.float32)
        # fit scaler on train only
        mean = X_train_raw.mean(axis=0); std = X_train_raw.std(axis=0)+1e-8
        X_train = (X_train_raw-mean)/std
        X_test  = (X_test_raw-mean)/std
        X_val   = (X_val_raw-mean)/std
        print(f"Train {len(X_train)}  Val {len(X_val)}  Test {len(X_test)}  features 21  (scaler fit on train only)")
    else:
        dataset_path = "data/synthetic/lightcurves.npz"
        if not os.path.exists(dataset_path):
            from src.simulator.lightcurve_generator import generate_dataset as gen_fn
            gen_fn(n_per_class=args.n_per_class, noise_std=0.02, seed=SEED, save_path=dataset_path)
        data = np.load(dataset_path, allow_pickle=True)
        curves = data["curves"]; labels = data["labels"].astype(int)
        print(f"Extracting features from {len(curves)} curves...")
        X = np.array([extract_all_features(c) for c in curves], dtype=np.float32)
        mean = X.mean(axis=0); std = X.std(axis=0)+1e-8
        X = (X-mean)/std
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=SEED)
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples (fallback)")
        X_val, y_val = X_test, y_test

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Build hybrid model
    model = HybridQuantumClassifier(
        n_features=21,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classical_hidden=16,
        n_classes=5,
        device_name="default.qubit",
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    n_train = len(X_train_t)
    best_accuracy = 0.0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_train)

        for start in range(0, n_train, args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch_x = X_train_t[batch_indices]
            batch_y = y_train_t[batch_indices]

            optimizer.zero_grad()
            logits, probs = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            logits, probs = model(X_test_t)
            _, predicted = torch.max(probs, 1)
            correct = (predicted == y_test_t).sum().item()
            test_accuracy = correct / len(X_test_t) * 100

        # Per-class metrics
        y_test_np = y_test_t.numpy()
        preds_np = predicted.numpy()
        metrics_result = compute_metrics(y_test_np, preds_np, average=None, labels=[0, 1, 2, 3, 4])

        print(
            f"Epoch {epoch:02d}: Loss={avg_loss:.4f}, Test Acc={test_accuracy:.2f}%, "
            f"F1-macro: {metrics_result['f1_macro']:.4f}"
        )

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, "hybrid_quantum_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_features": 21,
                    "n_qubits": args.n_qubits,
                    "n_layers": args.n_layers,
                    "n_classical_hidden": 16,
                    "accuracy": test_accuracy,
                    "epoch": epoch,
                },
                model_path,
            )
            print(f"  -> Saved best model to {model_path} (acc: {test_accuracy:.2f}%)")

    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")

    # Generate evaluation report
    save_metrics_report(
        y_true=y_test_np,
        y_pred=preds_np,
        class_names=["Intact Satellite", "Dead Satellite", "Rocket Body",
                     "Fragmentation Debris", "Spoofed Satellite"],
        output_path="results/reports/evaluation_report.md",
    )


if __name__ == "__main__":
    main()
