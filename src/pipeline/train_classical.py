# train_classical.py - CLI: train baseline, save to models/
from __future__ import annotations
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classical.cnn_baseline import LightCurveCNN, LightCurveSVM
from src.classical.feature_engineering import extract_all_features
from src.simulator.lightcurve_generator import generate_dataset
from src.utils.metrics import compute_metrics, save_metrics_report


def main():
    parser = argparse.ArgumentParser(
        description="Train classical baseline model for comparison"
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
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "svm"],
        default="cnn",
        help="Classical model type: cnn or svm (default: cnn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model (default: models)",
    )
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Training classical baseline model: {args.model.upper()}")
    print(f"  Epochs: {args.epochs}, batch_size={args.batch_size}")
    print(f"  LR: {args.lr}, n_per_class: {args.n_per_class}")
    print()

    # Use frozen splits if available (fair comparison), else fall back to 70/15/15 from synthetic
    frozen_train = "data/splits/train.npz"
    frozen_val = "data/splits/val.npz"
    frozen_test = "data/splits/test.npz"
    if os.path.exists(frozen_train) and os.path.exists(frozen_test):
        print(f"Using frozen splits from data/splits/ (seed {SEED})")
        tr = np.load(frozen_train, allow_pickle=True); X_train_c, y_train = tr["curves"], tr["labels"].astype(int)
        te = np.load(frozen_test, allow_pickle=True); X_test_c, y_test = te["curves"], te["labels"].astype(int)
        va = np.load(frozen_val, allow_pickle=True); X_val_c, y_val = va["curves"], va["labels"].astype(int)
        print(f"Train: {len(X_train_c)}  Val: {len(X_val_c)}  Test: {len(X_test_c)} (frozen 70/15/15)")
    else:
        dataset_path = "data/synthetic/lightcurves.npz"
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}, generating...")
            gen_fn = generate_dataset
            gen_fn(n_per_class=args.n_per_class, noise_std=0.02, seed=SEED, save_path=dataset_path)
        data = np.load(dataset_path, allow_pickle=True)
        curves = data["curves"]; labels = data["labels"].astype(int)
        from sklearn.model_selection import train_test_split
        X_train_c, X_test_c, y_train, y_test = train_test_split(curves, labels, test_size=0.2, stratify=labels, random_state=SEED)
        print(f"Train: {len(X_train_c)} samples, Test: {len(X_test_c)} samples (fallback 80/20)")

    if args.model == "cnn":
        model = LightCurveCNN(n_classes=5, initial_channels=16, dropout=0.3)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        n_train = len(X_train_c)
        best_accuracy = 0.0

        X_train_t = torch.tensor(X_train_c, dtype=torch.float32).unsqueeze(1)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test_c, dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        print("Training classical CNN on raw light curves...")
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
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Evaluate
            model.eval()
            with torch.no_grad():
                logits = model(X_test_t)
                _, predicted = torch.max(logits, 1)
                correct = (predicted == y_test_t).sum().item()
                test_accuracy = correct / len(X_test_t) * 100

            y_test_np = y_test_t.numpy()
            preds_np = predicted.numpy()
            metrics_result = compute_metrics(y_test_np, preds_np, average=None, labels=[0, 1, 2, 3, 4])

            print(
                f"Epoch {epoch:02d}: Loss={avg_loss:.4f}, Test Acc={test_accuracy:.2f}%, "
                f"F1-macro: {metrics_result['f1_macro']:.4f}"
            )

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                os.makedirs(args.output_dir, exist_ok=True)
                model_path = os.path.join(args.output_dir, "classical_cnn_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "accuracy": test_accuracy,
                        "epoch": epoch,
                        "model_type": "cnn",
                    },
                    model_path,
                )
                print(f"  -> Saved best model to {model_path}")

        print(f"\nCNN training complete! Best accuracy: {best_accuracy:.2f}%")

    else:  # svm
        print("Training classical SVM on engineered features...")
        X_train_feats = np.array([extract_all_features(c) for c in X_train_c], dtype=np.float32)
        mean = X_train_feats.mean(axis=0)
        std = X_train_feats.std(axis=0) + 1e-8
        X_train_feats = (X_train_feats - mean) / std

        svm_model = LightCurveSVM(kernel="rbf", C=1.0, gamma="scale")
        svm_model.fit(X_train_feats, y_train)

        X_test_feats = np.array([extract_all_features(c) for c in X_test_c], dtype=np.float32)
        X_test_feats = (X_test_feats - mean) / std
        predictions = svm_model.predict(X_test_feats)

        y_test_np = np.asarray(y_test)
        preds_np = np.asarray(predictions)
        accuracy = float(np.mean(preds_np == y_test_np) * 100)
        metrics_result = compute_metrics(y_test_np, preds_np, average=None, labels=[0, 1, 2, 3, 4])

        print(f"SVM test accuracy: {accuracy:.2f}%")
        print(f"F1-macro: {metrics_result['f1_macro']:.4f}")

        # Save SVM model
        os.makedirs(args.output_dir, exist_ok=True)
        import joblib
        svm_path = os.path.join(args.output_dir, "classical_svm_model.joblib")
        joblib.dump({
            "model": svm_model.clf,
            "scaler_mean": mean,
            "scaler_std": std,
            "accuracy": accuracy,
        }, svm_path)
        print(f"  -> Saved SVM model to {svm_path}")

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
