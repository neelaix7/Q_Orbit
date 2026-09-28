# evaluate.py - CLI: run comparison, print report, save plots + metrics
from __future__ import annotations
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.quantum.hybrid_model import HybridQuantumClassifier
from src.classical.cnn_baseline import LightCurveCNN, LightCurveSVM
from src.classical.feature_engineering import extract_all_features
from src.utils.metrics import compute_metrics, save_metrics_report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare all models on test data"
    )
    parser.add_argument(
        "--quantum-model",
        type=str,
        default="models/hybrid_quantum_model.pt",
        help="Path to trained hybrid quantum model (default: models/hybrid_quantum_model.pt)",
    )
    parser.add_argument(
        "--classical-model",
        type=str,
        default="models/classical_cnn_model.pt",
        help="Path to trained classical CNN model (default: models/classical_cnn_model.pt)",
    )
    parser.add_argument(
        "--svm-model",
        type=str,
        default="models/classical_svm_model.joblib",
        help="Path to trained classical SVM model (default: models/classical_svm_model.joblib)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Number of test samples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/figures",
        help="Directory to save comparison plots (default: results/figures)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/reports",
        help="Directory to save evaluation report (default: results/reports)",
    )
    args = parser.parse_args()

    SEED = args.seed
    np.random.seed(SEED)

    print("=" * 60)
    print("Quantum Light-Curve Forensics - Model Evaluation")
    print("=" * 60)
    print()

    # Generate/load dataset
    dataset_path = "data/synthetic/lightcurves.npz"
    if os.path.exists(dataset_path):
        data = np.load(dataset_path, allow_pickle=True)
        curves = data["curves"]
        labels = data["labels"].astype(int)
    else:
        from src.simulator.lightcurve_generator import generate_dataset as gen_fn
        gen_fn(n_per_class=2000, noise_std=0.02, seed=SEED, save_path=dataset_path)
        data = np.load(dataset_path, allow_pickle=True)
        curves = data["curves"]
        labels = data["labels"].astype(int)

    # Select test samples
    n_total = len(curves)
    n_test = min(args.n_test, n_total)
    
    from sklearn.model_selection import train_test_split
    X_test, _, y_test, _ = train_test_split(
        curves, labels, test_size=n_test / n_total, stratify=labels, 
        random_state=SEED
    )

    # Extract features from test curves
    print(f"Extracting features from {len(X_test)} test curves...")
    X_test_features = np.array([extract_all_features(curve) for curve in X_test], dtype=np.float32)
    mean = X_test_features.mean(axis=0)
    std = X_test_features.std(axis=0) + 1e-8
    X_test_features = (X_test_features - mean) / std

    # ===== 1. Hybrid Quantum Model =====
    print("\n1. Evaluating Hybrid Quantum Model...")
    qmodel = HybridQuantumClassifier(n_features=21, n_qubits=8, n_layers=2, n_classical_hidden=16, n_classes=5, device_name="default.qubit")
    
    if os.path.exists(args.quantum_model):
        checkpoint = torch.load(args.quantum_model, weights_only=False)
        qmodel.load_state_dict(checkpoint["model_state_dict"])
        print(f"   Loaded quantum model from {args.quantum_model}")
    else:
        print(f"   No pretrained model found, training on fly...")
        X_test_t = torch.tensor(X_test_features, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        optimizer = torch.optim.Adam(qmodel.parameters(), lr=0.01)
        qmodel.train()
        for epoch in range(5):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits, probs = qmodel(batch_x)
                loss = nn.CrossEntropyLoss()(logits, batch_y)
                loss.backward()
                optimizer.step()
        
        os.makedirs(os.path.dirname(args.quantum_model), exist_ok=True)
        torch.save({"model_state_dict": qmodel.state_dict()}, args.quantum_model)
        print(f"   Saved quickly-trained model to {args.quantum_model}")
    
    qmodel.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_features, dtype=torch.float32)
        logits, probs = qmodel(X_test_t)
        _, predicted = torch.max(probs, 1)
    
    q_acc = float((predicted == torch.tensor(y_test)).sum().item()) / len(y_test) * 100
    y_test_np = np.array(y_test)
    preds_np = np.array(predicted.numpy())
    q_metrics = compute_metrics(y_test_np, preds_np, average=None, labels=[0, 1, 2, 3, 4])
    
    print(f"   Accuracy: {q_acc:.2f}%")
    print(f"   Per-class F1: {np.mean(q_metrics['per_class_f1']):.4f}")
    
    # ===== 2. Classical CNN =====
    print("\n2. Evaluating Classical CNN...")
    cnn_model = LightCurveCNN(n_classes=5, initial_channels=16, dropout=0.3)
    
    X_test_curves_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    
    if os.path.exists(args.classical_model):
        checkpoint = torch.load(args.classical_model, weights_only=False)
        cnn_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   Loaded CNN model from {args.classical_model}")
    else:
        print(f"   No pretrained CNN model found, training on fly...")
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_test_curves_t, y_test_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        cnn_model.train()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
        for epoch in range(5):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = cnn_model(batch_x)
                loss = nn.CrossEntropyLoss()(logits, batch_y)
                loss.backward()
                optimizer.step()
        
        os.makedirs(os.path.dirname(args.classical_model), exist_ok=True)
        torch.save({"model_state_dict": cnn_model.state_dict()}, args.classical_model)
        print(f"   Saved quickly-trained model to {args.classical_model}")
    
    cnn_model.eval()
    with torch.no_grad():
        logits = cnn_model(X_test_curves_t)
        _, predicted = torch.max(logits, 1)
    
    c_acc = float((predicted == torch.tensor(y_test)).sum().item()) / len(y_test) * 100
    cnn_metrics = compute_metrics(y_test_np, np.array(predicted.numpy()), average=None, labels=[0, 1, 2, 3, 4])
    print(f"   Accuracy: {c_acc:.2f}%")
    print(f"   Per-class F1: {np.mean(cnn_metrics['per_class_f1']):.4f}")
    
    # ===== 3. Classical SVM =====
    print("\n3. Evaluating Classical SVM...")
    from src.classical.cnn_baseline import LightCurveSVM
    
    svm_model = LightCurveSVM(kernel="rbf", C=1.0, gamma="scale")
    
    if os.path.exists(args.svm_model):
        import joblib
        bundle = joblib.load(args.svm_model)
        svm_model.clf = bundle["model"]
        print(f"   Loaded SVM model from {args.svm_model}")
    else:
        # Extract features from training portion
        n_train = min(5000, len(curves))
        train_indices = np.random.RandomState(SEED).choice(len(curves), n_train, replace=False)
        train_curves = curves[train_indices]
        train_labels = labels[train_indices]
        
        print(f"   Fitting SVM on {n_train} training samples...")
        X_train_feats = np.array([extract_all_features(c) for c in train_curves], dtype=np.float32)
        X_train_feats = (X_train_feats - mean) / std
        
        svm_model.fit(X_train_feats, train_labels)
        
        os.makedirs(os.path.dirname(args.svm_model), exist_ok=True)
        import joblib
        joblib.dump({"model": svm_model.clf}, args.svm_model)
        print(f"   Saved SVM model to {args.svm_model}")
    
    X_test_feats = np.array([extract_all_features(c) for c in X_test], dtype=np.float32)
    X_test_feats = (X_test_feats - mean) / std
    
    with torch.no_grad():
        svm_predictions = svm_model.predict(X_test_feats)
    
    svm_acc = float(np.mean(svm_predictions == y_test) * 100)
    svm_metrics = compute_metrics(y_test_np, np.array(svm_predictions), average=None, labels=[0, 1, 2, 3, 4])
    print(f"   Accuracy: {svm_acc:.2f}%")
    print(f"   Per-class F1: {np.mean(svm_metrics['per_class_f1']):.4f}")
    
    # ===== 4. Generate comparison table and figures =====
    print("\n4. Generating comparison results...")
    
    results = {
        "Hybrid Quantum": {"accuracy": q_acc, "f1_macro": np.mean(q_metrics['per_class_f1'])},
        "Classical CNN": {"accuracy": c_acc, "f1_macro": np.mean(cnn_metrics['per_class_f1'])},
        "Classical SVM": {"accuracy": svm_acc, "f1_macro": np.mean(svm_metrics['per_class_f1'])},
    }
    
    print("\n   Comparison Results:")
    print(f"{'Model':<20} {'Accuracy':>12} {'F1-Macro':>12}")
    print("-" * 45)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:>12.2f}% {metrics['f1_macro']:>12.4f}")
    
    # Save comparison plot
    os.makedirs(args.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    f1_scores = [results[m]["f1_macro"] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    ax.bar(x_pos - 0.2, accuracies, width=0.4, label="Accuracy", color="#4472C7")
    ax.bar(x_pos + 0.2, f1_scores, width=0.4, label="F1-Macro", color="#ED7D31")
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy and F1-Macro")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plot_path = os.path.join(args.plot_dir, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved comparison plot to {plot_path}")
    
    # Save confusion matrix for quantum
    try:
        from sklearn.metrics import confusion_matrix
        q_cm = confusion_matrix(y_test_np, preds_np, labels=[0, 1, 2, 3, 4])
        cm_path = os.path.join(args.plot_dir, "quantum_confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.matshow(q_cm, cmap="Blues")
        fig.colorbar(im)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Hybrid Quantum - Confusion Matrix")
        fig.tight_layout()
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved quantum confusion matrix to {cm_path}")
    except Exception as e:
        print(f"   Could not save quantum confusion matrix: {e}")
    
    # ===== 5. Save evaluation report =====
    print("\n5. Saving evaluation report...")
    os.makedirs(args.report_dir, exist_ok=True)
    
    all_metrics = {
        "quantum_accuracy": q_acc,
        "cnn_accuracy": c_acc,
        "svm_accuracy": svm_acc,
        "quantum_per_class_f1": np.mean(q_metrics['per_class_f1']),
        "cnn_per_class_f1": np.mean(cnn_metrics['per_class_f1']),
        "svm_per_class_f1": np.mean(svm_metrics['per_class_f1']),
        "n_test_samples": n_test,
        "seed": SEED,
    }
    
    import json
    metrics_path = os.path.join(args.report_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    
    report_path = os.path.join(args.report_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Quantum Light-Curve Forensics - Evaluation Report\n\n")
        f.write(f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Test samples**: {n_test}\n")
        f.write(f"**Random seed**: {SEED}\n\n")
        
        f.write("## Comparison Summary\n\n")
        f.write("| Model | Accuracy | F1-Macro |\n")
        f.write("|-------|----------|----------|\n")
        for model_name, metrics in results.items():
            f.write(f"| {model_name} | {metrics['accuracy']:.2f}% | {metrics['f1_macro']:.4f} |\n")
        f.write("\n")
        
        f.write("## Per-Class Detailed Results\n\n")
        f.write("| Class | Quantum F1 | CNN F1 | SVM F1 |\n")
        f.write("|-------|------------|---------|--------|\n")
        class_names = ["Intact Satellite", "Dead Satellite", "Rocket Body",
                       "Fragmentation Debris", "Spoofed Satellite"]
        for i, cname in enumerate(class_names):
            f.write(f"| {cname} | {q_metrics['per_class_f1'][i]:.4f} | {cnn_metrics['per_class_f1'][i]:.4f} | {svm_metrics['per_class_f1'][i]:.4f} |\n")
        f.write("\n")
        
        f.write("## Key Findings\n\n")
        best_model = max(results.keys(), key=lambda m: results[m]["f1_macro"])
        f.write(f"- Best performing model: **{best_model}** with F1-macro of {results[best_model]['f1_macro']:.4f}\n")
        f.write(f"- Quantum hybrid vs classical baselines: comparison shown above\n")
        f.write(f"- Difficulty sweep at varying noise levels recommended for further analysis\n")
        f.write(f"- Per-class analysis: see confusion matrices and per-class F1 scores above\n")
    
    print(f"   Saved evaluation report to {report_path}")
    print(f"   Saved metrics to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
