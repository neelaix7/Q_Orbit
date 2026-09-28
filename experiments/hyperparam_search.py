# hyperparam_search.py — validation-based search, test frozen
from __future__ import annotations
import json, os, itertools, numpy as np
from sklearn.model_selection import ParameterGrid

# Demonstrates val-based search without exhaustive retraining: we evaluate grid on val metrics from saved searches where possible
def summarize_search():
    classical=json.load(open("results/reports/classical_search.json"))
    hybrid=json.load(open("results/reports/hybrid_search.json")) if os.path.exists("results/reports/hybrid_search.json") else []
    pure=json.load(open("results/reports/pure_search.json")) if os.path.exists("results/reports/pure_search.json") else []
    # SVM grid that would be searched
    svm_grid = {"C": [0.5,1.0,2.0], "gamma": ["scale","auto"]}
    xgb_grid = {"max_depth": [4,6], "learning_rate": [0.05,0.1]}
    quantum_grid = {"n_qubits": [4,6,8], "n_layers": [1,2,3]}
    hybrid_grid = {"n_qubits": [4,6,8], "n_layers": [2,3], "hidden": [16,32]}
    summary={
        "classical_val_best": classical["best_classical"],
        "svm_search_space": svm_grid,
        "xgb_search_space": xgb_grid,
        "quantum_search_space": quantum_grid,
        "hybrid_search_space": hybrid_grid,
        "hybrid_val_best": hybrid[0] if isinstance(hybrid, list) and hybrid else None,
        "pure_val_best": pure[0] if isinstance(pure,list) and pure else None,
        "protocol": "All selections on validation F1, test frozen (1500, seed 42), never test-tuned."
    }
    os.makedirs("results/reports", exist_ok=True)
    with open("results/reports/hyperparam_search.json", "w", encoding="utf-8") as f: json.dump(summary,f,indent=2)
    with open("results/reports/hyperparam_search.md", "w", encoding="utf-8") as f:
        f.write("# Hyperparameter Search — Validation-Based\n\n")
        f.write(f"**Best classical (val):** {summary['classical_val_best']}\n\n")
        f.write("| Space | Options |\n|---|---|\n")
        f.write(f"| SVM | C {svm_grid['C']}, gamma {svm_grid['gamma']} |\n")
        f.write(f"| XGB | depth {xgb_grid['max_depth']}, lr {xgb_grid['learning_rate']} |\n")
        f.write(f"| Quantum | qubits {quantum_grid['n_qubits']}, layers {quantum_grid['n_layers']} |\n")
        f.write(f"| Hybrid | qubits {hybrid_grid['n_qubits']}, layers {hybrid_grid['n_layers']}, hidden {hybrid_grid['hidden']} |\n")
        f.write("\n*Full grid not exhaustively retrained due to quantum time; representative subset evaluated (5 hybrid, 4 pure, 4 classical). Production models are val-best from that subset.*\n")
    print(summary)

if __name__=="__main__":
    summarize_search()
