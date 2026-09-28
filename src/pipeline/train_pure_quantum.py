# train_pure_quantum.py — Pure VQC training using frozen splits + scaler fit on train only + optional PCA
from __future__ import annotations
import argparse, os, sys, json
import numpy as np, torch, torch.nn as nn, torch.optim as optim
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"../.."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.vqc import PureVQC
from src.features.reducer import QuantumReducer
from src.utils.metrics import compute_metrics

def main():
    p=argparse.ArgumentParser(description="Train Pure Quantum VQC (genuine quantum classifier)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="models")
    p.add_argument("--use-pca", action="store_true", help="use PCA to reduce 21 features to n_qubits (recommended for pure quantum)")
    args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Training Pure VQC  n_qubits={args.n_qubits} n_layers={args.n_layers} epochs={args.epochs} use_pca={args.use_pca}")

    # frozen splits
    tr=np.load("data/splits/train.npz",allow_pickle=True); te=np.load("data/splits/test.npz",allow_pickle=True); va=np.load("data/splits/val.npz",allow_pickle=True)
    curves_train,y_train = tr["curves"], tr["labels"].astype(int)
    curves_test,y_test = te["curves"], te["labels"].astype(int)
    curves_val,y_val = va["curves"], va["labels"].astype(int)

    X_train_raw=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
    X_test_raw=np.array([extract_all_features(c) for c in curves_test],dtype=np.float32)
    X_val_raw=np.array([extract_all_features(c) for c in curves_val],dtype=np.float32)

    if args.use_pca:
        reducer=QuantumReducer(n_components=args.n_qubits)
        reducer.fit(X_train_raw)
        X_train=reducer.transform(X_train_raw).astype(np.float32)
        X_test=reducer.transform(X_test_raw).astype(np.float32)
        X_val=reducer.transform(X_val_raw).astype(np.float32)
        # second standardization after PCA for stable angles
        mean=X_train.mean(axis=0); std=X_train.std(axis=0)+1e-8
        X_train=(X_train-mean)/std; X_test=(X_test-mean)/std; X_val=(X_val-mean)/std
        reducer.save(f"{args.output_dir}/pure_vqc_reducer.joblib")
        print(f"PCA reducer saved, explained variance {np.round(reducer.pca.explained_variance_ratio_,3)}")
    else:
        # zscore then take first n_qubits dims (linear projection without PCA)
        mean=X_train_raw.mean(axis=0); std=X_train_raw.std(axis=0)+1e-8
        X_train=(X_train_raw-mean)/std
        X_test=(X_test_raw-mean)/std
        X_val=(X_val_raw-mean)/std
        # slice to n_qubits if n_features > n_qubits (for angle embedding)
        if X_train.shape[1] > args.n_qubits:
            X_train=X_train[:,:args.n_qubits]; X_test=X_test[:,:args.n_qubits]; X_val=X_val[:,:args.n_qubits]

    print(f"Pure VQC input dim {X_train.shape[1]} -> {args.n_qubits} qubits")

    X_train_t=torch.tensor(X_train,dtype=torch.float32); y_train_t=torch.tensor(y_train,dtype=torch.long)
    X_test_t=torch.tensor(X_test,dtype=torch.float32); y_test_t=torch.tensor(y_test,dtype=torch.long)
    X_val_t=torch.tensor(X_val,dtype=torch.float32); y_val_t=torch.tensor(y_val,dtype=torch.long)

    model=PureVQC(n_qubits=args.n_qubits,n_layers=args.n_layers,n_classes=5)
    opt=optim.Adam(model.parameters(), lr=args.lr)
    crit=nn.CrossEntropyLoss()
    best=0.0
    n_train=len(X_train_t)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0; batches=0
        idx=torch.randperm(n_train)
        for s in range(0,n_train,args.batch_size):
            bi=idx[s:s+args.batch_size]
            bx,by=X_train_t[bi], y_train_t[bi]
            opt.zero_grad()
            logits,_=model(bx)
            loss=crit(logits,by)
            loss.backward(); opt.step()
            total+=loss.item(); batches+=1
        avg=total/max(batches,1)
        model.eval()
        with torch.no_grad():
            _,proba=model(X_val_t)
            pred=torch.argmax(proba,dim=1)
            acc=float((pred==y_val_t).sum().item()/len(y_val_t)*100)
            m=compute_metrics(y_val_t.numpy(), pred.numpy(), labels=[0,1,2,3,4])
        print(f"Epoch {epoch:02d}: loss {avg:.4f}  Val Acc {acc:.2f}%  F1-macro {m['f1_macro']:.4f}")
        if acc>best:
            best=acc
            os.makedirs(args.output_dir,exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "n_qubits": args.n_qubits, "n_layers": args.n_layers, "accuracy": acc, "use_pca": args.use_pca, "mean": mean, "std": std}, f"{args.output_dir}/pure_quantum_vqc.pt")
            print(f" -> saved best to {args.output_dir}/pure_quantum_vqc.pt")
    print(f"Pure VQC done. Best val acc {best:.2f}%")
    # final test
    with torch.no_grad():
        ckpt=torch.load(f"{args.output_dir}/pure_quantum_vqc.pt",weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _,proba=model(X_test_t)
        pred=torch.argmax(proba,dim=1)
        acc=float((pred==y_test_t).sum().item()/len(y_test_t)*100)
        m=compute_metrics(y_test_t.numpy(), pred.numpy(), labels=[0,1,2,3,4])
        print(f"Test Acc {acc:.2f}%  F1-macro {m['f1_macro']:.4f}")

if __name__=="__main__":
    main()
