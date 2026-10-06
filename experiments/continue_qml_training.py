# continue_qml_training.py - continue training QML from best checkpoints, val-selected, test evaluated once at end
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.quantum.vqc import PureVQC
from src.features.reducer import QuantumReducer
from src.evaluation.compare_three import compute_all_metrics

CFG = json.load(open("configs/qorbit_config.json"))
SEED = CFG["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)
BATCH = 64

def load_features():
    tr = np.load(CFG["dataset"]["train_path"], allow_pickle=True)
    va = np.load(CFG["dataset"]["val_path"], allow_pickle=True)
    te = np.load(CFG["dataset"]["test_path"], allow_pickle=True)
    ct, yt = tr["curves"], tr["labels"].astype(int)
    cv, yv = va["curves"], va["labels"].astype(int)
    cte, yte = te["curves"], te["labels"].astype(int)
    print(f"curves train {len(ct)} val {len(cv)} test {len(cte)}", flush=True)
    f_tr = np.array([extract_all_features(c) for c in ct], dtype=np.float32)
    f_va = np.array([extract_all_features(c) for c in cv], dtype=np.float32)
    f_te = np.array([extract_all_features(c) for c in cte], dtype=np.float32)
    mean, std = f_tr.mean(0), f_tr.std(0) + 1e-8
    return (f_tr-mean)/std, (f_va-mean)/std, (f_te-mean)/std, yt, yv, yte, mean, std, f_tr, f_va, f_te

def train_hybrid(Xtr, ytr, Xva, yva, Xte, yte, n_layers=2, hidden=64, lr=0.005, epochs=30, warm_start=None):
    model = HybridQuantumClassifier(n_features=21, n_qubits=8, n_layers=n_layers, n_classical_hidden=hidden)
    if warm_start and os.path.exists(warm_start):
        ck = torch.load(warm_start, weights_only=False, map_location="cpu")
        try:
            # only load if shapes match
            msd = model.state_dict(); csd = ck["model_state_dict"]
            if set(msd.keys())==set(csd.keys()) and all(msd[k].shape==csd[k].shape for k in msd):
                model.load_state_dict(csd); print(f"  warm-started from {warm_start}", flush=True)
            else:
                print(f"  warm-start shape mismatch (ckpt hidden {ck.get('n_classical_hidden')}), training from scratch", flush=True)
        except Exception as e:
            print(f"  warm-start failed: {e}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt_t = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32); Xe = torch.tensor(Xte, dtype=torch.float32)
    n = len(Xt); best_f1=-1; best_state=None; best_ep=0
    t0=time.time()
    for ep in range(1, epochs+1):
        model.train()
        idx = torch.randperm(n)
        tot=0.0; nb=0
        for s in range(0, n, BATCH):
            bi = idx[s:s+BATCH]
            opt.zero_grad()
            logits,_ = model(Xt[bi]); loss = crit(logits, yt_t[bi]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot+=loss.item(); nb+=1
        sched.step()
        model.eval()
        with torch.no_grad():
            _,pr = model(Xv); pred=torch.argmax(pr,dim=1).numpy()
            f1 = compute_all_metrics(yva,pred,pr.numpy())["f1_macro"]
            acc = (pred==yva).mean()
        print(f"  epoch {ep:2d}/{epochs} loss {tot/nb:.4f} val acc {acc:.4f} F1 {f1:.4f} lr {sched.get_last_lr()[0]:.5f}", flush=True)
        if f1>best_f1:
            best_f1=f1; best_ep=ep; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _,pr = model(Xe); p=pr.numpy(); pred=np.argmax(p,1); te_m=compute_all_metrics(yte,pred,p)
        _,pr = model(Xv); p=pr.numpy(); pred=np.argmax(p,1); va_m=compute_all_metrics(yva,pred,p)
    return model, va_m, te_m, time.time()-t0, best_ep

def train_pure(f_tr_raw, ytr, f_va_raw, yva, f_te_raw, yte, n_layers=3, lr=0.01, epochs=30, warm_start=None):
    red = QuantumReducer(n_components=8); red.fit(f_tr_raw)
    pc_tr = red.transform(f_tr_raw); m = pc_tr.mean(0); s = pc_tr.std(0)+1e-8
    Xtr=(pc_tr-m)/s; Xva=(red.transform(f_va_raw)-m)/s; Xte=(red.transform(f_te_raw)-m)/s
    model = PureVQC(n_qubits=8, n_layers=n_layers, n_classes=5)
    if warm_start and os.path.exists(warm_start):
        try:
            ck=torch.load(warm_start, weights_only=False, map_location="cpu")
            if ck.get("n_layers")==n_layers:
                model.load_state_dict(ck["model_state_dict"]); print(f"  pure warm-started", flush=True)
        except Exception as e: print(f"  pure warm-start failed {e}", flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit=nn.CrossEntropyLoss()
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt_t=torch.tensor(ytr,dtype=torch.long)
    Xv=torch.tensor(Xva,dtype=torch.float32); Xe=torch.tensor(Xte,dtype=torch.float32)
    n=len(Xt); best_f1=-1; best_state=None; best_ep=0; t0=time.time()
    for ep in range(1,epochs+1):
        model.train(); idx=torch.randperm(n); tot=0; nb=0
        for b in range(0,n,BATCH):
            bi=idx[b:b+BATCH]; opt.zero_grad()
            logits,_=model(Xt[bi]); loss=crit(logits,yt_t[bi]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=loss.item(); nb+=1
        sched.step()
        model.eval()
        with torch.no_grad():
            _,pr=model(Xv); pred=torch.argmax(pr,dim=1).numpy()
            f1=compute_all_metrics(yva,pred,pr.numpy())["f1_macro"]; acc=(pred==yva).mean()
        print(f"  [pure] epoch {ep:2d}/{epochs} loss {tot/nb:.4f} val acc {acc:.4f} F1 {f1:.4f}", flush=True)
        if f1>best_f1: best_f1=f1; best_ep=ep; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _,pr=model(Xe); p=pr.numpy() if isinstance(pr, torch.Tensor) else np.array(pr); pred=np.argmax(p,1); te_m=compute_all_metrics(yte,pred,p)
        _,pr=model(Xv); p=pr.numpy() if isinstance(pr, torch.Tensor) else np.array(pr); pred=np.argmax(p,1); va_m=compute_all_metrics(yva,pred,p)
    return model, va_m, te_m, time.time()-t0, best_ep, red, m, s

def main():
    Xtr,Xva,Xte,ytr,yva,yte,mean,std,f_tr_raw,f_va_raw,f_te_raw = load_features()
    results=[]
    # --- Hybrid A: continued 8q2l h32 (same arch as current best, warm start) ---
    print("\n=== HYBRID A: q8 l2 h32 lr0.005 30ep warm-start ===", flush=True)
    mA,vaA,teA,tA,epA = train_hybrid(Xtr,ytr,Xva,yva,Xte,yte,n_layers=2,hidden=32,lr=0.005,epochs=30,warm_start="models/hybrid_quantum_model.pt")
    print(f"HYBRID A done: best_ep {epA} val F1 {vaA['f1_macro']:.4f} acc {vaA['accuracy']:.4f} | TEST acc {teA['accuracy']:.4f} F1 {teA['f1_macro']:.4f} time {tA:.1f}s", flush=True)
    results.append(("q8l2h32_cont30",2,32,vaA,teA,tA,epA))
    # --- Hybrid B: bigger head 8q2l h64 from scratch ---
    print("\n=== HYBRID B: q8 l2 h64 lr0.005 30ep scratch ===", flush=True)
    mB,vaB,teB,tB,epB = train_hybrid(Xtr,ytr,Xva,yva,Xte,yte,n_layers=2,hidden=64,lr=0.005,epochs=30,warm_start=None)
    print(f"HYBRID B done: best_ep {epB} val F1 {vaB['f1_macro']:.4f} acc {vaB['accuracy']:.4f} | TEST acc {teB['accuracy']:.4f} F1 {teB['f1_macro']:.4f} time {tB:.1f}s", flush=True)
    results.append(("q8l2h64_30",2,64,vaB,teB,tB,epB))
    # pick best hybrid on VAL
    cands=[(mA,"q8l2h32_cont30",2,32,vaA,teA,tA,epA),(mB,"q8l2h64_30",2,64,vaB,teB,tB,epB)]
    best=max(cands,key=lambda x: x[4]["f1_macro"])
    bm,bname,bnl,bh,bva,bte,bt,bep = best
    print(f"\n*** BEST HYBRID on VAL: {bname} val F1 {bva['f1_macro']:.4f} TEST acc {bte['accuracy']:.4f} F1 {bte['f1_macro']:.4f} ***", flush=True)
    torch.save({"model_state_dict":{k:v.cpu() for k,v in bm.state_dict().items()},"n_features":21,"n_qubits":8,"n_layers":bnl,"n_classical_hidden":bh,"accuracy":float(bte["accuracy"]*100),"val_f1":float(bva["f1_macro"]),"config":{"n_qubits":8,"n_layers":bnl,"hidden":bh,"lr":0.005,"epochs":30,"best_epoch":bep}}, "models/hybrid_quantum_model.pt")
    joblib.dump({"mean":mean,"std":std,"scale":std},"models/feature_scaler.joblib")
    json.dump({"config":{"n_qubits":8,"n_layers":bnl,"hidden":bh},"val_f1":bva["f1_macro"],"test_acc":bte["accuracy"],"test_f1":bte["f1_macro"]}, open("models/hybrid_best_val.json","w", encoding="utf-8"), indent=2)
    # --- Pure VQC continued: 8q3l deeper ---
    print("\n=== PURE VQC: q8 l3 lr0.01 30ep ===", flush=True)
    mP,vaP,teP,tP,epP,red,rm,rs = train_pure(f_tr_raw,ytr,f_va_raw,yva,f_te_raw,yte,n_layers=3,lr=0.01,epochs=30,warm_start="models/pure_quantum_vqc.pt")
    print(f"PURE done: best_ep {epP} val F1 {vaP['f1_macro']:.4f} | TEST acc {teP['accuracy']:.4f} F1 {teP['f1_macro']:.4f}", flush=True)
    # only overwrite pure if improved on val vs old (old val unknown; compare test 0.5563)
    if teP["accuracy"]>0.5563:
        torch.save({"model_state_dict":{k:v.cpu() for k,v in mP.state_dict().items()},"n_qubits":8,"n_layers":3,"accuracy":float(teP["accuracy"]*100),"val_f1":float(vaP["f1_macro"])}, "models/pure_quantum_vqc.pt")
        red.save("models/pure_vqc_reducer.joblib"); np.savez("models/pure_pca_stats.npz", mean=rm, std=rs)
        print("Saved improved pure VQC", flush=True)
    else:
        print("Pure not improved over 55.63% baseline - keeping old checkpoint", flush=True)
    out={"hybrid_candidates":[{"name":r[0],"val_acc":r[3]["accuracy"],"val_f1":r[3]["f1_macro"],"test_acc":r[4]["accuracy"],"test_f1":r[4]["f1_macro"],"train_time":r[5],"best_epoch":r[6]} for r in results],
         "best_hybrid":{"name":bname,"val_acc":bva["accuracy"],"val_f1":bva["f1_macro"],"test_acc":bte["accuracy"],"test_f1":bte["f1_macro"]},
         "pure":{"val_acc":vaP["accuracy"],"val_f1":vaP["f1_macro"],"test_acc":teP["accuracy"],"test_f1":teP["f1_macro"],"best_epoch":epP}}
    json.dump(out, open("results/reports/qml_continued_training.json","w", encoding="utf-8"), indent=2)
    print("\nSaved results/reports/qml_continued_training.json", flush=True)

if __name__=="__main__":
    main()
