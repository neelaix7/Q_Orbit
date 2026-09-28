# parameter_overlap_experiment.py — Easy (current) vs Overlapping synthetic dataset
from __future__ import annotations
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from src.simulator.config import SEED
import itertools

# Overlapping regime: widen and overlap tumble/reflectivity/aspect to prevent memorization
OVERLAP_PARAMS = {
    0: {"tumble": (60, 180), "aspect": (1.0, 4.0), "reflect": (0.15, 0.45), "faces":4},
    1: {"tumble": (80, 250), "aspect": (1.2, 5.0), "reflect": (0.10, 0.35), "faces":6},
    2: {"tumble": (100, 400), "aspect": (2.0, 8.0), "reflect": (0.08, 0.25), "faces":8},
    3: {"tumble": (40, 200), "aspect": (0.7, 3.0), "reflect": (0.06, 0.20), "faces":3},
    4: {"tumble": (90, 380), "aspect": (1.8, 6.0), "reflect": (0.06, 0.22), "faces":6},
}

def generate_overlapping_dataset(n_per_class=500, noise_std=0.02, seed=42, save_path="data/synthetic/overlapping_lightcurves.npz"):
    from src.simulator.physics import ObjectState, compute_light_curve_from_faces
    from src.simulator.lightcurve_generator import generate_sun_observer_geometry
    from src.simulator.config import N_SAMPLES, TOTAL_DURATION_SEC, N_TIME_STEPS, EARTH_SHADOW_CONE_ANGLE
    rng = np.random.default_rng(seed)
    curves=[]; labels=[]
    # temporarily override CLASS_PARAMS via local sampling
    times = np.linspace(0, TOTAL_DURATION_SEC, N_SAMPLES)
    sun_dirs, obs_dirs = generate_sun_observer_geometry(times)
    for cls in range(5):
        cfg = OVERLAP_PARAMS[cls]
        for i in range(n_per_class):
            tp = float(rng.uniform(*cfg["tumble"])); ar=float(rng.uniform(*cfg["aspect"])); rf=float(rng.uniform(*cfg["reflect"]))
            pp = max(tp*3.0, 200+rng.uniform(50,150)); axis=rng.normal(0,1,3); axis/=np.linalg.norm(axis); phase=rng.uniform(0,2*np.pi)
            state = ObjectState(400.0,0.001,rng.uniform(0,360),rng.uniform(0,360),rng.uniform(0,360),rng.uniform(0,360),tp,pp,axis,phase,ar,rf)
            n_raw = max(N_TIME_STEPS,100); raw_times=np.linspace(0,TOTAL_DURATION_SEC,n_raw)
            raw_sun, raw_obs = generate_sun_observer_geometry(raw_times)
            curve = compute_light_curve_from_faces(raw_times, state, raw_sun, raw_obs, n_faces=cfg["faces"])
            # resample
            from scipy.interpolate import interp1d
            f = interp1d(np.linspace(0,TOTAL_DURATION_SEC,len(curve)), curve, kind='linear', bounds_error=False, fill_value=0.0)
            curve = f(np.linspace(0,TOTAL_DURATION_SEC,N_SAMPLES))
            curve = curve + rng.normal(0, noise_std, N_SAMPLES)
            n_drop = max(1,int(0.05*N_SAMPLES)); idx=rng.choice(N_SAMPLES,n_drop,replace=False); curve[idx]=np.nan
            mask=~np.isnan(curve)
            if mask.sum()>1:
                xp=np.where(mask)[0]; fp=curve[mask]; curve=np.interp(np.arange(N_SAMPLES),xp,fp)
            cmin,cmax=curve.min(),curve.max()
            if cmax>cmin: curve=(curve-cmin)/(cmax-cmin)
            else: curve=np.zeros(N_SAMPLES)
            curve=np.clip(curve,0,1)
            curves.append(curve.astype(np.float32)); labels.append(cls)
    curves=np.array(curves); labels=np.array(labels, dtype=np.int64)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, curves=curves, labels=labels)
    print(f"Saved overlapping dataset {save_path}: {curves.shape}")
    return {"curves":curves, "labels":labels}

if __name__ == "__main__":
    generate_overlapping_dataset()
    # quick evaluation: train-test split and report separability
    data=np.load("data/synthetic/overlapping_lightcurves.npz",allow_pickle=True)
    curves,labels=data["curves"],data["labels"]
    from sklearn.model_selection import train_test_split
    from src.classical.feature_engineering import extract_all_features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    X_train,X_test,y_train,y_test=train_test_split(curves,labels,test_size=0.3, stratify=labels, random_state=SEED)
    f_train=np.array([extract_all_features(c) for c in X_train]); f_test=np.array([extract_all_features(c) for c in X_test])
    mean=f_train.mean(0); std=f_train.std(0)+1e-8; f_train=(f_train-mean)/std; f_test=(f_test-mean)/std
    clf=RandomForestClassifier(200, random_state=SEED, n_jobs=-1); clf.fit(f_train,y_train); pred=clf.predict(f_test)
    acc=accuracy_score(y_test,pred)
    print(f"Overlapping RF accuracy (separability proxy): {acc:.4f} (lower = harder, more overlap)")
    # also compare easy dataset via same method
    from pathlib import Path
    if Path("data/synthetic/lightcurves.npz").exists():
        d2=np.load("data/synthetic/lightcurves.npz",allow_pickle=True); c2,l2=d2["curves"],d2["labels"]
        Xtr,Xte,ytr,yte=train_test_split(c2,l2,test_size=0.3, stratify=l2, random_state=SEED)
        ft=np.array([extract_all_features(c) for c in Xtr]); fte=np.array([extract_all_features(c) for c in Xte])
        m=ft.mean(0); s=ft.std(0)+1e-8; ft=(ft-m)/s; fte=(fte-m)/s
        clf2=RandomForestClassifier(200, random_state=SEED, n_jobs=-1); clf2.fit(ft,ytr); acc2=accuracy_score(yte,clf2.predict(fte))
        print(f"Easy (current) RF accuracy: {acc2:.4f}")
        with open("results/reports/overlap_separability.json", "w", encoding="utf-8") as f: json.dump({"overlapping_acc": float(acc), "easy_acc": float(acc2)}, f, indent=2)
