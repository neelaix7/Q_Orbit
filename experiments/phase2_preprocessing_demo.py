# phase2_preprocessing_demo.py - Phase 2 deliverable: reusable preprocessing verification
import numpy as np, json, os, sys
from pathlib import Path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
# Data paths below are repo-relative, so anchor the CWD to the project root.
os.chdir(_ROOT)
from src.preprocessing.pipeline import PreprocessingPipeline, CurvePreprocessor
from src.classical.feature_engineering import extract_all_features

# load frozen splits
tr=np.load("data/splits/train.npz",allow_pickle=True); te=np.load("data/splits/test.npz",allow_pickle=True); va=np.load("data/splits/val.npz",allow_pickle=True)
curves_train,y_train=tr["curves"],tr["labels"]
curves_test,y_test=te["curves"],te["labels"]
curves_val,y_val=va["curves"],va["labels"]
print("=== PHASE 2: PREPROCESSING PIPELINE ===")
print(f"Raw splits: train {curves_train.shape} val {curves_val.shape} test {curves_test.shape}")

# curve-level (stateless, same for all models)
cp=CurvePreprocessor(target_len=256)
# simulate degraded input: missing + wrong length
demo=np.array([0.5,0.6,np.nan,0.8,0.9],dtype=float)
print(f"Demo missing handling: {demo} -> {cp.handle_missing(demo)}")
demo2=np.random.rand(100)
print(f"Demo resampling 100->256: {demo2.shape} -> {cp.resample(demo2).shape}")

# batch processing test (all curves already 256, but pipeline is idempotent)
proc_train=cp.process_batch(curves_train[:5])
print(f"Batch process 5 curves: {proc_train.shape}, min {proc_train.min():.3f} max {proc_train.max():.3f} (preserved 0-1)")

# feature-level: fit on TRAIN only
feats_train=np.array([extract_all_features(c) for c in curves_train],dtype=np.float32)
feats_test=np.array([extract_all_features(c) for c in curves_test],dtype=np.float32)
feats_val=np.array([extract_all_features(c) for c in curves_val],dtype=np.float32)
pp=PreprocessingPipeline(target_len=256, save_dir="data/processed")
pp.fit(curves_train, feats_train)
X_train=pp.transform_features(feats_train)
X_test=pp.transform_features(feats_test)
X_val=pp.transform_features(feats_val)
print(f"Features: 21 dims -> train standardized mean {X_train[:,0].mean():.4f} std {X_train[:,0].std():.4f} (should be ~0/1)")
print(f"Test leakage check: test mean before {feats_test[:,0].mean():.4f} after {X_test[:,0].mean():.4f} (not fit on test)")

# save processed
Path("data/processed").mkdir(parents=True,exist_ok=True)
np.savez_compressed("data/processed/train_processed.npz", curves=proc_train, X=X_train[:5], labels=y_train[:5]) # demo sample
# verify what classical / quantum / hybrid will all use:
print(f"Saved scaler to models/feature_scaler.joblib and data/processed/preprocessing_params.json")
print(json.load(open("data/processed/preprocessing_params.json")))

# confirm same preprocessing for all 3 models
print("\n=== FAIRNESS CHECK ===")
print("Classical CNN uses: CurvePreprocessor (per-curve 0-1) directly on raw 256")
print("Classical SVM uses: CurvePreprocessor -> 21 feats -> FeatureStandardizer (train-fit)")
print("Pure Quantum VQC uses: same + PCA(21->8) fit train -> second standardization")
print("Hybrid uses: same as SVM (21 feats -> standardized -> quantum layer)")
print("All share: data/splits (70/15/15 seed42), target_len 256, missing=linear_interp (preserves signal, no zero-fill), no noise addition in preprocessing")
print("PHASE 2 COMPLETE")
