"""CLI: train both AI models -> models/ (link_predictor.joblib, eve_detector.joblib).

Usage:
    python -m src.pipeline.train_ai --qkd data/qkd --orbits data/orbits --out models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from ..ai.eve_detector import EveDetector, extract_features, extract_qber_features
    from ..ai.link_predictor import LinkPredictor, link_features_from_pass
    from ..orbit.geometry import GroundStation
    from ..quantum.attacks import EveAttack
    from ..pipeline.run_qkd import qkd_over_pass
    from ..quantum.params import QKDParams
except (ImportError, ValueError):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.ai.eve_detector import EveDetector, extract_features, extract_qber_features
    from src.ai.link_predictor import LinkPredictor, link_features_from_pass
    from src.orbit.geometry import GroundStation
    from src.quantum.attacks import EveAttack
    from src.pipeline.run_qkd import qkd_over_pass
    from src.quantum.params import QKDParams

GS = GroundStation(lat_deg=13.0827, lon_deg=80.2707, alt_m=10.0, name="Chennai-ISRO")
CONFIG_GRID = [(0.2, 30.0, 0.3), (0.2, 30.0, 0.6), (0.2, 30.0, 0.9),
               (0.5, 100.0, 0.3), (0.5, 100.0, 0.6), (0.5, 100.0, 0.9),
               (0.8, 300.0, 0.3), (0.8, 300.0, 0.6), (0.8, 300.0, 0.9)]


def build_link_dataset(orbits_dir: Path, passes_meta: pd.DataFrame,
                       grid: list[tuple], n_passes: int = 12,
                       seed: int = 42, n_photons: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run QKD at every grid config on a sample of passes to train the predictor.

    Honest by construction: the regressor learns from genuinely re-run QKD
    exchanges, so the (link, config) -> (QBER, secure bps) mapping is measured,
    not assumed.
    """
    rng = np.random.default_rng(seed)
    meta = passes_meta.sample(min(n_passes, len(passes_meta)), random_state=seed)
    X, yq, yb = [], [], []
    for _, m in meta.iterrows():
        df = pd.read_csv(orbits_dir / f"{m['pass_id']}.csv")
        feats = link_features_from_pass(df, sat_idx=int(m["sat_idx"]))
        for (mu, dark, eta) in grid:
            cfg = QKDParams(laser_intensity=mu, dark_count_rate=dark,
                            detector_eta=eta, basis_bias=0.5)
            rows = qkd_over_pass(df, cfg, EveAttack("none", 0.0), rng,
                                 n_photons=n_photons)
            X.append(list(feats) + [mu, dark, eta, 0.5])
            yq.append(float(np.mean([r["qber"] for r in rows])))
            yb.append(float(np.mean([r["secure_bps"] for r in rows])))
    return np.array(X, dtype=float), np.array(yq), np.array(yb)


def build_eve_dataset(qkd_dir: Path, lp: LinkPredictor | None = None,
                      default_config=QKDParams()) -> tuple[np.ndarray, np.ndarray]:
    """Features + labels from clean and attacked QKD runs (fused with AI link context).

    The fused feature includes ``qber_residual`` = measured QBER minus the QBER
    the link predictor expects under the pass's link conditions.  For attack runs
    the residual is large and positive (Eve pushes QBER above atmosphere), which
    is exactly what the fusion engine is designed to exploit.
    """
    X, y = [], []
    for fname in sorted(qkd_dir.glob("qkd_eve_*.csv")):
        df = pd.read_csv(fname)
        attacked = df["attack_name"].iloc[0] != "none"
        for (pid, sat), g in df.groupby(["pass_id", "sat_idx"]):
            q = g["qber"].to_numpy()
            if len(q) < 4:
                continue
            mean_link_q = float(g["link_quality"].mean()) if "link_quality" in g else 0.5
            residual = 0.0
            if lp is not None and g["link_quality"].notna().all():
                feats = link_features_from_pass(
                    pd.DataFrame({
                        "elevation_deg": g["elevation_deg"].to_numpy(),
                        "mjd": g["mjd"].to_numpy(),
                        "link_quality": g["link_quality"].to_numpy(),
                    }), sat_idx=int(sat))
                Xc = np.concatenate([feats, default_config.as_array()])[None, :]
                qber_pred, _ = lp.predict(Xc)
                residual = float(q.mean() - qber_pred[0])
            X.append(extract_features(q, mean_link_q, residual))
            y.append(1 if attacked else 0)
    return np.array(X, dtype=float), np.array(y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train link predictor + eve detector")
    ap.add_argument("--qkd", default="data/qkd")
    ap.add_argument("--orbits", default="data/orbits")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-passes", type=int, default=12)
    args = ap.parse_args()

    qkd_dir = Path(args.qkd)
    orbits_dir = Path(args.orbits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Link predictor ----
    passes_meta = pd.read_csv(orbits_dir / "passes_meta.csv")
    print(f"Running QKD at {len(CONFIG_GRID)} configs on {args.n_passes} passes "
          f"to build the link-predictor training set ...")
    X, yq, yb = build_link_dataset(orbits_dir, passes_meta, CONFIG_GRID,
                                   n_passes=args.n_passes, seed=args.seed)
    lp = LinkPredictor(seed=args.seed)
    lp.fit(X, yq, yb)
    lp.save(out_dir / "link_predictor.joblib")
    print(f"LinkPredictor trained: {X.shape[0]} rows -> {out_dir / 'link_predictor.joblib'}")

    # ---- Eve detector ----
    Xe, ye = build_eve_dataset(qkd_dir, lp=lp)
    if Xe.shape[0] < 30:
        raise RuntimeError(f"Too few QKD runs ({Xe.shape[0]}); generate data/qkd first.")
    # Proper protocol: train (fit) / val (pick operating threshold) / test (metrics).
    Xtr, Xte, ytr, yte = train_test_split(Xe, ye, test_size=0.25, random_state=args.seed,
                                          stratify=ye)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.25,
                                              random_state=args.seed, stratify=ytr)
    det = EveDetector(seed=args.seed, low_fp=True)
    det.fit(Xfit, yfit)                     # fit on train
    det.operating_threshold = det._pick_threshold(Xval, yval)   # choose threshold on val
    det.save(out_dir / "eve_detector.joblib")
    te_prob = det.predict_proba(Xte)
    print(f"EveDetector trained: {Xe.shape[0]} samples "
          f"(fit={len(yfit)} val={len(yval)} test={len(yte)}), "
          f"hold-out AUC = {roc_auc_score(yte, te_prob):.3f}, "
          f"threshold = {det.operating_threshold:.3f} "
          f"-> {out_dir / 'eve_detector.joblib'}")
    # Save the held-out test set for evaluate.py (avoids train/test leakage).
    np.savez(out_dir / "eve_test_split.npz", Xte=Xte, yte=yte)


if __name__ == "__main__":
    main()
