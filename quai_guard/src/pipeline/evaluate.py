"""CLI: full evaluation -> results/reports/evaluation_report.md + figures + metrics.json.

Usage:
    python -m src.pipeline.evaluate --orbits data/orbits --qkd data/qkd \
        --models models --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from ..ai.eve_detector import EveDetector, extract_features, extract_qber_features
    from ..ai.fusion import GuardEngine
    from ..ai.link_predictor import LinkPredictor, link_features_from_pass
    from ..orbit.geometry import GroundStation
    from ..orbit.link import LinkModel
    from ..quantum.attacks import EveAttack
    from ..quantum.bb84 import run_bb84_round
    from ..quantum.params import QKDParams, parameter_grid
    from ..utils.metrics import (classifier_metrics, key_rate_improvement, mean_qber,
                                 mean_secure_bps, save_metrics, secure_fraction,
                                 write_report)
    from ..utils.viz import (plot_attack_sweep, plot_confusion, plot_key_rate_comparison,
                             plot_qber_series, plot_roc, save_fig)
except (ImportError, ValueError):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.ai.eve_detector import EveDetector, extract_features, extract_qber_features
    from src.ai.fusion import GuardEngine
    from src.ai.link_predictor import LinkPredictor, link_features_from_pass
    from src.orbit.geometry import GroundStation
    from src.orbit.link import LinkModel
    from src.quantum.attacks import EveAttack
    from src.quantum.bb84 import run_bb84_round
    from src.quantum.params import QKDParams, parameter_grid
    from src.utils.metrics import (classifier_metrics, key_rate_improvement, mean_qber,
                                 mean_secure_bps, save_metrics, secure_fraction,
                                 write_report)
    from src.utils.viz import (plot_attack_sweep, plot_confusion, plot_key_rate_comparison,
                             plot_qber_series, plot_roc, save_fig)

GS = GroundStation(lat_deg=13.0827, lon_deg=80.2707, alt_m=10.0, name="Chennai-ISRO")
N_PHOTONS = 96


def run_one_pass(pass_df, config: QKDParams, eve: EveAttack,
                 rng) -> pd.DataFrame:
    """Run QKD over one pass with the given config and optional Eve attack."""
    link = LinkModel()
    rows = []
    n = len(pass_df)
    for i, (_, row) in enumerate(pass_df.iterrows()):
        channel = link.channel_noise(row["elevation_deg"], row["mjd"],
                                     GS.lat_deg, GS.lon_deg, rng)
        round_ = run_bb84_round(N_PHOTONS, channel, config, rng)
        frac_t = i / max(n - 1, 1)
        from ..quantum.attacks import run_attacked_sift
        run_attacked_sift(round_, eve, rng, frac_t=frac_t)
        rows.append({"t_sec": float(row["t_sec"]),
                     "qber": float(round_.qber),
                     "secure_bps": float(round_.secure_key_bps),
                     "raw_bps": float(round_.raw_key_bps),
                     "secure_bits": round_.secure_key_bits,
                     "mean_link_q": float(channel["link_quality"]),
                     "attacked": int(round_.attacked)})
    return pd.DataFrame(rows)


def eval_tuned_vs_fixed(passes_meta: pd.DataFrame, orbits_dir: Path,
                        lp: LinkPredictor, rng, n_passes: int) -> dict:
    """Compare AI-tuned vs fixed config secure key rate on held-out passes."""
    meta = passes_meta.sample(min(n_passes, len(passes_meta)), random_state=7)
    fixed_rows, tuned_rows = [], []
    for _, m in meta.iterrows():
        df = pd.read_csv(orbits_dir / f"{m['pass_id']}.csv")
        feats = link_features_from_pass(df, sat_idx=int(m["sat_idx"]))
        best, _, _ = lp.recommend(feats)
        r_fixed = run_one_pass(df, QKDParams(), EveAttack("none", 0.0), rng)
        r_tuned = run_one_pass(df, best, EveAttack("none", 0.0), rng)
        fixed_rows.append(r_fixed["secure_bps"].mean())
        tuned_rows.append(r_tuned["secure_bps"].mean())
    improvement = key_rate_improvement(fixed_rows, tuned_rows)
    return {
        "fixed_bps_mean": mean_secure_bps(fixed_rows),
        "tuned_bps_mean": mean_secure_bps(tuned_rows),
        "improvement_pct": improvement,
        "n_passes": len(fixed_rows),
        "fixed_bps_list": fixed_rows,
        "tuned_bps_list": tuned_rows,
    }


def eval_detector_holdout(qkd_dir: Path, models_dir: Path, det: EveDetector) -> dict:
    """Honest Eve-detector metrics on the *held-out test split* saved by train_ai."""
    split_path = models_dir / "eve_test_split.npz"
    if not split_path.exists():
        raise FileNotFoundError("Run train_ai.py first (it saves eve_test_split.npz).")
    z = np.load(split_path)
    X = z["Xte"]
    y = z["yte"]
    prob = det.predict_proba(X)
    metrics = classifier_metrics(y, prob, threshold=det.operating_threshold)
    fpr, tpr, _ = roc_curve(y, prob)
    return {
        "detector": metrics,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "n_clean": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()),
    }


def eval_attack_sweep(passes_meta: pd.DataFrame, orbits_dir: Path,
                      det: EveDetector, lp: LinkPredictor, rng,
                      strengths=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0),
                      n_passes_each: int = 6) -> dict:
    """What-if sweep: attack strength vs detection rate."""
    meta = passes_meta.sample(min(n_passes_each, len(passes_meta)), random_state=3)
    results = {"strengths": list(strengths), "detection_rates": [],
               "mean_qber": []}
    for s in strengths:
        dets, qbers = [], []
        for _, m in meta.iterrows():
            df = pd.read_csv(orbits_dir / f"{m['pass_id']}.csv")
            r = run_one_pass(df, QKDParams(), EveAttack("intercept_resend", s), rng)
            mean_link_q = float(r["mean_link_q"].mean()) if "mean_link_q" in r else 0.5
            feats_link = link_features_from_pass(
                pd.DataFrame({
                    "elevation_deg": df["elevation_deg"].to_numpy(),
                    "mjd": df["mjd"].to_numpy(),
                    "link_quality": df["link_quality"].to_numpy(),
                }), sat_idx=int(m["sat_idx"]))
            qber_pred, _ = lp.predict(np.concatenate(
                [feats_link, QKDParams().as_array()])[None, :])
            residual = float(r["qber"].mean() - qber_pred[0])
            det_feat = extract_features(r["qber"].to_numpy(), mean_link_q, residual)
            dets.append(int(det.predict(det_feat[None, :])[0]))
            qbers.append(r["qber"].mean())
        results["detection_rates"].append(float(np.mean(dets)))
        results["mean_qber"].append(float(np.mean(qbers)))
    return results


def eval_per_type_detection(qkd_dir: Path, models_dir: Path,
                            lp: LinkPredictor, det: EveDetector) -> dict:
    """Detection rate at the operating threshold, per attack type."""
    rows = []
    for fname in sorted(qkd_dir.glob("qkd_eve_*.csv")):
        df = pd.read_csv(fname)
        aname = df["attack_name"].iloc[0]
        for (pid, sat), g in df.groupby(["pass_id", "sat_idx"]):
            q = g["qber"].to_numpy()
            if len(q) < 4:
                continue
            mean_link_q = float(g["link_quality"].mean())
            feats = link_features_from_pass(pd.DataFrame({
                "elevation_deg": g["elevation_deg"].to_numpy(),
                "mjd": g["mjd"].to_numpy(),
                "link_quality": g["link_quality"].to_numpy(),
            }), sat_idx=int(sat))
            qber_pred, _ = lp.predict(np.concatenate(
                [feats, QKDParams().as_array()])[None, :])
            residual = float(q.mean() - qber_pred[0])
            p = float(det.predict_proba(
                extract_features(q, mean_link_q, residual)[None, :])[0])
            rows.append({"attack": aname, "prob": p})
    r = pd.DataFrame(rows)
    out = {}
    for name, grp in r.groupby("attack"):
        out[name] = {
            "n": int(len(grp)),
            "detection_rate": float((grp["prob"] >= det.operating_threshold).mean()),
            "mean_alarm_conf": float(grp["prob"].mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Full QU-AI-GUARD evaluation")
    ap.add_argument("--orbits", default="data/orbits")
    ap.add_argument("--qkd", default="data/qkd")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    orbits_dir = Path(args.orbits)
    qkd_dir = Path(args.qkd)
    models_dir = Path(args.models)
    out_dir = Path(args.out)
    figs_dir = out_dir / "figures"
    reports_dir = out_dir / "reports"
    figs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    passes_meta = pd.read_csv(orbits_dir / "passes_meta.csv")
    lp = LinkPredictor.load(models_dir / "link_predictor.joblib")
    det = EveDetector.load(models_dir / "eve_detector.joblib")

    print("[1/4] AI-tuned vs fixed config comparison ...")
    tuned = eval_tuned_vs_fixed(passes_meta, orbits_dir, lp, rng, n_passes=12)
    save_fig(plot_key_rate_comparison(tuned["fixed_bps_list"], tuned["tuned_bps_list"]),
             figs_dir / "key_rate_comparison.png")

    print("[2/4] Eve detector hold-out metrics + ROC ...")
    det_ev = eval_detector_holdout(qkd_dir, models_dir, det)
    save_fig(plot_roc(det_ev["roc_fpr"], det_ev["roc_tpr"], det_ev["detector"]["auc"]),
             figs_dir / "eve_detector_roc.png")
    cm = np.array(det_ev["detector"]["confusion"])
    save_fig(plot_confusion(cm), figs_dir / "eve_detector_confusion.png")

    print("[3/4] QBER signature plot (clean vs attack) ...")
    clean_csv = qkd_dir / "qkd_eve_none_s0.0.csv"
    attack_csv = qkd_dir / "qkd_eve_intercept_resend_s0.4.csv"
    if clean_csv.exists() and attack_csv.exists():
        c = pd.read_csv(clean_csv); a = pd.read_csv(attack_csv)
        q_clean = c.groupby("pass_id")["qber"].first().to_numpy()[:30]
        q_att = a.groupby("pass_id")["qber"].first().to_numpy()[:30]
        save_fig(plot_qber_series(q_clean, q_att), figs_dir / "qber_signature.png")

    print("[4/4] Attack-strength sweep ...")
    sweep = eval_attack_sweep(passes_meta, orbits_dir, det, lp, rng)
    save_fig(plot_attack_sweep(sweep["strengths"], sweep["detection_rates"]),
             figs_dir / "attack_sweep.png")

    print("[5/5] Per-type detection table ...")
    per_type = eval_per_type_detection(qkd_dir, models_dir, lp, det)

    all_metrics = {"tuned_vs_fixed": {k: v for k, v in tuned.items()
                                      if not k.endswith("_list")},
                   "detector_holdout": det_ev,
                   "attack_sweep": sweep,
                   "per_type_detection": per_type,
                   "config": {"n_photons": N_PHOTONS, "seed": args.seed}}
    save_metrics(all_metrics, reports_dir / "metrics.json")

    m = det_ev["detector"]
    lines = [
            "# QU-AI-GUARD — Evaluation Report",
        "",
        "## 1. AI auto-tune vs fixed QKD config",
        f"- Fixed config mean secure key rate: **{tuned['fixed_bps_mean']:.1f} bps**",
        f"- AI-tuned mean secure key rate: **{tuned['tuned_bps_mean']:.1f} bps**",
        f"- Improvement: **+{tuned['improvement_pct']:.1f}%** "
        f"(n = {tuned['n_passes']} held-out passes)",
        "",
        "## 2. Eve detector (hold-out)",
        f"- AUC = **{m['auc']:.3f}**",
        f"- Precision = **{m['precision']:.3f}**, Recall = **{m['recall']:.3f}**, "
        f"F1 = **{m['f1']:.3f}**",
        f"- False-alarm rate = **{m['false_alarm_rate']*100:.2f}%** "
        f"at threshold {m['threshold']:.2f}",
        f"- Accuracy = **{m['accuracy']*100:.1f}%** "
        f"(n_clean={det_ev['n_clean']}, n_attack={det_ev['n_attack']})",
        "",
        "## 3. Attack-strength sweep (intercept-resend)",
        "",
        "| Strength | Detection rate | Mean QBER |",
        "|---|---|---|",
    ]
    for s, d, q in zip(sweep["strengths"], sweep["detection_rates"], sweep["mean_qber"]):
        lines.append(f"| {s:.2f} | {d*100:.0f}% | {q:.3f} |")
    lines += [
        "",
        "## 4. Detection by attack type (operating threshold)",
        "",
        "| Attack | Pct detected | Mean alarm confidence |",
        "|---|---|---|",
    ]
    for name, v in per_type.items():
        lines.append(f"| {name} | {v['detection_rate']*100:.0f}% | {v['mean_alarm_conf']:.2f} |")
    lines += [
        "",
        "## 5. Figures",
        "- `figures/key_rate_comparison.png`",
        "- `figures/eve_detector_roc.png`",
        "- `figures/eve_detector_confusion.png`",
        "- `figures/qber_signature.png`",
        "- `figures/attack_sweep.png`",
        "",
        "_All data is simulated. Validation on real QKD missions (EAGLE-1, "
        "QEYSSat) is future work._",
    ]
    write_report("\n".join(lines), reports_dir / "evaluation_report.md")
    print(f"Wrote report + figures to {out_dir}")


if __name__ == "__main__":
    main()
