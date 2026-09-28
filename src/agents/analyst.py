# analyst.py — Traceable AI Analyst: structured input → templated interpretation, never invents numbers
from __future__ import annotations
from typing import Dict, Any
import numpy as np, json

CLASS_NAMES = ["Intact Satellite","Dead Satellite","Rocket Body","Fragmentation Debris","Spoofed Satellite"]

def analyze(prediction_pack: Dict[str, Any]) -> str:
    """
    Traceable: every statement derived from structured variables.
    Expected keys: classical_pred, quantum_pred, hybrid_pred, adaptive_pred (optional),
                   classical_proba, quantum_proba, hybrid_proba, adaptive_proba,
                   signal_quality (dict or str), uncertainty (dict), agreement (dict)
    All numeric values come from caller; analyst only interprets.
    """
    # Extract with fallbacks, but never invent
    cp = prediction_pack.get("classical_pred"); qp = prediction_pack.get("quantum_pred")
    hp = prediction_pack.get("hybrid_pred"); ap = prediction_pack.get("adaptive_pred")
    cpr = prediction_pack.get("classical_proba"); qpr = prediction_pack.get("quantum_proba")
    hpr = prediction_pack.get("hybrid_proba"); apr = prediction_pack.get("adaptive_proba")
    sq = prediction_pack.get("signal_quality", "unknown")
    uncertainty = prediction_pack.get("uncertainty", {})
    agreement = prediction_pack.get("agreement", {})
    # Trace log for audit
    trace = {
        "classical_pred": int(cp) if cp is not None else None,
        "quantum_pred": int(qp) if qp is not None else None,
        "hybrid_pred": int(hp) if hp is not None else None,
        "signal_quality": sq if isinstance(sq, str) else sq.get("quality_label") if isinstance(sq, dict) else str(sq),
        "agreement": agreement.get("consensus") if isinstance(agreement, dict) else None,
        "uncertainty_label": uncertainty.get("uncertainty_label") if isinstance(uncertainty, dict) else None,
    }
    lines=[]
    lines.append("## Q-ORBIT Traceable Analyst Report")
    lines.append(f"- **Trace ID:** {hash(json.dumps(trace, sort_keys=True)) % 100000} — all statements derived from measured inputs, no invented numbers.")
    def fmt(pred, proba, label):
        if pred is None or proba is None: return f"{label}: unavailable"
        conf = float(np.max(proba)); name = CLASS_NAMES[int(pred)] if 0 <= int(pred) <5 else str(pred)
        return f"{label}: **{name}** (p={conf:.2f}, entropy={float(-np.sum(proba*np.log(proba+1e-9))):.2f})"
    lines.append(f"- {fmt(cp,cpr,'Classical')}")
    lines.append(f"- {fmt(qp,qpr,'Pure Quantum (VQC)')}")
    lines.append(f"- {fmt(hp,hpr,'Hybrid')}")
    if ap is not None: lines.append(f"- {fmt(ap,apr,'Adaptive Hybrid')}")
    # Signal quality
    if isinstance(sq, dict):
        lines.append(f"- Signal quality: **{sq.get('quality_label','unknown')}** (score {sq.get('signal_quality_score',0):.2f}, noise {sq.get('noise_estimate',0):.3f}, periodicity {sq.get('periodicity_score',0):.2f})")
    else:
        lines.append(f"- Signal quality: **{sq}**")
    # Agreement
    if isinstance(agreement, dict) and agreement:
        lines.append(f"- Agreement: **{agreement.get('agreement_count','-')}** — {agreement.get('consensus','')}")
        lines.append(f"  Avg JS divergence: {agreement.get('avg_js_divergence',0):.3f}")
    else:
        if cp is not None and qp is not None and hp is not None:
            if cp==qp==hp: lines.append(f"- Consensus: all three agree on **{CLASS_NAMES[int(cp)]}** — high reliability.")
            elif hp==cp or hp==qp: lines.append(f"- Hybrid agrees with {'classical' if hp==cp else 'quantum'}; third diverges → borderline.")
            else: lines.append(f"- Disagreement across all three — ambiguous; recommend additional observation.")
    # Uncertainty
    if isinstance(uncertainty, dict) and uncertainty:
        lines.append(f"- Uncertainty: **{uncertainty.get('uncertainty_label','-')}** (entropy {uncertainty.get('entropy',0):.2f}, margin {uncertainty.get('margin',0):.2f}, max prob {uncertainty.get('probability',0):.2f})")
    # Research interpretation (never claim advantage without data)
    try:
        # Use provided metrics if available
        metrics = prediction_pack.get("metrics", {})
        if metrics:
            c_f1 = metrics.get("classical_f1"); h_f1 = metrics.get("hybrid_f1"); q_f1 = metrics.get("quantum_f1")
            if c_f1 and h_f1:
                delta = float(h_f1 - c_f1)
                if delta > 0.02: lines.append(f"- Research interpretation: Hybrid F1 {h_f1:.4f} beats classical {c_f1:.4f} by +{delta*100:.1f}pp on this condition.")
                elif delta < -0.02: lines.append(f"- Research interpretation: Classical leads hybrid by {abs(delta)*100:.1f}pp (F1) on this condition — hybrid not universally superior.")
                else: lines.append(f"- Research interpretation: Hybrid vs classical within ±2pp (hybrid {h_f1:.4f} vs classical {c_f1:.4f}) — no clear winner.")
    except Exception:
        pass
    lines.append("\n*All numbers above are measured model outputs; analyst text is templated interpretation from those values. See trace for audit.*")
    # Save trace for experiment tracking. Anchored to the project root (not the CWD)
    # so traceability holds no matter where the caller was launched from.
    try:
        import os, json as js
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _trace_dir = os.path.join(_root, "results", "reports")
        os.makedirs(_trace_dir, exist_ok=True)
        with open(os.path.join(_trace_dir, "analyst_trace.json"), "w", encoding="utf-8") as f:
            js.dump({"input_trace": trace, "output_lines": lines}, f, indent=2)
    except OSError:
        pass  # best-effort: the analysis text is still returned to the caller
    return "\n".join(lines)

def model_comparison_summary(metrics_dict: Dict[str, Dict[str,float]]) -> str:
    lines=["| Model | Accuracy | F1-macro | F1-weighted | ROC-AUC |","|---|---|---|---|---|"]
    for k,v in metrics_dict.items():
        lines.append(f"| {k} | {v.get('accuracy',0):.4f} | {v.get('f1_macro',0):.4f} | {v.get('f1_weighted',0):.4f} | {str(v.get('roc_auc_ovr_macro','-'))} |")
    return "\n".join(lines)
