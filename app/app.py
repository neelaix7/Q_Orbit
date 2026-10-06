# app.py — Q-ORBIT Home Page
# Clean, professional space-research website. All results from real data.
from __future__ import annotations
import os, sys, json, base64
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# Guard: if a stray top-level `app` module (e.g. app.py itself, when the
# working directory is app/) shadowed the package, drop it and re-import.
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── local imports ─────────────────────────────────────────────────────────────
from app.space_theme import (
    css, stars_html, sidebar_logo, apply_layout, CLASS_COLORS, CLASS_NAMES,
    PLOTLY_LAYOUT,
)
from src.simulator.lightcurve_generator import generate_single_light_curve
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.evaluation.robustness_score import robustness_score

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(ROOT, "models", "hybrid_quantum_model.pt")
IMG_DIR    = os.path.join(ROOT, "assets", "images", "web")
REPORTS    = os.path.join(ROOT, "results", "reports")

CLASS_EMOJIS = ["🛰️", "💀", "🚀", "💥", "🕵️"]

st.set_page_config(
    page_title="Q-ORBIT — Space Object Classification",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="collapsed",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def img_b64(fname: str) -> str:
    p = os.path.join(IMG_DIR, fname)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_results() -> dict:
    fc_path = os.path.join(REPORTS, "fair_comparison.json")
    if os.path.exists(fc_path):
        with open(fc_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_robustness(fname: str) -> list:
    p = os.path.join(REPORTS, fname)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model_bundle():
    """Load all trained models. Cached across sessions."""
    import torch, joblib
    from src.classical.cnn_baseline import LightCurveCNN

    bundle: dict = {}

    # Hybrid
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
            m = HybridQuantumClassifier(
                n_features=ckpt.get("n_features", 21),
                n_qubits=ckpt.get("n_qubits", 8),
                n_layers=ckpt.get("n_layers", 2),
                n_classical_hidden=ckpt.get("n_classical_hidden", 16),
                n_classes=5,
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            bundle["hybrid"] = m
            bundle["hybrid_acc"] = ckpt.get("accuracy")
        except Exception as e:
            bundle["hybrid_error"] = str(e)

    # CNN
    cnn_p = os.path.join(ROOT, "models", "classical_cnn_model.pt")
    if os.path.exists(cnn_p):
        try:
            ckpt = torch.load(cnn_p, weights_only=False, map_location="cpu")
            m = LightCurveCNN(n_classes=5)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            bundle["cnn"] = m
            bundle["cnn_acc"] = ckpt.get("accuracy")
        except Exception as e:
            bundle["cnn_error"] = str(e)

    # Scaler
    sc_p = os.path.join(ROOT, "models", "feature_scaler.joblib")
    if os.path.exists(sc_p):
        try:
            bundle["scaler"] = joblib.load(sc_p)
        except Exception:
            pass

    return bundle


# Training regime constant — MUST match frozen-split training data:
# clean regime (fixed noise/dropout handling, no exposure jitter, no class
# overlap injection) and feature sampling interval 5.0s used by every
# training script (extract_all_features default). See data/splits/manifest.json.
TRAIN_SAMPLING_INTERVAL = 5.0


def _sanitize(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def _resample(c: np.ndarray, n: int = 256) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if len(c) == n:
        return c.astype(np.float32)
    if len(c) < 2:
        return np.full(n, float(c[0]) if len(c) == 1 else 0.0, dtype=np.float32)
    xo = np.linspace(0, 1, len(c))
    xn = np.linspace(0, 1, n)
    return np.interp(xn, xo, np.nan_to_num(c)).astype(np.float32)


def predict_all(bundle: dict, curve: np.ndarray) -> dict:
    import torch
    curve = _sanitize(curve)
    # NOTE: sampling interval must equal the training value (5.0s). The
    # previous code used 720/(len-1) ≈ 2.82s, which rescales dominant_freq /
    # harmonic_energy at inference only — a train/inference mismatch.
    try:
        feats = extract_all_features(curve, sampling_interval=TRAIN_SAMPLING_INTERVAL)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        feats = np.zeros(21, dtype=np.float32)

    def _scale(fv):
        sc = bundle.get("scaler")
        if sc is None:
            return fv
        try:
            m = np.asarray(sc.get("mean", np.zeros(21)))
            s = np.asarray(sc.get("scale", sc.get("std", np.ones(21))))
            s = np.where(s == 0, 1.0, s)
            fv = (fv - m[:len(fv)]) / s[:len(fv)]
            return np.nan_to_num(fv, nan=0.0).astype(np.float32)
        except Exception:
            return fv

    def _norm(p):
        p = np.nan_to_num(np.clip(np.asarray(p, dtype=float), 0, 1), nan=0.2)
        s = p.sum()
        return (p / s if s > 0 else np.full_like(p, 0.2)).astype(np.float32)

    out: dict = {}

    if bundle.get("hybrid") is not None:
        try:
            f = _scale(feats)
            with torch.no_grad():
                _, p = bundle["hybrid"](torch.tensor(f[None, :], dtype=torch.float32))
            out["hybrid"] = _norm(p[0].numpy())
        except Exception as e:
            out["hybrid_error"] = str(e)

    if bundle.get("cnn") is not None:
        try:
            c256 = _sanitize(_resample(curve, 256))
            with torch.no_grad():
                logits = bundle["cnn"](
                    torch.tensor(c256[None, None, :], dtype=torch.float32)
                )
                p = torch.softmax(logits, dim=1).numpy()[0]
            out["cnn"] = _norm(p)
        except Exception as e:
            out["cnn_error"] = str(e)

    return out


# ── page layout ───────────────────────────────────────────────────────────────

def render_hero(sat_b64: str, fc: dict):
    cnn_acc = fc.get("Classical_CNN", {}).get("accuracy")
    hyb_acc = fc.get("Hybrid_Quantum", {}).get("accuracy")

    def _fmt(v):
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"
    img_tag  = (
        f'<img src="data:image/jpeg;base64,{sat_b64}" '
        f'alt="satellite" style="width:100%;border-radius:10px;'
        f'border:1px solid #1E2E4A;display:block"/>'
        if sat_b64 else
        '<div style="background:#0D1628;border:1px solid #1E2E4A;'
        'border-radius:10px;height:240px;display:grid;place-items:center;'
        'color:#6B84A8;font-size:0.85rem">Image not found</div>'
    )
    st.markdown(f"""
<div style="padding:2.4rem 0 1.6rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:2rem">
  <div style="display:grid;grid-template-columns:1fr auto;gap:2.5rem;align-items:center">
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                  letter-spacing:3px;color:#38BDF8;margin-bottom:0.7rem">
        QUANTUM-CLASSICAL SPACE RESEARCH
      </div>
      <h1 style="font-size:clamp(2.4rem,5vw,3.8rem);font-weight:700;
                  color:#F0F6FF;line-height:1.05;margin:0 0 0.5rem 0">
        Q-ORBIT
      </h1>
      <h2 style="font-size:clamp(1rem,2.2vw,1.35rem);font-weight:400;
                  color:#6B84A8;margin:0 0 1rem 0;line-height:1.4">
        Quantum-Classical AI for Space Object Classification
      </h2>
      <p style="color:#8A9FBF;font-size:0.95rem;line-height:1.65;
                 max-width:60ch;margin:0 0 1.4rem 0">
        Q-ORBIT investigates whether quantum representations can complement
        classical machine-learning representations for classifying space objects
        from photometric light curves. Two approaches — classical and hybrid —
        are evaluated on the same data, same split, same metrics.
      </p>
      <div style="display:flex;gap:0.6rem;flex-wrap:wrap">
        <span style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                     color:#38BDF8;border-radius:4px;font-family:'Space Mono',monospace;
                     font-size:0.62rem;letter-spacing:1px;padding:0.25rem 0.6rem">
          25,000 LIGHT CURVES
        </span>
        <span style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                     color:#38BDF8;border-radius:4px;font-family:'Space Mono',monospace;
                     font-size:0.62rem;letter-spacing:1px;padding:0.25rem 0.6rem">
          5 OBJECT CLASSES
        </span>
        <span style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                     color:#38BDF8;border-radius:4px;font-family:'Space Mono',monospace;
                     font-size:0.62rem;letter-spacing:1px;padding:0.25rem 0.6rem">
          8 QUBITS
        </span>
        <span style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                     color:#38BDF8;border-radius:4px;font-family:'Space Mono',monospace;
                     font-size:0.62rem;letter-spacing:1px;padding:0.25rem 0.6rem">
           SEED 123 REPRODUCIBLE
        </span>
      </div>
    </div>
    <div style="min-width:220px;max-width:280px">
      {img_tag}
      <div style="margin-top:0.6rem;display:grid;grid-template-columns:1fr 1fr;gap:0.4rem">
        <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
                    padding:0.6rem;text-align:center">
          <div style="font-weight:700;font-size:1.1rem;color:#38BDF8">
            {_fmt(cnn_acc)}
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                      letter-spacing:1px;color:#6B84A8;margin-top:2px">CLASSICAL CNN</div>
        </div>
        <div style="background:#0D1628;border:1px solid rgba(56,189,248,0.25);border-radius:8px;
                    padding:0.6rem;text-align:center">
          <div style="font-weight:700;font-size:1.1rem;color:#38BDF8">
            {_fmt(hyb_acc)}
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                      letter-spacing:1px;color:#6B84A8;margin-top:2px">HYBRID</div>
        </div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_workflow():
    st.markdown("""
<div style="margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
              color:#38BDF8;margin-bottom:0.5rem">ANALYSIS PIPELINE</div>
  <h3 style="font-size:1.1rem;font-weight:600;color:#F0F6FF;margin:0 0 0.7rem 0">
    Same Data · Same Split · Same Metrics
  </h3>
  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;
              background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
              padding:0.85rem 1.1rem;overflow-x:auto">
    <div style="padding:0.4rem 0.8rem;border-radius:5px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;white-space:nowrap">
      Light Curve<br/><span style="color:#6B84A8">256 observations</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem;font-size:1rem">→</span>
    <div style="padding:0.4rem 0.8rem;border-radius:5px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;white-space:nowrap">
      Preprocessing<br/><span style="color:#6B84A8">interp + min-max</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem;font-size:1rem">→</span>
    <div style="padding:0.4rem 0.8rem;border-radius:5px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;white-space:nowrap">
      Feature Extraction<br/><span style="color:#6B84A8">21 features</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem;font-size:1rem">→</span>
    <div style="padding:0.4rem 0.8rem;border-radius:5px;
                background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#38BDF8;white-space:nowrap">
      Classical<br/><span style="color:#6B84A8">CNN / SVM / RF / XGB</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem;font-size:1rem">→</span>
    <div style="padding:0.4rem 0.8rem;border-radius:5px;
                background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.3);
                font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#38BDF8;white-space:nowrap">
      Hybrid ★<br/><span style="color:#6B84A8">classical + quantum · proposed</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem;font-size:1rem">→</span>
    <div style="padding:0.4rem 0.8rem;border-radius:5px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;white-space:nowrap">
      Comparison<br/><span style="color:#6B84A8">accuracy / F1 / robustness</span>
    </div>
  </div>
  <div style="margin-top:0.5rem;font-family:'Space Mono',monospace;font-size:0.6rem;
              color:#6B84A8;letter-spacing:0.8px">
    SAME dataset · SAME 70/15/15 split (seed 123) · SAME 3,750 test samples · 
    NO test leakage · Results from real experiments
  </div>
</div>""", unsafe_allow_html=True)


def render_results_summary(fc: dict):
    """Render 2-column comparison with real metrics from fair_comparison.json."""
    def pct(v):
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) and 0 < v <= 1 else "—"

    def _val(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    cnn_acc = fc.get("Classical_CNN", {}).get("accuracy")
    cnn_f1 = fc.get("Classical_CNN", {}).get("f1_macro")
    hyb_acc = fc.get("Hybrid_Quantum", {}).get("accuracy")
    hyb_f1 = fc.get("Hybrid_Quantum", {}).get("f1_macro")
    svm_acc = fc.get("Classical_SVM", {}).get("accuracy")
    rf_acc = fc.get("Classical_RF", {}).get("accuracy")
    xgb_acc = fc.get("Classical_XGB", {}).get("accuracy")
    pure_acc = fc.get("Pure_Quantum_VQC", {}).get("accuracy")
    pure_f1 = fc.get("Pure_Quantum_VQC", {}).get("f1_macro")

    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
        'letter-spacing:2px;color:#38BDF8;margin-bottom:0.5rem">'
        "EXPERIMENTAL RESULTS</div>"
        '<h3 style="font-size:1.1rem;font-weight:600;color:#F0F6FF;margin:0 0 0.8rem 0">'
        "Two Approaches — One Test Set (3,750 samples, seed 123)</h3>",
        unsafe_allow_html=True,
    )

    col_c, col_h = st.columns(2)
    with col_c:
        st.markdown(
            '<div style="background:#0D1628;border:1px solid #1E2E4A;'
            'border-radius:12px;padding:1.2rem">'
            '<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
            'letter-spacing:2px;color:#6B84A8;margin-bottom:0.5rem">CLASSICAL</div>'
            f'<div style="font-size:1.7rem;font-weight:700;color:#38BDF8;line-height:1">'
            f"{pct(cnn_acc)}</div>"
            '<div style="font-size:0.78rem;color:#8A9FBF;margin-top:0.2rem">'
            "CNN accuracy (best)</div>"
            '<div style="margin-top:0.8rem;padding-top:0.8rem;'
            'border-top:1px solid #1E2E4A;font-size:0.8rem;color:#C8D8F0;line-height:1.8">'
            f"CNN: {pct(cnn_acc)} · F1 {_val(cnn_f1)}<br/>"
            f"SVM: {pct(svm_acc)}<br/>"
            f"RF: {pct(rf_acc)}<br/>"
            f"XGB: {pct(xgb_acc)}</div>"
            '<div style="margin-top:0.7rem;font-family:\'Space Mono\',monospace;'
            'font-size:0.6rem;color:#6B84A8">~280k params · 0 qubits</div></div>',
            unsafe_allow_html=True,
        )
    with col_h:
        st.markdown(
            '<div style="background:#0D1628;border:1px solid rgba(56,189,248,0.25);'
            'border-radius:12px;padding:1.2rem">'
            '<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
            'letter-spacing:2px;color:#38BDF8;margin-bottom:0.5rem">'
            "HYBRID — PROPOSED ★</div>"
            f'<div style="font-size:1.7rem;font-weight:700;color:#38BDF8;line-height:1">'
            f"{pct(hyb_acc)}</div>"
            '<div style="font-size:0.78rem;color:#8A9FBF;margin-top:0.2rem">'
            "Hybrid accuracy</div>"
            '<div style="margin-top:0.8rem;padding-top:0.8rem;'
            'border-top:1px solid rgba(56,189,248,0.15);font-size:0.8rem;'
            'color:#C8D8F0;line-height:1.8">'
            f"Accuracy: {pct(hyb_acc)}<br/>"
            f"F1-macro: {_val(hyb_f1)}<br/>"
            "Architecture: 21→8q+head<br/>"
            "Params: 1,798 total</div>"
            '<div style="margin-top:0.7rem;font-family:\'Space Mono\',monospace;'
            'font-size:0.6rem;color:#38BDF8">'
            "1,798 params · 8 qubits · depth 5</div></div>",
            unsafe_allow_html=True,
        )

    # QML-hero spotlight — best quantum results, honest framing
    st.markdown(
        '<div style="margin-top:1rem;background:rgba(56,189,248,0.05);'
        'border:1px solid rgba(56,189,248,0.25);border-radius:10px;'
        'padding:0.9rem 1.2rem;font-size:0.82rem;color:#C8D8F0;line-height:1.7">'
        '<span style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
        'letter-spacing:2px;color:#38BDF8">BEST QUANTUM RESULTS ★ </span>'
        f"Hybrid <strong style='color:#38BDF8'>{pct(hyb_acc)}</strong> "
        f"(F1 {_val(hyb_f1)}, +2.45pp after continued training) · "
        f"Pure VQC <strong style='color:#38BDF8'>{pct(pure_acc)}</strong> "
        f"(F1 {_val(pure_f1)}, +6.45pp) · "
        "1,798 params — 156× fewer than the CNN. "
        "CNN remains the overall best on clean data; quantum wins on efficiency.</div>",
        unsafe_allow_html=True,
    )


def render_conclusion(fc: dict):
    """Dynamic research conclusion — never hardcoded."""
    hyb_f1  = fc.get("Hybrid_Quantum",   {}).get("f1_macro", 0.0)
    cnn_f1  = fc.get("Classical_CNN",    {}).get("f1_macro", 0.0)
    svm_f1  = fc.get("Classical_SVM",    {}).get("f1_macro", 0.0)
    hyb_acc = fc.get("Hybrid_Quantum",   {}).get("accuracy", 0.0)

    best_cl = max(cnn_f1, svm_f1)
    hybrid_wins = hyb_f1 > best_cl

    if hybrid_wins:
        color  = "rgba(56,189,248,0.08)"
        border = "rgba(56,189,248,0.25)"
        title  = "Hybrid Advantage Demonstrated"
        body   = (
            f"Hybrid F1 {hyb_f1:.4f} surpasses best classical F1 {best_cl:.4f} "
            f"(+{(hyb_f1-best_cl)*100:.2f} pp) on the same test set."
        )
        badge_color = "#38BDF8"
    else:
        color  = "rgba(107,132,168,0.07)"
        border = "#1E2E4A"
        title  = "Hybrid Advantage Not Demonstrated on Clean Data"
        body   = (
            f"Best classical F1 {best_cl:.4f}, Hybrid F1 {hyb_f1:.4f} "
            f"({(hyb_f1-best_cl)*100:+.2f} pp). "
            f"See Experiments page for robustness analysis under degraded observations."
        )
        badge_color = "#6B84A8"

    st.markdown(f"""
<div style="background:{color};border:1px solid {border};border-radius:10px;
            padding:1.1rem 1.3rem;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
              color:{badge_color};margin-bottom:0.35rem">RESEARCH CONCLUSION</div>
  <div style="font-size:0.95rem;font-weight:600;color:#F0F6FF;margin-bottom:0.4rem">
    {title}
  </div>
  <div style="font-size:0.85rem;color:#8A9FBF;line-height:1.6">{body}</div>
  <div style="margin-top:0.6rem;font-family:'Space Mono',monospace;font-size:0.58rem;
              color:#6B84A8">
    All metrics on 3,750-sample frozen test set · seed 123 · PennyLane default.qubit 
    · Never hardcoded · Generated from results/reports/fair_comparison.json
  </div>
</div>""", unsafe_allow_html=True)


def render_comparison_chart(fc: dict):
    """Bar chart: Accuracy + F1 for all models."""
    models = []
    accs   = []
    f1s    = []
    colors = []

    model_map = [
        ("Classical_CNN",    "CNN",          "#38BDF8"),
        ("Classical_SVM",    "SVM",          "#64D8A8"),
        ("Classical_RF",     "RF",           "#64D8A8"),
        ("Classical_XGB",    "XGB",          "#64D8A8"),
        ("Hybrid_Quantum",   "Hybrid ★",    "#38BDF8"),
    ]

    for key, label, color in model_map:
        if key in fc:
            models.append(label)
            accs.append(fc[key].get("accuracy", 0))
            f1s.append(fc[key].get("f1_macro", 0))
            colors.append(color)

    if not models:
        st.info("No results data found. Run experiments first.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=models, y=accs,
        marker_color=colors, opacity=0.9,
        hovertemplate="%{x}: %{y:.3f}<extra>Accuracy</extra>",
    ))
    fig.add_trace(go.Bar(
        name="F1-macro", x=models, y=f1s,
        marker_color=colors, opacity=0.55,
        hovertemplate="%{x}: %{y:.3f}<extra>F1-macro</extra>",
    ))
    apply_layout(fig, height=340, title="Model Comparison — Accuracy & F1-macro (test set)")
    fig.update_layout(barmode="group",
                      yaxis=dict(range=[0, 1.05], title="Score",
                                 gridcolor="rgba(30,46,74,0.6)"),
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_robustness_chart():
    """Robustness noise sweep — real data from JSON."""
    rob = load_robustness("robustness_noise.json")
    if not rob:
        st.caption("Robustness data not found.")
        return

    fig = go.Figure()
    series = [
        ("Classical_CNN", "CNN",         "#38BDF8"),
        ("Hybrid",        "Hybrid ★",   "#F5A623"),
    ]
    for key, label, color in series:
        x = [r.get("noise", r.get("noise_std", 0)) for r in rob if key in r]
        y = [r[key] for r in rob if key in r]
        if x:
            lw = 2.5 if "Hybrid" in label else 1.8
            try:
                rs = robustness_score(y[0], y[1:]) if len(y) > 1 and y[0] else 0.0
                legend_label = f"{label} (RS {rs:.2f})"
            except Exception:
                legend_label = label
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers", name=legend_label,
                line=dict(color=color, width=lw),
                marker=dict(size=6),
                hovertemplate=f"{label}<br>Noise %{{x:.0%}}<br>Acc %{{y:.3f}}<extra></extra>",
            ))
    apply_layout(fig, height=300, title="Robustness — Accuracy vs Noise Level")
    fig.update_layout(
        xaxis=dict(title="Noise std", tickformat=".0%",
                   gridcolor="rgba(30,46,74,0.6)"),
        yaxis=dict(title="Accuracy", range=[0, 1],
                   gridcolor="rgba(30,46,74,0.6)"),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── Inference Lab ─────────────────────────────────────────────────────────────

def render_inference_lab(bundle: dict):
    st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
            color:#38BDF8;margin-bottom:0.4rem">INFERENCE LAB</div>
<h3 style="font-size:1.1rem;font-weight:600;color:#F0F6FF;margin:0 0 0.3rem 0">
  Generate & Classify — Live Pipeline
</h3>
<p style="color:#6B84A8;font-size:0.85rem;margin:0 0 1rem 0">
  Simulate a light curve, extract features, run both models.
  Results come from the actual trained models, not hardcoded values.
</p>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1, 0.8])
    with c1:
        cls_choice = st.selectbox(
            "Object class to simulate",
            options=list(range(5)),
            format_func=lambda c: f"{CLASS_EMOJIS[c]}  {CLASS_NAMES[c]}",
            key="home_cls",
        )
    with c2:
        noise = st.slider("Noise level (std)", 0.0, 0.15, 0.02, 0.01, key="home_noise")
    with c3:
        st.write("")
        launch = st.button("Run Pipeline →", type="primary", key="home_launch")

    if not launch:
        st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1.4rem;text-align:center;color:#6B84A8;font-size:0.85rem">
  Select a class and click <strong style="color:#C8D8F0">Run Pipeline</strong> 
  to generate a light curve and classify it with both models.
</div>""", unsafe_allow_html=True)
        return

    # Generate — clean=True matches the frozen training regime
    # (data/splits/manifest.json: fixed noise 0.02, fixed 5% dropout, no
    # exposure-gain jitter, no class-2/4 boundary overlap). The old default
    # clean=False applied per-curve gain jitter + overlap injection that the
    # trained models never saw, shifting live samples off-distribution.
    with st.spinner("Simulating light curve…"):
        try:
            curve, true_label = generate_single_light_curve(
                int(cls_choice), noise_std=float(noise), n_samples=256,
                clean=True,
            )
            curve = _sanitize(curve)
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            curve = np.clip(np.random.default_rng(0).random(256) * 0.4 + 0.3, 0, 1).astype(np.float32)
            true_label = int(cls_choice)

    # Plot light curve
    t = np.linspace(0, 720, len(curve))
    col_lc, col_res = st.columns([1.3, 1])

    with col_lc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=curve.tolist(),
            mode="lines",
            line=dict(color=CLASS_COLORS[cls_choice], width=1.8),
            fill="tozeroy",
            fillcolor=f"rgba({int(CLASS_COLORS[cls_choice][1:3],16)},"
                      f"{int(CLASS_COLORS[cls_choice][3:5],16)},"
                      f"{int(CLASS_COLORS[cls_choice][5:7],16)},0.08)",
            hovertemplate="t=%{x:.0f}s<br>brightness=%{y:.3f}<extra></extra>",
            name=CLASS_NAMES[cls_choice],
        ))
        apply_layout(fig, height=260,
                     title=f"Light Curve — {CLASS_NAMES[cls_choice]} (noise={noise:.2f})")
        fig.update_layout(
            xaxis=dict(title="Time (s)", gridcolor="rgba(30,46,74,0.6)"),
            yaxis=dict(title="Normalized brightness", range=[0, 1.1],
                       gridcolor="rgba(30,46,74,0.6)"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Inference
    with st.spinner("Running models…"):
        results = predict_all(bundle, curve)

    with col_res:
        # Ground truth is the SELECTED simulation class — kept separate from
        # either model's prediction. Never confuse the two.
        st.markdown(
            f'<div style="background:rgba(100,216,168,0.06);border:1px solid '
            f'rgba(100,216,168,0.25);border-radius:8px;padding:0.7rem;'
            f'margin-bottom:0.6rem">'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
            f'letter-spacing:1.5px;color:#64D8A8;margin-bottom:0.3rem">'
            f'GROUND TRUTH (SIMULATED)</div>'
            f'<div style="font-size:0.92rem;font-weight:600;color:#F0F6FF">'
            f'{CLASS_EMOJIS[int(cls_choice)]} {CLASS_NAMES[int(cls_choice)]}</div>'
            f'<div style="font-size:0.72rem;color:#6B84A8">'
            f'class index {int(cls_choice)} · selected, not predicted</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Check for errors
        for ek in ["hybrid_error", "cnn_error"]:
            if ek in results:
                st.toast(f"{ek.replace('_error','').upper()}: {str(results[ek])[:80]}", icon="⚠️")

        def _pred_block(label: str, key: str, color: str):
            proba = results.get(key)
            if proba is None:
                st.markdown(
                    f'<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;'
                    f'padding:0.7rem;margin-bottom:0.6rem;font-size:0.78rem;color:#6B84A8">'
                    f'{label}: Model not loaded</div>',
                    unsafe_allow_html=True,
                )
                return
            pred = int(np.argmax(proba))
            conf = float(proba[pred]) * 100
            emoji = CLASS_EMOJIS[pred]
            match = (pred == int(cls_choice))
            badge = ("✓ MATCHES ground truth" if match else "✗ DIFFERS from ground truth")
            badge_color = "#64D8A8" if match else "#F5A623"
            st.markdown(
                f'<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;'
                f'padding:0.7rem;margin-bottom:0.6rem">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
                f'letter-spacing:1.5px;color:{color};margin-bottom:0.3rem">{label} — PREDICTION</div>'
                f'<div style="font-size:0.92rem;font-weight:600;color:#F0F6FF">'
                f'{emoji} {CLASS_NAMES[pred]}</div>'
                f'<div style="font-size:0.75rem;color:#6B84A8">'
                f'Confidence: {conf:.1f}% (probability, not accuracy)</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
                f'color:{badge_color};margin-top:0.2rem">{badge}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _pred_block("CNN (Classical)",  "cnn",    "#38BDF8")
        _pred_block("Hybrid",           "hybrid", "#38BDF8")

        def _prob_bars(title: str, proba):
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
                f'letter-spacing:1.5px;color:#6B84A8;margin:0.4rem 0 0.3rem 0">'
                f'{title}</div>',
                unsafe_allow_html=True,
            )
            for i, (cname, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
                pval = float(proba[i]) * 100
                is_pred = (i == int(np.argmax(proba)))
                border = f"1px solid {color}55" if is_pred else "1px solid #1E2E4A"
                st.markdown(
                    f'<div style="background:#111D32;border:{border};border-radius:4px;'
                    f'height:24px;margin:3px 0;overflow:hidden;position:relative">'
                    f'<div style="background:{color};opacity:{"0.75" if is_pred else "0.45"};'
                    f'width:{pval:.1f}%;height:100%;border-radius:4px;'
                    f'display:flex;align-items:center;padding-left:8px;'
                    f'font-size:0.7rem;color:#F0F6FF;white-space:nowrap;overflow:hidden">'
                    f'{cname} — {pval:.1f}%</div></div>',
                    unsafe_allow_html=True,
                )

        # Probability bars for both models (index i always maps to CLASS_NAMES[i])
        if results.get("hybrid") is not None:
            _prob_bars("HYBRID CLASS PROBABILITIES", results["hybrid"])
        if results.get("cnn") is not None:
            _prob_bars("CNN CLASS PROBABILITIES", results["cnn"])

        # Full pipeline trace: true class → features → input → prediction
        with st.expander("Pipeline trace (audit)"):
            try:
                dt = TRAIN_SAMPLING_INTERVAL
                raw_feats = extract_all_features(curve, sampling_interval=dt)
                sc = bundle.get("scaler")
                if sc is not None:
                    m = np.asarray(sc.get("mean", np.zeros(21)))
                    s = np.asarray(sc.get("scale", sc.get("std", np.ones(21))))
                    s = np.where(s == 0, 1.0, s)
                    x_feats = (raw_feats - m[:len(raw_feats)]) / s[:len(raw_feats)]
                else:
                    x_feats = raw_feats
                trace_lines = [
                    f"true class: {CLASS_NAMES[int(cls_choice)]} (index {int(cls_choice)})",
                    f"curve shape: {np.asarray(curve).shape}, range [{float(np.min(curve)):.3f}, {float(np.max(curve)):.3f}]",
                    f"feature order (21): mean..min_value | dominant_freq..n_harmonics | flash_count..period_estimate",
                    f"raw features[:5]: {np.asarray(raw_feats).ravel()[:5].round(4).tolist()}",
                    f"scaled features[:5]: {np.asarray(x_feats).ravel()[:5].round(4).tolist()}",
                    f"hybrid input shape: (1, {np.asarray(x_feats).ravel().shape[0]})",
                    f"cnn input shape: (1, 1, {len(np.asarray(curve).ravel())})",
                ]
                for key, mname in (("cnn", "CNN"), ("hybrid", "Hybrid")):
                    p = results.get(key)
                    if p is not None:
                        pi = int(np.argmax(p))
                        trace_lines.append(
                            f"{mname}: pred index {pi} = {CLASS_NAMES[pi]} | "
                            + " | ".join(f"{CLASS_NAMES[i]}={float(p[i])*100:.1f}%" for i in range(5))
                        )
                st.code("\n".join(trace_lines), language="text")
            except Exception as e:
                st.caption(f"Trace unavailable: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Inject CSS + stars
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(), unsafe_allow_html=True)
    sidebar_logo(st)

    # Sidebar nav hint
    with st.sidebar:
        st.markdown("""
<div style="padding:0.5rem;font-size:0.8rem;color:#6B84A8;line-height:1.7">
  <strong style="color:#C8D8F0;font-size:0.82rem">Navigation</strong><br/>
  Use the links above to explore:<br/>
  • <strong style="color:#38BDF8">Data</strong> — Dataset explorer<br/>
  • <strong style="color:#38BDF8">Classical AI</strong> — ML models<br/>
  • <strong style="color:#38BDF8">Hybrid</strong> — Classical + quantum approach<br/>
  • <strong style="color:#38BDF8">Experiments</strong> — Full comparison<br/>
</div>""", unsafe_allow_html=True)

    # Load data
    fc     = load_results()
    bundle = load_model_bundle()
    sat_b64 = img_b64("surveillance_satellite.jpg") or img_b64("satellite_other.jpg")

    # ── Hero ──────────────────────────────────────────────────────────────────
    render_hero(sat_b64, fc)

    # ── Workflow ──────────────────────────────────────────────────────────────
    render_workflow()

    # ── Dataset overview strip ────────────────────────────────────────────────
    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0.6rem;margin-bottom:1.6rem">
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.75rem;text-align:center">
    <div style="font-size:1.35rem;font-weight:700;color:#F0F6FF">25,000</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">LIGHT CURVES</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.75rem;text-align:center">
    <div style="font-size:1.35rem;font-weight:700;color:#F0F6FF">5</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">CLASSES</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.75rem;text-align:center">
    <div style="font-size:1.35rem;font-weight:700;color:#F0F6FF">21</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">FEATURES</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.75rem;text-align:center">
    <div style="font-size:1.35rem;font-weight:700;color:#F0F6FF">256</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">OBSERVATIONS</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.75rem;text-align:center">
    <div style="font-size:1.35rem;font-weight:700;color:#38BDF8">8</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">QUBITS</div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
              padding:0.75rem;text-align:center">
    <div style="font-size:1.35rem;font-weight:700;color:#F0F6FF">720s</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                letter-spacing:1px;color:#6B84A8;margin-top:2px">OBS WINDOW</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if fc:
        render_results_summary(fc)
        render_conclusion(fc)

        col_chart, col_rob = st.columns(2, gap="large")
        with col_chart:
            render_comparison_chart(fc)
        with col_rob:
            render_robustness_chart()
    else:
        st.info(
            "No experiment results found. Run `python experiments/run_comprehensive_fair.py` "
            "to generate results, then refresh."
        )

    # ── Inference Lab ─────────────────────────────────────────────────────────
    st.divider()
    render_inference_lab(bundle)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin-top:2.5rem;padding:1.2rem;background:#0D1628;
            border:1px solid #1E2E4A;border-radius:10px;
            font-size:0.78rem;color:#6B84A8;line-height:1.7">
  <strong style="color:#C8D8F0">Q-ORBIT</strong> — 
  Quantum-Classical AI Framework for Space Object Classification<br/>
   B.Tech Capstone 2026 · Physics-informed synthetic dataset · 
   PennyLane default.qubit simulator · No paid APIs · Seed 123 reproducible<br/>
  Stack: Python · PyTorch · PennyLane · scikit-learn · XGBoost · Streamlit · Plotly
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
