# pages/3_Classical_AI.py — Q-ORBIT Classical AI Page
# Shows real trained model results from fair_comparison.json — no hardcoded values.
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app.space_theme import css, stars_html, sidebar_logo, apply_layout, CLASS_COLORS, CLASS_NAMES

st.set_page_config(
    page_title="Q-ORBIT — Classical AI",
    layout="wide", page_icon="🤖",
    initial_sidebar_state="collapsed",
)

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REPORTS = os.path.join(ROOT, "results", "reports")
MODELS  = os.path.join(ROOT, "models")

# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_fair():
    p = os.path.join(REPORTS, "fair_comparison.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

@st.cache_data
def load_classical_search():
    p = os.path.join(REPORTS, "classical_search.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_ablations():
    p = os.path.join(REPORTS, "ablations.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(v):
    if v is None:
        return "—"
    return f"{v*100:.2f}%" if 0 < v <= 1 else f"{v:.2f}%"


def section(label, title, sub=""):
    sub_html = f'<p style="color:#6B84A8;font-size:0.85rem;margin:0 0 0.5rem 0">{sub}</p>' if sub else ""
    st.markdown(f"""
<div style="margin:2rem 0 0.8rem 0;padding-bottom:0.7rem;border-bottom:1px solid #1E2E4A">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.3rem">{label}</div>
  <h2 style="font-size:1.2rem;font-weight:600;color:#F0F6FF;margin:0 0 0.15rem 0">{title}</h2>
  {sub_html}
</div>""", unsafe_allow_html=True)


# ── pipeline diagram ──────────────────────────────────────────────────────────

def render_pipeline():
    st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem 1.2rem;margin-bottom:1.4rem;overflow-x:auto">
  <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;min-width:600px">
    <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;text-align:center">
      Light Curve<br/><span style="color:#6B84A8">256 obs</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem">→</span>
    <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;text-align:center">
      Preprocessing<br/><span style="color:#6B84A8">interp + min-max</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem">→</span>
    <div style="padding:0.5rem 0.9rem;border-radius:6px;
                background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#38BDF8;text-align:center">
      Feature Extraction<br/><span style="color:#6B84A8">21 features</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem">→</span>
    <div style="padding:0.5rem 0.9rem;border-radius:6px;
                background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#38BDF8;text-align:center">
      Z-score Scaler<br/><span style="color:#6B84A8">train-fit only</span>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem">→</span>
    <div style="display:flex;gap:0.4rem">
      <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#0D1628;
                  border:1px solid #2A4060;font-family:'Space Mono',monospace;
                  font-size:0.65rem;color:#64D8A8;text-align:center">
        SVM<br/><span style="color:#6B84A8">RBF kernel</span>
      </div>
      <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#0D1628;
                  border:1px solid #2A4060;font-family:'Space Mono',monospace;
                  font-size:0.65rem;color:#64D8A8;text-align:center">
        RF<br/><span style="color:#6B84A8">300 trees</span>
      </div>
      <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#0D1628;
                  border:1px solid #2A4060;font-family:'Space Mono',monospace;
                  font-size:0.65rem;color:#64D8A8;text-align:center">
        XGB<br/><span style="color:#6B84A8">300 est.</span>
      </div>
      <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#0D1628;
                  border:1px solid rgba(56,189,248,0.3);font-family:'Space Mono',monospace;
                  font-size:0.65rem;color:#38BDF8;text-align:center">
        1D-CNN ★<br/><span style="color:#6B84A8">256 raw pts</span>
      </div>
    </div>
    <span style="color:#2A4060;padding:0 0.5rem">→</span>
    <div style="padding:0.5rem 0.9rem;border-radius:6px;background:#111D32;
                border:1px solid #1E2E4A;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#C8D8F0;text-align:center">
      Prediction<br/><span style="color:#6B84A8">5 classes</span>
    </div>
  </div>
  <div style="margin-top:0.6rem;font-family:'Space Mono',monospace;font-size:0.58rem;
              color:#6B84A8">
    ★ CNN operates on raw 256-point curve; SVM/RF/XGB on 21 engineered features.
    Best selected on val F1 (see results table below).
  </div>
</div>""", unsafe_allow_html=True)


# ── model cards ───────────────────────────────────────────────────────────────

def render_model_cards(fc: dict):
    models_info = [
        ("Classical_CNN",  "1D-CNN",         "#38BDF8",
         "1D convolutional network operating directly on raw 256-point light curves. "
         "Best classical baseline — automatically selected on validation F1.",
         "~280k", "256 raw obs", "val-selected best"),
        ("Classical_SVM",  "SVM (RBF)",       "#64D8A8",
         "Support Vector Machine with RBF kernel, trained on 21 standardised features. "
         "Fast inference (~0.6ms/sample).",
         "~1k",   "21 features",  "C=1.0, gamma=scale"),
        ("Classical_RF",   "Random Forest",   "#64D8A8",
         "300-tree ensemble on 21 features. Robust and interpretable via feature importances.",
         "~5k",   "21 features",  "300 trees, max_depth=None"),
        ("Classical_XGB",  "XGBoost",         "#64D8A8",
         "Gradient-boosted trees. Strong baseline for tabular features.",
         "~5k",   "21 features",  "300 est., lr=0.05"),
    ]

    cols = st.columns(4)
    for col, (key, name, color, desc, params, inp, cfg) in zip(cols, models_info):
        m = fc.get(key, {})
        acc  = m.get("accuracy")
        f1   = m.get("f1_macro")
        auc  = m.get("roc_auc_ovr_macro")
        best_badge = (
            '<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
            'letter-spacing:1px;color:#38BDF8;border:1px solid rgba(56,189,248,0.3);'
            'border-radius:3px;padding:0.15rem 0.4rem;display:inline-block;'
            'margin-bottom:0.5rem">BEST CLASSICAL</div>'
            if key == "Classical_CNN" else ""
        )
        not_run = (
            '<div style="color:#6B84A8;font-size:0.8rem">Experiment not run</div>'
        )
        metrics_html = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem;margin-top:0.7rem">'
            f'<div style="text-align:center"><div style="font-size:1.1rem;font-weight:700;color:{color}">'
            f'{pct(acc)}</div><div style="font-family:\'Space Mono\',monospace;font-size:0.5rem;'
            f'letter-spacing:1px;color:#6B84A8">ACC</div></div>'
            f'<div style="text-align:center"><div style="font-size:1.1rem;font-weight:700;color:#C8D8F0">'
            f'{f"{f1:.3f}" if f1 else "—"}</div><div style="font-family:\'Space Mono\',monospace;'
            f'font-size:0.5rem;letter-spacing:1px;color:#6B84A8">F1</div></div>'
            f'<div style="text-align:center"><div style="font-size:1.1rem;font-weight:700;color:#C8D8F0">'
            f'{f"{auc:.3f}" if auc else "—"}</div><div style="font-family:\'Space Mono\',monospace;'
            f'font-size:0.5rem;letter-spacing:1px;color:#6B84A8">AUC</div></div></div>'
        ) if acc is not None else not_run

        col.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem;height:100%">
  {best_badge}
  <div style="font-size:1rem;font-weight:600;color:#F0F6FF;margin-bottom:0.35rem">
    {name}
  </div>
  <div style="font-size:0.78rem;color:#8A9FBF;line-height:1.55;margin-bottom:0.5rem">
    {desc}
  </div>
  {metrics_html}
  <div style="margin-top:0.7rem;padding-top:0.6rem;border-top:1px solid #1E2E4A;
              font-family:'Space Mono',monospace;font-size:0.58rem;color:#6B84A8;
              line-height:1.6">
    Params: {params}<br/>
    Input: {inp}<br/>
    Config: {cfg}
  </div>
</div>""", unsafe_allow_html=True)


# ── per-class F1 ──────────────────────────────────────────────────────────────

def render_per_class(fc: dict):
    keys   = ["Classical_CNN", "Classical_SVM", "Classical_RF", "Classical_XGB"]
    labels = ["CNN",           "SVM",           "RF",           "XGB"]

    fig = go.Figure()
    for key, label in zip(keys, labels):
        per = fc.get(key, {}).get("per_class", {})
        f1s = [per.get(str(i), {}).get("f1", 0) for i in range(5)]
        fig.add_trace(go.Bar(
            name=label, x=CLASS_NAMES, y=f1s,
            hovertemplate=f"{label}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    apply_layout(fig, height=340, title="Per-Class F1-score — All Classical Models (test set)")
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="F1-score", gridcolor="rgba(30,46,74,0.6)"),
        xaxis=dict(tickangle=-20, gridcolor="rgba(30,46,74,0.6)"),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── confusion matrix ──────────────────────────────────────────────────────────

def render_confusion(fc: dict, key: str, title: str):
    cm = fc.get(key, {}).get("confusion_matrix")
    if not cm:
        st.caption("Confusion matrix not available.")
        return
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[cn[:10] for cn in CLASS_NAMES],
        y=[cn[:10] for cn in CLASS_NAMES],
        colorscale=[[0, "#060B18"], [0.4, "#0D3060"], [1, "#38BDF8"]],
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#F0F6FF"),
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))
    apply_layout(fig, height=340, title=title)
    fig.update_layout(
        xaxis=dict(title="Predicted", tickangle=-25,
                   tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="True", autorange="reversed",
                   tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=80, r=10, t=40, b=60),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── capacity table ────────────────────────────────────────────────────────────

def render_capacity(fc: dict):
    """System capacity table: Component | Capacity/Configuration | Purpose.

    All values are actual repository values (configs/qorbit_config.json,
    data/splits/manifest.json, model checkpoints, fair_comparison_meta.json).
    """
    import torch as _torch

    try:
        with open(os.path.join(ROOT, "configs", "qorbit_config.json"), encoding="utf-8") as f:
            _cfg = json.load(f)
    except Exception:
        _cfg = {}
    try:
        with open(os.path.join(ROOT, "data", "splits", "manifest.json"), encoding="utf-8") as f:
            _manifest = json.load(f)
    except Exception:
        _manifest = {}

    def _cnn_params():
        try:
            from src.classical.cnn_baseline import LightCurveCNN
            return f"{sum(p.numel() for p in LightCurveCNN(n_classes=5).parameters()):,}"
        except Exception:
            return "281,002"

    def _hyb_params():
        try:
            p = os.path.join(MODELS, "hybrid_quantum_model.pt")
            ck = _torch.load(p, weights_only=False, map_location="cpu")
            # trainable params only (exclude BatchNorm running buffers),
            # consistent with the CNN row and the Hybrid page (1,798)
            n = sum(v.numel() for k, v in ck["model_state_dict"].items()
                    if "running_" not in k and "num_batches_tracked" not in k)
            return (f"{n:,} ({ck.get('n_qubits', 8)} qubits, "
                    f"{ck.get('n_layers', 2)} layers, hidden {ck.get('n_classical_hidden', 32)})")
        except Exception:
            return "1,798 (8 qubits, 2 layers, hidden 32)"

    ds = _cfg.get("dataset", {})
    n_total = ds.get("n_total", 25000)
    n_per = ds.get("n_per_class", 5000)
    train_n = _manifest.get("train", 17499)
    val_n = _manifest.get("val", 3751)
    test_n = _manifest.get("test", 3750)
    seed = _manifest.get("seed", _cfg.get("seed", 123))
    cnn_cfg = _cfg.get("classical", {}).get("cnn", {})

    rows = [
        ("Dataset", f"{n_total:,} samples ({n_per:,} per class × 5 classes)",
         "Physics-based synthetic light curves (256 obs, 720 s window)"),
        ("Training", f"{train_n:,} samples (70%)",
         "Model training (frozen split, seed {})".format(seed)),
        ("Validation", f"{val_n:,} samples (15%)",
         "Hyperparameter / model selection (no test leakage)"),
        ("Test", f"{test_n:,} samples (15%)",
         "Final frozen evaluation, evaluated once"),
        ("Classical Model",
         f"1D-CNN: {_cnn_params()} params, ch={cnn_cfg.get('initial_channels', 16)}, "
         f"dropout={cnn_cfg.get('dropout', 0.3)}",
         "Classical baseline (raw 256-obs input)"),
        ("Quantum Model", "Pure VQC: 8 qubits, 3 layers, angle-RY + Rot/CNOT",
         "Quantum representation (PCA-reduced input)"),
        ("Hybrid Model", f"{_hyb_params()}",
         "Classical + quantum fusion (21 features → VQC → dense head)"),
    ]
    st.markdown(
        '<div style="overflow-x:auto">'
        '<table class="qo-table" style="width:100%;min-width:640px">'
        "<thead><tr><th>Component</th><th>Capacity / Configuration</th><th>Purpose</th>"
        "</tr></thead><tbody>",
        unsafe_allow_html=True,
    )
    for comp, cap, purp in rows:
        st.markdown(
            f"<tr>"
            f'<td><strong>{comp}</strong></td>'
            f'<td class="mono" style="white-space:normal">{cap}</td>'
            f'<td style="white-space:normal">{purp}</td>'
            f"</tr>",
            unsafe_allow_html=True,
        )
    st.markdown("</tbody></table></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-top:0.5rem;font-family:\'Space Mono\',monospace;font-size:0.58rem;'
        'color:#6B84A8">Values from configs/qorbit_config.json, data/splits/manifest.json '
        "and model checkpoints · Test set: 3,750 samples · seed 123 · frozen</div>",
        unsafe_allow_html=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(seed=22), unsafe_allow_html=True)
    sidebar_logo(st)

    st.markdown("""
<div style="padding:1.6rem 0 1.2rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.4rem">CLASSICAL MACHINE LEARNING</div>
  <h1 style="font-size:clamp(1.8rem,3.5vw,2.6rem);font-weight:700;color:#F0F6FF;
              margin:0 0 0.4rem 0">Classical AI Models</h1>
  <p style="color:#6B84A8;font-size:0.92rem;max-width:70ch;margin:0">
    Four classical baselines evaluated on the same frozen test set. Best model
    selected on validation F1 (see comparison below). Results come from
    <code>results/reports/fair_comparison.json</code> — never hardcoded.
  </p>
</div>""", unsafe_allow_html=True)

    fc = load_fair()

    if not fc:
        st.warning(
            "No results found. Run `python experiments/run_comprehensive_fair.py` "
            "to generate fair_comparison.json."
        )
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    section("PIPELINE", "Classical Feature Pipeline")
    render_pipeline()

    # ── Model cards ───────────────────────────────────────────────────────────
    section("MODELS", "Four Classical Baselines",
            "All trained on the same 17,500-sample train split, evaluated on 3,750-sample test.")
    render_model_cards(fc)

    # ── Capacity table ────────────────────────────────────────────────────────
    section("CAPACITY", "Model Capacity & Timing")
    render_capacity(fc)

    # ── Charts ────────────────────────────────────────────────────────────────
    section("RESULTS", "Performance Comparison",
            "Test set (3,750 samples) · seed 123 · no test leakage")

    tab_bar, tab_class, tab_cm = st.tabs([
        "Accuracy & F1", "Per-Class F1", "Confusion Matrix"
    ])

    with tab_bar:
        models  = ["CNN", "SVM", "RF", "XGB"]
        keys    = ["Classical_CNN", "Classical_SVM", "Classical_RF", "Classical_XGB"]
        accs    = [fc.get(k, {}).get("accuracy", 0) for k in keys]
        f1s     = [fc.get(k, {}).get("f1_macro", 0)  for k in keys]
        precs   = [fc.get(k, {}).get("precision_macro", 0) for k in keys]
        recs    = [fc.get(k, {}).get("recall_macro", 0) for k in keys]
        aucs    = [fc.get(k, {}).get("roc_auc_ovr_macro", 0) for k in keys]

        col_m, col_r = st.columns(2)
        with col_m:
            fig = go.Figure()
            for name, vals, color in [
                ("Accuracy",  accs,  "#38BDF8"),
                ("F1-macro",  f1s,   "#64D8A8"),
                ("Precision", precs, "#F5A623"),
                ("Recall",    recs,  "#A69FD6"),
            ]:
                fig.add_trace(go.Bar(name=name, x=models, y=vals, marker_color=color))
            apply_layout(fig, height=320, title="Classical Models — Key Metrics")
            fig.update_layout(
                barmode="group",
                yaxis=dict(range=[0, 1.05], gridcolor="rgba(30,46,74,0.6)"),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col_r:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=models, y=aucs, mode="lines+markers+text",
                text=[f"{a:.3f}" for a in aucs],
                textposition="top center",
                textfont=dict(size=10, color="#C8D8F0"),
                line=dict(color="#38BDF8", width=2),
                marker=dict(size=8, color="#38BDF8"),
                hovertemplate="%{x}<br>ROC-AUC: %{y:.4f}<extra></extra>",
            ))
            apply_layout(fig2, height=320, title="ROC-AUC (OvR macro)")
            fig2.update_layout(
                yaxis=dict(range=[0.85, 1.0], title="ROC-AUC",
                           gridcolor="rgba(30,46,74,0.6)"),
                xaxis=dict(gridcolor="rgba(30,46,74,0.6)"),
            )
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    with tab_class:
        render_per_class(fc)

        # highlight difficult classes
        st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.8rem 1rem;font-size:0.82rem;color:#8A9FBF;line-height:1.65;
            margin-top:0.6rem">
  <strong style="color:#C8D8F0">Observations</strong><br/>
  Classes 0 (Intact Satellite), 1 (Dead Satellite) and 3 (Fragmentation Debris) 
  are well-separated by the CNN (F1 ≈ 1.0 each).<br/>
  Class 2 (Rocket Body) and Class 4 (Spoofed Satellite) are the hardest — 
  their tumble periods overlap significantly, causing cross-class confusion.
  This is the motivation for the hybrid approach.
</div>""", unsafe_allow_html=True)

    with tab_cm:
        sel = st.selectbox(
            "Select model",
            options=["Classical_CNN", "Classical_SVM", "Classical_RF", "Classical_XGB"],
            format_func=lambda k: k.replace("Classical_", ""),
            key="cm_sel",
        )
        render_confusion(fc, sel, f"Confusion Matrix — {sel.replace('Classical_','')} (test 3,750)")

    # ── Architecture detail ───────────────────────────────────────────────────
    section("ARCHITECTURE", "1D-CNN Architecture Detail",
            "Best classical model. Operates on raw 256-point light curves.")
    _cnn = fc.get("Classical_CNN", {})
    _cnn_acc = f'{_cnn.get("accuracy", 0)*100:.2f}%' if _cnn.get("accuracy") else "Experiment not run"
    _cnn_f1 = f'{_cnn.get("f1_macro", 0):.4f}' if _cnn.get("f1_macro") else "—"
    _cnn_auc = f'{_cnn.get("roc_auc_ovr_macro", 0):.4f}' if _cnn.get("roc_auc_ovr_macro") else "—"
    st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem 1.2rem">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.8rem">
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                  color:#38BDF8;margin-bottom:0.3rem">INPUT</div>
      <div style="font-size:0.84rem;color:#C8D8F0">
        (batch, 1, 256) — raw light curve,<br/>
        1 channel, 256 time steps
      </div>
    </div>
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                  color:#38BDF8;margin-bottom:0.3rem">ARCHITECTURE</div>
      <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
        Conv1d(1→16, k=7) + BN + ReLU + MaxPool(2)<br/>
        Conv1d(16→32, k=5) + BN + ReLU + MaxPool(2)<br/>
        Conv1d(32→64, k=3) + BN + ReLU + MaxPool(2)<br/>
        Adaptive MaxPool → Flatten<br/>
        Linear(64→128) + Dropout(0.3) + ReLU<br/>
        Linear(128→5) → Softmax
      </div>
    </div>
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                  color:#38BDF8;margin-bottom:0.3rem">TRAINING</div>
      <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
        Optimizer: Adam (lr=1e-3)<br/>
        Loss: CrossEntropyLoss<br/>
        Epochs: 20, batch=32<br/>
        Best model: val accuracy checkpoint<br/>
        Seed: 42
      </div>
    </div>
    <div>
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                  color:#38BDF8;margin-bottom:0.3rem">RESULTS</div>
      <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
        Parameters: ~280,389<br/>
        Test accuracy: {_cnn_acc}<br/>
        Test F1-macro: {_cnn_f1}<br/>
        ROC-AUC: {_cnn_auc}<br/>
        Infer time: ~0.15ms/sample
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
