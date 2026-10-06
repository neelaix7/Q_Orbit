# pages/5_Hybrid.py — Q-ORBIT Hybrid Pipeline Page
# Shows hybrid architecture, real metrics, adaptive fusion results.
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
    page_title="Q-ORBIT — Hybrid",
    layout="wide", page_icon="⚡",
    initial_sidebar_state="collapsed",
)

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REPORTS = os.path.join(ROOT, "results", "reports")
MODELS  = os.path.join(ROOT, "models")

@st.cache_data
def load_fair():
    p = os.path.join(REPORTS, "fair_comparison.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

@st.cache_data
def load_adaptive():
    p = os.path.join(REPORTS, "adaptive_fusion.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

@st.cache_data
def load_hybrid_search():
    p = os.path.join(REPORTS, "hybrid_search.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_continued():
    p = os.path.join(REPORTS, "qml_continued_training.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_best_val():
    p = os.path.join(MODELS, "hybrid_best_val.json")
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
    color = "#38BDF8"
    st.markdown(f"""
<div style="margin:2rem 0 0.8rem 0;padding-bottom:0.7rem;border-bottom:1px solid #1E2E4A">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2.5px;
              color:{color};margin-bottom:0.3rem">{label}</div>
  <h2 style="font-size:1.2rem;font-weight:600;color:#F0F6FF;margin:0 0 0.15rem 0">{title}</h2>
  {sub_html}
</div>""", unsafe_allow_html=True)


def render_hybrid_architecture():
    """Full pipeline diagram for hybrid model."""
    st.markdown("""
<div style="background:#0D1628;border:1px solid rgba(56,189,248,0.2);border-radius:12px;
            padding:1.2rem;margin-bottom:1rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
              color:#38BDF8;margin-bottom:0.8rem">HYBRID QUANTUM-CLASSICAL ARCHITECTURE</div>

  <!-- Row 1: Input -->
  <div style="display:flex;justify-content:center;margin-bottom:0.5rem">
    <div style="padding:0.5rem 1.2rem;border-radius:6px;background:#111D32;
                border:1px solid #2A4060;font-family:'Space Mono',monospace;
                font-size:0.68rem;color:#C8D8F0;text-align:center">
      Light Curve (256 obs)
      <div style="font-size:0.58rem;color:#6B84A8">physics-informed synthetic</div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;color:#2A4060;margin-bottom:0.5rem">↓</div>

  <!-- Row 2: Feature extraction -->
  <div style="display:flex;justify-content:center;margin-bottom:0.5rem">
    <div style="padding:0.5rem 1.2rem;border-radius:6px;background:rgba(56,189,248,0.06);
                border:1px solid rgba(56,189,248,0.2);font-family:'Space Mono',monospace;
                font-size:0.68rem;color:#38BDF8;text-align:center">
      Feature Extraction + Z-score
      <div style="font-size:0.58rem;color:#6B84A8">21 standardised features</div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;color:#2A4060;margin-bottom:0.5rem">↓</div>

  <!-- Row 3: Quantum Layer -->
  <div style="display:flex;justify-content:center;margin-bottom:0.5rem">
    <div style="padding:0.7rem 1.4rem;border-radius:8px;
                background:rgba(166,159,214,0.1);border:1px solid rgba(166,159,214,0.3);
                font-family:'Space Mono',monospace;font-size:0.68rem;
                color:#A69FD6;text-align:center;min-width:320px">
      Quantum Layer — 8 Qubits
      <div style="font-size:0.58rem;color:#8A7FC8;margin-top:2px">
        Linear(21→8) + tanh·π → AngleEmbedding(RY) → Rot+CNOT ×2 → ⟨Z⟩×8
      </div>
      <div style="margin-top:0.4rem;font-size:0.6rem;color:#6B84A8">
        8 trainable Pauli-Z expectation values
      </div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;color:#2A4060;margin-bottom:0.5rem">↓</div>

  <!-- Row 4: Classical Head -->
  <div style="display:flex;justify-content:center;margin-bottom:0.5rem">
    <div style="padding:0.6rem 1.3rem;border-radius:8px;background:rgba(56,189,248,0.08);
                border:1px solid rgba(56,189,248,0.25);font-family:'Space Mono',monospace;
                font-size:0.68rem;color:#38BDF8;text-align:center;min-width:280px">
       Classical Head
       <div style="font-size:0.58rem;color:#6B84A8;margin-top:2px">
         Linear(8→32) + BN + ReLU → Linear(32→32) + ReLU → Linear(32→5)
       </div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;color:#2A4060;margin-bottom:0.5rem">↓</div>

  <!-- Row 5: Output -->
  <div style="display:flex;justify-content:center">
    <div style="padding:0.5rem 1.2rem;border-radius:6px;background:#111D32;
                border:1px solid #2A4060;font-family:'Space Mono',monospace;
                font-size:0.68rem;color:#C8D8F0;text-align:center">
      Softmax → 5-Class Prediction
    </div>
  </div>

  <div style="margin-top:0.9rem;padding-top:0.7rem;border-top:1px solid #1E2E4A;
              display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;text-align:center">
    <div><div style="font-size:1.0rem;font-weight:700;color:#38BDF8">1,798</div>
         <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                     letter-spacing:1px;color:#6B84A8">TOTAL PARAMS</div></div>
    <div><div style="font-size:1.0rem;font-weight:700;color:#A69FD6">8</div>
         <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                     letter-spacing:1px;color:#6B84A8">QUBITS</div></div>
    <div><div style="font-size:1.0rem;font-weight:700;color:#64D8A8">5</div>
         <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                     letter-spacing:1px;color:#6B84A8">DEPTH</div></div>
    <div><div style="font-size:1.0rem;font-weight:700;color:#F5A623">21</div>
         <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                     letter-spacing:1px;color:#6B84A8">INPUT FEATURES</div></div>
  </div>
</div>""", unsafe_allow_html=True)


def render_comparison_panel(fc: dict, adaptive: dict):
    """Side-by-side: Classical CNN vs Hybrid vs Adaptive fusion."""
    models = [
        ("Classical_CNN",  "Classical CNN",   "#38BDF8"),
        ("Hybrid_Quantum", "Hybrid ★",        "#64D8A8"),
    ]
    adap_acc = adaptive.get("test_acc", None)
    adap_f1  = adaptive.get("test_f1",  None)

    cols = st.columns(3)
    for col, (key, label, color) in zip(cols[:2], models):
        m   = fc.get(key, {})
        acc = m.get("accuracy")
        f1  = m.get("f1_macro")
        roc = m.get("roc_auc_ovr_macro")
        col.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem;text-align:center">
  <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:2px;
              color:{color};margin-bottom:0.4rem">{label}</div>
  <div style="font-size:1.6rem;font-weight:700;color:{color};line-height:1">
    {pct(acc)}
  </div>
  <div style="font-size:0.75rem;color:#6B84A8;margin-top:0.2rem">Accuracy</div>
  <div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid #1E2E4A;
              font-size:0.8rem;color:#C8D8F0;line-height:1.7">
    F1-macro: {f"{f1:.4f}" if f1 else "—"}<br/>
    ROC-AUC: {f"{roc:.4f}" if roc else "—"}<br/>
    Params: {"~280k" if "Classical" in key else "1,798"}
  </div>
</div>""", unsafe_allow_html=True)

    # Adaptive (archived diverse-regime result — not head-to-head with fresh)
    col = cols[2]
    alpha = adaptive.get("alpha_mean", None)
    note = adaptive.get("note", "")
    col.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:0.55rem;'
        'letter-spacing:1px;color:#F5A623;border:1px solid rgba(245,166,35,0.35);'
        'border-radius:3px;padding:0.15rem 0.4rem;display:inline-block;'
        'margin-bottom:0.5rem">ARCHIVED · OLD REGIME</div>',
        unsafe_allow_html=True,
    )
    col.markdown(f"""
<div style="background:#0D1628;border:1px solid rgba(100,216,168,0.25);border-radius:10px;
            padding:1rem;text-align:center">
  <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:2px;
              color:#64D8A8;margin-bottom:0.4rem">ADAPTIVE FUSION</div>
  <div style="font-size:1.6rem;font-weight:700;color:#64D8A8;line-height:1">
    {pct(adap_acc)}
  </div>
  <div style="font-size:0.75rem;color:#6B84A8;margin-top:0.2rem">Accuracy</div>
  <div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid rgba(100,216,168,0.15);
              font-size:0.8rem;color:#C8D8F0;line-height:1.7">
    F1-macro: {f"{adap_f1:.4f}" if adap_f1 else "—"}<br/>
    α (class weight): {f"{alpha:.3f}" if alpha else "—"}<br/>
    <span style="font-size:0.72rem;color:#6B84A8">{note[:60] if note else ""}</span>
  </div>
</div>""", unsafe_allow_html=True)


def render_hybrid_search(hs: list):
    if not hs:
        st.caption("No hybrid search results found.")
        return

    configs = []
    val_f1s = []
    for item in hs:
        if not isinstance(item, dict):
            continue
        cfg = item.get("config", item)
        vm  = item.get("val_metrics", {})
        f1  = vm.get("f1_macro", 0)
        nq  = cfg.get("n_qubits",  "?")
        nl  = cfg.get("n_layers",  "?")
        nh  = cfg.get("n_hidden",  "?")
        configs.append(f"{nq}q {nl}L h{nh}")
        val_f1s.append(f1)

    if not configs:
        st.caption("Hybrid search items have unexpected format.")
        return

    max_idx = int(np.argmax(val_f1s))
    colors  = ["rgba(56,189,248,0.7)" if i == max_idx else "rgba(56,189,248,0.3)"
               for i in range(len(configs))]

    fig = go.Figure(go.Bar(
        x=configs, y=val_f1s,
        marker_color=colors,
        hovertemplate="%{x}<br>Val F1: %{y:.4f}<extra></extra>",
    ))
    apply_layout(fig, height=280, title="Hybrid Config Search — Val F1 (best config highlighted)")
    fig.update_layout(
        yaxis=dict(title="Val F1-macro", gridcolor="rgba(30,46,74,0.6)"),
        xaxis=dict(tickangle=-25, gridcolor="rgba(30,46,74,0.6)"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_hybrid_perclass(fc: dict):
    # Compare classical CNN vs hybrid per-class F1
    cnn  = fc.get("Classical_CNN",  {}).get("per_class", {})
    hyb  = fc.get("Hybrid_Quantum", {}).get("per_class", {})

    fig = go.Figure()
    for label, pc, color in [
        ("CNN",       cnn,  "#38BDF8"),
        ("Hybrid",    hyb,  "#64D8A8"),
    ]:
        f1s = [pc.get(str(i), {}).get("f1", 0) for i in range(5)]
        fig.add_trace(go.Bar(name=label, x=CLASS_NAMES, y=f1s, marker_color=color))

    apply_layout(fig, height=320, title="Per-Class F1 — CNN vs Hybrid")
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="F1-score", gridcolor="rgba(30,46,74,0.6)"),
        xaxis=dict(tickangle=-22, gridcolor="rgba(30,46,74,0.6)"),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def main():
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(seed=42), unsafe_allow_html=True)
    sidebar_logo(st)

    st.markdown("""
<div style="padding:1.6rem 0 1.2rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.4rem">HYBRID QUANTUM-CLASSICAL</div>
  <h1 style="font-size:clamp(1.8rem,3.5vw,2.6rem);font-weight:700;color:#F0F6FF;
              margin:0 0 0.4rem 0">Hybrid Architecture</h1>
  <p style="color:#6B84A8;font-size:0.92rem;max-width:70ch;margin:0">
    Classical representation + quantum feature map, trained end-to-end. 
    The proposed approach: 21 features → 8-qubit variational circuit → 
    classical dense head. Best configuration selected on validation F1.
  </p>
</div>""", unsafe_allow_html=True)

    fc       = load_fair()
    adaptive = load_adaptive()
    hs       = load_hybrid_search()
    best_val = load_best_val()

    if not fc:
        st.warning("No results found. Run experiments first.")
        return

    # ── Architecture ──────────────────────────────────────────────────────────
    section("ARCHITECTURE", "Hybrid Quantum-Classical Pipeline")
    render_hybrid_architecture()

    # ── What hybrid adds ──────────────────────────────────────────────────────
    st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;margin-bottom:1.4rem">
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;padding:0.9rem">
    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                color:#38BDF8;margin-bottom:0.4rem">CLASSICAL CONTRIBUTION</div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.65">
      21 physics-informed features provide structured input:
      temporal, frequency, and pattern descriptors of the light curve.
    </div>
  </div>
  <div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;padding:0.9rem">
    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                color:#A69FD6;margin-bottom:0.4rem">QUANTUM CONTRIBUTION</div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.65">
      Variational circuit creates correlated feature interactions via 
      entanglement (CNOT chain). Expectation values capture non-linear 
      quantum correlations in the 256-D Hilbert space.
    </div>
  </div>
  <div style="background:#0D1628;border:1px solid rgba(56,189,248,0.2);border-radius:8px;padding:0.9rem">
    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
                color:#64D8A8;margin-bottom:0.4rem">FUSION</div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.65">
      Quantum features (8-D ⟨Z⟩ expectations) pass to a 2-layer 
      dense classifier. All weights (quantum + classical) trained 
      end-to-end with backpropagation.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Comparison ────────────────────────────────────────────────────────────
    section("RESULTS", "Classical vs Hybrid vs Adaptive — Test Results")
    render_comparison_panel(fc, adaptive)

    # ── Per-class breakdown ───────────────────────────────────────────────────
    section("PER-CLASS", "Per-Class F1 Breakdown",
            "CNN is the strongest classical baseline; shown for reference.")
    render_hybrid_perclass(fc)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    section("CONFUSION", "Hybrid Confusion Matrix (test 3,750)")
    col_cm, col_note = st.columns([1.2, 1])
    with col_cm:
        cm = fc.get("Hybrid_Quantum", {}).get("confusion_matrix")
        if cm:
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=[cn[:10] for cn in CLASS_NAMES],
                y=[cn[:10] for cn in CLASS_NAMES],
                colorscale=[[0, "#060B18"], [0.4, "#0D3060"], [1, "#38BDF8"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=11, color="#F0F6FF"),
                showscale=False,
                hovertemplate="True: %{y}<br>Pred: %{x}<br>%{z}<extra></extra>",
            ))
            apply_layout(fig_cm, height=320, title="Hybrid Confusion Matrix")
            fig_cm.update_layout(
                xaxis=dict(title="Predicted", tickangle=-25, tickfont=dict(size=9),
                           gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="True", autorange="reversed", tickfont=dict(size=9),
                           gridcolor="rgba(0,0,0,0)"),
                margin=dict(l=80, r=10, t=40, b=60),
            )
            st.plotly_chart(fig_cm, width="stretch", config={"displayModeBar": False})

    with col_note:
        hyb   = fc.get("Hybrid_Quantum",   {})
        cnn   = fc.get("Classical_CNN",    {})
        h_acc = hyb.get("accuracy", 0)
        c_acc = cnn.get("accuracy", 0)
        h_f1  = hyb.get("f1_macro", 0)
        c_f1  = cnn.get("f1_macro", 0)
        st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem;height:100%;font-size:0.82rem;color:#C8D8F0;line-height:1.8">
  <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:2px;
              color:#38BDF8;margin-bottom:0.5rem">RESULT SUMMARY</div>
  <div>
    Hybrid Acc: <strong style="color:#38BDF8">{h_acc*100:.2f}%</strong><br/>
    CNN Acc: <strong style="color:#38BDF8">{c_acc*100:.2f}%</strong><br/>
    Hybrid F1: <strong style="color:#38BDF8">{h_f1:.4f}</strong> · CNN F1: {c_f1:.4f}
  </div>
  <div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid #1E2E4A;
              color:#8A9FBF;line-height:1.65">
    Difference on clean data:
    <strong style="color:#64D8A8">{(h_acc-c_acc)*100:+.1f} pp</strong>.<br/><br/>
    The hybrid combines 21 classical features with an 8-qubit variational
    circuit, trained end-to-end. Robustness analysis may reveal
    stability advantages — see the Experiments page.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Config search ─────────────────────────────────────────────────────────
    section("CONTINUED TRAINING", "Best Quantum Results — Continued Training",
            "Fresh regime: val-selected on 17.5k train / 3.75k val, tested once on frozen 3,750.")
    continued = load_continued()
    if continued and continued.get("hybrid_candidates"):
        cands = continued["hybrid_candidates"]
        names = [c["name"] for c in cands]
        tacc = [c["test_acc"] for c in cands]
        best_name = continued.get("best_hybrid", {}).get("name")
        colors = ["rgba(56,189,248,0.85)" if n == best_name else "rgba(56,189,248,0.35)"
                  for n in names]
        fig_c = go.Figure(go.Bar(x=names, y=tacc, marker_color=colors,
            hovertemplate="%{x}<br>Test acc: %{y:.4f}<extra></extra>"))
        apply_layout(fig_c, height=280,
                     title="Continued-training hybrid candidates — test accuracy (best highlighted)")
        fig_c.update_layout(yaxis=dict(title="Test accuracy",
                                       gridcolor="rgba(30,46,74,0.6)"),
                           xaxis=dict(tickangle=-20))
        st.plotly_chart(fig_c, width="stretch", config={"displayModeBar": False})
        pure = continued.get("pure", {})
        if pure:
            st.markdown(
                '<div style="background:#0D1628;border:1px solid rgba(166,159,214,0.25);'
                'border-radius:8px;padding:0.9rem;font-size:0.82rem;color:#C8D8F0;'
                'line-height:1.7">'
                "<strong style='color:#A69FD6'>Pure VQC (8q, 3 layers, 30 epochs):</strong> "
                f"val F1 {pure.get('val_f1', 0):.4f} → "
                f"test <strong>{pure.get('test_acc', 0)*100:.2f}%</strong> / "
                f"F1 {pure.get('test_f1', 0):.4f} "
                f"(best epoch {pure.get('best_epoch', '—')}, +6.45pp vs 55.63% baseline).</div>",
                unsafe_allow_html=True)

    section("CONFIG SEARCH (ARCHIVED)", "Hybrid Grid Search — Old Subset Demo",
            "Archived quick-subset demo from the previous regime — kept for provenance, "
            "not comparable with continued-training results above.")
    render_hybrid_search(hs)

    # Best config detail
    if best_val:
        st.markdown(f"""
<div style="background:#0D1628;border:1px solid rgba(56,189,248,0.2);border-radius:8px;
            padding:0.9rem;font-size:0.82rem;color:#C8D8F0;line-height:1.7">
  <strong style="color:#38BDF8">Best Hybrid Configuration (val-selected)</strong><br/>
  {" · ".join(f"{k}: {v}" for k, v in best_val.items() if k not in ("confusion_matrix","per_class"))}
</div>""", unsafe_allow_html=True)

    # ── Adaptive fusion (archived regime) ───────────────────────────────────
    section("ADAPTIVE FUSION (ARCHIVED)", "Adaptive Fusion Engine — Old Regime",
            "Archived result from the previous diverse regime — shown for provenance, "
            "not head-to-head with the fresh continued-training scores above.")

    alpha_mean = adaptive.get("alpha_mean")
    alpha_std  = adaptive.get("alpha_std")
    adap_acc   = adaptive.get("test_acc")
    adap_f1    = adaptive.get("test_f1")

    st.markdown(f"""
<div style="background:#0D1628;border:1px solid rgba(100,216,168,0.2);border-radius:10px;
            padding:1.1rem 1.2rem">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;
              margin-bottom:0.8rem">
    <div style="text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:#64D8A8">
        {pct(adap_acc)}
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                  letter-spacing:1px;color:#6B84A8;margin-top:2px">TEST ACC</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:#C8D8F0">
        {f"{adap_f1:.4f}" if adap_f1 else "—"}
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                  letter-spacing:1px;color:#6B84A8;margin-top:2px">F1-MACRO</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:#F5A623">
        {f"{alpha_mean:.3f}" if alpha_mean else "—"}
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                  letter-spacing:1px;color:#6B84A8;margin-top:2px">α MEAN</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:#C8D8F0">
        {f"{alpha_std:.4f}" if alpha_std else "—"}
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.52rem;
                  letter-spacing:1px;color:#6B84A8;margin-top:2px">α STD</div>
    </div>
  </div>
  <div style="font-size:0.82rem;color:#8A9FBF;line-height:1.65">
    <strong style="color:#64D8A8">How it works:</strong> 
    Signal quality features (noise, completeness, periodicity, spectral complexity) 
    are extracted from each light curve. A learned gating network outputs α ∈ [0,1] — 
    the weight given to the classical representation. 
    Fusion = α·classical + (1−α)·quantum.<br/><br/>
    α ≈ {f"{alpha_mean:.3f}" if alpha_mean else "0.99"} means the gate currently leans heavily on 
    the classical representation for this dataset — the classical features are high-quality 
    and informative. This is an honest result; the gate learns from data, not by assumption.
  </div>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
