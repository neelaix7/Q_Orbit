# pages/6_Experiments.py — Q-ORBIT Experiments Comparison Page
# Full experimental matrix: clean + robustness. All values from real JSON files.
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.space_theme import css, stars_html, sidebar_logo, apply_layout, CLASS_COLORS, CLASS_NAMES

st.set_page_config(
    page_title="Q-ORBIT — Experiments",
    layout="wide", page_icon="🔬",
    initial_sidebar_state="collapsed",
)

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REPORTS = os.path.join(ROOT, "results", "reports")

# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_json(fname):
    p = os.path.join(REPORTS, fname)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_exp_matrix():
    p = os.path.join(REPORTS, "experimental_matrix.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


MODEL_SERIES = [
    ("Classical_CNN", "CNN",       "#38BDF8", 2.5),
    ("Classical_SVM", "SVM",       "#64D8A8", 1.8),
    ("Hybrid",        "Hybrid ★", "#F5A623", 2.5),
]


def section(label, title, sub=""):
    sub_html = f'<p style="color:#6B84A8;font-size:0.85rem;margin:0 0 0.5rem 0">{sub}</p>' if sub else ""
    st.markdown(f"""
<div style="margin:2rem 0 0.8rem 0;padding-bottom:0.7rem;border-bottom:1px solid #1E2E4A">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.3rem">{label}</div>
  <h2 style="font-size:1.2rem;font-weight:600;color:#F0F6FF;margin:0 0 0.15rem 0">{title}</h2>
  {sub_html}
</div>""", unsafe_allow_html=True)


def render_protocol():
    st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1rem 1.2rem;margin-bottom:1rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
              color:#38BDF8;margin-bottom:0.5rem">EXPERIMENTAL PROTOCOL</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.8rem">
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
      <strong style="color:#38BDF8">Dataset</strong><br/>
      25,000 synthetic light curves<br/>
      5 classes × 5,000 · seed 123
    </div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
      <strong style="color:#38BDF8">Split</strong><br/>
      70/15/15 stratified<br/>
      Train: 17.5k · Val: 3.75k · Test: 3.75k<br/>
      Frozen · Never tune on test
    </div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
      <strong style="color:#38BDF8">Fairness</strong><br/>
      Same test set (3,750) for all<br/>
      Val used only for model selection<br/>
      No test leakage
    </div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
      <strong style="color:#38BDF8">Robustness</strong><br/>
      Training frozen · Test degraded<br/>
      Noise / Missing / Short obs<br/>
      Combinations tested
    </div>
    <div style="font-size:0.82rem;color:#C8D8F0;line-height:1.7">
      <strong style="color:#38BDF8">Simulator</strong><br/>
      PennyLane default.qubit<br/>
      No real hardware required<br/>
      Backprop differentiation
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_clean_comparison(fc: dict):
    """Full comparison table for clean (no degradation) results."""
    model_keys = [
        ("Classical_CNN",   "CNN",           "Classical"),
        ("Classical_SVM",   "SVM",           "Classical"),
        ("Classical_RF",    "Random Forest", "Classical"),
        ("Classical_XGB",   "XGBoost",       "Classical"),
        ("Pure_Quantum_VQC","Pure VQC",      "Quantum"),
        ("Hybrid_Quantum",  "Hybrid ★",      "Hybrid"),
    ]

    def fmt(v, pct=True):
        if v is None:
            return "—"
        return f"{v*100:.2f}%" if pct and 0 < v <= 1 else f"{v:.4f}" if not pct else f"{v:.2f}%"

    st.markdown('<table class="qo-table" style="width:100%"><thead><tr>'
                '<th>Type</th><th>Model</th>'
                '<th>Accuracy</th><th>Precision</th><th>Recall</th>'
                '<th>F1-macro</th><th>ROC-AUC</th><th>Params</th>'
                '</tr></thead><tbody>', unsafe_allow_html=True)

    meta = load_json("fair_comparison_meta.json") or {}

    def _params(key, fallback):
        try:
            return f"{meta.get(key, {}).get('params', fallback):,}"
        except Exception:
            return fallback

    param_map = {
        "Classical_CNN": _params("Classical_CNN", "~280k"),
        "Classical_SVM": _params("Classical_SVM", "~1k"),
        "Classical_RF": _params("Classical_RF", "~5k"),
        "Classical_XGB": _params("Classical_XGB", "~5k"),
        "Pure_Quantum_VQC": _params("Pure_Quantum_VQC", "118"),
        "Hybrid_Quantum": _params("Hybrid_Quantum", "1,798"),
    }
    best_acc = max(
        (fc.get(k, {}).get("accuracy", 0) for k, _, _ in model_keys),
        default=0,
    )
    for key, name, mtype in model_keys:
        m   = fc.get(key, {})
        acc = m.get("accuracy")
        prec = m.get("precision_macro")
        rec  = m.get("recall_macro")
        f1   = m.get("f1_macro")
        auc  = m.get("roc_auc_ovr_macro")
        is_best = acc is not None and abs(acc - best_acc) < 1e-6
        hl = 'class="highlight"' if is_best else ""
        color_map = {"Classical": "#38BDF8", "Quantum": "#A69FD6", "Hybrid": "#64D8A8"}
        tc = color_map.get(mtype, "#C8D8F0")
        st.markdown(
            f'<tr>'
            f'<td><span style="color:{tc};font-size:0.75rem">{mtype}</span></td>'
            f'<td><strong>{name}</strong>{"&nbsp;★" if is_best else ""}</td>'
            f'<td class="mono" {hl}>{fmt(acc)}</td>'
            f'<td class="mono">{fmt(prec)}</td>'
            f'<td class="mono">{fmt(rec)}</td>'
            f'<td class="mono">{fmt(f1,False)}</td>'
            f'<td class="mono">{fmt(auc,False)}</td>'
            f'<td class="mono" style="font-size:0.75rem">{param_map.get(key,"—")}</td>'
            f'</tr>',
            unsafe_allow_html=True,
        )
    st.markdown("</tbody></table>", unsafe_allow_html=True)
    st.markdown("""
<div style="margin-top:0.4rem;font-family:'Space Mono',monospace;font-size:0.57rem;
            color:#6B84A8">
  ★ Best overall · Test: 3,750 samples · seed 123 · frozen · PennyLane default.qubit
</div>""", unsafe_allow_html=True)


def render_robustness_chart(title, fname, x_key, x_label):
    data = load_json(fname)
    if not data:
        st.caption(f"{fname} not found.")
        return

    fig = go.Figure()
    for key, label, color, lw in MODEL_SERIES:
        x = [r.get(x_key, 0) for r in data if key in r]
        y = [r[key] for r in data if key in r]
        if x:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers", name=label,
                line=dict(color=color, width=lw),
                marker=dict(size=6),
                hovertemplate=f"{label}<br>{x_label}: %{{x}}<br>Acc: %{{y:.3f}}<extra></extra>",
            ))
    apply_layout(fig, height=300, title=title)
    fig.update_layout(
        xaxis=dict(title=x_label, gridcolor="rgba(30,46,74,0.6)"),
        yaxis=dict(title="Accuracy", range=[0, 1], gridcolor="rgba(30,46,74,0.6)"),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_robustness_score(fc: dict):
    """Compute and display robustness scores."""
    try:
        from src.evaluation.robustness_score import robustness_score as rs_fn
    except Exception:
        st.caption("Robustness score module not available.")
        return

    model_map = {
        "Classical_CNN": "CNN",
        "Classical_SVM": "SVM",
        "Hybrid":        "Hybrid ★",
    }
    fc_key_map = {
        "Classical_CNN": "Classical_CNN",
        "Classical_SVM": "Classical_SVM",
        "Hybrid":        "Hybrid_Quantum",
    }

    scores = {}
    for rob_key, label in model_map.items():
        clean, degraded = None, []
        for fname in ("robustness_noise.json", "robustness_observation.json",
                      "robustness_missing.json"):
            d = load_json(fname)
            if not d or not isinstance(d, list) or rob_key not in d[0]:
                continue
            if clean is None:
                clean = d[0][rob_key]
            degraded += [r[rob_key] for r in d[1:] if rob_key in r]
        if clean is not None and degraded:
            try:
                scores[label] = rs_fn(clean, degraded)
            except Exception:
                pass

    if not scores:
        return

    fig = go.Figure(go.Bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        marker_color=["#38BDF8", "#64D8A8", "#F5A623"][:len(scores)],
        hovertemplate="%{x}<br>RS: %{y:.4f}<extra></extra>",
        text=[f"{v:.3f}" for v in scores.values()],
        textposition="outside",
        textfont=dict(color="#C8D8F0", size=10),
    ))
    apply_layout(fig, height=280, title="Robustness Score (RS) — mean(acc_degraded / acc_clean)")
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], title="RS", gridcolor="rgba(30,46,74,0.6)"),
        xaxis=dict(gridcolor="rgba(30,46,74,0.6)"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_ablations():
    ab = load_json("ablations.json")
    if not ab:
        st.info("No ablation results found.")
        return

    # Flatten ablation dict to table
    rows = []
    if isinstance(ab, dict):
        for ablation_name, result in ab.items():
            if isinstance(result, dict):
                acc = result.get("accuracy") or result.get("test_acc")
                f1  = result.get("f1_macro") or result.get("test_f1")
                rows.append({
                    "Ablation": ablation_name.replace("_", " "),
                    "Accuracy": f"{acc*100:.2f}%" if acc and acc <= 1 else (str(acc) if acc else "—"),
                    "F1-macro": f"{f1:.4f}" if f1 else "—",
                })
    elif isinstance(ab, list):
        for item in ab:
            if isinstance(item, dict):
                cfg = item.get("config", item.get("name", ""))
                vm  = item.get("test_metrics", item)
                acc = vm.get("accuracy", 0)
                f1  = vm.get("f1_macro",  0)
                rows.append({
                    "Ablation": str(cfg),
                    "Accuracy": f"{acc*100:.2f}%" if acc else "—",
                    "F1-macro": f"{f1:.4f}" if f1 else "—",
                })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("Ablation data has unexpected format.")


def render_statistical_validation():
    sv = load_json("statistical_validation.json")
    if not sv:
        st.info("Statistical validation results not found.")
        return

    if isinstance(sv, dict):
        for key, val in sv.items():
            if isinstance(val, dict):
                acc_mean = val.get("accuracy_mean") or val.get("mean_accuracy")
                acc_std  = val.get("accuracy_std")  or val.get("std_accuracy")
                f1_mean  = val.get("f1_mean") or val.get("mean_f1")
                f1_std   = val.get("f1_std")  or val.get("std_f1")
                pval     = val.get("p_value")

                display = ""
                if acc_mean is not None:
                    display += f"Acc: {acc_mean*100:.2f}%"
                    if acc_std:
                        display += f" ± {acc_std*100:.2f}%"
                if f1_mean is not None:
                    display += f" · F1: {f1_mean:.4f}"
                    if f1_std:
                        display += f" ± {f1_std:.4f}"
                if pval is not None:
                    display += f" · p={pval:.4f}"
                if display:
                    st.markdown(
                        f'<div style="background:#0D1628;border:1px solid #1E2E4A;'
                        f'border-radius:6px;padding:0.55rem 0.8rem;margin:0.3rem 0;'
                        f'font-size:0.82rem;color:#C8D8F0">'
                        f'<strong style="color:#38BDF8">{key}</strong> — {display}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(seed=55), unsafe_allow_html=True)
    sidebar_logo(st)

    st.markdown("""
<div style="padding:1.6rem 0 1.2rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.4rem">SCIENTIFIC EXPERIMENTS</div>
  <h1 style="font-size:clamp(1.8rem,3.5vw,2.6rem);font-weight:700;color:#F0F6FF;
              margin:0 0 0.4rem 0">Experiment Results</h1>
  <p style="color:#6B84A8;font-size:0.92rem;max-width:70ch;margin:0">
    Full comparison: Classical vs Hybrid (pure-quantum VQC as baseline row).
    All experiments use the same frozen test set (3,750 samples, seed 123).
    Results loaded from <code>results/reports/</code> — no hardcoded values.
  </p>
</div>""", unsafe_allow_html=True)

    fc = load_json("fair_comparison.json") or {}

    # ── Protocol ──────────────────────────────────────────────────────────────
    section("PROTOCOL", "Experimental Protocol")
    render_protocol()

    # ── Clean comparison ──────────────────────────────────────────────────────
    section("CLEAN", "Clean Data — Full Comparison Table",
            "No degradation applied. Standard test conditions.")
    if fc:
        render_clean_comparison(fc)
    else:
        st.warning("No fair_comparison.json found. Run experiments first.")

    # ── Radar chart ───────────────────────────────────────────────────────────
    section("RADAR", "Multi-Metric Radar")
    if fc:
        cats = ["Accuracy", "Precision", "F1-macro", "ROC-AUC", "Recall"]
        fig_r = go.Figure()
        for key, label, color, fill in [
            ("Classical_CNN",   "CNN",       "#38BDF8", "rgba(56,189,248,0.08)"),
            ("Hybrid_Quantum",  "Hybrid ★", "#F5A623", "rgba(245,166,35,0.08)"),
        ]:
            m = fc.get(key, {})
            vals = [
                m.get("accuracy", 0),
                m.get("precision_macro", 0),
                m.get("f1_macro", 0),
                m.get("roc_auc_ovr_macro", 0),
                m.get("recall_macro", 0),
            ]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself", name=label,
                line=dict(color=color, width=2),
                fillcolor=fill,
                opacity=0.8,
            ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(30,46,74,0.8)",
                                color="#6B84A8", tickfont=dict(size=9)),
                angularaxis=dict(color="#C8D8F0", gridcolor="rgba(30,46,74,0.6)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#C8D8F0", family="Space Grotesk"),
            height=380,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", y=-0.08, bgcolor="rgba(0,0,0,0)"),
            title=dict(text="CNN vs Hybrid — 5-metric radar (test set)",
                       font=dict(size=12, color="#C8D8F0"), x=0.02),
        )
        st.plotly_chart(fig_r, width="stretch", config={"displayModeBar": False})

    # ── Robustness ────────────────────────────────────────────────────────────
    section("ROBUSTNESS", "Robustness Experiments",
            "Training frozen. Only test-time degradation applied. "
            "Noise std / missing fraction / observation fraction varied.")

    tab_noise, tab_obs, tab_miss, tab_rs = st.tabs([
        "Noise Sweep", "Observation Length", "Missing Data", "Robustness Score"
    ])

    with tab_noise:
        render_robustness_chart(
            "Accuracy vs Noise Level (σ)",
            "robustness_noise.json",
            "noise", "Noise std (σ)",
        )
        st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.7rem 0.9rem;font-size:0.8rem;color:#8A9FBF;line-height:1.6">
  Noise levels: 0% (clean), 10%, 20%, 30%. Added to test curves at inference time. 
  Training weights frozen. All models degrade; relative ordering tells which is more 
  noise-robust. CNN (deep features) tends to be most resilient on raw curves.
</div>""", unsafe_allow_html=True)

    with tab_obs:
        render_robustness_chart(
            "Accuracy vs Observation Fraction",
            "robustness_observation.json",
            "observation_fraction", "Observation fraction",
        )
        st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.7rem 0.9rem;font-size:0.8rem;color:#8A9FBF;line-height:1.6">
  Observation fractions: 100%, 75%, 50%, 25%. Truncation + last-value padding simulates 
  early termination of the pass. CNN on raw curve is most sensitive; feature-based 
  methods (SVM/Hybrid) can be more stable at shorter windows.
</div>""", unsafe_allow_html=True)

    with tab_miss:
        render_robustness_chart(
            "Accuracy vs Missing Data Fraction",
            "robustness_missing.json",
            "missing_fraction", "Missing fraction",
        )
        st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.7rem 0.9rem;font-size:0.8rem;color:#8A9FBF;line-height:1.6">
  Missing fractions: 0%, 5%, 10%, 20%. Random dropout with linear interpolation. 
  CNN is less affected (interpolated gaps preserve shape); feature extraction may 
  degrade gracefully.
</div>""", unsafe_allow_html=True)

    with tab_rs:
        render_robustness_score(fc)
        st.markdown("""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.7rem 0.9rem;font-size:0.8rem;color:#8A9FBF;line-height:1.6;margin-top:0.6rem">
  <strong style="color:#C8D8F0">Robustness Score (RS)</strong> = 
  mean(acc_degraded / acc_clean) across all degradation conditions (noise + observation + missing).
  RS ∈ [0,1]; higher = more robust across degraded conditions relative to clean baseline.
  Formula from <code>src/evaluation/robustness_score.py</code>.
</div>""", unsafe_allow_html=True)

    # ── Experimental matrix ───────────────────────────────────────────────────
    section("MATRIX", "Full Experimental Matrix",
            "Rows: degradation conditions · Columns: models")
    em = load_exp_matrix()
    if em is not None:
        st.dataframe(em, width="stretch", height=400)
        # Download
        csv_bytes = em.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Experimental Matrix (CSV)",
            data=csv_bytes,
            file_name="QORBIT_experimental_matrix.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Experimental matrix not found. "
            "Run `python experiments/run_comprehensive_fair.py` to generate it."
        )

    # ── Ablations ─────────────────────────────────────────────────────────────
    section("ABLATIONS", "Ablation Study",
            "7 ablations testing component contributions: "
            "classical only / quantum only / hybrid combinations / adaptive fusion.")
    render_ablations()

    # ── Statistical validation ────────────────────────────────────────────────
    section("STATISTICS", "Statistical Validation",
            "Multi-seed (seeds 42–46) reproducibility + paired t-tests.")
    render_statistical_validation()

    # ── Export ────────────────────────────────────────────────────────────────
    section("EXPORT", "Download Results")
    ec1, ec2, ec3 = st.columns(3)
    for col, fname, label in [
        (ec1, "fair_comparison.json",  "Fair Comparison (JSON)"),
        (ec2, "fair_comparison.md",    "Fair Comparison (MD)"),
        (ec3, "ablations.json",        "Ablations (JSON)"),
    ]:
        fpath = os.path.join(REPORTS, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                col.download_button(
                    f"⬇ {label}", data=f.read(),
                    file_name=fname,
                    mime="application/json" if fname.endswith(".json") else "text/markdown",
                    width="stretch",
                )
        else:
            col.caption(f"{label} — not found")


if __name__ == "__main__":
    main()
else:
    main()
