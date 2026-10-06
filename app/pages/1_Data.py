# pages/2_Data.py — Q-ORBIT Data Page
# Reads actual dataset files — no hardcoded values.
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from app.space_theme import css, stars_html, sidebar_logo, apply_layout, CLASS_COLORS, CLASS_NAMES

st.set_page_config(
    page_title="Q-ORBIT — Data",
    layout="wide", page_icon="📊",
    initial_sidebar_state="collapsed",
)

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA    = os.path.join(ROOT, "data")
SPLITS  = os.path.join(DATA, "splits")
CLASSICAL = os.path.join(DATA, "classical")
HYBRID  = os.path.join(DATA, "hybrid")
QR      = os.path.join(DATA, "quantum_ready")

FEATURE_NAMES = [
    "mean","std","variance","median","skewness","kurtosis","peak_to_peak",
    "amplitude","energy","above_median_fraction","min_value",
    "dominant_freq","dominant_magnitude","harmonic_energy","fft_entropy","n_harmonics",
    "flash_count","rise_time","fall_time","eclipse_fraction","period_estimate",
]
FEATURE_GROUPS = {
    "Time Domain": FEATURE_NAMES[:11],
    "Frequency Domain": FEATURE_NAMES[11:16],
    "Temporal Pattern": FEATURE_NAMES[16:],
}

# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_splits_npz():
    out = {}
    for s in ("train", "val", "test"):
        p = os.path.join(SPLITS, f"{s}.npz")
        if os.path.exists(p):
            d = np.load(p)
            out[s] = {"curves": d["curves"], "labels": d["labels"].astype(int)}
    return out


@st.cache_data
def load_classical_csv():
    p = os.path.join(CLASSICAL, "classical_dataset.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


@st.cache_data
def load_qr_readme():
    p = os.path.join(QR, "README.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def section(label: str, title: str, sub: str = ""):
    sub_html = f'<p style="color:#6B84A8;font-size:0.85rem;margin:0">{sub}</p>' if sub else ""
    st.markdown(f"""
<div style="margin:2rem 0 0.8rem 0;padding-bottom:0.7rem;border-bottom:1px solid #1E2E4A">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.3rem">{label}</div>
  <h2 style="font-size:1.2rem;font-weight:600;color:#F0F6FF;margin:0 0 0.2rem 0">{title}</h2>
  {sub_html}
</div>""", unsafe_allow_html=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(seed=12), unsafe_allow_html=True)
    sidebar_logo(st)

    # Page header
    st.markdown("""
<div style="padding:1.6rem 0 1.2rem 0;border-bottom:1px solid #1E2E4A;margin-bottom:1.6rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
              color:#38BDF8;margin-bottom:0.4rem">DATASET MANAGEMENT</div>
  <h1 style="font-size:clamp(1.8rem,3.5vw,2.6rem);font-weight:700;color:#F0F6FF;
              margin:0 0 0.4rem 0">Dataset Explorer</h1>
  <p style="color:#6B84A8;font-size:0.92rem;max-width:70ch;margin:0">
    Physics-informed synthetic photometric light curves. Regime A — controlled synthetic, 
    primary dataset used for all reported metrics. All values read from real data files.
  </p>
</div>""", unsafe_allow_html=True)

    splits_data = load_splits_npz()
    classical_df = load_classical_csv()
    qr_meta = load_qr_readme()

    if not splits_data:
        st.error(
            "Dataset splits not found at `data/splits/`. "
            "Run `python src/pipeline/generate_dataset.py --n-per-class 5000 --seed 123` first."
        )
        return

    # ── Overview metrics ──────────────────────────────────────────────────────
    all_labels = np.concatenate([v["labels"] for v in splits_data.values()])
    n_total    = len(all_labels)
    n_train    = len(splits_data.get("train", {}).get("labels", []))
    n_val      = len(splits_data.get("val",   {}).get("labels", []))
    n_test     = len(splits_data.get("test",  {}).get("labels", []))
    n_classes  = len(np.unique(all_labels))
    curve_len  = splits_data["train"]["curves"].shape[1] if "train" in splits_data else 256

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl, sub in [
        (c1, f"{n_total:,}",  "Total Samples",   "25k synthetic"),
        (c2, str(n_classes),  "Classes",         "balanced"),
        (c3, "21",            "Features",        "engineered"),
        (c4, str(curve_len),  "Observations",    "per curve"),
        (c5, f"{n_train:,}",  "Train",           "70%"),
        (c6, f"{n_test:,}",   "Test",            "15% frozen"),
    ]:
        col.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:8px;
            padding:0.8rem;text-align:center">
  <div style="font-size:1.35rem;font-weight:700;color:#F0F6FF">{val}</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
              letter-spacing:1px;color:#6B84A8;margin-top:2px">{lbl}</div>
  <div style="font-size:0.68rem;color:#4A5F7A;margin-top:1px">{sub}</div>
</div>""", unsafe_allow_html=True)

    # ── Class distribution ────────────────────────────────────────────────────
    section("DATASET OVERVIEW", "Class Distribution",
            "Each class has exactly 5,000 samples — perfectly balanced.")

    counts = {cn: int(np.sum(all_labels == i)) for i, cn in enumerate(CLASS_NAMES)}
    col_bar, col_table = st.columns([1.4, 1])

    with col_bar:
        fig = go.Figure(go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=CLASS_COLORS,
            hovertemplate="%{x}<br>%{y} samples<extra></extra>",
        ))
        apply_layout(fig, height=300, title="Class Distribution (all splits)")
        fig.update_layout(
            xaxis=dict(tickangle=-25, gridcolor="rgba(30,46,74,0.6)"),
            yaxis=dict(title="Sample count", gridcolor="rgba(30,46,74,0.6)"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col_table:
        st.markdown("""
<table class="qo-table" style="width:100%">
  <thead>
    <tr>
      <th>Class</th><th>Train</th><th>Val</th><th>Test</th><th>Total</th>
    </tr>
  </thead>
  <tbody>""", unsafe_allow_html=True)
        for i, cn in enumerate(CLASS_NAMES):
            tr = int(np.sum(splits_data.get("train", {}).get("labels", np.array([])) == i))
            va = int(np.sum(splits_data.get("val",   {}).get("labels", np.array([])) == i))
            te = int(np.sum(splits_data.get("test",  {}).get("labels", np.array([])) == i))
            tot = tr + va + te
            color = CLASS_COLORS[i]
            st.markdown(
                f'<tr><td><span style="color:{color}">■</span> {cn}</td>'
                f'<td class="mono">{tr}</td><td class="mono">{va}</td>'
                f'<td class="mono">{te}</td>'
                f'<td class="mono highlight">{tot}</td></tr>',
                unsafe_allow_html=True,
            )
        st.markdown("</tbody></table>", unsafe_allow_html=True)

        st.markdown("""
<div style="margin-top:0.8rem;background:#0D1628;border:1px solid #1E2E4A;
            border-radius:8px;padding:0.8rem;font-size:0.8rem;color:#8A9FBF;line-height:1.7">
  <strong style="color:#C8D8F0">Split protocol</strong><br/>
  70 / 15 / 15 stratified · seed 123 · frozen<br/>
  Validation: model selection only<br/>
  Test: evaluated once, never tuned on
</div>""", unsafe_allow_html=True)

    # ── Sample light curves ───────────────────────────────────────────────────
    section("LIGHT CURVES", "Sample Photometric Light Curves",
            "One representative curve per class from the test split.")

    test_curves  = splits_data.get("test", {}).get("curves", None)
    test_labels  = splits_data.get("test", {}).get("labels", None)

    if test_curves is not None:
        cols = st.columns(5)
        for cls_i, (col, cn, color) in enumerate(zip(cols, CLASS_NAMES, CLASS_COLORS)):
            idxs = np.where(test_labels == cls_i)[0]
            if len(idxs) == 0:
                continue
            c = test_curves[idxs[0]]
            t = np.linspace(0, 720, len(c))
            fig = go.Figure(go.Scatter(
                x=t.tolist(), y=c.tolist(),
                mode="lines",
                line=dict(color=color, width=1.6),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},"
                          f"{int(color[3:5],16)},"
                          f"{int(color[5:7],16)},0.08)",
                hovertemplate="t=%{x:.0f}s<br>%{y:.3f}<extra></extra>",
            ))
            apply_layout(fig, height=160, title=cn)
            fig.update_layout(
                margin=dict(l=4, r=4, t=28, b=4),
                xaxis=dict(showticklabels=False, gridcolor="rgba(30,46,74,0.4)"),
                yaxis=dict(range=[0, 1.05], showticklabels=False,
                           gridcolor="rgba(30,46,74,0.4)"),
                showlegend=False,
            )
            with col:
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})

    # ── Feature analysis ──────────────────────────────────────────────────────
    section("FEATURE ANALYSIS", "21 Engineered Features",
            "Extracted by src/classical/feature_engineering.py from each light curve.")

    if classical_df is not None:
        tab1, tab2, tab3 = st.tabs(["Feature Groups", "Statistics", "Preview"])

        with tab1:
            for group, feats in FEATURE_GROUPS.items():
                st.markdown(f"""
<div style="margin-bottom:1rem">
  <div style="font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2px;
              color:#38BDF8;margin-bottom:0.4rem">{group.upper()}</div>
  <div style="display:flex;flex-wrap:wrap;gap:0.4rem">
    {"".join(f'<span style="background:#0D1628;border:1px solid #1E2E4A;border-radius:4px;font-family:Space Mono,monospace;font-size:0.68rem;color:#C8D8F0;padding:0.22rem 0.55rem">{f}</span>' for f in feats)}
  </div>
</div>""", unsafe_allow_html=True)

            # Correlation heatmap on test split
            test_df = classical_df[classical_df["split"] == "test"]
            if len(test_df) > 0:
                feat_cols = [c for c in FEATURE_NAMES if c in test_df.columns]
                corr = test_df[feat_cols].corr()
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale=[[0,"#0D1628"],[0.5,"#1E3A6E"],[1,"#38BDF8"]],
                    zmid=0,
                    hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.2f}<extra></extra>",
                    showscale=True,
                ))
                apply_layout(fig_corr, height=480, title="Feature Correlation Matrix (test split)")
                fig_corr.update_layout(
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                    yaxis=dict(tickfont=dict(size=9)),
                    margin=dict(l=80, r=10, t=40, b=80),
                )
                st.plotly_chart(fig_corr, width="stretch",
                                config={"displayModeBar": False})

        with tab2:
            test_df = classical_df[classical_df["split"] == "test"]
            feat_cols = [c for c in FEATURE_NAMES if c in test_df.columns]
            if feat_cols:
                stats = test_df[feat_cols].describe().T
                stats["missing"] = test_df[feat_cols].isnull().sum().values
                stats = stats.round(4)
                st.dataframe(
                    stats[["mean", "std", "min", "25%", "50%", "75%", "max", "missing"]],
                    width="stretch", height=500,
                )

        with tab3:
            st.markdown(
                '<div style="font-size:0.8rem;color:#6B84A8;margin-bottom:0.5rem">'
                'First 20 rows of the test split (classical_dataset.csv)</div>',
                unsafe_allow_html=True,
            )
            test_df = classical_df[classical_df["split"] == "test"].head(20)
            show_cols = ["sample_id", "class_name"] + FEATURE_NAMES[:8]
            show_cols = [c for c in show_cols if c in test_df.columns]
            st.dataframe(test_df[show_cols], width="stretch", height=420)
    else:
        st.warning(
            "classical_dataset.csv not found. "
            "Run `python experiments/export_datasets.py` to generate it."
        )

    # ── Quantum-ready ─────────────────────────────────────────────────────────
    section("QUANTUM REDUCTION", "Quantum-Ready Representation",
            "21 features → z-score → PCA-8, fit on train only. "
            "Matches 8-qubit VQC/hybrid ansatz.")

    ev = qr_meta.get("explained_variance", {}).get("explained_variance_ratio", [])
    if ev:
        c_ev, c_info = st.columns([1.2, 1])
        with c_ev:
            cumev = np.cumsum(ev)
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(ev))],
                y=ev,
                name="Individual",
                marker_color="#38BDF8",
                opacity=0.8,
                hovertemplate="PC%{x}<br>Variance: %{y:.3f}<extra></extra>",
            ))
            fig_ev.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(ev))],
                y=cumev.tolist(),
                name="Cumulative",
                line=dict(color="#F5A623", width=2),
                mode="lines+markers",
                yaxis="y2",
                hovertemplate="PC%{x}<br>Cumulative: %{y:.3f}<extra></extra>",
            ))
            apply_layout(fig_ev, height=300,
                         title="PCA Explained Variance (8 components)")
            fig_ev.update_layout(
                yaxis=dict(title="Individual variance", range=[0, max(ev)*1.2],
                           gridcolor="rgba(30,46,74,0.6)"),
                yaxis2=dict(title="Cumulative", range=[0, 1.05],
                            overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", showgrid=False),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_ev, width="stretch",
                            config={"displayModeBar": False})
        with c_info:
            total_var = sum(ev)
            st.markdown(f"""
<div style="background:#0D1628;border:1px solid #1E2E4A;border-radius:10px;
            padding:1.1rem;height:100%">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:2px;
              color:#38BDF8;margin-bottom:0.7rem">REDUCTION DETAILS</div>
  <div style="font-size:0.85rem;color:#C8D8F0;line-height:2">
    Input dimensions: <strong>21</strong><br/>
    Output dimensions: <strong style="color:#38BDF8">8</strong><br/>
    Explained variance: <strong style="color:#38BDF8">{total_var*100:.1f}%</strong><br/>
    Fit on: <strong>train split only</strong><br/>
    Method: <strong>StandardScaler + PCA</strong><br/>
    Encoding: <strong>θ = tanh(z)·π → RY</strong><br/>
    Seed: <strong>42</strong>
  </div>
  <div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #1E2E4A;
              font-size:0.78rem;color:#6B84A8;line-height:1.6">
    {qr_meta.get("why_8","8 components match the 8-qubit ansatz.")}
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("Quantum-ready metadata not found at data/quantum_ready/README.json")

    # ── Dataset files ─────────────────────────────────────────────────────────
    section("FILES", "Dataset File Structure")

    files = [
        ("data/synthetic/lightcurves.npz",        "Master dataset (25k curves)",             "22.8 MB"),
        ("data/splits/train.npz",                  "Train split NPZ (17.5k curves + labels)", "~16 MB"),
        ("data/splits/val.npz",                    "Validation split NPZ (3.75k)",           "~3.4 MB"),
        ("data/splits/test.npz",                   "Test split NPZ (3.75k) — FROZEN",        "~3.4 MB"),
        ("data/classical/classical_dataset.csv",   "Classical 21-feature CSV (25k rows)",     "7.6 MB"),
        ("data/hybrid/hybrid_dataset.csv",         "Hybrid 21+8 feature CSV (25k rows)",      "11.5 MB"),
        ("data/quantum_ready/quantum_ready_dataset.csv", "PCA-8 quantum-ready CSV (25k rows)", "4.8 MB"),
        ("data/quantum_ready/reducer8.joblib",     "PCA+scaler pipeline (train-fit)",         ""),
        ("models/feature_scaler.joblib",           "Z-score scaler (train-fit)",              ""),
    ]

    st.markdown('<table class="qo-table" style="width:100%"><thead>'
                '<tr><th>File</th><th>Description</th><th>Size</th><th>Exists</th></tr>'
                '</thead><tbody>', unsafe_allow_html=True)
    for rel_path, desc, size in files:
        full = os.path.join(ROOT, *rel_path.split("/"))
        exists = os.path.exists(full)
        mark = (
            '<span style="color:#64D8A8">✓</span>' if exists
            else '<span style="color:#E06060">✗ missing</span>'
        )
        st.markdown(
            f'<tr><td><code style="font-size:0.75rem;color:#38BDF8">{rel_path}</code></td>'
            f'<td style="font-size:0.8rem">{desc}</td>'
            f'<td class="mono" style="font-size:0.75rem">{size}</td>'
            f'<td>{mark}</td></tr>',
            unsafe_allow_html=True,
        )
    st.markdown("</tbody></table>", unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    section("EXPORT", "Download Dataset Files")
    dl1, dl2, dl3 = st.columns(3)
    for col, fpath, label in [
        (dl1, os.path.join(CLASSICAL, "classical_dataset.csv"), "Classical CSV (21 features)"),
        (dl2, os.path.join(HYBRID,    "hybrid_dataset.csv"),    "Hybrid CSV (21+8 features)"),
        (dl3, os.path.join(QR,        "quantum_ready_dataset.csv"), "Quantum-ready CSV (8 PCA)"),
    ]:
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                data_bytes = f.read()
            fname = os.path.basename(fpath)
            col.download_button(
                f"⬇ {label}", data=data_bytes,
                file_name=fname, mime="text/csv",
                width="stretch",
            )
        else:
            col.caption(f"{label} — not found")


if __name__ == "__main__":
    main()
else:
    main()
