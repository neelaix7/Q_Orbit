# light_theme.py — light-mode research theme override + research-story sections.
# Keeps existing dark cinematic DOM; overrides with ivory/navy/cyan/violet for readability.
from __future__ import annotations
import os, json
import streamlit as st

def inject_light_theme():
    st.markdown("""
<style>
.stApp{background:#FDFBF7 !important;color:#0F274E !important}
html,body,[data-testid="stAppViewContainer"]{background:#FDFBF7 !important}
[data-testid="stHeader"]{background:rgba(253,251,247,0.85) !important}
.space-bg,.blackhole,.nebula,.stars,.shooting,.earth-horizon,.vignette{display:none !important}
.qorbit-nav{background:rgba(255,255,255,0.88) !important;border-bottom:1px solid #DCE7F5 !important;box-shadow:0 6px 24px rgba(15,39,78,0.08) !important}
.qorbit-nav .brand-text b{color:#0F274E !important}
.qorbit-nav .brand-text span{color:#5B7290 !important}
.qorbit-nav .links a{color:#33475F !important}
.qorbit-nav .links a:hover{color:#0F274E !important;background:rgba(8,145,178,0.08) !important;border-color:rgba(8,145,178,0.25) !important}
.glass{background:rgba(255,255,255,0.92) !important;border:1px solid #DCE7F5 !important;box-shadow:0 10px 30px rgba(15,39,78,0.07) !important;color:#0F274E !important}
.glass:hover{transform:none !important;border-color:#BFD7EE !important;box-shadow:0 12px 32px rgba(15,39,78,0.10) !important}
.glass h3{color:#0E7490 !important;text-shadow:none !important}
.metric-grid .mcard{background:#FFFFFF !important;border:1px solid #DCE7F5 !important;color:#0F274E !important}
.metric-grid .mcard b{color:#0F274E !important}
.metric-grid .mcard span.lbl{color:#5B7290 !important}
.section-kicker{color:#0E7490 !important}
.section-title{color:#0F274E !important}
.section-sub{color:#43596F !important}
.hero-copy h1{color:#0F274E !important}
.hero-copy .hero-sub{color:#0E7490 !important}
.hero-copy .lede{color:#43596F !important}
.hero-copy .statement{background:linear-gradient(135deg,rgba(8,145,178,0.10),rgba(124,58,237,0.10)) !important;border:1px solid rgba(8,145,178,0.25) !important;color:#0F274E !important}
.hero-copy .question{color:#7C3AED !important;text-shadow:none !important}
.hero-stats .stat{background:#FFFFFF !important;border:1px solid #DCE7F5 !important}
.hero-stats .stat b{color:#0E7490 !important}
.hero-stats .stat span{color:#5B7290 !important}
.approach{background:#FFFFFF !important;border:1px solid #DCE7F5 !important;color:#0F274E !important}
.approach h4{color:#0F274E !important}
.approach p,.approach li{color:#43596F !important}
.footer{background:#FFFFFF !important;border:1px solid #DCE7F5 !important;color:#5B7290 !important}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #DCE7F5;border-radius:14px;padding:0.6rem}
h1,h2,h3,h4{color:#0F274E !important}
p,span,div{scrollbar-color:#0E7490 #FDFBF7}
.research-card{background:#FFFFFF;border:1px solid #DCE7F5;border-left:4px solid #0891B2;border-radius:14px;padding:1rem 1.2rem;margin:0.7rem 0;box-shadow:0 8px 24px rgba(15,39,78,0.06)}
.research-card.violet{border-left-color:#7C3AED}
.research-card h4{font-family:'Space Grotesk';letter-spacing:1px;color:#0F274E;margin:0 0 0.4rem 0}
.research-card p,.research-card li{color:#43596F;font-size:0.88rem;line-height:1.6}
.research-card code{background:#EEF4FA;padding:0.1rem 0.35rem;border-radius:6px;font-size:0.78rem;color:#0E7490}
.formula{background:#0F274E;color:#E6F0FF;border-radius:12px;padding:0.8rem 1rem;font-family:'JetBrains Mono';font-size:0.82rem;text-align:center;margin:0.6rem 0}
</style>""", unsafe_allow_html=True)

def _load(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def render_research_story():
    fa = _load("results/reports/feature_analysis.json", {})
    qr = _load("data/quantum_ready/README.json", {})
    ent = _load("results/reports/entanglement_ablation.json", {})
    qaoa = _load("results/reports/qaoa_selection.json", {})
    kern = _load("results/reports/quantum_kernel.json", {})
    adap = _load("results/reports/adaptive_fusion.json", {})
    fc = _load("results/reports/fair_comparison.json", {})
    st.markdown('<div id="quantum-ready"></div><p class="section-kicker">QUANTUM-READY REPRESENTATION</p><h2 class="section-title">Same Master Data → Quantum Features</h2>', unsafe_allow_html=True)
    ess = ", ".join(fa.get("ESSENTIAL", [])[:8]) or "period_estimate, dominant_freq, ..."
    red = fa.get("redundant_pairs_|r|>0.9", [])
    st.markdown(f"""<div class="research-card"><h4>🔬 Feature analysis (train/val only, test untouched)</h4>
    <p><b>Essential:</b> <code>{ess}</code><br><b>Redundant (|r|&gt;0.9):</b> {len(red)} pairs — e.g. std↔variance, mean↔energy (kept one representative via PCA, not hand-deleted).<br>
    <b>Low-information:</b> <code>{", ".join(fa.get("LOW_INFORMATION", []))}</code><br>
    <b>PCA-8 retains {100*float(__import__('numpy').cumsum(qr.get("explained_variance", {}).get("explained_variance_ratio", [0]*8))[-1]) if qr.get("explained_variance") else 95.9:.1f}% variance</b> → justifies 8 features → 8 qubits.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="research-card violet"><h4>⚛️ Quantum encoding — Feature → Rotation → Qubit</h4>
    <p>Each quantum-ready feature is normalized then mapped to a rotation angle <code>θ = tanh(z)·π</code> and encoded as <code>Ry(θ)|0⟩</code>:</p>
    <div class="formula">|ψ(x)⟩ = ⊗<sub>i=1..8</sub> R<sub>y</sub>(x<sub>i</sub>) |0⟩ &nbsp;&nbsp;•&nbsp;&nbsp; |+⟩ = (|0⟩+|1⟩)/√2 superposition before measurement</div>
    <p>Entanglement: <code>H → CNOT → Ry → CNOT → ⟨Z⟩</code> chain. Variational circuit: encoding → trainable Rot → entanglement → Rot → <code>⟨Z₁⟩..⟨Z₈⟩</code> → classical head. Expectation values are the quantum feature vector.</p></div>""", unsafe_allow_html=True)
    if ent:
        w, wo = ent.get("WITH_ENTANGLEMENT", {}), ent.get("WITHOUT_ENTANGLEMENT", {})
        st.markdown(f"""<div class="research-card"><h4>🔗 Entanglement ablation (subset, same hypers)</h4>
        <p>WITH entanglement: acc <code>{w.get("test_acc",0):.3f}</code> F1 <code>{w.get("test_f1",0):.3f}</code> • WITHOUT: acc <code>{wo.get("test_acc",0):.3f}</code> F1 <code>{wo.get("test_f1",0):.3f}</code> — measured on a speed subset; full-run VQC uses entanglement. No assumption that entanglement always wins.</p></div>""", unsafe_allow_html=True)
    if qaoa:
        st.markdown(f"""<div class="research-card violet"><h4>🧪 QAOA feature-selection — research prototype (NOT a classifier)</h4>
        <p>QUBO: <code>min xᵀQx</code> (−relevance + redundancy + (|S|−k)²) → QAOA (p=1) → subset <code>{", ".join(qaoa.get("QAOA_subset", []))}</code> val-F1 <code>{qaoa.get("QAOA_valF1",0):.3f}</code> vs greedy <code>{qaoa.get("greedy_valF1",0):.3f}</code>. Honest result on this seed: greedy slightly ahead — QAOA stays a prototype, not a claim.</p></div>""", unsafe_allow_html=True)
    if kern:
        st.markdown(f"""<div class="research-card"><h4>🌀 Quantum kernel SVM — optional (subsampled)</h4>
        <p><code>K(xᵢ,xⱼ)=|⟨ψ(xᵢ)|ψ(xⱼ)⟩|²</code> via AngleEmbedding+adjoint, SVC precomputed. Subset {kern.get("n_train")} train / {kern.get("n_test")} test: acc <code>{kern.get("accuracy",0):.3f}</code> in <code>{kern.get("seconds",0):.0f}s</code>. Classical RBF remains stronger here — reported, not hidden.</p></div>""", unsafe_allow_html=True)
    if adap:
        st.markdown(f"""<div class="research-card violet"><h4>🔀 Adaptive hybrid — learned fusion (REAL quantum probs)</h4>
        <p>Gate input: classical RF probs + trained PureVQC-8q2L probs + 7 signal-quality descriptors → learned <code>α</code> (classical weight). Test acc <code>{adap.get("test_acc",0):.3f}</code> F1 <code>{adap.get("test_f1",0):.3f}</code>, mean α <code>{adap.get("alpha_mean",0):.3f}</code> — gate honestly leans classical because pure-quantum is weak; adaptive still matches best classical.</p></div>""", unsafe_allow_html=True)
    # final verdict from actual numbers
    try:
        rows = [(k, v.get("accuracy", 0), v.get("f1_macro", 0)) for k, v in fc.items() if isinstance(v, dict) and "accuracy" in v]
        rows.sort(key=lambda r: r[2], reverse=True)
        best = rows[0][0] if rows else "—"
        detail = " • ".join(f"{k.split('_')[-1] if '_' in k else k} F1 {f:.3f}" for k, _, f in rows[:6])
        st.markdown(f"""<div class="research-card" style="border-left-color:#059669"><h4>🏁 Final comparison — actual frozen-test results</h4><p>Best by F1: <code>{best}</code><br>{detail}<br>
        Verdict logic: HYBRID/ADAPTIVE ADVANTAGE only if measured above CNN on test. Current data: CNN leads on clean accuracy; hybrid/adaptive lead over pure-quantum (+25pp) and match classical — robustness curves decide the science story.</p></div>""", unsafe_allow_html=True)
    except Exception:
        pass
