# space_theme.py — Q-ORBIT shared design system
# Clean, professional deep-space research website theme
# NO cyberpunk/gaming aesthetics. Scientific. Spacious. Readable.
from __future__ import annotations
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "bg":           "#060B18",   # deep space near-black
    "bg2":          "#0A1020",   # slightly lighter panel bg
    "surface":      "#0D1628",   # card/panel surface
    "surface2":     "#111D32",   # elevated surface
    "border":       "#1E2E4A",   # subtle border
    "border_bright":"#2A4060",   # hover/active border
    "white":        "#F0F6FF",   # primary text
    "text":         "#C8D8F0",   # body text
    "muted":        "#6B84A8",   # muted/secondary text
    "cyan":         "#38BDF8",   # primary accent
    "cyan_dim":     "#1E6B8C",   # dim cyan
    "blue":         "#3B82F6",   # blue accent
    "violet":       "#7C6FBB",   # subtle violet
    "star":         "#A8C4E8",   # star color
    "class0":       "#38BDF8",   # Intact Satellite
    "class1":       "#64D8A8",   # Dead Satellite
    "class2":       "#F5A623",   # Rocket Body
    "class3":       "#E06060",   # Fragmentation Debris
    "class4":       "#A78BFA",   # Spoofed Satellite
}

CLASS_COLORS = [COLORS["class0"], COLORS["class1"], COLORS["class2"],
                COLORS["class3"], COLORS["class4"]]
CLASS_NAMES  = ["Intact Satellite", "Dead Satellite", "Rocket Body",
                "Fragmentation Debris", "Spoofed Satellite"]


# ── Core CSS ─────────────────────────────────────────────────────────────────
def css() -> str:
    """Full page CSS for the clean space theme."""
    return """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
  --bg:      #060B18;
  --bg2:     #0A1020;
  --surface: #0D1628;
  --surface2:#111D32;
  --border:  #1E2E4A;
  --border2: #2A4060;
  --white:   #F0F6FF;
  --text:    #C8D8F0;
  --muted:   #6B84A8;
  --cyan:    #38BDF8;
  --blue:    #3B82F6;
  --violet:  #7C6FBB;
  --ff-sans: 'Space Grotesk', 'Inter', sans-serif;
  --ff-mono: 'Space Mono', monospace;
}

/* ── Base ── */
html, body { background: var(--bg) !important; }
.stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--ff-sans) !important;
}
[data-testid="stHeader"]       { background: rgba(6,11,24,0.85) !important;
                                  backdrop-filter: blur(12px) !important;
                                  border-bottom: 1px solid var(--border) !important; }
[data-testid="stSidebar"]      { background: var(--bg2) !important;
                                  border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarContent"] { padding-top: 0.5rem !important; }
.block-container               { padding-top: 1.2rem !important;
                                  max-width: 1280px !important; }
h1, h2, h3, h4 { color: var(--white) !important;
                  font-family: var(--ff-sans) !important;
                  font-weight: 600 !important; }
p, li, label   { color: var(--text) !important; }
hr             { border-color: var(--border) !important; }
a              { color: var(--cyan) !important; }

/* ── Stars background (injected via render_stars()) ── */
.qo-stars { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
.qo-star  { position: absolute; border-radius: 50%; background: var(--star);
             animation: qo-twinkle ease-in-out infinite; }
@keyframes qo-twinkle {
  0%, 100% { opacity: 0.15; } 50% { opacity: 0.8; }
}
.qo-nebula {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 80vw 50vh at 15% 20%, rgba(56,189,248,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60vw 60vh at 85% 70%, rgba(124,111,187,0.05) 0%, transparent 60%),
    radial-gradient(ellipse 100vw 40vh at 50% 100%, rgba(10,20,50,0.8) 0%, transparent 80%);
}

/* ── Nav ── */
.qo-nav {
  position: sticky; top: 0; z-index: 100;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.65rem 1.2rem; margin: -1rem -1rem 1.5rem -1rem;
  background: rgba(6,11,24,0.92); backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}
.qo-brand { display: flex; align-items: center; gap: 0.7rem; text-decoration: none; }
.qo-logo  {
  width: 34px; height: 34px; border-radius: 8px;
  background: linear-gradient(135deg, #38BDF8 0%, #3B5FC0 100%);
  display: grid; place-items: center;
  font-family: var(--ff-mono); font-weight: 700; font-size: 0.78rem;
  color: #060B18; letter-spacing: 1px;
}
.qo-brand-name {
  font-family: var(--ff-sans); font-weight: 700; font-size: 1.05rem;
  color: var(--white); letter-spacing: 1px;
}
.qo-brand-sub {
  font-family: var(--ff-mono); font-size: 0.55rem;
  color: var(--muted); letter-spacing: 1.5px; display: block; margin-top: 1px;
}
.qo-nav-links { display: flex; gap: 0.1rem; align-items: center; flex-wrap: wrap; }
.qo-nav-links a {
  font-size: 0.78rem; color: var(--muted); text-decoration: none;
  padding: 0.38rem 0.7rem; border-radius: 6px;
  border: 1px solid transparent; transition: all 0.15s;
  font-family: var(--ff-sans); font-weight: 500;
}
.qo-nav-links a:hover { color: var(--white); background: rgba(56,189,248,0.06);
                         border-color: var(--border); }
.qo-nav-links a.active { color: var(--cyan); background: rgba(56,189,248,0.08);
                          border-color: rgba(56,189,248,0.2); }

/* ── Page Header ── */
.qo-page-header {
  padding: 1.8rem 0 1.2rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.6rem;
}
.qo-eyebrow {
  font-family: var(--ff-mono); font-size: 0.65rem;
  letter-spacing: 2.5px; color: var(--cyan);
  text-transform: uppercase; margin-bottom: 0.4rem;
}
.qo-page-title {
  font-size: clamp(1.8rem, 3.5vw, 2.8rem) !important;
  font-weight: 700 !important; color: var(--white) !important;
  line-height: 1.1; margin: 0 0 0.5rem 0 !important;
}
.qo-page-sub {
  color: var(--muted); font-size: 0.95rem; max-width: 70ch; line-height: 1.6;
}

/* ── Cards / Panels ── */
.qo-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.2rem 1.4rem;
  transition: border-color 0.2s;
}
.qo-card:hover { border-color: var(--border2); }
.qo-card-label {
  font-family: var(--ff-mono); font-size: 0.6rem;
  letter-spacing: 2px; color: var(--muted);
  text-transform: uppercase; margin-bottom: 0.5rem;
}
.qo-card-title {
  font-size: 1.05rem; font-weight: 600; color: var(--white);
  margin: 0 0 0.4rem 0;
}
.qo-card-body { color: var(--text); font-size: 0.88rem; line-height: 1.6; }

/* ── Metric boxes ── */
.qo-metric {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.9rem 1rem;
  text-align: center;
}
.qo-metric-value {
  font-family: var(--ff-sans); font-weight: 700;
  font-size: 1.7rem; color: var(--white); line-height: 1;
}
.qo-metric-value.cyan  { color: var(--cyan) !important; }
.qo-metric-value.blue  { color: var(--blue) !important; }
.qo-metric-value.muted { color: var(--muted) !important; }
.qo-metric-label {
  font-family: var(--ff-mono); font-size: 0.58rem;
  letter-spacing: 1.5px; color: var(--muted);
  text-transform: uppercase; margin-top: 0.3rem;
}
.qo-metric-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Section headings ── */
.qo-section-label {
  font-family: var(--ff-mono); font-size: 0.62rem;
  letter-spacing: 2.5px; color: var(--cyan);
  text-transform: uppercase; margin: 0 0 0.3rem 0;
}
.qo-section-title {
  font-size: 1.25rem !important; font-weight: 600 !important;
  color: var(--white) !important; margin: 0 0 0.2rem 0 !important;
}
.qo-section-sub {
  color: var(--muted); font-size: 0.88rem; line-height: 1.6;
}

/* ── Tags / Badges ── */
.qo-tag {
  display: inline-block;
  background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18);
  color: var(--cyan); border-radius: 4px;
  font-family: var(--ff-mono); font-size: 0.62rem; letter-spacing: 1px;
  padding: 0.22rem 0.55rem;
}
.qo-tag.violet {
  background: rgba(124,111,187,0.1); border-color: rgba(124,111,187,0.22);
  color: #A69FD6;
}
.qo-tag.green {
  background: rgba(100,216,168,0.08); border-color: rgba(100,216,168,0.2);
  color: #64D8A8;
}
.qo-tag.orange {
  background: rgba(245,166,35,0.08); border-color: rgba(245,166,35,0.2);
  color: #F5A623;
}

/* ── Orbit line accent (used in hero) ── */
.qo-orbit-ring {
  position: absolute; border-radius: 50%;
  border: 1px solid rgba(56,189,248,0.10);
  pointer-events: none;
}

/* ── Simulator Mode badge ── */
.qo-sim-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(100,216,168,0.07); border: 1px solid rgba(100,216,168,0.18);
  color: #64D8A8; border-radius: 999px;
  font-family: var(--ff-mono); font-size: 0.62rem; letter-spacing: 1.2px;
  padding: 0.3rem 0.7rem;
}
.qo-sim-dot { width: 6px; height: 6px; border-radius: 50%;
               background: #64D8A8; animation: qo-pulse 2s infinite; }
@keyframes qo-pulse {
  0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Workflow strip ── */
.qo-flow {
  display: flex; align-items: center; gap: 0; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.85rem 1rem;
  overflow-x: auto;
}
.qo-flow-node {
  padding: 0.45rem 0.85rem; border-radius: 6px;
  background: var(--surface2); border: 1px solid var(--border);
  font-family: var(--ff-mono); font-size: 0.68rem;
  letter-spacing: 0.8px; color: var(--text);
  white-space: nowrap;
}
.qo-flow-node.active {
  background: rgba(56,189,248,0.08); border-color: rgba(56,189,248,0.25);
  color: var(--cyan);
}
.qo-flow-arrow {
  color: var(--border2); font-size: 0.9rem;
  padding: 0 0.4rem; flex-shrink: 0;
}

/* ── Probability bars ── */
.qo-prob-track {
  height: 30px; border-radius: 4px;
  background: var(--surface2); border: 1px solid var(--border);
  overflow: hidden; margin: 5px 0; position: relative;
}
.qo-prob-fill {
  height: 100%; border-radius: 4px;
  display: flex; align-items: center; padding-left: 10px;
  font-family: var(--ff-sans); font-size: 0.78rem;
  font-weight: 500; color: var(--white);
  white-space: nowrap; overflow: hidden;
  transition: width 0.6s cubic-bezier(0.2,0.8,0.2,1);
}

/* ── Result highlight box ── */
.qo-result-box {
  padding: 1rem 1.2rem; border-radius: 10px;
  border: 1px solid var(--border2);
  background: var(--surface2);
}
.qo-result-box.success {
  border-color: rgba(56,189,248,0.3);
  background: rgba(56,189,248,0.05);
}
.qo-result-box.warning {
  border-color: rgba(245,166,35,0.3);
  background: rgba(245,166,35,0.04);
}
.qo-result-box.info {
  border-color: rgba(124,111,187,0.3);
  background: rgba(124,111,187,0.05);
}

/* ── Table ── */
.qo-table { width: 100%; border-collapse: collapse; }
.qo-table th {
  font-family: var(--ff-mono); font-size: 0.62rem; letter-spacing: 1.5px;
  color: var(--muted); text-transform: uppercase;
  padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border);
  text-align: left;
}
.qo-table td {
  padding: 0.55rem 0.8rem; border-bottom: 1px solid rgba(30,46,74,0.5);
  font-size: 0.84rem; color: var(--text);
}
.qo-table tr:hover td { background: rgba(56,189,248,0.03); }
.qo-table .mono { font-family: var(--ff-mono); font-size: 0.8rem; }
.qo-table .highlight { color: var(--cyan); font-weight: 600; }

/* ── Streamlit overrides ── */
.stButton > button {
  background: rgba(56,189,248,0.08) !important;
  border: 1px solid rgba(56,189,248,0.3) !important;
  color: var(--cyan) !important;
  border-radius: 8px !important;
  font-family: var(--ff-sans) !important;
  font-weight: 600 !important;
  letter-spacing: 0.5px !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: rgba(56,189,248,0.14) !important;
  border-color: rgba(56,189,248,0.45) !important;
}
.stButton > button[kind="primary"] {
  background: rgba(56,189,248,0.12) !important;
  border-color: rgba(56,189,248,0.45) !important;
}
[data-baseweb="select"] > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
[data-testid="stSlider"] label { color: var(--text) !important; }
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 4px !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 6px !important;
  color: var(--muted) !important;
  font-family: var(--ff-sans) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(56,189,248,0.1) !important;
  color: var(--cyan) !important;
}
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; padding: 0.8rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--white) !important; }
[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  font-family: var(--ff-mono) !important;
  font-size: 0.62rem !important;
  letter-spacing: 1px !important;
}
[data-testid="stDataFrameResizable"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.stAlert { border-radius: 8px !important; }
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
}
/* Sidebar nav */
[data-testid="stSidebarNavItems"] a {
  font-family: var(--ff-sans) !important;
  font-size: 0.88rem !important;
  color: var(--muted) !important;
  border-radius: 6px !important;
}
[data-testid="stSidebarNavItems"] [aria-selected="true"] {
  background: rgba(56,189,248,0.08) !important;
  color: var(--cyan) !important;
}
/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* Plotly dark overrides */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Responsive: tablet & mobile ── */
@media (max-width: 980px) {
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .qo-nav { flex-direction: column; align-items: flex-start; gap: 0.6rem; }
  .qo-nav-links { width: 100%; overflow-x: auto; padding-bottom: 0.2rem; }
  .qo-table { display: block; overflow-x: auto; white-space: nowrap; }
}
@media (max-width: 640px) {
  .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
  .qo-page-title { font-size: 1.6rem !important; }
  .qo-card { padding: 0.9rem 1rem; }
  .qo-metric-value { font-size: 1.3rem; }
  .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
  img { max-width: 100%; height: auto; }
}
</style>
"""


def stars_html(seed: int = 7, count: int = 160) -> str:
    """Generate subtle star field HTML."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(count):
        x   = float(rng.uniform(0, 100))
        y   = float(rng.uniform(0, 100))
        sz  = round(float(rng.uniform(0.8, 2.2)), 1)
        dur = round(float(rng.uniform(3, 8)), 1)
        dl  = round(float(rng.uniform(-6, 0)), 1)
        op  = round(float(rng.uniform(0.2, 0.7)), 2)
        parts.append(
            f'<div class="qo-star" style="left:{x:.1f}%;top:{y:.1f}%'
            f';width:{sz}px;height:{sz}px'
            f';animation-duration:{dur}s;animation-delay:{dl}s;opacity:{op}"></div>'
        )
    return f'<div class="qo-stars">{"".join(parts)}</div><div class="qo-nebula"></div>'


def nav_html(active: str = "home") -> str:
    pages = [
        ("home",        "Home",        "/"),
        ("data",        "Data",        "/Data"),
        ("classical",   "Classical AI","/Classical_AI"),
        ("hybrid",      "Hybrid",      "/Hybrid"),
        ("experiments", "Experiments", "/Experiments"),
    ]
    links = ""
    for key, label, href in pages:
        cls = " active" if key == active else ""
        links += f'<a href="{href}" class="{cls.strip()}">{label}</a>'
    return f"""
<div class="qo-nav">
  <a class="qo-brand" href="/">
    <div class="qo-logo">Q</div>
    <div>
      <div class="qo-brand-name">Q-ORBIT</div>
      <span class="qo-brand-sub">QUANTUM-CLASSICAL SPACE RESEARCH</span>
    </div>
  </a>
  <div class="qo-nav-links">{links}</div>
</div>"""


def page_header(eyebrow: str, title: str, subtitle: str) -> str:
    return f"""
<div class="qo-page-header">
  <div class="qo-eyebrow">{eyebrow}</div>
  <h1 class="qo-page-title">{title}</h1>
  <p class="qo-page-sub">{subtitle}</p>
</div>"""


def metric_html(value: str, label: str, sub: str = "", color: str = "") -> str:
    color_cls = f" {color}" if color else ""
    return f"""
<div class="qo-metric">
  <div class="qo-metric-value{color_cls}">{value}</div>
  <div class="qo-metric-label">{label}</div>
  {"" if not sub else f'<div class="qo-metric-sub">{sub}</div>'}
</div>"""


def card_html(title: str, body: str, label: str = "") -> str:
    lbl = f'<div class="qo-card-label">{label}</div>' if label else ""
    return f"""
<div class="qo-card">
  {lbl}
  <div class="qo-card-title">{title}</div>
  <div class="qo-card-body">{body}</div>
</div>"""


def section_heading(label: str, title: str, subtitle: str = "") -> str:
    sub = f'<p class="qo-section-sub">{subtitle}</p>' if subtitle else ""
    return f"""
<div style="margin: 1.8rem 0 0.8rem 0;">
  <div class="qo-section-label">{label}</div>
  <h3 class="qo-section-title">{title}</h3>
  {sub}
</div>"""


def sim_mode_badge() -> str:
    return '<span class="qo-sim-badge"><span class="qo-sim-dot"></span>SIMULATOR MODE — PennyLane default.qubit</span>'


def inject(st_obj) -> None:
    """Inject all CSS + stars into a Streamlit page."""
    import streamlit as st
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(stars_html(), unsafe_allow_html=True)


# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(6,11,24,0.0)",
    font=dict(family="Space Grotesk, Inter, sans-serif", color="#C8D8F0", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(30,46,74,0.6)", zeroline=False, linecolor="rgba(30,46,74,0.8)"),
    yaxis=dict(gridcolor="rgba(30,46,74,0.6)", zeroline=False, linecolor="rgba(30,46,74,0.8)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(30,46,74,0.6)", borderwidth=1),
    colorway=["#38BDF8", "#64D8A8", "#F5A623", "#E06060", "#A78BFA"],
)


def apply_layout(fig, height: int = 360, title: str = "") -> None:
    """Apply standard plotly layout to a figure in-place."""
    kw = {**PLOTLY_LAYOUT, "height": height}
    if title:
        kw["title"] = dict(text=title, font=dict(size=13, color="#C8D8F0"), x=0.01)
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor="rgba(30,46,74,0.6)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(30,46,74,0.6)", zeroline=False)


# ── Sidebar branding ─────────────────────────────────────────────────────────
def sidebar_logo(st_obj) -> None:
    import streamlit as st
    st.sidebar.markdown(f"""
<div style="padding:0.8rem 0.5rem 1rem 0.5rem;border-bottom:1px solid var(--border);margin-bottom:0.5rem">
  <div style="display:flex;align-items:center;gap:0.6rem">
    <div style="width:28px;height:28px;border-radius:6px;background:linear-gradient(135deg,#38BDF8,#3B5FC0);
                display:grid;place-items:center;font-family:'Space Mono';font-weight:700;
                font-size:0.7rem;color:#060B18">Q</div>
    <div>
      <div style="font-weight:700;font-size:0.88rem;color:#F0F6FF">Q-ORBIT</div>
      <div style="font-family:'Space Mono';font-size:0.52rem;color:#6B84A8;letter-spacing:1px">
        SPACE RESEARCH
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
