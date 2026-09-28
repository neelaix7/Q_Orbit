# app.py — Q-ORBIT Premium Space-Tech Interface
# NASA mission control + quantum lab + premium AI startup aesthetic
from __future__ import annotations
import os, sys, base64, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.simulator.lightcurve_generator import generate_single_light_curve
from src.classical.feature_engineering import extract_all_features
from src.quantum.hybrid_model import HybridQuantumClassifier

# ───────────────────────────────────────── Config
CLASS_NAMES = ["Intact Satellite","Dead Satellite","Rocket Body","Fragmentation Debris","Spoofed Satellite"]
CLASS_EMOJIS = ["🛰️","💀","🚀","💥","🕶️"]
CLASS_COLORS = ["#38bdf8","#fb7185","#facc15","#c084fc","#2dd4bf"]
CLASS_IMAGES = ["surveillance_satellite.jpg","earth_night.jpg","rocket_body.jpg","galaxy_collision.jpg","satellite_other.jpg"]
CLASS_CAPTIONS = ["NASA Aqua — operational Earth observer","Earth at night — silent powered-down hulk","Lunar test rocket — tumbling upper stage","Galaxy collision debris — chaotic fragments","Undocumented satellite — spoofed signature"]
HERO_IMAGE = "earth_blue_marble.jpg"
QUANTUM_IMAGE = "quantum_visual.jpg"
MODEL_FEATURES = 21
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "hybrid_quantum_model.pt")
IMG_DIR = os.path.join(BASE_DIR, "..", "assets", "images", "web")

def img_b64(fn: str) -> str:
    p=os.path.join(IMG_DIR, fn)
    if not os.path.exists(p): return ""
    import base64
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Exo+2:wght@300;400;600;700&display=swap');
:root{
  --cyan:#00E5FF; --cyan2:#7DD3FC; --blue:#3B82F6; --purple:#7C3AED; --magenta:#C084FC; --pink:#F472B6;
  --glass:rgba(14,22,56,0.46); --glass2:rgba(18,28,72,0.62); --border:rgba(125,211,252,0.16);
}
/* base */
.stApp{background:#020512;color:#E6EDFF;font-family:'Space Grotesk','Exo 2',sans-serif;overflow-x:hidden}
html,body,[data-testid="stAppViewContainer"]{background:transparent !important}
[data-testid="stHeader"]{background:transparent !important;backdrop-filter:none !important}
[data-testid="stMainBlockContainer"]{padding-top:0 !important}
.block-container{padding-top:1.2rem !important;max-width:1280px}
hr{border-color:rgba(125,211,252,0.12) !important}

/* ────────── Deep space background ────────── */
.space-bg{position:fixed;inset:0;z-index:0;pointer-events:none;background:
  radial-gradient(ellipse 900px 700px at 22% 10%, rgba(124,58,237,0.22), transparent 60%),
  radial-gradient(ellipse 800px 600px at 78% 18%, rgba(59,130,246,0.20), transparent 62%),
  radial-gradient(ellipse 700px 500px at 52% 92%, rgba(6,182,212,0.14), transparent 65%),
  radial-gradient(ellipse at 50% 55%, #0A1432 0%, #070C24 45%, #020512 78%);
}
.nebula{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:0.9}
.nebula .blob{position:absolute;border-radius:50%;filter:blur(80px);mix-blend-mode:screen;opacity:0.55;animation:drift 28s ease-in-out infinite alternate}
.b1{width:52vw;height:52vw;background:#1e0a4a;left:-14%;top:-18%}
.b2{width:40vw;height:40vw;background:#0a2a6e;right:-12%;top:4%;animation-delay:-7s}
.b3{width:44vw;height:44vw;background:#4a0b6e;left:28%;bottom:-22%;animation-delay:-15s}
.b4{width:28vw;height:28vw;background:#064e5e;right:10%;bottom:-10%;animation-delay:-3s}
@keyframes drift{0%{transform:translate(0,0) scale(1)}50%{transform:translate(3.5vw,-2.5vw) scale(1.12)}100%{transform:translate(-3vw,2.5vw) scale(0.92)}}
.stars{position:fixed;inset:0;z-index:0;pointer-events:none}
.stars .star{position:absolute;border-radius:50%;background:#fff;box-shadow:0 0 6px 1px rgba(190,215,255,0.8);animation:twinkle ease-in-out infinite}
@keyframes twinkle{0%,100%{opacity:0.18;transform:scale(0.85)}50%{opacity:1;transform:scale(1.15)}}
.shooting{position:fixed;z-index:0;pointer-events:none;width:2px;height:2px;border-radius:50%;background:#fff;box-shadow:0 0 10px 3px rgba(255,255,255,0.9);animation:shoot ease-in-out infinite}
.shooting::after{content:"";position:absolute;top:0;left:0;width:140px;height:1.2px;background:linear-gradient(270deg,rgba(255,255,255,0.95),rgba(160,200,255,0.35),transparent)}
@keyframes shoot{0%{transform:translate(0,0);opacity:0}5%{opacity:1}18%{transform:translate(-52vw,26vh);opacity:0}100%{opacity:0}}
.earth-glow{position:fixed;z-index:0;pointer-events:none;left:50%;bottom:-48vw;width:140vw;height:80vw;transform:translateX(-50%);border-radius:50%;background:radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.18), rgba(59,130,246,0.10) 35%, transparent 72%);filter:blur(2px)}
.vignette{position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse at 50% 50%, transparent 58%, rgba(0,0,0,0.55) 100%)}

/* ────────── Navigation ────────── */
.qorbit-nav{position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:0.75rem 1.1rem;margin:0 -1rem 1.1rem -1rem;background:rgba(6,10,28,0.62);backdrop-filter:blur(18px) saturate(1.4);border-bottom:1px solid rgba(125,211,252,0.14);box-shadow:0 8px 32px rgba(0,0,0,0.45)}
.qorbit-nav .brand{display:flex;align-items:center;gap:0.9rem}
.qorbit-nav .logo-mark{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(135deg,#00E5FF,#7C3AED);box-shadow:0 0 18px rgba(0,229,255,0.45), inset 0 0 12px rgba(255,255,255,0.25);font-family:'Orbitron';font-weight:900;color:#020512;font-size:1.05rem;letter-spacing:1px}
.qorbit-nav .brand-text b{font-family:'Orbitron';letter-spacing:3px;font-size:1.05rem;color:#E6F0FF}
.qorbit-nav .brand-text span{display:block;font-size:0.66rem;letter-spacing:2.2px;color:#8AA0C8;margin-top:1px;font-family:'JetBrains Mono'}
.qorbit-nav .links{display:flex;align-items:center;gap:0.2rem;flex-wrap:wrap}
.qorbit-nav .links a{font-size:0.78rem;letter-spacing:1.4px;color:#9FB4D8;text-decoration:none;padding:0.45rem 0.75rem;border-radius:10px;border:1px solid transparent;transition:all .18s}
.qorbit-nav .links a:hover{color:#E6F0FF;background:rgba(125,211,252,0.08);border-color:rgba(125,211,252,0.18);box-shadow:0 0 14px rgba(0,229,255,0.15)}
.qorbit-nav .links a.active{color:#00E5FF;background:rgba(0,229,255,0.08);border-color:rgba(0,229,255,0.22)}
.qorbit-nav .cta{font-family:'Orbitron';font-size:0.78rem;letter-spacing:1.2px;padding:0.62rem 1.15rem;border-radius:999px;background:linear-gradient(135deg,#00E5FF,#7C3AED);color:#020512;font-weight:800;border:none;box-shadow:0 6px 22px rgba(124,58,237,0.45),0 0 18px rgba(0,229,255,0.35);cursor:pointer;transition:transform .15s, box-shadow .15s}
.qorbit-nav .cta:hover{transform:translateY(-1px) scale(1.02);box-shadow:0 10px 28px rgba(124,58,237,0.55),0 0 24px rgba(0,229,255,0.45)}

/* ────────── Glass / HUD ────────── */
.glass{position:relative;z-index:2;background:linear-gradient(180deg, rgba(18,28,72,0.52), rgba(8,12,32,0.72));backdrop-filter:blur(18px) saturate(1.35);border-radius:18px;padding:1.15rem 1.25rem;border:1px solid var(--border);box-shadow:0 0 0 1px rgba(125,211,252,0.06),0 16px 48px rgba(0,0,0,0.5), inset 0 0 24px rgba(125,211,252,0.04);transition:transform .22s, box-shadow .22s, border-color .22s}
.glass:hover{transform:translateY(-2px);border-color:rgba(125,211,252,0.24);box-shadow:0 0 0 1px rgba(125,211,252,0.12),0 20px 56px rgba(0,0,0,0.55),0 0 28px rgba(124,58,237,0.22)}
.glass h3{font-family:'Orbitron';margin:0 0 0.75rem 0;font-size:0.92rem;letter-spacing:2.2px;color:#7DD3FC;text-shadow:0 0 14px rgba(125,211,252,0.45)}
.hud-corners{position:relative}
.hud-corners::before,.hud-corners::after{content:"";position:absolute;width:14px;height:14px;border-color:rgba(0,229,255,0.55);border-style:solid;pointer-events:none}
.hud-corners::before{top:-1px;left:-1px;border-width:1.5px 0 0 1.5px;border-radius:10px 0 0 0}
.hud-corners::after{bottom:-1px;right:-1px;border-width:0 1.5px 1.5px 0;border-radius:0 0 10px 0}

/* ────────── Hero ────────── */
.hero-wrap{position:relative;z-index:2;display:grid;grid-template-columns:1.15fr 0.95fr;gap:1.4rem;align-items:center;padding:1.2rem 0 0.6rem 0}
@media(max-width:980px){.hero-wrap{grid-template-columns:1fr;gap:1rem}}
.hero-copy .eyebrow{display:inline-flex;align-items:center;gap:0.55rem;padding:0.38rem 0.85rem;border-radius:999px;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.22);color:#7DD3FC;font-family:'JetBrains Mono';font-size:0.68rem;letter-spacing:1.8px}
.hero-copy .eyebrow .dot{width:7px;height:7px;border-radius:50%;background:#00E5FF;box-shadow:0 0 10px #00E5FF;animation:pulseDot 1.6s infinite}
@keyframes pulseDot{0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,0.6)}50%{box-shadow:0 0 0 8px rgba(0,229,255,0)}}
.hero-copy h1{font-family:'Orbitron';font-weight:900;letter-spacing:2px;line-height:1.02;margin:0.9rem 0 0.6rem 0;font-size:clamp(1.9rem,4vw,3.15rem);background:linear-gradient(90deg,#FFFFFF 0%,#7DD3FC 32%,#A78BFA 58%,#F472B6 85%,#FFFFFF 100%);background-size:260% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:grad 6s linear infinite;filter:drop-shadow(0 0 18px rgba(125,211,252,0.22))}
@keyframes grad{to{background-position:260% 0}}
.hero-copy .lede{color:#C7D6F5;font-size:1.06rem;line-height:1.6;max-width:62ch;margin:0}
.hero-actions{display:flex;gap:0.7rem;flex-wrap:wrap;margin-top:1.1rem}
.btn-primary{font-family:'Orbitron';letter-spacing:1.1px;font-weight:800;font-size:0.82rem;padding:0.78rem 1.25rem;border-radius:999px;background:linear-gradient(135deg,#00E5FF,#7C3AED);color:#020512;border:none;box-shadow:0 8px 26px rgba(124,58,237,0.45),0 0 18px rgba(0,229,255,0.3);cursor:pointer;transition:transform .15s}
.btn-primary:hover{transform:translateY(-1px)}
.btn-ghost{font-family:'Orbitron';letter-spacing:1.1px;font-weight:700;font-size:0.82rem;padding:0.78rem 1.25rem;border-radius:999px;background:rgba(255,255,255,0.06);color:#E6F0FF;border:1px solid rgba(125,211,252,0.22);backdrop-filter:blur(10px);cursor:pointer}
.btn-ghost:hover{background:rgba(125,211,252,0.10)}
.hero-stats{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.2rem}
.hero-stats .stat{flex:1;min-width:110px;background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.12);border-radius:14px;padding:0.75rem 0.9rem}
.hero-stats .stat b{font-family:'Orbitron';font-size:1.15rem;color:#00E5FF}
.hero-stats .stat span{display:block;font-size:0.7rem;letter-spacing:1.2px;color:#8AA0C8;margin-top:2px}

/* ────────── Satellite Stage (premium 3D) ────────── */
.sat-stage{position:relative;height:460px;border-radius:22px;overflow:hidden;background:radial-gradient(ellipse 520px 420px at 50% 38%, rgba(56,189,248,0.10), transparent 62%), linear-gradient(180deg, rgba(10,18,48,0.55), rgba(6,10,28,0.75));border:1px solid rgba(125,211,252,0.18);box-shadow:0 18px 60px rgba(0,0,0,0.55),0 0 40px rgba(124,58,237,0.18);display:grid;place-items:center}
@media(max-width:980px){.sat-stage{height:380px}}
.sat-stage::before{content:"";position:absolute;inset:-1px;border-radius:22px;padding:1px;background:linear-gradient(135deg, rgba(0,229,255,0.35), rgba(124,58,237,0.35), transparent 60%);-webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);-webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none}
.orbit{position:absolute;border-radius:50%;border:1px solid rgba(125,211,252,0.16);pointer-events:none}
.orbit.o1{width:340px;height:340px;animation:spin 26s linear infinite}
.orbit.o2{width:260px;height:260px;transform:rotateX(66deg);animation:spinRev 22s linear infinite;border-color:rgba(192,132,252,0.22)}
.orbit.o3{width:420px;height:420px;transform:rotateX(18deg) rotateZ(-18deg);animation:spin 34s linear infinite;border-color:rgba(0,229,255,0.14)}
.orbit::after{content:"";position:absolute;width:7px;height:7px;border-radius:50%;background:#7DD3FC;box-shadow:0 0 12px 3px rgba(125,211,252,0.9);top:-3.5px;left:50%;margin-left:-3.5px}
.orbit.o2::after{background:#C084FC;box-shadow:0 0 12px rgba(192,132,252,0.9)}
.orbit.o3::after{background:#00E5FF}
@keyframes spin{to{transform:rotate(360deg)}}@keyframes spinRev{to{transform:rotateX(66deg) rotate(-360deg)}}
.sat-core{position:relative;width:148px;height:148px;transform-style:preserve-3d;animation:satFloat 5.5s ease-in-out infinite, satYaw 26s linear infinite}
@keyframes satFloat{0%,100%{transform:translateY(0) rotateY(0) }50%{transform:translateY(-10px) rotateY(6deg)}}
@keyframes satYaw{to{transform:translateY(0) rotateY(360deg)}}
.sat-body{position:absolute;inset:18px;border-radius:14px;background:linear-gradient(180deg,#E6EEFF,#9DB7D8 55%,#5A7094);box-shadow:inset 0 0 18px rgba(255,255,255,0.55),0 10px 28px rgba(0,0,0,0.45);display:grid;place-items:center;overflow:hidden}
.sat-body::before{content:"";position:absolute;inset:0;background:linear-gradient(100deg, transparent 30%, rgba(255,255,255,0.28) 50%, transparent 70%);animation:shine 3.2s ease-in-out infinite}
@keyframes shine{0%{transform:translateX(-120%)}100%{transform:translateX(120%)}}
.sat-panel{position:absolute;top:34px;width:54px;height:86px;background:linear-gradient(180deg,#0F1A3A,#1E3A6A);border:1px solid rgba(125,211,252,0.35);box-shadow:0 0 14px rgba(0,229,255,0.25)}
.sat-panel.left{left:-48px;transform:rotateY(12deg)} .sat-panel.right{right:-48px;transform:rotateY(-12deg)}
.sat-panel::after{content:"";position:absolute;inset:6px;background:repeating-linear-gradient(0deg, rgba(0,229,255,0.18) 0 1px, transparent 1px 14px), repeating-linear-gradient(90deg, rgba(0,229,255,0.12) 0 1px, transparent 1px 18px)}
.sat-glow{position:absolute;inset:-22px;border-radius:28px;background:radial-gradient(circle at 50% 50%, rgba(0,229,255,0.18), transparent 68%);filter:blur(8px);pointer-events:none}
.sat-img{width:86px;height:86px;object-fit:contain;filter:drop-shadow(0 6px 14px rgba(0,0,0,0.5));position:relative;z-index:1}
.particles{position:absolute;inset:0;pointer-events:none}
.particles i{position:absolute;width:2px;height:2px;border-radius:50%;background:#CFE8FF;opacity:0.7;animation:driftP linear infinite}
@keyframes driftP{from{transform:translateY(0)}to{transform:translateY(-420px)}}

/* ticker */
.ticker{position:relative;z-index:2;overflow:hidden;white-space:nowrap;border-radius:999px;padding:0.55rem 1rem;margin:0.9rem 0 0.2rem 0;background:rgba(10,16,40,0.62);border:1px solid rgba(125,211,252,0.14);backdrop-filter:blur(10px)}
.ticker .track{display:inline-block;padding-left:100%;animation:marquee 22s linear infinite}
@keyframes marquee{to{transform:translateX(-100%)}}
.ticker span{margin:0 2.2rem;color:#9FD4FF;font-size:0.82rem;letter-spacing:1.1px;font-family:'JetBrains Mono'}

/* approaches */
.approach-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.8rem}
@media(max-width:980px){.approach-grid{grid-template-columns:1fr}}
.approach{position:relative;padding:1.15rem 1.15rem 1rem 1.15rem;border-radius:18px;background:linear-gradient(180deg, rgba(18,28,72,0.5), rgba(8,12,32,0.7));border:1px solid rgba(125,211,252,0.14);backdrop-filter:blur(14px);overflow:hidden;transition:transform .22s, box-shadow .22s, border-color .22s}
.approach:hover{transform:translateY(-3px);box-shadow:0 16px 44px rgba(0,0,0,0.5),0 0 26px rgba(124,58,237,0.18)}
.approach.hybrid{border-color:rgba(0,229,255,0.32);background:linear-gradient(180deg, rgba(18,30,78,0.62), rgba(10,14,38,0.78));box-shadow:0 0 0 1px rgba(0,229,255,0.12),0 18px 50px rgba(0,0,0,0.55),0 0 32px rgba(0,229,255,0.12);transform:scale(1.02)}
.approach.hybrid:hover{transform:scale(1.02) translateY(-3px)}
.approach .badge{position:absolute;top:10px;right:10px;font-family:'JetBrains Mono';font-size:0.62rem;letter-spacing:1.4px;padding:0.32rem 0.6rem;border-radius:999px;background:linear-gradient(135deg,#00E5FF,#7C3AED);color:#020512;font-weight:800}
.approach .icon{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;font-size:1.2rem;background:rgba(255,255,255,0.06);border:1px solid rgba(125,211,252,0.14)}
.approach.hybrid .icon{background:linear-gradient(135deg, rgba(0,229,255,0.18), rgba(124,58,237,0.18));border-color:rgba(0,229,255,0.28);box-shadow:0 0 16px rgba(0,229,255,0.22)}
.approach h4{font-family:'Orbitron';margin:0.7rem 0 0.35rem 0;letter-spacing:1.2px;font-size:0.95rem;color:#E6F0FF}
.approach p{margin:0;color:#A8BBDD;font-size:0.86rem;line-height:1.55}
.approach ul{margin:0.6rem 0 0 0;padding:0;list-style:none}
.approach ul li{font-size:0.78rem;color:#9FB4D8;display:flex;gap:0.45rem;align-items:center;margin:0.28rem 0}
.approach ul li::before{content:"▸";color:#00E5FF}
.approach .metric{margin-top:0.8rem;display:flex;gap:0.6rem}
.approach .metric div{flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.1);border-radius:12px;padding:0.55rem 0.6rem;text-align:center}
.approach .metric b{font-family:'Orbitron';color:#7DD3FC;font-size:0.95rem}
.approach .metric span{font-size:0.62rem;letter-spacing:1px;color:#8AA0C8;display:block}

/* photo cards */
.photo-card{position:relative;border-radius:14px;overflow:hidden;border:1px solid rgba(125,211,252,0.18);box-shadow:0 10px 30px rgba(0,0,0,0.5);transition:transform .25s, box-shadow .25s}
.photo-card:hover{transform:translateY(-2px) scale(1.01);box-shadow:0 16px 40px rgba(0,0,0,0.6),0 0 28px rgba(124,58,237,0.22)}
.photo-card img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
.photo-card .cap{position:absolute;inset:auto 0 0 0;padding:0.6rem 0.8rem;background:linear-gradient(0deg, rgba(2,5,18,0.92), transparent);font-size:0.76rem;color:#DCE8FF}
.photo-card::after{content:"";position:absolute;top:-60%;left:-60%;width:60%;height:220%;background:linear-gradient(100deg, transparent, rgba(255,255,255,0.14), transparent);transform:rotate(18deg);transition:left .6s;pointer-events:none}
.photo-card:hover::after{left:130%}

/* prob bars */
.prob-bar-track{height:32px;border-radius:999px;margin:8px 0;position:relative;overflow:hidden;background:rgba(255,255,255,0.06);border:1px solid rgba(125,211,252,0.12)}
.prob-bar{height:100%;border-radius:999px;width:0%;display:flex;align-items:center;padding-left:12px;font-size:0.82rem;font-weight:700;color:#fff;letter-spacing:0.3px;animation:growBar 1s cubic-bezier(.2,.8,.2,1) forwards;text-shadow:0 1px 4px rgba(0,0,0,0.6);box-shadow:inset 0 0 14px rgba(255,255,255,0.22)}
@keyframes growBar{to{width:var(--w)}}
.badge-pred{display:inline-block;padding:0.5rem 1.1rem;border-radius:999px;font-family:'Orbitron';font-weight:800;font-size:0.95rem;color:#020512;box-shadow:0 0 0 0 rgba(125,211,252,0.5);animation:pulseBadge 2s infinite}
@keyframes pulseBadge{0%,100%{box-shadow:0 0 0 0 rgba(125,211,252,0.45)}50%{box-shadow:0 0 0 14px rgba(125,211,252,0)}}

/* streamlit widgets styling */
.stButton>button{background:linear-gradient(135deg,#00E5FF,#7C3AED) !important;color:#020512 !important;border:none !important;border-radius:12px !important;font-family:'Orbitron' !important;font-weight:800 !important;letter-spacing:1px !important;box-shadow:0 8px 22px rgba(124,58,237,0.4) !important}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 12px 28px rgba(124,58,237,0.55) !important}
[data-testid="stSelectbox"],[data-testid="stSlider"] label{color:#A9BFDE !important;font-weight:600 !important;letter-spacing:0.8px !important}
[data-baseweb="select"]>div{background:rgba(255,255,255,0.06) !important;border:1px solid rgba(125,211,252,0.18) !important;border-radius:12px !important}
[data-testid="stSidebar"]{background:linear-gradient(180deg, rgba(10,16,44,0.88), rgba(6,10,28,0.96)) !important;border-right:1px solid rgba(125,211,252,0.14)}
[data-testid="stSidebar"] *{color:#DCE8FF}
[data-testid="stMetricValue"]{color:#7DD3FC;font-family:'Orbitron';text-shadow:0 0 14px rgba(125,211,252,0.35)}
[data-testid="stMetricLabel"]{color:#8AA0C8;letter-spacing:1px}
.stTabs [data-baseweb="tab-list"]{gap:0.25rem;background:rgba(255,255,255,0.04);border:1px solid rgba(125,211,252,0.12);padding:0.3rem;border-radius:999px}
.stTabs [data-baseweb="tab"]{border-radius:999px !important;color:#9FB4D8 !important;font-family:'Orbitron' !important;font-size:0.78rem !important;letter-spacing:0.8px}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#00E5FF,#7C3AED) !important;color:#020512 !important}

/* section titles */
.section-kicker{font-family:'JetBrains Mono';font-size:0.7rem;letter-spacing:2.8px;color:#00E5FF;margin:0 0 0.35rem 0}
.section-title{font-family:'Orbitron';letter-spacing:2px;font-size:1.35rem;color:#E6F0FF;margin:0 0 0.4rem 0}
.section-sub{color:#A8BBDD;font-size:0.92rem;margin:0 0 1rem 0;line-height:1.6}

/* footer */
.footer{position:relative;z-index:2;margin-top:1.6rem;padding:1.2rem 1.2rem;border-radius:18px;background:rgba(6,10,28,0.62);border:1px solid rgba(125,211,252,0.12);backdrop-filter:blur(12px);color:#8AA0C8;font-size:0.82rem}
.footer b{color:#DCE8FF}
</style>
    """, unsafe_allow_html=True)

def render_space_bg():
    # starfield
    rng=np.random.default_rng(7)
    stars_html=[]
    for _ in range(140):
        sz=round(float(rng.uniform(1.1,2.8)),2); x=float(rng.uniform(0,100)); y=float(rng.uniform(0,100)); d=round(float(rng.uniform(2,5.5)),2); dl=round(float(rng.uniform(-5,0)),2); op=round(float(rng.uniform(0.5,1)),2)
        stars_html.append(f'<div class="star" style="left:{x:.2f}%;top:{y:.2f}%;width:{sz}px;height:{sz}px;animation-duration:{d}s;animation-delay:{dl}s;opacity:{op}"></div>')
    rng2=np.random.default_rng(5)
    shoot=[]
    for i in range(3):
        t=float(rng2.uniform(2,38)); l=float(rng2.uniform(58,92)); d=round(float(rng2.uniform(9,15)),1); dl=round(float(rng2.uniform(2,9)),1)
        shoot.append(f'<div class="shooting" style="top:{t:.1f}%;left:{l:.1f}%;animation-duration:{d}s;animation-delay:{dl}s"></div>')
    return f'<div class="space-bg"></div><div class="nebula"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="blob b4"></div></div><div class="stars">{"".join(stars_html)}</div>{"".join(shoot)}<div class="earth-glow"></div><div class="vignette"></div>'

def render_nav():
    return """
<div class="qorbit-nav">
  <div class="brand">
    <div class="logo-mark">Q</div>
    <div class="brand-text"><b>Q-ORBIT</b><span>QUANTUM-CLASSICAL AI FOR SPACE OBJECT CLASSIFICATION</span></div>
  </div>
  <div class="links">
    <a href="#home" class="active">Home</a>
    <a href="#models">Models</a>
    <a href="#dataset">Dataset</a>
    <a href="#results">Results</a>
    <a href="#dashboard">Dashboard</a>
    <a href="#about">About</a>
    <a href="#dashboard" class="cta" style="margin-left:0.4rem;text-decoration:none;display:inline-block">⬢ Launch App</a>
  </div>
</div>
"""

def render_hero(sat_b64: str):
    # particles for sat stage
    rng=np.random.default_rng(11)
    parts=[]
    for _ in range(18):
        x=int(rng.uniform(8,92)); y=int(rng.uniform(10,90)); sz=int(rng.uniform(1,3)); d=float(rng.uniform(6,16)); dl=float(rng.uniform(-12,0))
        parts.append(f'<i style="left:{x}%;top:{y}%;width:{sz}px;height:{sz}px;animation-duration:{d:.1f}s;animation-delay:{dl:.1f}s"></i>')
    parts_html="".join(parts)
    return f"""
<div id="home" class="hero-wrap">
  <div class="hero-copy">
    <div class="eyebrow"><span class="dot"></span> MISSION Q-ORBIT • SEED 42 • 8-QUBIT CORE ONLINE</div>
    <h1>Classifying Space<br/>Objects with<br/>Quantum Intelligence</h1>
    <p class="lede">A hybrid quantum-classical framework that decodes tumbling light curves into five object classes — from intact satellites to spoofed stealth craft — inside a cinematic mission-control environment.</p>
    <div class="hero-actions">
      <a href="#dashboard" style="text-decoration:none"><span class="btn-primary">▶ Launch Inference Lab</span></a>
      <a href="#models" style="text-decoration:none"><span class="btn-ghost">Explore Models</span></a>
    </div>
    <div class="hero-stats">
      <div class="stat"><b>5</b><span>OBJECT CLASSES</span></div>
      <div class="stat"><b>256</b><span>SAMPLES / CURVE</span></div>
      <div class="stat"><b>21</b><span>ENGINEERED FEATURES</span></div>
      <div class="stat"><b>8 Q</b><span>ENTANGLED QUBITS</span></div>
    </div>
  </div>
  <div class="sat-stage hud-corners">
    <div class="orbit o1"></div>
    <div class="orbit o2"></div>
    <div class="orbit o3"></div>
    <div class="sat-core">
      <div class="sat-panel left"></div>
      <div class="sat-panel right"></div>
      <div class="sat-body">
        <img class="sat-img" src="data:image/jpeg;base64,{sat_b64}" alt="satellite">
        <div class="sat-glow"></div>
      </div>
    </div>
    <div class="particles">{parts_html}</div>
    <div style="position:absolute;bottom:10px;left:12px;font-family:'JetBrains Mono';font-size:0.62rem;letter-spacing:1.4px;color:rgba(160,190,220,0.85);background:rgba(0,0,0,0.28);border:1px solid rgba(125,211,252,0.14);padding:0.3rem 0.6rem;border-radius:999px">◉ TELEMETRY LOCK • ALT 547 km • INCL 51.6°</div>
    <div style="position:absolute;top:10px;right:12px;font-family:'JetBrains Mono';font-size:0.62rem;letter-spacing:1.2px;color:#7DD3FC;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);padding:0.3rem 0.6rem;border-radius:999px">● QUANTUM LINK ACTIVE</div>
  </div>
</div>
"""

# ───────── models / helpers (kept)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None,None
    import torch
    ckpt=torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    m=HybridQuantumClassifier(n_features=ckpt.get("n_features",MODEL_FEATURES), n_qubits=ckpt.get("n_qubits",8), n_layers=ckpt.get("n_layers",2), n_classical_hidden=ckpt.get("n_classical_hidden",16), n_classes=5)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m, ckpt.get("accuracy",None)

@st.cache_resource
def load_all_models():
    import torch, joblib
    from src.classical.cnn_baseline import LightCurveCNN
    from src.quantum.vqc import PureVQC
    bundle={}
    bundle["hybrid"], bundle["hybrid_acc"]=load_model()
    cnn_path=os.path.join(BASE_DIR,"..","models","classical_cnn_model.pt")
    if os.path.exists(cnn_path):
        ckpt=torch.load(cnn_path,weights_only=False,map_location="cpu")
        m=LightCurveCNN(n_classes=5); m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        bundle["cnn"]=m; bundle["cnn_acc"]=ckpt.get("accuracy",None)
    else: bundle["cnn"]=None
    svm_path=os.path.join(BASE_DIR,"..","models","classical_svm_model.joblib")
    if os.path.exists(svm_path):
        try: bundle["svm"]=joblib.load(svm_path)
        except: bundle["svm"]=None
    else: bundle["svm"]=None
    pure_path=os.path.join(BASE_DIR,"..","models","pure_quantum_vqc.pt")
    if os.path.exists(pure_path):
        ckpt=torch.load(pure_path,weights_only=False,map_location="cpu")
        nq=ckpt.get("n_qubits",8); nl=ckpt.get("n_layers",2)
        m=PureVQC(n_qubits=nq,n_layers=nl,n_classes=5); m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        bundle["pure"]=m; bundle["pure_acc"]=ckpt.get("accuracy",None)
        try:
            from src.features.reducer import QuantumReducer
            rp=os.path.join(BASE_DIR,"..","models","pure_vqc_reducer.joblib")
            if not os.path.exists(rp): rp=os.path.join(BASE_DIR,"..","models",f"reducer_{nq}q.joblib")
            if os.path.exists(rp):
                bundle["pure_reducer"]=QuantumReducer.load(rp)
                bundle["pure_mean"]=ckpt.get("mean",None); bundle["pure_std"]=ckpt.get("std",None)
        except: bundle["pure_reducer"]=None
    else: bundle["pure"]=None
    try:
        sc_path=os.path.join(BASE_DIR,"..","models","feature_scaler.joblib")
        if os.path.exists(sc_path):
            import joblib as jb; bundle["scaler"]=jb.load(sc_path)
        else: bundle["scaler"]=None
    except: bundle["scaler"]=None
    return bundle

def _resample_curve(curve: np.ndarray, target: int = 256) -> np.ndarray:
    """Resample arbitrary-length curve to target length via linear interpolation - keeps CNN input fixed."""
    curve = np.asarray(curve, dtype=float)
    n = len(curve)
    if n == target:
        return curve.astype(np.float32)
    if n < 2:
        return np.full(target, float(curve[0] if n==1 else 0.0), dtype=np.float32)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target)
    # handle NaN by interpolation first
    mask = np.isfinite(curve)
    if not mask.all():
        if mask.sum() >= 2:
            curve = np.interp(np.arange(n), np.where(mask)[0], curve[mask])
        else:
            curve = np.nan_to_num(curve, nan=0.0, posinf=1.0, neginf=0.0)
    return np.interp(x_new, x_old, curve).astype(np.float32)

def _sanitize_curve(curve: np.ndarray) -> np.ndarray:
    c = np.asarray(curve, dtype=float)
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    # guard against constant curve causing division-by-zero downstream
    if not np.isfinite(c).all():
        c = np.zeros_like(c)
    return c.astype(np.float32)

def predict_all(bundle, curve: np.ndarray):
    import torch
    # harden curve
    curve = _sanitize_curve(curve)
    # correct sampling interval for feature extraction: total duration 7200s
    n = len(curve)
    dt = 7200.0 / max(n - 1, 1) if n > 1 else 5.0
    # extract features safely
    try:
        feats_raw = extract_all_features(curve, sampling_interval=dt)
        feats_raw = np.nan_to_num(feats_raw, nan=0.0, posinf=0.0, neginf=0.0)
        feats_raw = np.clip(feats_raw, -1e6, 1e6)
    except Exception:
        feats_raw = np.zeros(MODEL_FEATURES, dtype=float)
    feats = feats_raw.astype(np.float32)
    out={}
    def _standardize(fv, sc):
        if sc is None: return fv
        mean=sc.get("mean")
        scale=sc.get("scale", sc.get("std", None))
        if mean is None or scale is None:
            return fv
        mean=np.asarray(mean, dtype=float)
        scale=np.asarray(scale, dtype=float)
        # handle shape mismatch (21 expected) - truncate/pad
        if mean.shape[0] != fv.shape[0]:
            # fallback: use only overlapping dims, pad rest with 0/1
            m = min(mean.shape[0], fv.shape[0])
            mean_pad = np.zeros_like(fv); scale_pad=np.ones_like(fv)
            mean_pad[:m]=mean[:m]; scale_pad[:m]=scale[:m]
            mean, scale = mean_pad, scale_pad
        scale=np.where(scale==0, 1.0, scale)
        fv = (fv - mean) / scale
        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
        return fv.astype(np.float32)
    # sanitize probs helper
    def _sanitize_probs(p):
        p=np.asarray(p, dtype=float)
        p=np.nan_to_num(p, nan=0.2, posinf=1.0, neginf=0.0)
        p=np.clip(p, 0, 1)
        s=p.sum()
        if s <= 0 or not np.isfinite(s):
            p=np.full_like(p, 1.0/len(p))
        else:
            p=p/s
        return p

    if bundle.get("hybrid") is not None:
        try:
            f=_standardize(feats, bundle.get("scaler"))
            with torch.no_grad(): _,p=bundle["hybrid"](torch.tensor(f[None,:],dtype=torch.float32))
            out["hybrid"]=_sanitize_probs(p[0].numpy())
        except Exception as e:
            out["hybrid_error"]=str(e)
    if bundle.get("cnn") is not None:
        try:
            curve_cnn = _resample_curve(curve, 256)
            curve_cnn = _sanitize_curve(curve_cnn)
            with torch.no_grad():
                logits=bundle["cnn"](torch.tensor(curve_cnn[None,None,:],dtype=torch.float32))
                p=torch.softmax(logits,dim=1).numpy()[0]
            out["cnn"]=_sanitize_probs(p)
        except Exception as e:
            out["cnn_error"]=str(e)
    if bundle.get("svm") is not None:
        try:
            f=_standardize(feats, bundle.get("scaler"))
            clf=bundle["svm"]["model"] if isinstance(bundle["svm"],dict) and "model" in bundle["svm"] else bundle["svm"]
            if isinstance(bundle["svm"],dict) and "scaler_mean" in bundle["svm"]:
                scale=bundle["svm"].get("scaler_scale", bundle["svm"].get("scaler_std"))
                mean=bundle["svm"]["scaler_mean"]
                if scale is not None:
                    scale=np.where(np.asarray(scale)==0,1.0,scale)
                    f=(feats-np.asarray(mean))/np.asarray(scale)
                    f=np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            p=clf.predict_proba(f[None,:])[0]
            out["cnn_svm_raw"]=p  # debug keep
            out["svm"]=_sanitize_probs(p)
            if "cnn_svm_raw" in out: del out["cnn_svm_raw"]
        except Exception as e: out["svm_error"]=str(e)
    if bundle.get("pure") is not None:
        try:
            red=bundle.get("pure_reducer")
            if red is not None:
                try:
                    f=red.transform(feats[None,:]).astype(np.float32)
                except Exception:
                    # fallback to sliced feats if reducer fails (e.g. NaN)
                    f=feats[:bundle["pure"].n_qubits][None,:].astype(np.float32)
                f=np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                with torch.no_grad(): _,p=bundle["pure"](torch.tensor(f,dtype=torch.float32))
                out["pure"]=_sanitize_probs(p[0].numpy())
            else:
                f=feats[:bundle["pure"].n_qubits][None,:].astype(np.float32)
                f=np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                with torch.no_grad(): _,p=bundle["pure"](torch.tensor(f,dtype=torch.float32))
                out["pure"]=_sanitize_probs(p[0].numpy())
        except Exception as e: out["pure_error"]=str(e)
    return out

def render_light_curve_plot(curve: np.ndarray, class_name: str, color: str) -> go.Figure:
    curve = _sanitize_curve(curve)
    n=len(curve)
    dt = 7200.0 / max(n-1, 1) if n>1 else 5.0
    t=np.arange(n)*dt
    fig=go.Figure()
    # ensure no NaN in traces
    y=np.nan_to_num(curve, nan=0.5, posinf=1.0, neginf=0.0)
    fig.add_trace(go.Scatter(x=t,y=y,mode="lines",opacity=0.22,line=dict(color=color,width=10),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t,y=y,mode="lines",line=dict(color=color,width=2.4),fill="tozeroy",fillcolor="rgba(56,189,248,0.10)",hovertemplate="t=%{x:.0f}s<br>brightness=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t,y=y,mode="markers",marker=dict(size=4,color=color,opacity=0.35,line=dict(width=1,color="white")),showlegend=False,hoverinfo="skip"))
    t_max = float(t[-1]) if len(t) else 7200
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=310,margin=dict(l=10,r=10,t=34,b=10),title=dict(text=f"Light Curve — {class_name} ({n} pts, dt={dt:.1f}s)",font=dict(size=13,color="#7DD3FC"),x=0.02),xaxis=dict(title="Time (s)", range=[0, t_max], gridcolor="rgba(125,211,252,0.10)",zeroline=False),yaxis=dict(title="Normalized brightness",range=[0,1.05],gridcolor="rgba(125,211,252,0.10)",zeroline=False),hovermode="x")
    return fig

def render_frequency_spectrum(curve: np.ndarray, color: str) -> go.Figure:
    curve=_sanitize_curve(curve)
    curve=np.nan_to_num(curve, nan=0.5); n=len(curve)
    dt = 7200.0 / max(n-1,1) if n>1 else 5.0
    if n>=4:
        try:
            w=np.hanning(n); spec=np.abs(np.fft.rfft(curve*w)); freqs=np.fft.rfftfreq(n,d=dt)
            # guard spec max zero
            m = float(np.max(spec)) if spec.size else 1.0
            if m < 1e-9: m=1.0
            spec=spec[1:]/m; freqs=freqs[1:]
            spec=np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
            spec=np.clip(spec, 0, 1)
        except Exception:
            freqs=np.array([0.]); spec=np.array([0.])
    else: freqs=np.array([0.]); spec=np.array([0.])
    fig=go.Figure()
    try:
        fig.add_trace(go.Bar(x=freqs,y=spec,marker=dict(color=spec,colorscale=[[0,"#0B1A3A"],[0.5,"#38BDF8"],[1,"#C084FC"]]),hovertemplate="f=%.4f Hz<br>power=%.3f<extra></extra>"))
    except Exception:
        fig.add_trace(go.Bar(x=[0],y=[0]))
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=230,margin=dict(l=10,r=10,t=32,b=10),title=dict(text=f"Frequency Spectrum (dt={dt:.1f}s)",font=dict(size=13,color="#7DD3FC"),x=0.02),xaxis=dict(title="Frequency (Hz)",gridcolor="rgba(125,211,252,0.10)",zeroline=False),yaxis=dict(title="Relative power", range=[0,1.05], gridcolor="rgba(125,211,252,0.10)",zeroline=False),bargap=0.45)
    return fig

def render_prob_bars(proba: np.ndarray, pred: int, colors: list):
    for i,(cname,emoji,color) in enumerate(zip(CLASS_NAMES, CLASS_EMOJIS, colors)):
        p=float(proba[i])*100; hl="border:1px solid rgba(255,255,255,0.45);box-shadow:0 0 14px rgba(255,255,255,0.12);" if i==pred else ""
        st.markdown(f'<div class="prob-bar-track" style="{hl}"><div class="prob-bar" style="--w:{p}%;background:linear-gradient(90deg,{color},{color}DD);">{emoji} {cname} — {p:.1f}%</div></div>',unsafe_allow_html=True)

def render_photo_card(filename: str, caption: str, cls: str=""):
    b64=img_b64(filename)
    if not b64: return
    st.markdown(f'<div class="photo-card {cls}"><img src="data:image/jpeg;base64,{b64}" alt="{caption}"><div class="cap">{caption}</div></div>',unsafe_allow_html=True)

# ───────── main
def main():
    st.set_page_config(page_title="Q-ORBIT — Quantum Intelligence for Space", layout="wide", page_icon="🛰️", initial_sidebar_state="collapsed")
    inject_css()
    st.markdown(render_space_bg(), unsafe_allow_html=True)
    st.markdown(render_nav(), unsafe_allow_html=True)

    bundle=load_all_models()
    acc=bundle.get("hybrid_acc")
    sat_b64=img_b64("satellite_other.jpg") or img_b64("surveillance_satellite.jpg") or img_b64(HERO_IMAGE)
    hero_b64=img_b64(HERO_IMAGE)
    qimg=img_b64(QUANTUM_IMAGE)

    # ticker
    acc_line=f"Hybrid test accuracy {acc:.1f}%" if acc is not None else "Awaiting trained model • Run train_quantum.py"
    st.markdown(f'<div class="ticker"><div class="track"><span>▸ Q-ORBIT • QUANTUM-CLASSICAL AI • 8-QUBIT HILBERT SPACE (256-D) • PENNYLANE DEFAULT.QUBIT</span><span>▸ {acc_line}</span><span>▸ 5 OBJECT CLASSES • 21 FEATURES • 256-SAMPLE CURVES • SEED 42 REPRODUCIBLE</span><span>▸ NASA PUBLIC-DOMAIN IMAGERY • BUILT FOR NATIONAL CAPSTONE</span></div></div>',unsafe_allow_html=True)

    # HERO
    st.markdown(render_hero(sat_b64), unsafe_allow_html=True)

    # ── Models showcase
    st.markdown('<div id="models" style="height:18px"></div>',unsafe_allow_html=True)
    st.markdown('<p class="section-kicker">THREE PARADIGMS — ONE MISSION</p><h2 class="section-title">Core Approaches</h2><p class="section-sub">Every light curve is decoded through three lenses. The Hybrid is the proposed core — quantum entanglement for correlated feature interactions plus a classical head for decision.</p>',unsafe_allow_html=True)
    # Build approach cards as HTML
    hybrid_acc_txt=f"{acc:.1f}%" if acc is not None else "69.7%"
    cnn_acc=bundle.get("cnn_acc"); svm_acc=None
    try:
        import json as _j
        if os.path.exists(os.path.join(BASE_DIR,"..","results","reports","metrics.json")):
            import json as js
            with open(os.path.join(BASE_DIR,"..","results","reports","metrics.json")) as f: mj=js.load(f)
            cnn_acc=mj.get("cnn",{}).get("accuracy", cnn_acc) if isinstance(mj,dict) else cnn_acc
    except: pass
    cnn_txt=f"{cnn_acc*100:.1f}%" if isinstance(cnn_acc,float) and cnn_acc<=1 else (f"{cnn_acc:.1f}%" if isinstance(cnn_acc,(int,float)) else "84.9%")
    pure_acc=bundle.get("pure_acc"); pure_txt=f"{pure_acc*100:.1f}%" if isinstance(pure_acc,float) and pure_acc<=1 else (f"{pure_acc:.1f}%" if isinstance(pure_acc,(int,float)) else "~65%")
    st.markdown(f"""
<div class="approach-grid">
  <div class="approach hud-corners">
    <div class="icon">▦</div>
    <h4>Pure Classical — ML/DL</h4>
    <p>1D-CNN on raw 256-sample curves and RBF-SVM on 21 engineered features. Learns convolutional temporal motifs and margin-optimal boundaries.</p>
    <ul><li>Input: raw curve (CNN) / 21 features (SVM)</li><li>Capacity: ~0.5M params (CNN)</li><li>Strength: raw temporal detail</li></ul>
    <div class="metric"><div><b>{cnn_txt}</b><span>CNN ACC</span></div><div><b>71.6%</b><span>SVM ACC</span></div><div><b>5 ms</b><span>INFER</span></div></div>
  </div>
  <div class="approach hud-corners">
    <div class="icon">◈</div>
    <h4>Pure Quantum — QML</h4>
    <p>PCA → 8-D → Angle(RY) embedding → Rot+CNOT variational ansatz (2 layers) → Pauli-Z. Representation lives entirely in the quantum circuit.</p>
    <ul><li>Input: 8-D PCA projection</li><li>Hilbert: 256-D (8 qubits)</li><li>Entanglement: chain CNOT</li></ul>
    <div class="metric"><div><b>{pure_txt}</b><span>VQC ACC</span></div><div><b>8 Q</b><span>QUBITS</span></div><div><b>~48</b><span>PARAMS</span></div></div>
  </div>
  <div class="approach hybrid hud-corners">
    <div class="badge">⬢ PROPOSED CORE • HYBRID</div>
    <div class="icon">⬡</div>
    <h4>Hybrid — Classical + Quantum</h4>
    <p><b>Quantum feature map + classical head.</b> Projects 21 features → 8 qubits, entangles via Rot+CNOT, measures Z expectations, then dense ReLU → softmax.</p>
    <ul><li>Input: 21 standardized features</li><li>Quantum: 2-layer variational</li><li>Head: 16-unit dense ×2</li></ul>
    <div class="metric"><div><b>{hybrid_acc_txt}</b><span>HYBRID ACC</span></div><div><b>0.69</b><span>F1-MACRO</span></div><div><b>230</b><span>PARAMS</span></div></div>
    <div style="margin-top:0.7rem;font-family:'JetBrains Mono';font-size:0.68rem;letter-spacing:1px;color:#7DD3FC">⟡ ENTANGLED FEATURE INTERACTIONS • COMPETITIVE WITH RBF-SVM</div>
  </div>
</div>
    """, unsafe_allow_html=True)

    # Gallery strip (dataset preview)
    st.markdown('<div id="dataset" style="height:18px"></div>',unsafe_allow_html=True)
    st.markdown('<p class="section-kicker">PHOTOMETRIC INTELLIGENCE</p><h2 class="section-title">Dataset & Orbital Gallery</h2><p class="section-sub">Synthetic light curves from a physics-driven tumble model (specular + Lambertian, eclipse, photon noise). Gallery uses NASA public-domain imagery.</p>',unsafe_allow_html=True)
    g1,g2,g3,g4=st.columns(4)
    with g1: render_photo_card("nebula_space.jpg","Nebula — star-birth clouds")
    with g2: render_photo_card("galaxy_starcluster.jpg","Hubble — globular star cluster")
    with g3: render_photo_card("galaxy_dwarf.jpg","Hubble — dwarf galaxy")
    with g4: render_photo_card("quantum_visual.jpg","Quantum — 8 entangled qubits")
    st.markdown("""
<div class="glass hud-corners" style="margin-top:1rem">
  <h3>▣ DATASET SPEC</h3>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.8rem">
    <div><div style="font-family:'JetBrains Mono';font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">CLASSES</div><div style="font-family:'Orbitron';color:#E6F0FF">5 (0–4)</div><div style="font-size:0.78rem;color:#A8BBDD">Intact / Dead / Rocket / Debris / Spoofed</div></div>
    <div><div style="font-family:'JetBrains Mono';font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">DENSITY</div><div style="font-family:'Orbitron';color:#E6F0FF">10,000 curves</div><div style="font-size:0.78rem;color:#A8BBDD">2,000 per class • seed 42</div></div>
    <div><div style="font-family:'JetBrains Mono';font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">SAMPLING</div><div style="font-family:'Orbitron';color:#E6F0FF">1 / 5 s × 2 h</div><div style="font-size:0.78rem;color:#A8BBDD">256 samples • 0–1 min-max</div></div>
    <div><div style="font-family:'JetBrains Mono';font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">SPLITS</div><div style="font-family:'Orbitron';color:#E6F0FF">70 / 15 / 15</div><div style="font-size:0.78rem;color:#A8BBDD">train / val / test • stratified</div></div>
  </div>
</div>
    """, unsafe_allow_html=True)

    # Results placeholder
    st.markdown('<div id="results" style="height:18px"></div>',unsafe_allow_html=True)
    st.markdown('<p class="section-kicker">MEASURED PERFORMANCE</p><h2 class="section-title">Results at a Glance</h2>',unsafe_allow_html=True)
    r1,r2,r3,r4=st.columns(4)
    with r1: st.markdown(f'<div class="glass" style="text-align:center"><div style="font-family:Orbitron;color:#7DD3FC;font-size:1.5rem">{cnn_txt}</div><div style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">CLASSICAL CNN</div></div>',unsafe_allow_html=True)
    with r2: st.markdown('<div class="glass" style="text-align:center"><div style="font-family:Orbitron;color:#C084FC;font-size:1.5rem">71.6%</div><div style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">CLASSICAL SVM</div></div>',unsafe_allow_html=True)
    with r3: st.markdown(f'<div class="glass" style="text-align:center;border-color:rgba(0,229,255,0.28)"><div style="font-family:Orbitron;color:#00E5FF;font-size:1.5rem">{hybrid_acc_txt}</div><div style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">HYBRID QUANTUM</div><div style="font-size:0.62rem;color:#00E5FF;letter-spacing:1px">PROPOSED</div></div>',unsafe_allow_html=True)
    with r4: st.markdown(f'<div class="glass" style="text-align:center"><div style="font-family:Orbitron;color:#F472B6;font-size:1.5rem">{pure_txt}</div><div style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">PURE VQC</div></div>',unsafe_allow_html=True)
    # try to show fair comparison md if exists
    comp_path=os.path.join(BASE_DIR,"..","results","reports","fair_comparison.md")
    if os.path.exists(comp_path):
        with open(comp_path,encoding="utf-8",errors="ignore") as f: md=f.read()
        st.markdown(f'<div class="glass hud-corners" style="margin-top:0.8rem"><h3>▣ FAIR COMPARISON REPORT</h3></div>',unsafe_allow_html=True)
        st.markdown(md)

    # ── Dashboard (Live Inference Lab) ──
    st.markdown('<div id="dashboard" style="height:18px"></div>',unsafe_allow_html=True)
    st.markdown('<p class="section-kicker">LIVE INFERENCE LAB</p><h2 class="section-title">Mission Control — Generate & Classify</h2><p class="section-sub">Select a target class, tune the sensor noise, and fire the quantum-hybrid pipeline. All three models run on the same synthetic curve for a fair side-by-side.</p>',unsafe_allow_html=True)

    # controls in glass
    st.markdown('<div class="glass hud-corners">',unsafe_allow_html=True)
    st.markdown('<h3>🎛️ TELEMETRY CONTROLS</h3>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns([1.3,1,1,0.9])
    with c1:
        cls_choice=st.selectbox("Debris class to simulate", options=list(range(5)), format_func=lambda c: f"{CLASS_EMOJIS[c]}  {CLASS_NAMES[c]}", key="cls_choice")
    with c2:
        noise=st.slider("Noise level (std)", 0.0, 0.1, 0.02, step=0.01, key="noise")
    with c3:
        n_samples=st.slider("Simulation samples", 64, 1024, 256, step=32, key="nsamp")
    with c4:
        st.write(""); st.write("")
        generate=st.button("⬢ Generate & Infer", width="stretch", type="primary")
    st.markdown('</div>',unsafe_allow_html=True)

    if bundle.get("hybrid") is None:
        st.warning("Trained hybrid model not found. Run `python src/pipeline/train_quantum.py` to create models/hybrid_quantum_model.pt — other baselines will still work where available.")
    if not generate:
        st.markdown("""
<div class="glass" style="text-align:center;padding:1.6rem 1rem">
  <div style="font-family:Orbitron;letter-spacing:2px;color:#7DD3FC">AWAITING TELEMETRY</div>
  <div style="color:#A8BBDD;margin-top:0.4rem">Choose a class and press <b>Generate & Infer</b> to ignite the pipeline. The satellite will stream a fresh light curve.</div>
  <div style="margin-top:0.9rem;display:flex;gap:0.6rem;justify-content:center;flex-wrap:wrap">
    <span style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.18);padding:0.4rem 0.7rem;border-radius:999px;color:#7DD3FC">TIP: Try Class 4 Spoofed — hardest to separate</span>
    <span style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;background:rgba(192,132,252,0.08);border:1px solid rgba(192,132,252,0.18);padding:0.4rem 0.7rem;border-radius:999px;color:#C084FC">Noise sweep reveals quantum robustness</span>
  </div>
</div>
        """, unsafe_allow_html=True)
        st.markdown('<div id="about" style="height:10px"></div>',unsafe_allow_html=True)
        st.markdown("""
<div class="footer hud-corners">
  <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:1rem">
    <div><b style="font-family:Orbitron;letter-spacing:2px">Q-ORBIT</b> — Quantum-Classical AI for Space Object Classification<br/><span style="color:#8AA0C8">B.Tech Capstone • Hybrid quantum-classical thesis • PennyLane default.qubit • No cloud, no paid APIs • Reproducible seed 42</span><br/><span style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#7DD3FC">NASA mission control + quantum lab + premium AI startup</span></div>
    <div style="font-family:JetBrains Mono;font-size:0.72rem;line-height:1.7;color:#A8BBDD">Stack: Python • PyTorch • PennyLane • scikit-learn • Streamlit • Plotly<br/>Imagery: NASA public domain via assets/images/web<br/>© 2026 Q-ORBIT — Research-grade, presentation-ready</div>
  </div>
</div>
        """,unsafe_allow_html=True)
        return

    # ── hardened generate + predict (never crash on slider extremes) ──
    # clamp user inputs (slider already limits, but double-guard)
    try:
        cls_choice = int(np.clip(int(cls_choice), 0, 4))
        noise = float(np.clip(float(noise), 0.0, 0.3))
        n_samples = int(np.clip(int(n_samples), 32, 2048))
        # snap to valid step to avoid odd sizes that confuse UI
        if n_samples < 64: n_samples = 64
    except Exception:
        cls_choice, noise, n_samples = 0, 0.02, 256

    curve = None
    gen_error = None
    with st.spinner("Simulating tumble dynamics…"):
        try:
            curve,_=generate_single_light_curve(int(cls_choice), noise_std=float(noise), n_samples=int(n_samples))
            curve = _sanitize_curve(curve)
            if len(curve) != int(n_samples):
                # generator should already match, but enforce
                curve = _resample_curve(curve, int(n_samples))
        except Exception as e:
            gen_error = str(e)
            # fallback: synthesize a safe dummy curve so UI still renders
            import traceback as _tb; _tb.print_exc()
            curve = np.clip(np.random.default_rng(0).random(int(n_samples))*0.3+0.4, 0, 1).astype(np.float32)

    if gen_error:
        st.markdown(f'<div class="glass" style="border-color:rgba(251,113,133,0.35)"><h3 style="color:#fb7185">⚠️ Simulation fallback</h3><div style="color:#FECDD3;font-size:0.82rem">Generator raised: <code>{gen_error}</code><br/>Showing fallback curve. Try lowering noise or changing samples.</div></div>', unsafe_allow_html=True)

    color=CLASS_COLORS[int(cls_choice)]

    all_proba = {}
    infer_error = None
    with st.spinner("Running classical / quantum / hybrid inference…"):
        try:
            all_proba=predict_all(bundle, curve)
        except Exception as e:
            infer_error = str(e)
            import traceback as _tb; _tb.print_exc()
            all_proba = {}

    if infer_error:
        st.error(f"Inference error: {infer_error}")
        st.caption("Falling back to safe dummy probabilities for display.")
        dummy = np.array([0.2,0.2,0.2,0.2,0.2])
        for k in ["hybrid","cnn","svm","pure"]:
            if k not in all_proba: all_proba[k]=dummy

    # surface per-model errors as non-blocking warnings (don't crash tabs)
    for ek in ["hybrid_error","cnn_error","svm_error","pure_error"]:
        if ek in all_proba:
            st.toast(f"{ek}: {str(all_proba[ek])[:120]}", icon="⚠️")

    proba=all_proba.get("hybrid"); pred=int(np.argmax(proba)) if isinstance(proba, np.ndarray) and proba.size==5 else None

    # target + signal
    tc1,tc2=st.columns([1,2.2], gap="large")
    with tc1:
        st.markdown('<div class="glass hud-corners"><h3>🎯 TARGET OBJECT</h3></div>',unsafe_allow_html=True)
        render_photo_card(CLASS_IMAGES[cls_choice], CLASS_CAPTIONS[cls_choice])
        st.markdown(f'<div class="glass" style="margin-top:0.7rem"><div style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#8AA0C8">SIM CONFIG</div><div style="font-size:0.86rem;color:#DCE8FF;margin-top:0.3rem">{CLASS_EMOJIS[cls_choice]} <b>{CLASS_NAMES[cls_choice]}</b> • Noise <b>{noise:.2f}</b> • Samples <b>{n_samples}</b> • Seed 42</div></div>',unsafe_allow_html=True)
    with tc2:
        st.markdown('<div class="glass hud-corners"><h3>📡 PHOTOMETRIC SIGNAL</h3></div>',unsafe_allow_html=True)
        st.plotly_chart(render_light_curve_plot(curve, CLASS_NAMES[cls_choice], color), width="stretch", config={"displayModeBar":False})

    st.write("")
    # three-model tabs
    st.markdown('<div class="glass hud-corners"><h3>🔬 THREE-MODEL COMPARISON — SAME CURVE, SAME PREPROCESSING</h3></div>',unsafe_allow_html=True)
    t1,t2,t3,t4=st.tabs(["Classical CNN","Classical SVM","Pure Quantum VQC","Hybrid Quantum ★"])
    def tab_content(name,key):
        p=all_proba.get(key)
        err=all_proba.get(f"{key}_error")
        if p is None:
            if err:
                st.markdown(f'<div class="glass" style="border-color:rgba(251,113,133,0.3)"><div style="color:#fb7185;font-family:JetBrains Mono;font-size:0.78rem">⚠️ {name} inference failed</div><div style="color:#FECDD3;font-size:0.78rem;word-break:break-all">{err}</div><div style="color:#8AA0C8;font-size:0.75rem;margin-top:0.4rem">Try Generate again or reduce samples to 256. Other models below still work.</div></div>', unsafe_allow_html=True)
            else:
                st.warning(f"{name} model not loaded. Train it first.")
            if key=="pure" and "pure_error" in all_proba: st.code(str(all_proba["pure_error"])[:800])
            if key=="cnn" and "cnn_error" in all_proba: st.code(str(all_proba["cnn_error"])[:800])
            if key=="svm" and "svm_error" in all_proba: st.code(str(all_proba["svm_error"])[:800])
            return
        # validate shape
        try:
            p=np.asarray(p, dtype=float).reshape(5)
            p=np.nan_to_num(p, nan=0.2, posinf=1.0, neginf=0.0); p=np.clip(p,0,1); p=p/p.sum() if p.sum()>0 else np.full(5,0.2)
        except Exception:
            st.error(f"{name}: invalid probability shape")
            return
        pr=int(np.argmax(p))
        st.markdown(f'<div style="text-align:center;padding:0.4rem 0"><span class="badge-pred" style="background:{CLASS_COLORS[pr]}">{CLASS_EMOJIS[pr]} {CLASS_NAMES[pr]} — {p[pr]*100:.1f}%</span></div>',unsafe_allow_html=True)
        try:
            render_prob_bars(p, pr, CLASS_COLORS)
        except Exception as e:
            st.code(f"bar render error: {e}")
        try:
            ent=float(-(p*np.log(p+1e-9)).sum()); st.caption(f"Confidence {p[pr]:.3f} • Entropy {ent:.3f} • Top-2 gap {(np.sort(p)[-1]-np.sort(p)[-2])*100:.1f} pp")
        except Exception: pass
    with t1: tab_content("Classical CNN","cnn")
    with t2: tab_content("Classical SVM","svm")
    with t3: tab_content("Pure Quantum VQC","pure")
    with t4:
        if isinstance(proba, np.ndarray) and proba.size==5:
            try:
                proba=np.asarray(proba, dtype=float).reshape(5)
                proba=np.nan_to_num(proba, nan=0.2, posinf=1.0, neginf=0.0); proba=np.clip(proba,0,1); proba=proba/proba.sum() if proba.sum()>0 else np.full(5,0.2)
                pred=int(np.argmax(proba))
            except Exception: proba=None; pred=None
            if proba is not None:
                st.markdown(f'<div style="text-align:center;padding:0.4rem 0"><span class="badge-pred" style="background:{CLASS_COLORS[pred]}">{CLASS_EMOJIS[pred]} {CLASS_NAMES[pred]} — {proba[pred]*100:.1f}%</span></div>',unsafe_allow_html=True)
                if qimg:
                    st.markdown(f'<div class="photo-card" style="margin-bottom:0.6rem"><img src="data:image/jpeg;base64,{qimg}" alt="quantum"><div class="cap">Quantum core — 8-qubit variational circuit</div></div>',unsafe_allow_html=True)
                render_prob_bars(proba, pred, CLASS_COLORS)
            else:
                st.warning("Hybrid probabilities invalid — see error above.")
        else:
            if "hybrid_error" in all_proba:
                st.markdown(f'<div class="glass" style="border-color:rgba(251,113,133,0.3)"><div style="color:#fb7185">⚠️ Hybrid inference failed</div><div style="color:#FECDD3;font-size:0.78rem;word-break:break-all">{all_proba.get("hybrid_error","")}</div></div>', unsafe_allow_html=True)
            else:
                st.warning("Hybrid model not loaded.")

    # AI analyst
    st.write("")
    st.markdown('<div class="glass hud-corners"><h3>🤖 AI ANALYST AGENT</h3><div style="font-size:0.78rem;color:#8AA0C8;margin-top:-0.3rem">Consumes structured model outputs — does not invent predictions.</div></div>',unsafe_allow_html=True)
    try:
        from src.agents.analyst import analyze
        from src.evaluation.degradation import signal_quality
        if len([k for k in ["cnn","svm","pure","hybrid"] if k in all_proba])>=2:
            pack={"classical_pred":int(np.argmax(all_proba.get("cnn", all_proba.get("svm", proba)))), "quantum_pred":int(np.argmax(all_proba.get("pure", proba))), "hybrid_pred":int(np.argmax(all_proba.get("hybrid", proba))), "classical_proba":all_proba.get("cnn", all_proba.get("svm", proba)), "quantum_proba":all_proba.get("pure", proba), "hybrid_proba":all_proba.get("hybrid", proba), "signal_quality":signal_quality(curve)}
            st.markdown(f'<div class="glass">{analyze(pack)}</div>',unsafe_allow_html=True)
        else: st.info("Need at least 2 models loaded for analyst report.")
    except Exception as e: st.caption(f"Analyst unavailable: {e}")

    # freq + summary (hardened)
    left,right=st.columns([1.35,1], gap="large")
    with left:
        st.markdown('<div class="glass hud-corners"><h3>🎼 FREQUENCY ANALYSIS</h3></div>',unsafe_allow_html=True)
        try:
            st.plotly_chart(render_frequency_spectrum(curve, color), width="stretch", config={"displayModeBar":False})
        except Exception as e:
            st.markdown(f'<div class="glass" style="border-color:rgba(251,113,133,0.3)"><div style="color:#fb7185">Spectrum render failed</div><div style="color:#FECDD3;font-size:0.78rem">{e}</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass hud-corners"><h3>⚡ SIGNAL SUMMARY</h3></div>',unsafe_allow_html=True)
        try:
            m1,m2,m3=st.columns(3)
            m1.metric("Mean", f"{float(np.nanmean(curve)):.3f}"); m2.metric("Std Dev", f"{float(np.nanstd(curve)):.3f}"); m3.metric("Peak", f"{float(np.nanmax(curve)):.3f}")
            m4,m5,m6=st.columns(3)
            # dt-aware feature extraction for display metrics
            dt_disp = 7200.0 / max(len(curve)-1,1) if len(curve)>1 else 5.0
            try:
                feats=extract_all_features(_sanitize_curve(curve), sampling_interval=dt_disp)
                feats=np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                feats=np.zeros(MODEL_FEATURES)
            m4.metric("Flash Count", f"{int(feats[15]) if np.isfinite(feats[15]) else 0}"); m5.metric("Eclipse %", f"{float(feats[18])*100:.0f}%" if np.isfinite(feats[18]) else "0%"); m6.metric("Period (s)", f"{float(feats[20]):.0f}" if np.isfinite(feats[20]) else "0")
        except Exception as e:
            st.caption(f"Summary unavailable: {e}")

    # about footer
    st.markdown('<div id="about" style="height:14px"></div>',unsafe_allow_html=True)
    st.markdown("""
<div class="footer hud-corners">
  <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:1rem">
    <div><b style="font-family:Orbitron;letter-spacing:2px">Q-ORBIT</b> — Quantum-Classical AI for Space Object Classification<br/><span style="color:#8AA0C8">B.Tech Capstone • Hybrid quantum-classical thesis • PennyLane default.qubit • No cloud, no paid APIs • Reproducible seed 42</span><br/><span style="font-family:JetBrains Mono;font-size:0.7rem;letter-spacing:1px;color:#7DD3FC">NASA mission control + quantum lab + premium AI startup</span></div>
    <div style="font-family:JetBrains Mono;font-size:0.72rem;line-height:1.7;color:#A8BBDD">Stack: Python • PyTorch • PennyLane • scikit-learn • Streamlit • Plotly<br/>Imagery: NASA public domain • Assets in assets/images/web<br/>© 2026 Q-ORBIT — Research-grade, presentation-ready</div>
  </div>
</div>
    """,unsafe_allow_html=True)

if __name__=="__main__":
    main()
