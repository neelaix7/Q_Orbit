# UI Redesign Plan — Reference-Inspired Q-ORBIT (I1–I6)

## Inspection Summary
**Current Q-ORBIT UI** (app.py 809 lines, Streamlit + Plotly): deep dark #020512, 4 nebula blobs purple/blue, 160 stars, 3 shooting, small blackhole pulse 560px cyan/purple 0.18 opacity, earth-glow bottom, vignette, sticky glass nav (Home/Models/Dataset/Results/Dashboard/About + Launch App CTA), hero split 1.15/0.95 with Q-ORBIT gradient title, pills CLASSICAL vs QUANTUM vs HYBRID, satellite CSS (body+panels+3 orbits), glass cards 18px, thin cyan borders, hover lift.

**Reference Images:**
- **I1 (Gargantua):** Cinematic, huge off-center black hole (~85% viewport), razor-bright cream-orange accretion disk with light bending, lensing halo, true black center, no cyan — warm, high contrast, deep black. Our blackhole too small/subtle/cool.
- **I2 (Lab):** Pristine futuristic circular hall, overhead soft ring light, semi-transparent holographic Earth globe (wireframe + continents, cyan glow) center stage on raised dais, clean light grays, subtle Russian neon signage, depth via reflections. Our lab is dark only, no hologram.
- **I3 (Quantum HW):** Golden cryostat disc, vertical gold wiring loom, central blue plasma qubit column, warm metal vs cool core — premium hardware. Our quantum viz is flat grid.
- **I4 (Debris):** Earth curvature horizon, dozens of tumbling debris/satellites at multiple depths, motion trails, lens flare on panels, extreme depth, scattered. Our hero shows single satellite only.
- **I5 (Re-entry):** Capsule with intense orange plasma tail, Earth's atmosphere limb (thin cyan line), high velocity feel. Missing.
- **I6 (Starfield):** Ultra-dense monochrome starfield (~800 stars), random magnitudes/size, nebular dust, horizontal sensor ruler. Our stars 160, sparse.

## Gap Analysis
| Aspect | Current | Reference Target | Delta |
|---|---|---|---|
| Overall layout | Single column stack, tight | Expansive cinematic whitespace, large hero occupying 70% viewport | Increase hero height/padding, more breathing room |
| Space background | Purple-heavy nebula 55% opacity | Deepest black #01030A + subtle cool nebula 35% + warm Gargantua dominant | Reduce purple, add warm orange rim, darken base |
| Black hole | 560px, cyan/purple, 0.18 | 85vw, cream-orange biconvex disk, offset top, inner shadow | Scale 2.2×, warm gradient, double halo |
| Satellite/3D | CSS block, 3 thin rings | Debris field + realistic tumbling objects + re-entry plasma hint | Add 12 drifting debris dots + Earth limb image behind |
| Earth visual | Simple radial glow bottom | Horizon line with atmosphere scatter (thin cyan 2px) + cloud texture | Add Earth horizon arc |
| Holographic Earth | None | I2 wireframe globe behind satellite | Add CSS globe wireframe with continents |
| Quantum hardware | Grid of 8 pills | Gold disc with vertical wires descending to qubit | Add gold chasis styling |
| Typography | Orbitron 2.6-4rem Q-ORBIT | Larger hero headline 3.2-4.6rem, tighter Q-ORBIT kicker | Increase headline hierarchy |
| Glassmorphism | 18px blur 18px, subtle | More translucent (0.38), deeper blur 22px, thinner 0.8px border | Refine |
| Cards | 3 approach cards compact | Larger premium 22px radius, more padding 1.4rem, soft larger shadow | Increase spacing |
| Dashboard metrics | Hero stats 5 pills | 6 metric cards clean grid (10000+, 5, 3, 15+, 256, 8) | Add |
| Navigation | 6 links | 9 links (Home/Models/Dataset/Results/Dashboard/AI Analyst/Quantum Lab/Reports/About) | Add 3 |
| Charts | Dark, colored bars | Thinner lines, more whitespace | Tune |

## Reusable Components (Keep)
- Frozen split logic, predict_all, bundle loading
- Plotly bar/radar/line/heatmap generators
- Prob bars, photo cards, workflow/hybrid/quantum/conclusion sections
- Robustness JSON binding, flex grid system
- Glass/hud-corners, nav CTA pill style

## To Redesign (High Priority)
1. **space-bg** → Deepest black + Gargantua warm accretion (I1) + debris scatter (I4) + Earth horizon (I4/I5) + holographic globe (I2) + gold quantum loom (I3) hints. Stars → 240.
2. **Hero** → Left: kicker (Q-ORBIT small) → hero-sub → large headline "Classifying Space Objects with Quantum Intelligence" → lede → statement pill → three-vs pills → question → dual CTA [Run Analysis][Explore Models] → metric grid 6 cards (10k+,5,3,15+,256,8). Right: enlarged sat-stage 520px, Earth horizon image behind, debris field particles, holographic Earth wireframe behind satellite, enhanced plasma thrust tail.
3. **Navigation** → 9 items, thinner typography, Launch App pill with orange-cyan gradient.
4. **Three Approaches header** → "THREE APPROACHES. ONE GOAL." subtitle: "Two baselines. One proposed hybrid. All measured on same 1500 test."
5. **Quantum Section** → Gold cryostat styling for circuit container, blue core pulse.
6. **Overall** → Increase max-width 1280→1360, more vertical rhythm, cinematic depth, premium feel, fast (no WebGL).

## To Delete
- Duplicate web/ image folder reference, redundant ticker spans, old hero "Mission Control" phrasing.

## Missing Components (Add)
- 6-card dashboard metrics row directly under hero
- Holographic Earth globe element (pure CSS)
- Debris field particle layer (12 small tumbling rects with trail)
- Earth horizon arc (CSS radial)
- Quantum hardware gold ring visual

## Implementation Order (Critical Rule: No backend rewrite)
1. Backup app.py → app_backup.py (done)
2. Edit inject_css: new warm Gargantua + dense stars + horizon
3. Edit render_space_bg: 240 stars + Gargantua + debris + hologram
4. Edit render_nav: 9 links
5. Edit render_hero: new hierarchy + 6-metric grid + enlarged visual
6. Add dashboard metrics row function
7. Verify predict_all preserved, charts bound to real JSON, responsiveness via @media, performance via will-change, particle count capped.
