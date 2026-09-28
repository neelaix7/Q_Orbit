"""CLI: simulate many satellite passes -> data/orbits/.

Usage:
    python -m src.pipeline.simulate_passes --n-passes 200 --out data/orbits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ..orbit.geometry import (GroundStation, KeplerOrbit, compute_pass,
                                  extract_pass_segments)
    from ..orbit.link import LinkModel
except (ImportError, ValueError):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.orbit.geometry import (GroundStation, KeplerOrbit, compute_pass,
                                   extract_pass_segments)
    from src.orbit.link import LinkModel

# Two ISS-like LEO orbits + one mid-LEO, tweaked to give varied passes.
# (No GEO: a geostationary satellite is permanently visible — not a "pass".)
SAT_TEMPLATES = [
    {"name": "ISS-LEO", "a_km": 6781.0, "e": 0.0009, "i": 51.6},
    {"name": "QKD-LEO", "a_km": 6971.0, "e": 0.001, "i": 97.3},
    {"name": "STARLINK-LEO", "a_km": 6685.0, "e": 0.0002, "i": 53.0},
]
GS = GroundStation(lat_deg=13.0827, lon_deg=80.2707, alt_m=10.0, name="Chennai-ISRO")


def _subsample_segment(df: pd.DataFrame, max_samples: int = 30) -> pd.DataFrame:
    """Cap a pass segment to `max_samples` evenly-spaced rows (for runtime)."""
    if len(df) <= max_samples:
        return df
    idx = np.round(np.linspace(0, len(df) - 1, max_samples)).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def build_orbit(template: dict, raan_deg: float, argp_deg: float,
                m0_deg: float, epoch_mjd: float) -> KeplerOrbit:
    return KeplerOrbit(
        a=template["a_km"] * 1e3, e=template["e"], i_deg=template["i"],
        raan_deg=raan_deg, argp_deg=argp_deg, m0_deg=m0_deg, epoch_mjd=epoch_mjd,
        name=template["name"],
    )


def generate_pass_records(n_passes: int, rng: np.random.Generator) -> list[dict]:
    """Generate `n_passes` valid passes (elevation >= 15 deg for >= 6 samples)."""
    link = LinkModel()
    records = []
    mjd_anchor = 59000.0
    progress = tqdm(total=n_passes, desc="simulating passes")
    attempts = 0
    while len(records) < n_passes and attempts < n_passes * 50:
        attempts += 1
        template = SAT_TEMPLATES[int(rng.integers(0, len(SAT_TEMPLATES)))]
        raan = float(rng.uniform(0, 360))
        argp = float(rng.uniform(0, 360))
        m0 = float(rng.uniform(0, 360))
        start = mjd_anchor + float(rng.uniform(0, 1.0))
        # LEO satellites pass the station every ~90-100 min; propagate just
        # over one orbital period so valid passes are found fast.
        orbit = build_orbit(template, raan, argp, m0, start)
        period_days = orbit.period / 86400.0
        df = compute_pass(orbit, GS, start, start + period_days, step_s=15.0,
                          min_elevation=15.0)
        segs = extract_pass_segments(df, min_elevation=15.0, min_samples=6)
        if not segs:
            continue
        seg = max(segs, key=len)
        seg = _subsample_segment(seg, max_samples=30)
        aug = link.pass_summary(seg, GS.lat_deg, GS.lon_deg)
        el = aug["elevation_deg"].to_numpy()
        q = aug["link_quality"].to_numpy()
        dt = float(np.diff(seg["mjd"]).mean() * 86400.0) if len(seg) > 1 else 15.0
        sat_idx = int(next(i for i, s in enumerate(SAT_TEMPLATES)
                           if s["name"] == template["name"]))
        records.append({
            "pass_id": f"P{len(records):04d}",
            "satellite": template["name"],
            "sat_idx": sat_idx,
            "mjd_start": float(seg["mjd"].iloc[0]),
            "max_elev": float(el.max()),
            "mean_elev": float(el.mean()),
            "min_elev": float(el.min()),
            "pass_duration_s": float(len(seg) * dt),
            "mean_link_q": float(q.mean()),
            "n_samples": int(len(seg)),
            "template": dict(template),
            "table": aug,
        })
        progress.update(1)
    progress.close()
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate simulated satellite passes")
    ap.add_argument("--n-passes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/orbits")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    records = generate_pass_records(args.n_passes, rng)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for rec in records:
        fname = f"{rec['pass_id']}.csv"
        rec["table"].to_csv(out_dir / fname, index=False)
        meta_rows.append({k: rec[k] for k in rec if k not in ("table", "template")})
    pd.DataFrame(meta_rows).to_csv(out_dir / "passes_meta.csv", index=False)
    np.save(out_dir / "passes_records.npy", records, allow_pickle=True)
    print(f"Wrote {len(records)} passes to {out_dir}")


if __name__ == "__main__":
    main()
