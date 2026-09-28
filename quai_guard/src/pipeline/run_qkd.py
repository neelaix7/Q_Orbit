"""CLI: run QKD on simulated passes (clean or with Eve) -> data/qkd/.

Usage:
    python -m src.pipeline.run_qkd --eve none --out data/qkd
    python -m src.pipeline.run_qkd --eve intercept_resend --strength 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ..ai.link_predictor import link_features_from_pass
    from ..orbit.geometry import GroundStation
    from ..orbit.link import LinkModel
    from ..quantum.attacks import EveAttack, run_attacked_sift
    from ..quantum.bb84 import run_bb84_round
    from ..quantum.params import QKDParams
except (ImportError, ValueError):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.ai.link_predictor import link_features_from_pass
    from src.orbit.geometry import GroundStation
    from src.orbit.link import LinkModel
    from src.quantum.attacks import EveAttack, run_attacked_sift
    from src.quantum.bb84 import run_bb84_round
    from src.quantum.params import QKDParams

GS = GroundStation(lat_deg=13.0827, lon_deg=80.2707, alt_m=10.0, name="Chennai-ISRO")


def qkd_over_pass(pass_df: pd.DataFrame, config: QKDParams,
                  eve: EveAttack, rng: np.random.Generator,
                  n_photons: int = 96, seed_sim: int | None = None) -> list[dict]:
    """Run QKD on every sample of a pass; returns per-sample result dicts."""
    link = LinkModel()
    rows = []
    n = len(pass_df)
    for i, (_, row) in enumerate(pass_df.iterrows()):
        channel = link.channel_noise(row["elevation_deg"], row["mjd"],
                                     GS.lat_deg, GS.lon_deg, rng)
        round_ = run_bb84_round(n_photons, channel, config, rng, seed_sim=seed_sim)
        round_.meta["laser_intensity"] = config.laser_intensity
        frac_t = i / max(n - 1, 1)
        run_attacked_sift(round_, eve, rng, frac_t=frac_t)
        rows.append({
            "pass_id": row.get("pass_id", ""),
            "t_sec": float(row["t_sec"]),
            "mjd": float(row["mjd"]),
            "elevation_deg": float(row["elevation_deg"]),
            "link_quality": float(channel["link_quality"]),
            "qber": float(round_.qber),
            "secure_bps": float(round_.secure_key_bps),
            "raw_bps": float(round_.raw_key_bps),
            "n_sifted": round_.n_sifted,
            "n_errors": round_.n_errors,
            "secure_key_hex": round_.secure_key_hex,
            "secure_bits": round_.secure_key_bits,
            "attacked": int(round_.attacked),
            "attack_name": round_.attack_name,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run QKD over simulated passes")
    ap.add_argument("--orbits", default="data/orbits")
    ap.add_argument("--out", default="data/qkd")
    ap.add_argument("--eve", default="none",
                    choices=["none", "intercept_resend", "pns", "intermittent"])
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--n-pass-limit", type=int, default=0,
                    help="0 = use all passes; else cap the number")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-photons", type=int, default=96)
    args = ap.parse_args()

    orbits_dir = Path(args.orbits)
    meta = pd.read_csv(orbits_dir / "passes_meta.csv")
    if args.n_pass_limit:
        meta = meta.head(args.n_pass_limit)

    rng = np.random.default_rng(args.seed)
    config = QKDParams()
    eve = EveAttack(name=args.eve, strength=args.strength)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, features = [], []
    for _, m in tqdm(meta.iterrows(), total=len(meta), desc=f"QKD [{args.eve}]"):
        df = pd.read_csv(orbits_dir / f"{m['pass_id']}.csv")
        rows = qkd_over_pass(df, config, eve, rng, n_photons=args.n_photons,
                             seed_sim=args.seed)
        for r in rows:
            r["pass_id"] = m["pass_id"]
            r["satellite"] = m["satellite"]
            r["sat_idx"] = int(m["sat_idx"])
        all_rows.extend(rows)
        features.append(list(link_features_from_pass(df, sat_idx=int(m["sat_idx"]))))

    df_all = pd.DataFrame(all_rows)
    key = f"eve_{args.eve}_s{args.strength}"
    df_all.to_csv(out_dir / f"qkd_{key}.csv", index=False)
    np.save(out_dir / f"features_{key}.npy", np.array(features, dtype=float))
    np.save(out_dir / f"meta_{key}.npy", meta.to_dict("records"), allow_pickle=True)
    print(f"Wrote {len(df_all)} samples -> {out_dir / f'qkd_{key}.csv'}")
    print(f"  mean QBER = {df_all['qber'].mean():.4f}, "
          f"mean secure bps = {df_all['secure_bps'].mean():.1f}")


if __name__ == "__main__":
    main()
