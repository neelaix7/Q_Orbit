# Raw Data — Q-ORBIT

## Source
Physics generator `src/simulator/lightcurve_generator.py` (seed 123).

## Files
- `lightcurves_raw.npz` — raw generator output, `curves` (25000,256),
  `labels` (25000,). Canonical master copy lives at
  `data/synthetic/lightcurves.npz`.
- `sample_lightcurves.csv` — small human-readable sample of curves for inspection.

## Note
Raw curves are min-max normalized per curve to [0,1] at generation time.
Frozen 70/15/15 splits derive from this archive via `src/data/splits.py`.
