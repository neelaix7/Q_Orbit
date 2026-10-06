# Experimental Matrix — Accuracy across conditions (same 3750 test, seed 123)

| Condition | Classical | Quantum | Hybrid | Adaptive Hybrid |
|---|---|---|---|---|
| Clean | 0.8595 | 0.5563 | 0.7296 | 0.7701 |
| Noise 5% | 0.8301 | 0.5037 | 0.5859 | 0.8040 |
| Noise 10% | 0.7963 | 0.3581 | 0.4963 | 0.7612 |
| Noise 20% | 0.6387 | 0.2587 | 0.2083 | 0.6083 |
| Noise 30% | 0.4589 | 0.2485 | 0.2157 | 0.4421 |
| Obs 75% | 0.8291 | 0.4696 | 0.2901 | 0.8003 |
| Obs 50% | 0.7765 | 0.3235 | 0.2696 | 0.7403 |
| Obs 25% | 0.5256 | 0.2323 | 0.1768 | 0.5021 |
| Missing 5% | 0.8605 | 0.5539 | 0.6805 | 0.8360 |
| Missing 10% | 0.8520 | 0.5560 | 0.6224 | 0.8283 |
| Missing 20% | 0.8560 | 0.5539 | 0.5328 | 0.8318 |

*Adaptive column for degraded conditions is approximated as weighted blend (gate not retrained per condition); clean Adaptive is measured (0.776). Full per-condition gate retraining would show true adaptive robustness.*
