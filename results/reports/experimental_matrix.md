# Experimental Matrix — Accuracy across conditions (same 1500 test)

| Condition | Classical | Quantum | Hybrid | Adaptive Hybrid |
|---|---|---|---|---|
| Clean | 0.8520 | 0.4807 | 0.7220 | 0.7760 |
| Noise 10% | 0.5253 | 0.3840 | 0.3240 | 0.5140 |
| Noise 20% | 0.2907 | 0.2740 | 0.2033 | 0.2893 |
| Noise 30% | 0.2453 | 0.2580 | 0.2000 | 0.2463 |
| Obs 75% | 0.8113 | 0.3960 | 0.3067 | 0.7781 |
| Obs 50% | 0.7707 | 0.2700 | 0.2000 | 0.7306 |
| Obs 25% | 0.5867 | 0.2100 | 0.2000 | 0.5565 |
| Missing 5% | 0.8487 | 0.4820 | 0.6733 | 0.8193 |
| Missing 10% | 0.8460 | 0.4753 | 0.6080 | 0.8163 |
| Missing 20% | 0.8507 | 0.4700 | 0.5280 | 0.8202 |

*Adaptive column for degraded conditions is approximated as weighted blend (gate not retrained per condition); clean Adaptive is measured (0.776). Full per-condition gate retraining would show true adaptive robustness.*
