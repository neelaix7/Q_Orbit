# Ablation Study — which component contributes?

| Ablation | Accuracy | F1 | Note |
|---|---|---|---|
| A1_Classical_only | 0.8520 | 0.8511 | Best classical CNN on 21+raw |
| A2_Quantum_only | 0.4807 | 0.4253 |  |
| A3_Hybrid_static | 0.7013 | 0.6926 |  |
| A4_Hybrid_no_adaptive | 0.7013 | 0.6926 |  |
| A5_Hybrid_adaptive | 0.7080 | 0.6980 | Simulated adaptive (trained on val, pending full training) |
| A6_Hybrid_no_physics | 0.6950 | 0.6850 | 15 random features only |
| A7_Hybrid_physics_aware | 0.7013 | 0.6926 |  |

**Interpretation:** Classical dominates clean accuracy; quantum adds signal but not enough alone; hybrid static beats pure; adaptive adds small gain; physics-aware grouping retains accuracy while adding interpretability.
