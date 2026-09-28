# Hybrid Search (quick subset demo — validation-selected best, no test leakage)

Seed 42 | Grid 5 configs | Epochs 5 | Subset train 2000 val 500

| # | Qubits | Layers | Hidden | Val Acc | Val F1 | Test Acc | Test F1 | Params | Depth | Train(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 2 | 16 | 0.5680 | 0.5132 | 0.5287 | 0.4715 | 582 | 5 | 5.3 |
| 2 | 6 | 2 | 16 | 0.5900 | 0.5898 | 0.6060 | 0.6031 | 670 | 5 | 7.6 |
| 3 | 8 | 2 | 16 | 0.6400 | 0.5773 | 0.6393 | 0.5796 | 758 | 5 | 13.2 |
| 4 | 8 | 3 | 16 | 0.6380 | 0.5900 | 0.6207 | 0.5681 | 782 | 7 | 13.3 |
| 5 | 8 | 2 | 32 | 0.6220 | 0.5858 | 0.6300 | 0.5935 | 1798 | 5 | 5.5 |

**Best on VAL (subset):** qubits=8 layers=3 hidden=16 -> Val F1 0.5900 Test F1 0.5681

Note: Full production hybrid (8 qubits, 2 layers, 16 hidden) trained on full 7000-sample train set achieves 70.13% test accuracy (see fair_comparison.md). This quick search demonstrates systematic validation-based selection; production model remains the best validated on full data.
