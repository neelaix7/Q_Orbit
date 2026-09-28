# Hyperparameter Search — Validation-Based

**Best classical (val):** CNN

| Space | Options |
|---|---|
| SVM | C [0.5, 1.0, 2.0], gamma ['scale', 'auto'] |
| XGB | depth [4, 6], lr [0.05, 0.1] |
| Quantum | qubits [4, 6, 8], layers [1, 2, 3] |
| Hybrid | qubits [4, 6, 8], layers [2, 3], hidden [16, 32] |

*Full grid not exhaustively retrained due to quantum time; representative subset evaluated (5 hybrid, 4 pure, 4 classical). Production models are val-best from that subset.*
