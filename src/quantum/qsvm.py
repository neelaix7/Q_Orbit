# qsvm.py - DEPRECATED (broken duplicate). Canonical path:
# src.quantum.hybrid_model.QuantumKernelSVM (fidelity kernel + SVC precomputed).
from __future__ import annotations
import warnings as _w
_w.warn("src.quantum.qsvm is deprecated; use src.quantum.hybrid_model.QuantumKernelSVM", DeprecationWarning, stacklevel=2)
from src.quantum.hybrid_model import QuantumKernelSVM  # noqa: F401
