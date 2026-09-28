from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class BackendResult:
    statevector: np.ndarray  # (2**n,) complex
    n_qubits: int
    probs: np.ndarray = field(default=None)  # (2**n,) |amp|^2
    bloch_vectors: Optional[np.ndarray] = None  # (n_qubits,3)
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if self.probs is None:
            self.probs = np.abs(self.statevector)**2
        # normalize
        s=np.sum(self.probs)
        if s>1e-12:
            self.probs=self.probs/s

@dataclass
class StateSnapshot:
    label: str
    result: BackendResult
    description: str = ""
    code_qiskit: str = ""
    code_pennylane: str = ""
    code_stim: str = ""
