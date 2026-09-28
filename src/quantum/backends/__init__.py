"""Quantum simulator backends.

`AerBackend` is imported lazily. Qiskit-Aer initialises native libraries at import
time (`aer_initialize_libraries`), and a broken or conflicting Aer/OpenMP install
can hard-crash the interpreter there — a native fault that `except Exception`
cannot catch. Importing it lazily means a bad Aer install only affects code that
explicitly asks for Aer, instead of breaking every importer of this package.

Use `get_backend("aer" | "pennylane" | "stim")`, or import the concrete class
directly (`from src.quantum.backends.aer_backend import AerBackend`).
"""
from .base import BackendResult, StateSnapshot
from .pennylane_backend import PennylaneBackend
from .stim_backend import StimBackend

__all__ = ["BackendResult", "StateSnapshot", "PennylaneBackend", "AerBackend",
           "StimBackend", "get_backend"]


def __getattr__(name):
    # PEP 562: resolved on first attribute access, so `from ... import AerBackend`
    # still works without importing qiskit_aer when it is not needed.
    if name == "AerBackend":
        from .aer_backend import AerBackend
        return AerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_backend(name: str = "pennylane"):
    """Return a backend class by name. Aer is resolved only when requested."""
    key = (name or "").strip().lower()
    if key == "pennylane":
        return PennylaneBackend
    if key == "stim":
        return StimBackend
    if key == "aer":
        from .aer_backend import AerBackend
        return AerBackend
    raise ValueError(f"unknown backend {name!r}; expected 'pennylane', 'aer' or 'stim'")
