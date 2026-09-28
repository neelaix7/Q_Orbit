# src/cpp/__init__.py — C++ accelerator bridge for Q-ORBIT
# Provides pybind11-backed accelerated ops with Python fallback
try:
    from . import qorbit_cpp  # compiled pybind11 module
    HAS_CPP = True
except Exception as e:
    qorbit_cpp = None
    HAS_CPP = False
    _cpp_error = str(e)

def get_cpp_module():
    return qorbit_cpp

def has_cpp():
    return HAS_CPP
