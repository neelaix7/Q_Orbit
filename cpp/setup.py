from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "qorbit_cpp",
        ["qorbit_cpp_bindings.cpp", "qorbit_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="qorbit_cpp",
    ext_modules=ext_modules,
)
