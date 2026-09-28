#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "qorbit_core.h"

namespace py = pybind11;

py::array_t<double> py_extract_time_features(py::array_t<double, py::array::c_style | py::array::forcecast> curves) {
    auto buf = curves.request();
    if (buf.ndim != 2) throw std::runtime_error("curves must be 2D (N, L)");
    int N = buf.shape[0];
    int L = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    // output (N, 11)
    py::array_t<double> out({N, 11});
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    extract_time_features_batch(ptr, N, L, out_ptr);
    return out;
}

py::array_t<double> py_generate_synthetic(int N, int L, double noise_std, int seed) {
    py::array_t<double> out({N, L});
    auto buf = out.request();
    double* ptr = static_cast<double*>(buf.ptr);
    generate_synthetic_batch(ptr, N, L, noise_std, (uint32_t)seed);
    return out;
}

void py_add_noise(py::array_t<double, py::array::c_style> curves, double std, int seed) {
    auto buf = curves.request();
    int N = buf.shape[0];
    int L = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    add_gaussian_noise_batch(ptr, N, L, std, (uint32_t)seed);
}

py::array_t<double> py_bloch_vectors(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> state, int n_qubits) {
    auto buf = state.request();
    if (buf.ndim != 1) throw std::runtime_error("state must be 1D complex");
    int dim = buf.shape[0];
    if (dim != (1<<n_qubits)) throw std::runtime_error("state dim must be 2**n_qubits");
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);
    py::array_t<double> out({n_qubits, 3});
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    bloch_vectors_from_statevector(ptr, n_qubits, out_ptr);
    return out;
}

PYBIND11_MODULE(qorbit_cpp, m) {
    m.doc() = "Q-ORBIT C++ accelerated operations (pybind11)";
    m.def("extract_time_features", &py_extract_time_features, "Extract 11 time-domain features batch (C++ accelerated)",
          py::arg("curves"));
    m.def("generate_synthetic", &py_generate_synthetic, "Generate synthetic light curves batch (C++ accelerated)",
          py::arg("N"), py::arg("L")=256, py::arg("noise_std")=0.02, py::arg("seed")=42);
    m.def("add_gaussian_noise", &py_add_noise, "Add Gaussian noise in-place (C++ accelerated)",
          py::arg("curves"), py::arg("std"), py::arg("seed")=42);
    m.def("bloch_vectors", &py_bloch_vectors, "Compute Bloch vectors (n_qubits,3) from statevector (C++ accelerated)",
          py::arg("state"), py::arg("n_qubits"));
}
