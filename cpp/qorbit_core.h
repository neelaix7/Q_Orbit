#pragma once
#include <vector>
#include <cstdint>
#include <complex>

std::vector<double> extract_time_features_single(const double* curve, int n);
void extract_time_features_batch(const double* curves, int N, int n_samples, double* out_features);
std::vector<double> extract_freq_features_single(const double* curve, int n, double dt);
void add_gaussian_noise_batch(double* curves, int N, int n_samples, double std, uint32_t seed);
void truncate_observation_batch(double* curves, int N, int n_samples, double fraction);
void generate_synthetic_batch(double* out_curves, int N, int n_samples, double noise_std, uint32_t seed);

// Bloch sphere extraction: statevector -> (n_qubits,3) bloch vectors
void bloch_vectors_from_statevector(const std::complex<double>* state, int n_qubits, double* out_bloch); // out (n_qubits*3) row-major x,y,z
std::vector<double> bloch_vectors_cpp(const std::vector<std::complex<double>>& state, int n_qubits);
double bloch_purity_from_statevector(const std::complex<double>* state, int n_qubits, int target_qubit);
void expectation_pauli_batch(const std::complex<double>* state, int n_qubits, double* out_exp); // (n_qubits*3)

