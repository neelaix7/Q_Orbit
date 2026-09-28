#include "qorbit_core.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    const int N = 2000;
    const int L = 256;
    const double dt = 720.0/255.0;
    std::cout << "Q-ORBIT C++ Standalone Benchmark\n";
    std::cout << "N=" << N << " curves, L=" << L << "\n";

    // generate synthetic batch via C++
    std::vector<double> curves(N * L);
    auto t0 = std::chrono::high_resolution_clock::now();
    generate_synthetic_batch(curves.data(), N, L, 0.02, 42);
    auto t1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "C++ synthetic generation: " << gen_ms << " ms (" << gen_ms/N << " ms/curve)\n";

    // time features
    std::vector<double> feats(N * 11);
    t0 = std::chrono::high_resolution_clock::now();
    extract_time_features_batch(curves.data(), N, L, feats.data());
    t1 = std::chrono::high_resolution_clock::now();
    double feat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "C++ time-feature extraction (11 feats): " << feat_ms << " ms (" << feat_ms/N << " ms/curve)\n";

    // freq features (naive DFT O(n^2) — intentionally CPU-heavy demonstration)
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<std::min(N,200);++i){
        auto f = extract_freq_features_single(curves.data()+i*L, L, dt);
        (void)f;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double freq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "C++ freq-feature (200 curves, naive DFT): " << freq_ms << " ms\n";

    // noise
    t0 = std::chrono::high_resolution_clock::now();
    add_gaussian_noise_batch(curves.data(), N, L, 0.05, 42);
    t1 = std::chrono::high_resolution_clock::now();
    double noise_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "C++ Gaussian noise (std 0.05): " << noise_ms << " ms\n";

    // truncate
    t0 = std::chrono::high_resolution_clock::now();
    truncate_observation_batch(curves.data(), N, L, 0.5);
    t1 = std::chrono::high_resolution_clock::now();
    double trunc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "C++ truncate 50%: " << trunc_ms << " ms\n";

    std::cout << "BENCHMARK_DONE\n";
    return 0;
}
