#include "qorbit_core.h"
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <numeric>
#include <random>

// Single curve time-domain features (11 features matching Python's extract_time_domain_features)
// Returns vector size 11: mean, std, var, median, skew, kurtosis, ptp, amp, energy, above_median_frac, min_val

static double median_of_sorted(const std::vector<double>& sorted) {
    size_t n = sorted.size();
    if (n % 2 == 1) return sorted[n/2];
    return 0.5 * (sorted[n/2 - 1] + sorted[n/2]);
}

std::vector<double> extract_time_features_single(const double* curve, int n) {
    double sum = 0, sum2 = 0;
    double mn = curve[0], mx = curve[0];
    double energy = 0;
    for (int i=0;i<n;++i){ double v=curve[i]; sum+=v; sum2+=v*v; if(v<mn) mn=v; if(v>mx) mx=v; energy+=v*v; }
    double mean = sum / n;
    double var = 0; double skew_m3=0, kurt_m4=0;
    for (int i=0;i<n;++i){ double d=curve[i]-mean; var+=d*d; skew_m3+=d*d*d; kurt_m4+=d*d*d*d; }
    var/=n;
    double std = std::sqrt(var + 1e-12);
    // skewness/kurtosis
    double skew=0, kurt=-3;
    if (std > 1e-9) {
        skew = (skew_m3 / n) / (std*std*std);
        kurt = (kurt_m4 / n) / (std*std*std*std) - 3.0;
    }
    // median: copy & sort
    std::vector<double> sorted(curve, curve+n);
    std::sort(sorted.begin(), sorted.end());
    double median = median_of_sorted(sorted);
    double ptp = mx - mn;
    double amp = mx;
    int above=0;
    for(int i=0;i<n;++i) if(curve[i] > median) ++above;
    double above_frac = (double)above / n;
    std::vector<double> out(11);
    out[0]=mean; out[1]=std; out[2]=var; out[3]=median; out[4]=skew; out[5]=kurt;
    out[6]=ptp; out[7]=amp; out[8]=energy; out[9]=above_frac; out[10]=mn;
    return out;
}

// Batch: curves shape (N, n_samples) row-major
void extract_time_features_batch(const double* curves, int N, int n_samples, double* out_features) {
    // out_features shape (N, 11)
    for(int i=0;i<N;++i){
        auto f = extract_time_features_single(curves + i*n_samples, n_samples);
        for(int j=0;j<11;++j) out_features[i*11+j]=f[j];
    }
}

// FFT-related features: we implement a naive DFT for up to 256 points for benchmark fairness vs Python's FFT
// Dominant freq via magnitude peak (excluding DC). harmonic_energy sum |FFT|^2, fft_entropy, n_harmonics
std::vector<double> extract_freq_features_single(const double* curve, int n, double dt) {
    // Compute DFT magnitude for positive frequencies
    int n_pos = n/2;
    std::vector<double> mag(n_pos, 0.0);
    std::vector<double> freq(n_pos);
    for(int k=0;k<n_pos;++k) freq[k]= k / (n*dt);
    // naive O(n^2) DFT for benchmark (purposely demonstrates CPU-heavy)
    for(int k=1;k<n_pos;++k){
        double re=0, im=0;
        for(int t=0;t<n;++t){
            double angle = -2.0 * M_PI * k * t / n;
            re += curve[t] * std::cos(angle);
            im += curve[t] * std::sin(angle);
        }
        mag[k]=std::sqrt(re*re+im*im);
    }
    double dom_freq=0, dom_mag=0;
    int best=1;
    for(int k=1;k<n_pos;++k) if(mag[k]>dom_mag){dom_mag=mag[k]; best=k; dom_freq=freq[k];}
    double harm_energy=0;
    for(int k=1;k<n_pos;++k) harm_energy += mag[k]*mag[k];
    int n_harm=0;
    double thresh = dom_mag * 0.1;
    for(int k=1;k<n_pos;++k) if(mag[k]>thresh) ++n_harm;
    // fft entropy over all mag (including DC)
    double sum_mag=0; for(int k=0;k<n_pos;++k) sum_mag+=mag[k];
    if(sum_mag<1e-12) sum_mag=1.0;
    double entropy=0;
    for(int k=0;k<n_pos;++k){ double p=mag[k]/sum_mag; if(p>1e-12) entropy -= p*std::log(p); }
    std::vector<double> out(5);
    out[0]=dom_freq; out[1]=dom_mag; out[2]=harm_energy; out[3]=entropy; out[4]=(double)n_harm;
    return out;
}

// Add Gaussian noise in-place batch
void add_gaussian_noise_batch(double* curves, int N, int n_samples, double std, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, std);
    for(int i=0;i<N*n_samples;++i){
        curves[i] += dist(rng);
        if (curves[i] < 0) curves[i]=0;
        if (curves[i] > 1) curves[i]=1;
    }
}

// Truncate observation: keep fraction
void truncate_observation_batch(double* curves, int N, int n_samples, double fraction) {
    int keep = std::max(1, (int)(n_samples * fraction));
    for(int i=0;i<N;++i){
        double last = curves[i*n_samples + keep -1];
        for(int j=keep;j<n_samples;++j) curves[i*n_samples+j]=last;
    }
}

// Bloch sphere: compute reduced density matrix for each qubit via partial trace, then Pauli expectations
void bloch_vectors_from_statevector(const std::complex<double>* state, int n_qubits, double* out_bloch) {
    int dim = 1 << n_qubits;
    // For each qubit k, compute rho_k = Tr_{others} |psi><psi| -> 2x2
    // Do brute force O(n*4^n) which is fine for n<=8 (256^2=65536)
    for(int k=0;k<n_qubits;++k){
        // rho elements: rho[a][b] where a,b in {0,1} for qubit k
        std::complex<double> rho00(0,0), rho01(0,0), rho11(0,0);
        // iterate over all basis indices i,j where all qubits except k match
        for(int i=0;i<dim;++i){
            for(int j=0;j<dim;++j){
                // check other qubits equal: mask without bit k
                int mask = ~(1<<k);
                if( (i & mask) != (j & mask) ) continue;
                std::complex<double> contrib = state[i] * std::conj(state[j]);
                int a = (i>>k)&1;
                int b = (j>>k)&1;
                if(a==0 && b==0) rho00 += contrib;
                else if(a==0 && b==1) rho01 += contrib;
                else if(a==1 && b==1) rho11 += contrib;
                // rho10 = conj(rho01) -> not needed separately
            }
        }
        std::complex<double> rho10 = std::conj(rho01);
        double rho00_r = rho00.real();
        double rho11_r = rho11.real();
        // Pauli expectations: <X>=Tr(rho X)= rho01+rho10 ; <Y>= i(rho10 - rho01); <Z>= rho00 - rho11
        double x = (rho01 + rho10).real();
        double y = (std::complex<double>(0,1)*(rho10 - rho01)).real();
        double z = rho00_r - rho11_r;
        // clamp
        if(x>1) x=1; if(x<-1) x=-1;
        if(y>1) y=1; if(y<-1) y=-1;
        if(z>1) z=1; if(z<-1) z=-1;
        out_bloch[k*3+0]=x;
        out_bloch[k*3+1]=y;
        out_bloch[k*3+2]=z;
    }
}
std::vector<double> bloch_vectors_cpp(const std::vector<std::complex<double>>& state, int n_qubits){
    std::vector<double> out(n_qubits*3);
    bloch_vectors_from_statevector(state.data(), n_qubits, out.data());
    return out;
}
double bloch_purity_from_statevector(const std::complex<double>* state, int n_qubits, int target_qubit){
    double bloch[3];
    // reuse but compute only one
    int dim=1<<n_qubits;
    std::complex<double> rho00(0,0), rho01(0,0), rho11(0,0);
    for(int i=0;i<dim;++i) for(int j=0;j<dim;++j){
        int mask=~(1<<target_qubit);
        if((i&mask)!=(j&mask)) continue;
        std::complex<double> c=state[i]*std::conj(state[j]);
        int a=(i>>target_qubit)&1, b=(j>>target_qubit)&1;
        if(a==0&&b==0) rho00+=c; else if(a==0&&b==1) rho01+=c; else if(a==1&&b==1) rho11+=c;
    }
    std::complex<double> rho10=std::conj(rho01);
    double x=(rho01+rho10).real();
    double y=(std::complex<double>(0,1)*(rho10 - rho01)).real();
    double z=rho00.real()-rho11.real();
    double r2=x*x+y*y+z*z;
    // purity = Tr(rho^2) = (1+|r|^2)/2
    return 0.5*(1.0+r2);
}
void expectation_pauli_batch(const std::complex<double>* state, int n_qubits, double* out_exp){
    bloch_vectors_from_statevector(state, n_qubits, out_exp);
}

// Synthetic light-curve generation: simplified physics loop (CPU-intensive)
// We simulate specular+diffuse with tumble oscillation per sample
void generate_synthetic_batch(double* out_curves, int N, int n_samples, double noise_std, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0,1);
    std::normal_distribution<double> noise(0, noise_std);
    for(int i=0;i<N;++i){
        // per-curve random params
        double period = 50.0 + uni(rng)*450.0; // 50-500
        double amp = 0.3 + uni(rng)*0.4;
        double phase = uni(rng)*2*M_PI;
        double aspect = 1.0 + uni(rng)*4.0;
        for(int t=0;t<n_samples;++t){
            double time = 720.0 * t / (n_samples-1);
            double angle = 2*M_PI*time/period + phase;
            double specular = amp * std::pow(std::max(0.0, std::cos(angle)), 2) * aspect;
            double diffuse = 0.2 * std::abs(std::sin(angle*0.5));
            double v = 0.5 + 0.4*specular + 0.2*diffuse;
            v += noise(rng);
            // per-curve min-max will be done outside; just clip
            if(v<0) v=0; if(v>1.5) v=1.5;
            out_curves[i*n_samples+t]=v;
        }
        // per-curve min-max normalize to [0,1]
        double mn=out_curves[i*n_samples], mx=mn;
        for(int t=1;t<n_samples;++t){ double v=out_curves[i*n_samples+t]; if(v<mn) mn=v; if(v>mx) mx=v; }
        double range = mx - mn;
        if(range < 1e-9){ for(int t=0;t<n_samples;++t) out_curves[i*n_samples+t]=0.5; }
        else { for(int t=0;t<n_samples;++t) out_curves[i*n_samples+t]=(out_curves[i*n_samples+t]-mn)/range; }
        // 5% dropout interpolate (simple)
        int ndrop = std::max(1, (int)(0.05*n_samples));
        for(int k=0;k<ndrop;++k){
            int idx = (int)(uni(rng)*n_samples);
            // simple: replace with neighbor average
            double left = (idx>0)? out_curves[i*n_samples+idx-1] : out_curves[i*n_samples+idx+1];
            double right = (idx<n_samples-1)? out_curves[i*n_samples+idx+1] : left;
            out_curves[i*n_samples+idx]=0.5*(left+right);
        }
    }
}
