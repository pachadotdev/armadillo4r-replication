#ifndef TVP_CONVERGENCE_H
#define TVP_CONVERGENCE_H

/* ==========================================================================
   TVP-VAR Convergence Diagnostics
   Following Baumeister and Peersman (2013)

   Ported from MATLAB: convergence_check.m, mymatprog/convergence/
   ========================================================================== */

// ============================================================================
// SPECTRAL DENSITY ESTIMATION
// ============================================================================

// bartwind: Create Bartlett spectral window (bartwind.m)
inline vec bartwind(int M) {
  vec w(2 * M + 1);
  for (int i = 0; i <= 2 * M; ++i) {
    int j = i - M;
    w(i) = 1.0 - std::abs((double)j) / (M + 1);
  }
  return w;
}

// specsmooth: Smooth periodogram with spectral window (specsmooth.m)
inline vec specsmooth(const vec &Iy, int M, const vec &BART) {
  int n = Iy.n_elem;
  vec Py(n);

  for (int j = 0; j < n; ++j) {
    double sum_w = 0.0;
    double sum_wy = 0.0;

    for (int k = -M; k <= M; ++k) {
      int idx = j + k;
      if (idx >= 0 && idx < n) {
        double w = BART(k + M);
        sum_w += w;
        sum_wy += w * Iy(idx);
      }
    }

    Py(j) = sum_wy / sum_w;
  }

  return Py;
}

// beltbloom: Select optimal bandwidth (beltbloom.m simplified)
// Uses cross-validation in frequency domain
inline int beltbloom(const vec &y) {
  int n = y.n_elem;

  // Start with a reasonable range of M values
  int M_min = 1;
  int M_max = std::min(n / 4, 50);

  // Compute periodogram
  vec y_dm = y - mean(y);
  cx_vec Y = fft(y_dm);
  vec Iy(n / 2);
  for (int j = 1; j <= n / 2; ++j) {
    double re = std::real(Y(j)) / n;
    double im = std::imag(Y(j)) / n;
    Iy(j - 1) = 2 * (re * re + im * im) * n / (4 * M_PI);
  }

  // Simple heuristic: sqrt(n)
  int M_opt = (int)sqrt((double)n);
  M_opt = std::max(M_min, std::min(M_opt, M_max));

  return M_opt;
}

// SpecDensEstima: Spectral density estimation (SpecDensEstima.m simplified)
// Returns inefficiency factor (inverse of spectral density at zero normalized)
inline double spec_dens_estima(const vec &y) {
  int n = y.n_elem;
  if (n < 10)
    return 1.0;

  // Demean
  vec y_dm = y - mean(y);

  // Select bandwidth
  int M = beltbloom(y_dm);
  vec BART = bartwind(M);

  // Compute periodogram
  cx_vec Y = fft(y_dm);
  int n_freq = n / 2;
  vec Iy(n_freq);
  for (int j = 1; j <= n_freq; ++j) {
    double re = std::real(Y(j)) / n;
    double im = std::imag(Y(j)) / n;
    Iy(j - 1) = 2 * (re * re + im * im) * n / (4 * M_PI);
  }

  // Smooth
  vec Py = specsmooth(Iy, M, BART);

  // Compute inefficiency factor
  double P0 = Py(0);
  double sum_Py = sum(Py);

  // Inefficiency factor = 1 / (sum(Py) / (P0 * 2 * pi))
  if (sum_Py < 1e-10)
    return 1.0;
  double IF = (P0 * 2 * M_PI) / sum_Py;

  return IF;
}

// ============================================================================
// CONVERGENCE DIAGNOSTICS
// ============================================================================

// Compute inefficiency factors for a matrix of draws
// Input: draws (n_draws x n_params) or (n_params x n_draws)
// Returns: vector of inefficiency factors
inline vec compute_inefficiency_factors(const Mat<double> &draws,
                                        bool by_row = false) {
  int n_params = by_row ? draws.n_rows : draws.n_cols;

  vec IF(n_params);

  for (int i = 0; i < n_params; ++i) {
    vec x;
    if (by_row) {
      x = draws.row(i).t();
    } else {
      x = draws.col(i);
    }
    IF(i) = spec_dens_estima(x);
  }

  return IF;
}

// Effective sample size from inefficiency factors
inline vec effective_sample_size(const vec &IF, int n_draws) {
  return (double)n_draws / IF;
}

// Check convergence: returns true if max inefficiency factor is below threshold
inline bool check_convergence(const vec &IF, double threshold = 50.0) {
  return max(IF) < threshold;
}

// Summary statistics for MCMC draws
struct MCMCDiagnostics {
  vec mean;
  vec std_dev;
  vec inefficiency;
  vec ess; // Effective sample size
  bool converged;
};

inline MCMCDiagnostics compute_diagnostics(const Mat<double> &draws,
                                           double IF_threshold = 50.0) {
  MCMCDiagnostics diag;

  int n_draws = draws.n_rows;
  int n_params = draws.n_cols;

  diag.mean.set_size(n_params);
  diag.std_dev.set_size(n_params);
  diag.inefficiency.set_size(n_params);
  diag.ess.set_size(n_params);

  for (int i = 0; i < n_params; ++i) {
    vec x = draws.col(i);
    diag.mean(i) = mean(x);
    diag.std_dev(i) = stddev(x);
    diag.inefficiency(i) = spec_dens_estima(x);
    diag.ess(i) = n_draws / diag.inefficiency(i);
  }

  diag.converged = check_convergence(diag.inefficiency, IF_threshold);

  return diag;
}

#endif // TVP_CONVERGENCE_H
