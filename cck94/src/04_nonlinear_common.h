#pragma once
#include "01_parameters.h"
#include <armadillo4r.hpp>
#include <cmath>
#include <functional>

using namespace arma;

// =====================================================================
//  GridSpec - Specification for state-space grids
//
//  For the CCK94/BW Ramsey taxation model:
//    States: (k, z, g) - capital, technology shock, govt spending
//    Controls: (c, h) - consumption, hours
// =====================================================================
struct GridSpec {
  // Capital grid
  int n_k;      // number of capital grid points
  double k_min; // min capital (fraction of steady state)
  double k_max; // max capital (fraction of steady state)
  vec k_grid;   // capital grid values

  // Shock discretization (Tauchen/Rouwenhorst)
  int n_z;    // number of technology shock states
  int n_g;    // number of govt spending shock states
  vec z_grid; // technology shock grid (log deviations)
  vec g_grid; // govt spending grid (log deviations)
  mat P_z;    // transition matrix for z
  mat P_g;    // transition matrix for g

  // Total grid size
  int n_total() const { return n_k * n_z * n_g; }

  // Index mapping: (i_k, i_z, i_g) -> linear index
  int idx(int i_k, int i_z, int i_g) const {
    return i_k + n_k * (i_z + n_z * i_g);
  }

  // Reverse mapping: linear index -> (i_k, i_z, i_g)
  void idx_inv(int idx, int &i_k, int &i_z, int &i_g) const {
    i_k = idx % n_k;
    i_z = (idx / n_k) % n_z;
    i_g = idx / (n_k * n_z);
  }
};

// =====================================================================
//  NonlinearModel - Model equations for CCK94/BW Ramsey problem
//
//  Production: y = k^alpha (zh)^{1-alpha}
//  Utility:    u(c,h) = [c^{1-gamma}(1-h)^gamma]^phi / phi  (or log if phi=0)
//
//  Key FOCs:
//    1. Resource: y = c + g + k' - (1-delta)k
//    2. Euler: u_c = betaE[u_c'(1 + (1-tau^k')(f_k' - delta))]
//    3. Labor: u_h/u_c = -(1-tau^h)zf_h
//    4. Implementability (present-value budget constraint)
// =====================================================================
struct NonlinearModel {
  // Calibrated parameters
  double alpha;         // capital share
  double beta;          // discount factor (detrended)
  double delta;         // depreciation rate
  double gamma;         // utility weight on leisure
  double phi;           // utility curvature (0 = log)
  double tau_h_ss;      // steady-state labor tax
  double k_ss;          // steady-state capital
  double h_ss;          // steady-state hours
  double c_ss;          // steady-state consumption
  double g_ss;          // steady-state govt spending (level)
  double A;             // TFP normalizing factor (so y_ss = 1)
  double lambda_ramsey; // Ramsey implementability multiplier

  // Constructor from Params
  explicit NonlinearModel(const Params &p) {
    alpha = p.alpha;
    beta = p.beta_tilde;
    delta = p.delta_tilde;
    gamma = p.gamma_bw;
    phi = p.psi_u;
    tau_h_ss = p.tau_h_ss;

    // Compute steady-state quantities
    // k_ss is the capital-output ratio from BW
    k_ss = p.sk;
    h_ss = p.h_ss;
    c_ss = p.sc; // consumption-output ratio
    g_ss = p.sg; // govt spending-output ratio

    // Compute TFP so that A * k^alpha * h^{1-alpha} = 1 at steady state
    // This normalizes output to 1
    double y_raw = std::pow(k_ss, alpha) * std::pow(h_ss, 1.0 - alpha);
    A = 1.0 / y_raw;

    // Compute Ramsey multiplier lambda from steady-state tax rate
    // For log utility: tau^h = lambda/(1-h+lambda) => lambda = tau^h(1-h)/(1-tau^h)
    // This is derived from CCK94's primal approach to the Ramsey problem
    lambda_ramsey = tau_h_ss * (1.0 - h_ss) / (1.0 - tau_h_ss);
  }

  // Production function: f(k, zh) = A * k^alpha (zh)^{1-alpha}
  // With z=1 at steady state, f(k_ss, 1, h_ss) = 1
  double f(double k, double z, double h) const {
    return A * std::pow(k, alpha) * std::pow(z * h, 1.0 - alpha);
  }

  // Marginal product of capital: f_k = A * alpha k^{alpha-1} (zh)^{1-alpha}
  double f_k(double k, double z, double h) const {
    return A * alpha * std::pow(k, alpha - 1.0) * std::pow(z * h, 1.0 - alpha);
  }

  // Marginal product of labor: f_h = A * (1-alpha) k^alpha z (zh)^{-alpha} = (1-alpha)y/h
  double f_h(double k, double z, double h) const {
    return A * (1.0 - alpha) * std::pow(k, alpha) * z * std::pow(z * h, -alpha);
  }

  // Utility: u(c,h) = [c^{1-gamma}(1-h)^gamma]^phi / phi  (log utility if phi~=0)
  double u(double c, double h) const {
    if (std::abs(phi) < 1e-10) {
      // Log utility
      return (1.0 - gamma) * std::log(c) + gamma * std::log(1.0 - h);
    } else {
      double composite = std::pow(c, 1.0 - gamma) * std::pow(1.0 - h, gamma);
      return (std::pow(composite, phi) - 1.0) / phi;
    }
  }

  // Marginal utility of consumption: u_c
  double u_c(double c, double h) const {
    if (std::abs(phi) < 1e-10) {
      return (1.0 - gamma) / c;
    } else {
      double composite = std::pow(c, 1.0 - gamma) * std::pow(1.0 - h, gamma);
      return (1.0 - gamma) * std::pow(composite, phi) / c;
    }
  }

  // Marginal disutility of labor: u_h (negative)
  double u_h(double c, double h) const {
    if (std::abs(phi) < 1e-10) {
      return -gamma / (1.0 - h);
    } else {
      double composite = std::pow(c, 1.0 - gamma) * std::pow(1.0 - h, gamma);
      return -gamma * std::pow(composite, phi) / (1.0 - h);
    }
  }

  // Inverse marginal utility: given u_c, solve for c (holding h fixed)
  // Used in EGM
  double c_from_uc(double uc_val, double h) const {
    if (std::abs(phi) < 1e-10) {
      return (1.0 - gamma) / uc_val;
    } else {
      // u_c = (1-gamma) c^{(1-gamma)phi-1} (1-h)^{gammaphi}
      // c = [(u_c / (1-gamma)) (1-h)^{-gammaphi}]^{1/((1-gamma)phi-1)}
      double leisure_pow = std::pow(1.0 - h, gamma * phi);
      double exponent = 1.0 / ((1.0 - gamma) * phi - 1.0);
      return std::pow((uc_val / (1.0 - gamma)) / leisure_pow, exponent);
    }
  }

  // Labor tax from intratemporal FOC: tau^h = 1 - gamma(c/y)h/[(1-gamma)(1-h)(1-alpha)]
  // This uses the ratio-based formulation consistent with BW's normalization
  // Household FOC: (1-tau^h) w = -u_h/u_c where w = (1-alpha)y/h
  double tau_h_from_foc(double c, double h, double k, double z) const {
    double y = f(k, z, h);
    if (y <= 0.0 || h <= 0.0 || h >= 1.0)
      return tau_h_ss;
    double c_y = c / y; // consumption/output ratio
    return 1.0 -
           (gamma * c_y * h) / ((1.0 - gamma) * (1.0 - h) * (1.0 - alpha));
  }

  // Ramsey labor tax from the implementability constraint
  // For log utility: tau^h = lambda/(1-h+lambda) where lambda is the constant Ramsey multiplier
  // This formula comes from the CCK94 primal Ramsey FOCs
  double tau_h_ramsey(double h) const {
    return lambda_ramsey / (1.0 - h + lambda_ramsey);
  }

  // Solve for hours from Ramsey planner's intratemporal FOC
  // The Ramsey FOC (for log utility) is:
  //   (1-gamma)/c * z f_h = gamma(1-h+lambda)/[(1-h)^2]
  // This differs from competitive equilibrium by the term involving lambda
  double solve_hours_ramsey(double c, double k, double z, double tol = 1e-10,
                            int max_iter = 100) const {
    double h = h_ss; // initial guess

    for (int iter = 0; iter < max_iter; ++iter) {
      double fh = f_h(k, z, h);

      // Ramsey FOC residual:
      // (1-gamma)/c * z * f_h - gamma(1-h+lambda)/(1-h)^2 = 0
      double lhs = (1.0 - gamma) / c * z * fh;
      double rhs = gamma * (1.0 - h + lambda_ramsey) / ((1.0 - h) * (1.0 - h));
      double res = lhs - rhs;

      if (std::abs(res) < tol)
        break;

      // Numerical derivative
      const double dh = 1e-6;
      double fh_p = f_h(k, z, h + dh);
      double lhs_p = (1.0 - gamma) / c * z * fh_p;
      double rhs_p = gamma * (1.0 - h - dh + lambda_ramsey) /
                     ((1.0 - h - dh) * (1.0 - h - dh));
      double res_p = lhs_p - rhs_p;

      double dres = (res_p - res) / dh;
      if (std::abs(dres) < 1e-15)
        break;

      double h_new = h - res / dres;
      h_new = std::max(0.01, std::min(0.99, h_new));
      h = h_new;
    }

    return h;
  }

  // Capital tax from Euler: given R_k = E[u_c'(f_k' - delta)], compute tau^k
  // Euler: u_c = beta E[u_c'(1 + (1-tau^k)(f_k - delta))]
  // => tau^k = 1 - (u_c/beta - E[u_c']) / E[u_c'(f_k - delta)]
  double tau_k_from_euler(double uc, double E_uc_prime, double E_uc_fk_prime,
                          double E_uc_prime_delta) const {
    double E_uc_Rk = E_uc_fk_prime - E_uc_prime_delta;
    if (std::abs(E_uc_Rk) < 1e-15)
      return 0.0;
    return 1.0 - (uc / beta - E_uc_prime) / E_uc_Rk;
  }
};

// =====================================================================
//  Tauchen (1986) discretization for AR(1) processes
//
//  z' = rho z + eps, eps ~ N(0, sigma^2(1-rho^2))
//  Returns grid values and transition matrix
// =====================================================================
inline void tauchen(int n, double rho, double sigma, double m, vec &grid,
                    mat &P) {
  grid.set_size(n);
  P.set_size(n, n);

  if (n == 1) {
    grid(0) = 0.0;
    P(0, 0) = 1.0;
    return;
  }

  // Unconditional std dev of z
  const double sigma_z = sigma / std::sqrt(1.0 - rho * rho);

  // Grid spans +-m std deviations
  const double z_max = m * sigma_z;
  const double z_min = -z_max;
  const double dz = (z_max - z_min) / (n - 1);

  for (int i = 0; i < n; ++i) {
    grid(i) = z_min + i * dz;
  }

  // Transition probabilities using normal CDF
  auto norm_cdf = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double z_j = grid(j);
      double z_i = grid(i);
      if (j == 0) {
        P(i, j) = norm_cdf((z_j + 0.5 * dz - rho * z_i) / sigma);
      } else if (j == n - 1) {
        P(i, j) = 1.0 - norm_cdf((z_j - 0.5 * dz - rho * z_i) / sigma);
      } else {
        P(i, j) = norm_cdf((z_j + 0.5 * dz - rho * z_i) / sigma) -
                  norm_cdf((z_j - 0.5 * dz - rho * z_i) / sigma);
      }
    }
    // Normalize
    P.row(i) /= accu(P.row(i));
  }
}

// =====================================================================
//  setup_grid - Initialize state-space grid for the CCK94/BW model
// =====================================================================
inline GridSpec setup_grid(const Params &p, int n_k, int n_z, int n_g,
                           double k_width = 0.25, double m_tauchen = 3.0) {
  GridSpec g;

  // Capital grid setup - use k_ss = sk from Params
  g.n_k = n_k;
  const double k_ss = p.sk; // BW's capital-output ratio
  g.k_min = (1.0 - k_width) * k_ss;
  g.k_max = (1.0 + k_width) * k_ss;

  g.k_grid.set_size(n_k);
  for (int i = 0; i < n_k; ++i) {
    g.k_grid(i) = g.k_min + i * (g.k_max - g.k_min) / (n_k - 1);
  }

  // Technology shock discretization
  g.n_z = n_z;
  const double sigma_eps_z = p.sigma_z * std::sqrt(1.0 - p.rho_z * p.rho_z);
  tauchen(n_z, p.rho_z, sigma_eps_z, m_tauchen, g.z_grid, g.P_z);

  // Government spending shock discretization
  g.n_g = n_g;
  const double sigma_eps_g = p.sigma_g * std::sqrt(1.0 - p.rho_g * p.rho_g);
  tauchen(n_g, p.rho_g, sigma_eps_g, m_tauchen, g.g_grid, g.P_g);

  return g;
}

// =====================================================================
//  PolicyFunctions - Container for nonlinear policy approximations
//
//  Stores policy functions on the grid:
//    k'(k,z,g)  - capital
//    c(k,z,g)   - consumption
//    h(k,z,g)   - hours
//    tau^h(k,z,g) - labor tax
//    tau^k(k,z,g) - capital tax
// =====================================================================
struct PolicyFunctions {
  cube k_prime; // n_k * n_z * n_g: capital policy
  cube c;       // n_k * n_z * n_g: consumption
  cube h;       // n_k * n_z * n_g: hours
  cube tau_h;   // n_k * n_z * n_g: labor tax
  cube tau_k;   // n_k * n_z * n_g: capital tax
  cube V;       // n_k * n_z * n_g: value function (for VFI)
  cube V_k;     // n_k * n_z * n_g: marginal value (for ECM)

  void initialize(int n_k, int n_z, int n_g) {
    k_prime.zeros(n_k, n_z, n_g);
    c.zeros(n_k, n_z, n_g);
    h.zeros(n_k, n_z, n_g);
    tau_h.zeros(n_k, n_z, n_g);
    tau_k.zeros(n_k, n_z, n_g);
    V.zeros(n_k, n_z, n_g);
    V_k.zeros(n_k, n_z, n_g);
  }

  // Linear interpolation for k_prime given off-grid capital
  double interp_k_prime(const vec &k_grid, int i_z, int i_g, double k) const {
    const int n_k = k_grid.n_elem;
    if (k <= k_grid(0))
      return k_prime(0, i_z, i_g);
    if (k >= k_grid(n_k - 1))
      return k_prime(n_k - 1, i_z, i_g);

    // Find bracketing indices
    int i_lo = 0;
    while (i_lo < n_k - 2 && k_grid(i_lo + 1) < k)
      ++i_lo;
    int i_hi = i_lo + 1;

    // Linear interpolation weight
    double w = (k - k_grid(i_lo)) / (k_grid(i_hi) - k_grid(i_lo));
    return (1.0 - w) * k_prime(i_lo, i_z, i_g) + w * k_prime(i_hi, i_z, i_g);
  }

  // Same for other variables...
  double interp_c(const vec &k_grid, int i_z, int i_g, double k) const {
    const int n_k = k_grid.n_elem;
    if (k <= k_grid(0))
      return c(0, i_z, i_g);
    if (k >= k_grid(n_k - 1))
      return c(n_k - 1, i_z, i_g);

    int i_lo = 0;
    while (i_lo < n_k - 2 && k_grid(i_lo + 1) < k)
      ++i_lo;
    int i_hi = i_lo + 1;

    double w = (k - k_grid(i_lo)) / (k_grid(i_hi) - k_grid(i_lo));
    return (1.0 - w) * c(i_lo, i_z, i_g) + w * c(i_hi, i_z, i_g);
  }

  double interp_h(const vec &k_grid, int i_z, int i_g, double k) const {
    const int n_k = k_grid.n_elem;
    if (k <= k_grid(0))
      return h(0, i_z, i_g);
    if (k >= k_grid(n_k - 1))
      return h(n_k - 1, i_z, i_g);

    int i_lo = 0;
    while (i_lo < n_k - 2 && k_grid(i_lo + 1) < k)
      ++i_lo;
    int i_hi = i_lo + 1;

    double w = (k - k_grid(i_lo)) / (k_grid(i_hi) - k_grid(i_lo));
    return (1.0 - w) * h(i_lo, i_z, i_g) + w * h(i_hi, i_z, i_g);
  }

  double interp_tau_h(const vec &k_grid, int i_z, int i_g, double k) const {
    const int n_k = k_grid.n_elem;
    if (k <= k_grid(0))
      return tau_h(0, i_z, i_g);
    if (k >= k_grid(n_k - 1))
      return tau_h(n_k - 1, i_z, i_g);

    int i_lo = 0;
    while (i_lo < n_k - 2 && k_grid(i_lo + 1) < k)
      ++i_lo;
    int i_hi = i_lo + 1;

    double w = (k - k_grid(i_lo)) / (k_grid(i_hi) - k_grid(i_lo));
    return (1.0 - w) * tau_h(i_lo, i_z, i_g) + w * tau_h(i_hi, i_z, i_g);
  }

  double interp_V(const vec &k_grid, int i_z, int i_g, double k) const {
    const int n_k = k_grid.n_elem;
    if (k <= k_grid(0))
      return V(0, i_z, i_g);
    if (k >= k_grid(n_k - 1))
      return V(n_k - 1, i_z, i_g);

    int i_lo = 0;
    while (i_lo < n_k - 2 && k_grid(i_lo + 1) < k)
      ++i_lo;
    int i_hi = i_lo + 1;

    double w = (k - k_grid(i_lo)) / (k_grid(i_hi) - k_grid(i_lo));
    return (1.0 - w) * V(i_lo, i_z, i_g) + w * V(i_hi, i_z, i_g);
  }

  double interp_V_k(const vec &k_grid, int i_z, int i_g, double k) const {
    const int n_k = k_grid.n_elem;
    if (k <= k_grid(0))
      return V_k(0, i_z, i_g);
    if (k >= k_grid(n_k - 1))
      return V_k(n_k - 1, i_z, i_g);

    int i_lo = 0;
    while (i_lo < n_k - 2 && k_grid(i_lo + 1) < k)
      ++i_lo;
    int i_hi = i_lo + 1;

    double w = (k - k_grid(i_lo)) / (k_grid(i_hi) - k_grid(i_lo));
    return (1.0 - w) * V_k(i_lo, i_z, i_g) + w * V_k(i_hi, i_z, i_g);
  }
};

// =====================================================================
//  SolverStats - Convergence statistics
// =====================================================================
struct SolverStats {
  int iterations;
  double error;
  bool converged;
  double elapsed_ms;
};
