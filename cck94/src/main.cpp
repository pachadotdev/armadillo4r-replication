// cck94/src/main.cpp
//
// Replication of Benigno & Woodford (2005), NBER WP 11029,
// "Optimal Taxation in an RBC Model: A Linear-Quadratic Approach"
//
// Uses the BW Linear-Quadratic (LQ) method directly:
//   1. Quadratic loss function (BW eq 2.13)
//   2. Log-linear resource constraint (BW eq 2.4)
//   3. Discrete Riccati equation -> optimal decision rules
//   4. Tax formulas (BW eqs 2.14–2.17)
//
// Also implements nonlinear solution methods:
//   - Time Iteration (Policy Function Iteration)
//   - Endogenous Grid Method (EGM) - Carroll (2005)
//   - Envelope Condition Method (ECM) - Maliar & Maliar (2013)
//   - Projection/Collocation method - CCK94 style
//
// Exported:
//   cck94_stats       -> BW Table 5 (1st-order Monte Carlo)
//   cck94_stats_2nd   -> BW Table 6 (2nd-order Monte Carlo)
//   cck94_solve_ti    -> Time Iteration solver
//   cck94_solve_egm   -> EGM solver
//   cck94_solve_ecm   -> ECM solver
//   cck94_solve_proj  -> Projection method solver

#include <armadillo4r.hpp>
#include <cpp4r.hpp>
#include <random>

using namespace cpp4r;
using namespace arma;

#include "01_parameters.h"
#include "02_solver.h"
#include "03_second_order.h"
#include "04_nonlinear_common.h"
#include "05_time_iteration.h"
#include "06_egm.h"
#include "07_ecm.h"
#include "08_projection.h"

// -----------------------------------------------------------------------
//  cck94_stats
//
//  Monte Carlo of first-order LQ optimal policy rules.
//  BW footnote 31: T_total = 500,000; T_burn = 60,000.
//
//  Returns (5 × 3) matrix (percentage points):
//    rows: E | s.d. | autocorr | corr_g | corr_z
//    cols: τ^h | θ^e | τ^k

/* roxygen
@title BW Table 5: 1st-order Monte Carlo statistics
@description Simulates log-linearized Ramsey policy rules using BW's
  LQ approach and returns the same statistics as Table 5 of Benigno &
  Woodford (2005).
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> cck94_stats(list args, int n_k, int T_total,
                                         int T_burn, int seed) {
  Params p(args);
  DecisionRules rules = compute_decision_rules(p);

  mat sim = simulate(p, rules, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0; // fraction -> percentage points

  const mat stats = simulation_stats(sim);

  writable::doubles_matrix<> out(5, 3);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = stats(r, c);
  return out;
}

// -----------------------------------------------------------------------
//  cck94_params — NSSS + LQ diagnostics
//
//  Returns (1 × 10):
//    sk | sc | sb | h_ss | phi_bw | qc | qh | qk | theta_lq | theta_z

/* roxygen
@title NSSS and LQ diagnostics
@description Returns steady-state shares, BW utility coefficients, and
  LQ loss function coefficients for diagnostic purposes.
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> cck94_params(list args) {
  Params p(args);

  writable::doubles_matrix<> out(1, 10);
  out(0, 0) = p.sk;
  out(0, 1) = p.sc;
  out(0, 2) = p.sb;
  out(0, 3) = p.h_ss;
  out(0, 4) = p.phi_bw;
  out(0, 5) = p.qc;
  out(0, 6) = p.qh;
  out(0, 7) = p.qk;
  out(0, 8) = p.theta_lq;
  out(0, 9) = p.theta_z;
  return out;
}

// -----------------------------------------------------------------------
//  cck94_diag — Full decision rule diagnostics
//
//  Returns a flat matrix with all coefficients for debugging.
//  Row layout (29 rows × 3 cols, padded with zeros):
//    rows 0-1:   gx (2×3)
//    rows 2-4:   hx (3×3)
//    rows 5:     gamma_w (1×3)
//    rows 6:     gamma_b (1×3)
//    rows 7:     gamma_tau_h (1×3)
//    rows 8:     Gamma_k_0 (1×3)
//    rows 9:     Gamma_k_1 (1×3)
//    row 10:     [sb, bc, bh]
//    row 11:     [bk, btau, 0]
//    row 12:     [dc, dh, phi_bw]
//    row 13:     [sigma_inv, nu, psi_bw]

/* roxygen
@title Full decision rule diagnostics
@description Returns gx, hx, and all BW tax formula coefficients.
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> cck94_diag(list args) {
  Params p(args);
  DecisionRules rules = compute_decision_rules(p);

  writable::doubles_matrix<> out(14, 3);
  for (int r = 0; r < 14; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = 0.0;

  // gx (2×3)
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = rules.gx(r, c);

  // hx (3×3)
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out(r + 2, c) = rules.hx(r, c);

  // gamma_w
  for (int c = 0; c < 3; ++c)
    out(5, c) = rules.gamma_w(c);
  // gamma_b
  for (int c = 0; c < 3; ++c)
    out(6, c) = rules.gamma_b(c);
  // gamma_tau_h
  for (int c = 0; c < 3; ++c)
    out(7, c) = rules.gamma_tau_h(c);
  // Gamma_k_0
  for (int c = 0; c < 3; ++c)
    out(8, c) = rules.Gamma_k_0(c);
  // Gamma_k_1
  for (int c = 0; c < 3; ++c)
    out(9, c) = rules.Gamma_k_1(c);

  // leverage
  out(10, 0) = p.sb;
  out(10, 1) = p.bc_bw;
  out(10, 2) = p.bh_bw;
  out(11, 0) = p.bk_bw;
  out(11, 1) = p.btau_bw;
  out(11, 2) = 0.0;
  // dc, dh, phi
  out(12, 0) = p.dc;
  out(12, 1) = p.dh;
  out(12, 2) = p.phi_bw;
  // utility coeffs
  out(13, 0) = p.sigma_inv;
  out(13, 1) = p.nu;
  out(13, 2) = p.psi_bw;

  return out;
}

// -----------------------------------------------------------------------
//  cck94_stats_2nd
//
//  Monte Carlo of SECOND-ORDER optimal policy rules (BW Table 6).
//  Uses Schmitt-Grohé & Uribe (2004) perturbation approximation.
//
//  Returns (5 × 3) matrix (percentage points):
//    rows: E | s.d. | autocorr | corr_g | corr_z
//    cols: τ^h | θ^e | τ^k

/* roxygen
@title BW Table 6: 2nd-order Monte Carlo statistics
@description Simulates second-order Ramsey policy rules using SGU (2004)
  perturbation and returns the same statistics as Table 6 of Benigno &
  Woodford (2005).
@export
*/
[[cpp4r::register]]
writable::doubles_matrix<> cck94_stats_2nd(list args, int n_k, int T_total,
                                             int T_burn, int seed) {
  Params p(args);
  DecisionRules r1 = compute_decision_rules(p);
  SecondOrderRules r2 = compute_second_order_rules(p, r1);

  mat sim = simulate_second_order(p, r2, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0; // fraction -> percentage points

  const mat stats = simulation_stats(sim);

  writable::doubles_matrix<> out(5, 3);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = stats(r, c);
  return out;
}

// =======================================================================
//  NONLINEAR SOLVERS
// =======================================================================

// Helper: find closest grid index for shock discretization
static int find_closest_idx(const vec &grid, double val) {
  int n = grid.n_elem;
  int idx = 0;
  double min_dist = std::abs(grid(0) - val);
  for (int i = 1; i < n; ++i) {
    double dist = std::abs(grid(i) - val);
    if (dist < min_dist) {
      min_dist = dist;
      idx = i;
    }
  }
  return idx;
}

// Helper: simulate from solved NONLINEAR policy functions
//
// Uses actual policy function interpolation for state evolution.
// Tax rates computed from LQ formulas (BW eqs 2.17, 3.8) to match
// BW Table 4 methodology.
static mat simulate_nonlinear(const Params &p, const GridSpec &grid,
                              const PolicyFunctions &pol,
                              const DecisionRules &rules, int T_total,
                              int T_burn, int seed) {
  const int T_keep = T_total - T_burn;
  mat out(T_keep, 6, fill::zeros);

  NonlinearModel model(p);

  // BW innovation sizes (footnote 18)
  const double delta_z = p.sigma_z * std::sqrt(1.0 - p.rho_z * p.rho_z);
  const double delta_g = p.sigma_g * std::sqrt(1.0 - p.rho_g * p.rho_g);

  std::mt19937 rng(static_cast<unsigned>(seed));
  std::bernoulli_distribution flip(0.5);

  // State variables in LEVELS
  double k = model.k_ss;
  double z_hat = 0.0; // log(z)
  double g_hat = 0.0; // log(g/g_ss)

  // Lagged state for tau_k computation (BW eq 3.8)
  double k_hat_lag = 0.0;
  double z_hat_lag = 0.0;
  double g_hat_lag = 0.0;

  for (int t = 0; t < T_total; ++t) {
    // Find closest (z, g) grid indices for interpolation
    int i_z = find_closest_idx(grid.z_grid, z_hat);
    int i_g = find_closest_idx(grid.g_grid, g_hat);

    // Interpolate policy functions at current state
    double k_prime = pol.interp_k_prime(grid.k_grid, i_z, i_g, k);
    double c = pol.interp_c(grid.k_grid, i_z, i_g, k);
    double h = pol.interp_h(grid.k_grid, i_z, i_g, k);

    // Bound values
    k_prime = std::max(grid.k_min, std::min(grid.k_max, k_prime));
    c = std::max(0.01, c);
    h = std::max(0.01, std::min(0.99, h));

    // State deviations for LQ formulas
    double k_hat = std::log(k / model.k_ss);

    if (t >= T_burn) {
      const int row = t - T_burn;

      // col 0: τ^h - labor tax from LQ formula (BW eq 2.17)
      double tau_h_hat = rules.gamma_tau_h(0) * k_hat +
                         rules.gamma_tau_h(1) * z_hat +
                         rules.gamma_tau_h(2) * g_hat;
      out(row, 0) = p.tau_h_ss + (1.0 - p.tau_h_ss) * tau_h_hat;

      // col 1: θ^e - ex ante capital tax (BW eq 3.9)
      out(row, 1) = rules.gamma_theta_e(0) * k_hat +
                    rules.gamma_theta_e(1) * z_hat +
                    rules.gamma_theta_e(2) * g_hat;

      // col 2: τ^k - ex post capital tax (BW eq 3.8)
      //   hattau^k_t = Gamma^k_tau(0) v_t + Gamma^k_tau(1) v_{t-1}
      out(row, 2) =
          rules.Gamma_k_0(0) * k_hat + rules.Gamma_k_0(1) * z_hat +
          rules.Gamma_k_0(2) * g_hat + rules.Gamma_k_1(0) * k_hat_lag +
          rules.Gamma_k_1(1) * z_hat_lag + rules.Gamma_k_1(2) * g_hat_lag;

      // col 3: θ^{nc} - non-contingent debt component
      out(row, 3) = rules.Gamma_k_0(0) * k_hat + rules.Gamma_k_0(1) * z_hat +
                    rules.Gamma_k_0(2) * g_hat;

      out(row, 4) = z_hat; // z_hat
      out(row, 5) = g_hat; // g_hat
    }

    // Save current state as lagged for next period
    k_hat_lag = k_hat;
    z_hat_lag = z_hat;
    g_hat_lag = g_hat;

    // Advance state using NONLINEAR policy and exogenous shock processes
    k = k_prime;
    z_hat = p.rho_z * z_hat + (flip(rng) ? delta_z : -delta_z);
    g_hat = p.rho_g * g_hat + (flip(rng) ? delta_g : -delta_g);
  }

  return out;
}

// -----------------------------------------------------------------------
//  cck94_solve_ti - Time Iteration solver
//
//  Returns list with:
//    stats: (5 × 3) statistics matrix (same format as cck94_stats)
//    info: convergence info (iterations, error, converged, elapsed_ms)

/* roxygen
@title Solve using Time Iteration
@description Solves the CCK94/BW Ramsey taxation model using time iteration
  (policy function iteration) on the Euler equation.
@param args Model parameters (same as cck94_stats)
@param n_k Number of capital grid points (default 50)
@param n_z Number of technology shock states (default 5)
@param n_g Number of govt spending shock states (default 5)
@param max_iter Maximum iterations (default 500)
@param tol Convergence tolerance (default 1e-6)
@param T_total Simulation periods for statistics (default 500000)
@param T_burn Burn-in periods (default 60000)
@param seed Random seed (default 42)
@export
*/
[[cpp4r::register]]
writable::list cck94_solve_ti(list args, int n_k, int n_z, int n_g,
                                int max_iter, double tol, int T_total,
                                int T_burn, int seed) {
  Params p(args);
  GridSpec grid = setup_grid(p, n_k, n_z, n_g);
  PolicyFunctions pol;

  SolverStats stats = time_iteration_solve(p, grid, pol, max_iter, tol);

  // Compute LQ rules for tax formulas
  DecisionRules rules = compute_decision_rules(p);

  // Simulate and compute statistics
  mat sim = simulate_nonlinear(p, grid, pol, rules, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0; // to percentage points

  mat stat_mat = simulation_stats(sim);

  // Output
  writable::doubles_matrix<> out_stats(5, 4);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 4; ++c)
      out_stats(r, c) = stat_mat(r, c);

  writable::doubles info(4);
  info[0] = static_cast<double>(stats.iterations);
  info[1] = stats.error;
  info[2] = stats.converged ? 1.0 : 0.0;
  info[3] = stats.elapsed_ms;

  writable::list result;
  result.push_back({"stats"_nm = out_stats});
  result.push_back({"info"_nm = info});

  return result;
}

// -----------------------------------------------------------------------
//  cck94_solve_egm - Endogenous Grid Method solver

/* roxygen
@title Solve using Endogenous Grid Method
@description Solves the CCK94/BW Ramsey taxation model using Carroll's (2005)
  endogenous grid method.
@param args Model parameters (same as cck94_stats)
@param n_k Number of capital grid points (default 50)
@param n_z Number of technology shock states (default 5)
@param n_g Number of govt spending shock states (default 5)
@param max_iter Maximum iterations (default 500)
@param tol Convergence tolerance (default 1e-6)
@param T_total Simulation periods for statistics (default 500000)
@param T_burn Burn-in periods (default 60000)
@param seed Random seed (default 42)
@export
*/
[[cpp4r::register]]
writable::list cck94_solve_egm(list args, int n_k, int n_z, int n_g,
                                 int max_iter, double tol, int T_total,
                                 int T_burn, int seed) {
  Params p(args);
  GridSpec grid = setup_grid(p, n_k, n_z, n_g);
  PolicyFunctions pol;

  SolverStats stats = egm_solve(p, grid, pol, max_iter, tol);

  // Compute LQ rules for tax formulas
  DecisionRules rules = compute_decision_rules(p);

  mat sim = simulate_nonlinear(p, grid, pol, rules, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0;

  mat stat_mat = simulation_stats(sim);

  writable::doubles_matrix<> out_stats(5, 4);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 4; ++c)
      out_stats(r, c) = stat_mat(r, c);

  writable::doubles info(4);
  info[0] = static_cast<double>(stats.iterations);
  info[1] = stats.error;
  info[2] = stats.converged ? 1.0 : 0.0;
  info[3] = stats.elapsed_ms;

  writable::list result;
  result.push_back({"stats"_nm = out_stats});
  result.push_back({"info"_nm = info});

  return result;
}

// -----------------------------------------------------------------------
//  cck94_solve_ecm - Envelope Condition Method solver

/* roxygen
@title Solve using Envelope Condition Method
@description Solves the CCK94/BW Ramsey taxation model using Maliar & Maliar's
  (2013) envelope condition method.
@param args Model parameters (same as cck94_stats)
@param n_k Number of capital grid points (default 50)
@param n_z Number of technology shock states (default 5)
@param n_g Number of govt spending shock states (default 5)
@param max_iter Maximum iterations (default 500)
@param tol Convergence tolerance (default 1e-6)
@param T_total Simulation periods for statistics (default 500000)
@param T_burn Burn-in periods (default 60000)
@param seed Random seed (default 42)
@export
*/
[[cpp4r::register]]
writable::list cck94_solve_ecm(list args, int n_k, int n_z, int n_g,
                                 int max_iter, double tol, int T_total,
                                 int T_burn, int seed) {
  Params p(args);
  GridSpec grid = setup_grid(p, n_k, n_z, n_g);
  PolicyFunctions pol;

  SolverStats stats = ecm_solve(p, grid, pol, max_iter, tol);

  // Compute LQ rules for tax formulas
  DecisionRules rules = compute_decision_rules(p);

  mat sim = simulate_nonlinear(p, grid, pol, rules, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0;

  mat stat_mat = simulation_stats(sim);

  writable::doubles_matrix<> out_stats(5, 4);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 4; ++c)
      out_stats(r, c) = stat_mat(r, c);

  writable::doubles info(4);
  info[0] = static_cast<double>(stats.iterations);
  info[1] = stats.error;
  info[2] = stats.converged ? 1.0 : 0.0;
  info[3] = stats.elapsed_ms;

  writable::list result;
  result.push_back({"stats"_nm = out_stats});
  result.push_back({"info"_nm = info});

  return result;
}

// -----------------------------------------------------------------------
//  cck94_solve_proj - Projection method solver (CCK94 style)

/* roxygen
@title Solve using Projection Method
@description Solves the CCK94/BW Ramsey taxation model using polynomial
  projection (Chebyshev collocation), similar to CCK94's minimum-weighted
  residual method.
@param args Model parameters (same as cck94_stats)
@param n_k Number of capital grid points (default 50)
@param n_z Number of technology shock states (default 5)
@param n_g Number of govt spending shock states (default 5)
@param poly_degree Chebyshev polynomial degree (default 5)
@param max_iter Maximum iterations (default 500)
@param tol Convergence tolerance (default 1e-6)
@param T_total Simulation periods for statistics (default 500000)
@param T_burn Burn-in periods (default 60000)
@param seed Random seed (default 42)
@export
*/
[[cpp4r::register]]
writable::list cck94_solve_proj(list args, int n_k, int n_z, int n_g,
                                  int poly_degree, int max_iter, double tol,
                                  int T_total, int T_burn, int seed) {
  Params p(args);
  GridSpec grid = setup_grid(p, n_k, n_z, n_g);
  PolicyFunctions pol;
  ProjectionPolicies proj;

  SolverStats stats =
      projection_solve(p, grid, pol, proj, poly_degree, max_iter, tol);

  // Compute LQ rules for tax formulas
  DecisionRules rules = compute_decision_rules(p);

  mat sim = simulate_nonlinear(p, grid, pol, rules, T_total, T_burn, seed);
  sim.cols(0, 3) *= 100.0;

  mat stat_mat = simulation_stats(sim);

  writable::doubles_matrix<> out_stats(5, 4);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 4; ++c)
      out_stats(r, c) = stat_mat(r, c);

  writable::doubles info(4);
  info[0] = static_cast<double>(stats.iterations);
  info[1] = stats.error;
  info[2] = stats.converged ? 1.0 : 0.0;
  info[3] = stats.elapsed_ms;

  writable::list result;
  result.push_back({"stats"_nm = out_stats});
  result.push_back({"info"_nm = info});

  return result;
}
