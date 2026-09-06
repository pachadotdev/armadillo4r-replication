#pragma once
#include "01_parameters.h"
#include <armadillo4r.hpp>
#include <cmath>
#include <random>
#include <stdexcept>

using namespace arma;

// =====================================================================
//  DecisionRules - first-order optimal policy rules from BW LQ solver
//
//  State vector:   v_t = [hatk_t, hatz_t, hatg_t]'
//  Control vector: u_t = [hatc_t, hath_t]'
//
//  Rules: u_t  = gx * v_t     (2x3)
//         v_{t+1} = hx * v_t + eta epsilon_{t+1}  (3x3)
//
//  Tax rules (BW eq 3.8):
//    hattau^h_t = Gamma^h_tau v_t
//    hattau^k_t = Gamma^k_tau(0) v_t + Gamma^k_tau(1) v_{t-1}
// =====================================================================
struct DecisionRules {
  mat gx; // 2x3: controls [hatc, hath] = gx * v
  mat hx; // 3x3: states v' = hx * v + noise

  rowvec gamma_w;       // 1x3: tildeW_t = Gamma_w * v_t  (BW eq 2.14)
  rowvec gamma_b;       // 1x3: barb_s_t = Gamma_b * v_t  (BW eq 2.15)
  rowvec gamma_tau_h;   // 1x3: hattau_h_t = Gamma_tau_h * v_t  (BW eq 2.17)
  rowvec Gamma_k_0;     // 1x3: current-state part of hattau_k  (BW eq 3.8)
  rowvec Gamma_k_1;     // 1x3: lagged-state part of hattau_k
  rowvec gamma_theta_e; // 1x3: theta_e_t = E_t[tau_k_{t+1}] = (Gamma_k_0*hx +
                        // Gamma_k_1)*v_t
};

// =====================================================================
//  solve_dare - Discrete Algebraic Riccati Equation (iterative)
//
//  Solves for P in:
//    P = Q + beta A'PA - (N + beta A'PB)(R + beta B'PB)^{-1}(N' + beta B'PA)
//
//  Returns the 3x3 value matrix P.
// =====================================================================
static mat solve_dare(const Params &p, int max_iter = 5000,
                      double tol = 1e-14) {
  const mat &A = p.A_con;
  const mat &B = p.B_con;
  const mat &Q = p.Q_cost;
  const mat &R = p.R_cost;
  const mat &N = p.N_cost;
  const double beta = p.beta_tilde;

  mat P = Q; // initial guess

  for (int iter = 0; iter < max_iter; ++iter) {
    const mat BtPB = beta * B.t() * P * B;
    const mat BtPA = beta * B.t() * P * A;
    const mat AtPB = beta * A.t() * P * B;
    const mat AtPA = beta * A.t() * P * A;

    const mat S = R + BtPB; // 2x2
    const mat L = N + AtPB; // 3x2  (= N + beta A'PB)
    // L' = N' + beta B'PA, so solve(S, L') = S^{-1}(N' + beta B'PA)
    const mat S_inv_Lt = solve(S, L.t()); // 2x3

    const mat P_new = Q + AtPA - L * S_inv_Lt;

    if (norm(P_new - P, "fro") < tol) {
      return P_new;
    }
    P = P_new;
  }

  // Return best estimate even if not fully converged
  return P;
}

// =====================================================================
//  compute_decision_rules - BW LQ optimal policy
//
//  1. Solve the DARE for the value matrix P
//  2. Extract policy F -> gx = F, hx = A + BF
//  3. Compute BW tax rule coefficients (eqs 2.14-2.17)
// =====================================================================
inline DecisionRules compute_decision_rules(const Params &p) {
  const mat &A = p.A_con;
  const mat &B = p.B_con;
  const mat &R = p.R_cost;
  const mat &N = p.N_cost;

  // Step 1: Solve DARE ----
  const mat P = solve_dare(p);

  // Step 2: Extract policy F ----
  //  F = -(R + beta B'PB)^{-1}(N' + beta B'PA)
  const mat S = R + p.beta_tilde * B.t() * P * B;       // 2x2
  const mat rhs = N.t() + p.beta_tilde * B.t() * P * A; // 2x3
  const mat F = -solve(S, rhs);                         // 2x3

  DecisionRules rules;
  rules.gx = F;
  rules.hx = A + B * F;

  // Step 3: BW eq (2.14) - present-value implementability ----
  //  tildeW_t = [d_c, d_h] gx (I - tildebeta hx)^{-1} v_t := Gamma_w * v_t
  {
    rowvec dcdh = {p.dc, p.dh};
    const mat IminBH = eye<mat>(3, 3) - p.beta_tilde * rules.hx;
    // Gamma_w = (dcdh gx) (I-tildebetahx)^{-1}
    rules.gamma_w = solve(IminBH.t(), (dcdh * rules.gx).t()).t();
  }

  // Step 4: BW eq (2.15) - debt rule ----
  //  Log-linearising (1.21):  uc*k_{t+1} = tildebeta Et[W_{t+1} - b^s_t *
  //  uc_{t+1}] yields:  barb^s_t = (s_c/s_b) Gamma_w*hx*v_t
  //                       - (tildebeta^{-1} s_k/s_b) hx^{(0)}*v_t
  //                       - (tildebeta^{-1} s_k/s_b) A*v_t
  //                       - A*hx*v_t
  //  where A = -sigma^{-1} gx^{(0)} + psi gx^{(1)}
  {
    const double bt_inv = 1.0 / p.beta_tilde;
    const double c1 = p.sc / p.sb;
    const double c2 = bt_inv * p.sk / p.sb;

    rowvec A_row = -p.sigma_inv * rules.gx.row(0) + p.psi_bw * rules.gx.row(1);

    rules.gamma_b = c1 * rules.gamma_w * rules.hx - c2 * rules.hx.row(0) -
                    c2 * A_row - A_row * rules.hx;
  }

  // Step 5: BW eq (2.17) - labour tax rule ----
  //  hattau^h_t = (1-alpha)hatz + alpha hatk - (sigma^{-1} - phi^{-1}psi)hatc -
  //  (alpha+vee-psi)hath
  //         = [alpha, 1-alpha, 0] v + coefficients x gx v
  {
    const double phi_inv_psi = (p.phi_bw > 1e-14) ? p.psi_bw / p.phi_bw : 0.0;
    // BW eq (2.17): hattau^h = hatz + hatalpha_k - (sigma^-1 - phi^-1psi)hatc - (vee
    // - psi)hath where hatalpha_k = hatk - hatz - hath (eq 2.12, capital per effective
    // labor) Expanding: alphahatk + (1-alpha)hatz - (sigma^-1 - phi^-1psi)hatc -
    // (alpha + vee - psi)hath
    rowvec e_kz = {p.alpha, 1.0 - p.alpha, 0.0};
    rules.gamma_tau_h = e_kz - (p.sigma_inv - phi_inv_psi) * rules.gx.row(0) -
                        (p.alpha + p.nu - p.psi_bw) * rules.gx.row(1);
  }

  // Step 6: BW eq (2.16) - capital tax rule ----
  //  hattau^k_t = Gamma^k_tau(0) v_t + Gamma^k_tau(1) v_{t-1}  (BW eq 3.8)
  //
  //  From:  b_tau hattau^k = s_b barb_{t-1} - s_c tildeW_t - b_c hatc + b_h
  //  hath
  //                    + b_k hatk + s_c^{-1} alpha(1-alpha) hatz
  //
  //  Gamma^k_tau(0) v_t part:  (everything involving current v_t)
  //  Gamma^k_tau(1) v_{t-1} part:  s_b Gamma_b * v_{t-1}
  {
    const double inv_bt = (std::abs(p.btau_bw) > 1e-14) ? 1.0 / p.btau_bw : 0.0;

    // Current-state part
    rowvec zterm = {0.0, p.alpha * (1.0 - p.alpha), 0.0};
    rowvec kterm = {p.bk_bw, 0.0, 0.0};

    rules.Gamma_k_0 =
        inv_bt * (-p.sc * rules.gamma_w - p.bc_bw * rules.gx.row(0) +
                  p.bh_bw * rules.gx.row(1) + kterm + zterm);

    // Lagged-state part
    rules.Gamma_k_1 = inv_bt * p.sb * rules.gamma_b;
  }

  // Step 7: Ex-ante capital tax (linear first-order) ----
  //  theta^e_t = E_t[tau^k_{t+1}] = Gamma^k_0 hx v_t + Gamma^k_1 v_t
  //  BW Table 3: analytically zero mean for all phi.
  //  BW Table 5: MC simulation of this linear rule.
  rules.gamma_theta_e = rules.Gamma_k_0 * rules.hx + rules.Gamma_k_1;

  return rules;
}

// =====================================================================
//  simulate - BW Table 5 Monte Carlo (first-order, continuous AR(1))
//
//  BW footnote 18: epsilon^x_t %in% {+delta_x, -delta_x} with equal probability,
//  delta_x = sigma_x sqrt(1-rho^2_x), giving unconditional Var(x) = sigma^2_x.
//
//  BW footnote 31: T_total = 500,000; T_burn = 60,000.
//
//  Returns (T_keep x 6) matrix:
//    col 0: tau^h   col 1: theta^e   col 2: tau^k
//    col 3: theta^{nc} (non-contingent debt)
//    col 4: hatz_t   col 5: hatg_t
//
//  At first order:
//    E[tau^h] ~= bartau^h,  E[tau^k] = 0 analytically.
// =====================================================================
inline mat simulate(const Params &p, const DecisionRules &rules, int T_total,
                    int T_burn, int seed) {
  if (T_burn >= T_total)
    throw std::invalid_argument("simulate: T_burn must be < T_total");

  // BW innovation sizes (footnote 18)
  const double delta_z = p.sigma_z * std::sqrt(1.0 - p.rho_z * p.rho_z);
  const double delta_g = p.sigma_g * std::sqrt(1.0 - p.rho_g * p.rho_g);

  std::mt19937 rng(static_cast<unsigned>(seed));
  std::bernoulli_distribution flip(0.5);

  // State variables (initialise at NSSS: all deviations zero)
  double k_hat = 0.0, z = 0.0, g_hat = 0.0;
  double k_hat_p = 0.0, z_p = 0.0, g_hat_p = 0.0; // lagged state

  const int T_keep = T_total - T_burn;
  mat out(T_keep, 6, fill::zeros);

  for (int t = 0; t < T_total; ++t) {
    if (t >= T_burn) {
      const int row = t - T_burn;

      // col 0: tau^h - BW eq (2.17) + level conversion
      //  hattau^h is a log-like deviation, tau^h = bartau^h + (1-bartau^h)hattau^h
      const double tau_hat_h = rules.gamma_tau_h(0) * k_hat +
                               rules.gamma_tau_h(1) * z +
                               rules.gamma_tau_h(2) * g_hat;
      out(row, 0) = p.tau_h_ss + (1.0 - p.tau_h_ss) * tau_hat_h;

      // col 1: theta^e - BW eq (3.9) at first order
      //  theta^e_t = E_t[tau^k_{t+1}] = (Gamma^k_0 hx + Gamma^k_1) v_t
      //  BW Table 3: analytically zero unconditional mean for all phi.
      //  BW Table 5: MC of first-order rules gives E~=0 (0.002 is MC noise).
      out(row, 1) = rules.gamma_theta_e(0) * k_hat +
                    rules.gamma_theta_e(1) * z + rules.gamma_theta_e(2) * g_hat;

      // col 2: tau^k - BW eq (3.8) = Gamma^k_tau(0) v_t + Gamma^k_tau(1)
      // v_{t-1}
      out(row, 2) = rules.Gamma_k_0(0) * k_hat + rules.Gamma_k_0(1) * z +
                    rules.Gamma_k_0(2) * g_hat + rules.Gamma_k_1(0) * k_hat_p +
                    rules.Gamma_k_1(1) * z_p + rules.Gamma_k_1(2) * g_hat_p;

      // col 3: theta^{nc} - non-contingent debt (barb_{t-1} = 0)
      //  Same as tau^k but drop Gamma^k_tau(1) v_{t-1} term
      out(row, 3) = rules.Gamma_k_0(0) * k_hat + rules.Gamma_k_0(1) * z +
                    rules.Gamma_k_0(2) * g_hat;

      out(row, 4) = z;     // hatz_t
      out(row, 5) = g_hat; // hatg_t = log(g_t/barg)
    }

    // Save current state for next period's barb_{t-1}
    k_hat_p = k_hat;
    z_p = z;
    g_hat_p = g_hat;

    // Advance: capital from linear rule, shocks from AR(1)
    const double k_hat_next =
        rules.hx(0, 0) * k_hat + rules.hx(0, 1) * z + rules.hx(0, 2) * g_hat;
    const double z_next = p.rho_z * z + (flip(rng) ? delta_z : -delta_z);
    const double g_hat_next =
        p.rho_g * g_hat + (flip(rng) ? delta_g : -delta_g);

    k_hat = k_hat_next;
    z = z_next;
    g_hat = g_hat_next;
  }

  return out;
}

// =====================================================================
//  Pearson correlation (NaN for constant series)
// =====================================================================
static double safe_cor(const vec &a, const vec &b) {
  const double sa = stddev(a), sb = stddev(b);
  if (sa < 1e-15 || sb < 1e-15)
    return datum::nan;
  return dot(a - mean(a), b - mean(b)) / ((double)(a.n_elem - 1) * sa * sb);
}

// =====================================================================
//  simulation_stats - BW Tables 5/6 summary statistics
//
//  Input: (T x 6) simulation matrix (pre-scaling units)
//  Output: (5 x 4) matrix
//    rows: E | sd | autocorr | corr_g | corr_z
//    cols: tau^h | theta^e | tau^k | theta^{nc}
// =====================================================================
static mat simulation_stats(const mat &sim) {
  const int T = (int)sim.n_rows;
  mat out(5, 4);
  out.fill(datum::nan);

  const vec z_hat = sim.col(4);
  const vec g_hat = sim.col(5);

  for (int j = 0; j < 4; ++j) {
    const vec x = sim.col(j);
    out(0, j) = mean(x);
    out(1, j) = stddev(x);
    if (T > 1)
      out(2, j) = safe_cor(x.head(T - 1), x.tail(T - 1));
    out(3, j) = safe_cor(x, g_hat);
    out(4, j) = safe_cor(x, z_hat);
  }
  return out;
}
