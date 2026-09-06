#pragma once
#include "01_parameters.h"
#include "04_nonlinear_common.h"
#include "05_time_iteration.h" // for solve_hours
#include <armadillo4r.hpp>
#include <chrono>
#include <cmath>

using namespace arma;

// =====================================================================
//  Envelope Condition Method (ECM) - Maliar & Maliar (2013)
//
//  Uses the envelope condition instead of FOCs to identify policies.
//
//  Envelope condition:
//    V_k(k, z) = u_c(c, h) * [1 - delta + z * f_k(k, h)]
//
//  Key relation (eq 8 in M&M 2013):
//    V_k(k, z) = beta * [1 - delta + z * f_k(k, h)] * E[V_k(k', z')]
//
//  This allows us to iterate on V_k directly, which is often simpler
//  than solving the Euler equation.
//
//  Algorithm:
//    1. Initialize V_k on the grid
//    2. For each (k, z, g):
//       a. Compute E[V_k(k', z', g')]
//       b. From envelope: V_k = beta * [1-delta+fk] * E[V_k']
//       c. Recover c from: u_c = V_k / [1-delta+fk]
//       d. Solve for h from labor FOC
//       e. Update k' from resource constraint
//    3. Update V_k from envelope condition
//    4. Iterate until convergence
// =====================================================================

// =====================================================================
//  ecm_iteration_step - Single ECM iteration
//
//  Uses LQ tax formula for state-dependent tau_h (BW eq 2.17).
// =====================================================================
inline double ecm_iteration_step(const NonlinearModel &model,
                                 const GridSpec &grid, PolicyFunctions &pol,
                                 const DecisionRules &rules, double tau_h_ss) {
  const int n_k = grid.n_k;
  const int n_z = grid.n_z;
  const int n_g = grid.n_g;

  // New policy storage
  cube V_k_new(n_k, n_z, n_g);
  cube k_prime_new(n_k, n_z, n_g);
  cube c_new(n_k, n_z, n_g);
  cube h_new(n_k, n_z, n_g);
  cube tau_h_new(n_k, n_z, n_g);

  double max_diff = 0.0;

  for (int i_g = 0; i_g < n_g; ++i_g) {
    double g_hat = grid.g_grid(i_g);
    double g = std::exp(g_hat) * model.g_ss;

    for (int i_z = 0; i_z < n_z; ++i_z) {
      double z_hat = grid.z_grid(i_z);
      double z = std::exp(z_hat);

      for (int i_k = 0; i_k < n_k; ++i_k) {
        double k = grid.k_grid(i_k);
        double k_hat = std::log(k / model.k_ss);

        // State-dependent labor tax from LQ formula (BW eq 2.17)
        double tau_h_hat = rules.gamma_tau_h(0) * k_hat +
                           rules.gamma_tau_h(1) * z_hat +
                           rules.gamma_tau_h(2) * g_hat;
        double tau_h = tau_h_ss + (1.0 - tau_h_ss) * tau_h_hat;
        tau_h_new(i_k, i_z, i_g) = tau_h;

        // Current k' from previous iteration
        double k_p = pol.k_prime(i_k, i_z, i_g);

        // Compute E[V_k(k', z', g')] by interpolation
        double E_Vk = 0.0;

        for (int j_g = 0; j_g < n_g; ++j_g) {
          double prob_g = grid.P_g(i_g, j_g);

          for (int j_z = 0; j_z < n_z; ++j_z) {
            double prob_z = grid.P_z(i_z, j_z);
            double prob = prob_g * prob_z;

            // Interpolate V_k at k'
            double Vk_p = pol.interp_V_k(grid.k_grid, j_z, j_g, k_p);
            E_Vk += prob * Vk_p;
          }
        }

        // Current hours (from previous iteration)
        double h = pol.h(i_k, i_z, i_g);
        if (h <= 0 || h >= 1)
          h = model.h_ss;

        // Marginal product
        double fk = model.f_k(k, z, h);
        double Rk = 1.0 - model.delta + fk;

        // ECM equation (8): V_k = beta * Rk * E[V_k']
        double Vk_new = model.beta * Rk * E_Vk;

        // Recover u_c from envelope: V_k = u_c * Rk
        // => u_c = V_k / Rk
        double uc_target = Vk_new / Rk;
        if (uc_target <= 0)
          uc_target = model.u_c(model.c_ss, model.h_ss);

        // Solve for c given u_c and h
        double c = model.c_from_uc(uc_target, h);
        if (c <= 0)
          c = model.c_ss;

        // Update h from household FOC with state-dependent tau_h
        for (int inner = 0; inner < 5; ++inner) {
          double h_old = h;
          h = solve_hours(model, c, k, z, tau_h);
          c = model.c_from_uc(uc_target, h);
          if (std::abs(h - h_old) < 1e-6)
            break;
        }

        // tau_h already set from LQ formula above

        // Update k' from resource constraint
        double y = model.f(k, z, h);
        double k_p_new = (1.0 - model.delta) * k + y - c - g;
        k_p_new = std::max(grid.k_min, std::min(grid.k_max, k_p_new));

        V_k_new(i_k, i_z, i_g) = Vk_new;
        k_prime_new(i_k, i_z, i_g) = k_p_new;
        c_new(i_k, i_z, i_g) = c;
        h_new(i_k, i_z, i_g) = h;

        double diff = std::abs(k_p_new - pol.k_prime(i_k, i_z, i_g));
        if (diff > max_diff)
          max_diff = diff;
      }
    }
  }

  // Update with damping
  const double damp = 0.5;
  pol.V_k = (1.0 - damp) * pol.V_k + damp * V_k_new;
  pol.k_prime = (1.0 - damp) * pol.k_prime + damp * k_prime_new;
  pol.c = (1.0 - damp) * pol.c + damp * c_new;
  pol.h = (1.0 - damp) * pol.h + damp * h_new;
  pol.tau_h = tau_h_new; // No damping - direct update from LQ formula

  return max_diff;
}

// =====================================================================
//  ecm_solve - Full ECM solver
//
//  Solves the Ramsey planner's problem using envelope condition method.
//  Tax rates are implied by allocations, not computed from LQ formulas.
// =====================================================================
inline SolverStats ecm_solve(const Params &p, const GridSpec &grid,
                             PolicyFunctions &pol, int max_iter = 500,
                             double tol = 1e-6) {
  auto start = std::chrono::high_resolution_clock::now();

  NonlinearModel model(p);

  // Initialize
  pol.initialize(grid.n_k, grid.n_z, grid.n_g);

  // Initial V_k from steady state
  double c_ss = model.c_ss;
  double h_ss = model.h_ss;
  double uc_ss = model.u_c(c_ss, h_ss);

  for (int i_g = 0; i_g < grid.n_g; ++i_g) {
    for (int i_z = 0; i_z < grid.n_z; ++i_z) {
      double z = std::exp(grid.z_grid(i_z));
      for (int i_k = 0; i_k < grid.n_k; ++i_k) {
        double k = grid.k_grid(i_k);
        double fk = model.f_k(k, z, h_ss);
        double Rk = 1.0 - model.delta + fk;

        pol.k_prime(i_k, i_z, i_g) = k;
        pol.c(i_k, i_z, i_g) = c_ss;
        pol.h(i_k, i_z, i_g) = h_ss;
        pol.V_k(i_k, i_z, i_g) = uc_ss * Rk;
        pol.tau_h(i_k, i_z, i_g) = p.tau_h_ss;
      }
    }
  }

  // Compute LQ decision rules for state-dependent tau_h
  DecisionRules rules = compute_decision_rules(p);

  SolverStats stats;
  stats.converged = false;

  for (int iter = 0; iter < max_iter; ++iter) {
    double err = ecm_iteration_step(model, grid, pol, rules, p.tau_h_ss);

    if (err < tol) {
      stats.converged = true;
      stats.iterations = iter + 1;
      stats.error = err;
      break;
    }

    stats.iterations = iter + 1;
    stats.error = err;
  }

  // tau_h already computed from allocations during iteration
  // Set capital tax to zero (ex ante BW result)
  pol.tau_k.zeros();

  auto end = std::chrono::high_resolution_clock::now();
  stats.elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return stats;
}
