#pragma once
#include "01_parameters.h"
#include "02_solver.h"
#include "04_nonlinear_common.h"
#include <armadillo4r.hpp>
#include <chrono>
#include <cmath>

using namespace arma;

// =====================================================================
//  Time Iteration (Policy Function Iteration) Solver
//
//  Iterates on the Euler equation:
//    u_c(c,h) = beta E[u_c(c',h') * (1 + (1-tau^k')(f_k' - delta))]
//
//  For the CCK94/BW Ramsey problem, we solve for optimal allocations
//  {c, h, k'} and then back out tax rates from FOCs.
//
//  Algorithm:
//    1. Initialize policy functions (c, h, k') on grid
//    2. For each (k, z, g):
//       a. Compute E[u_c' * (1 + R_k')] using current policies
//       b. Solve for (c, h) from Euler + labor FOC
//       c. Update k' from resource constraint
//    3. Repeat until ||k'_new - k'_old|| < tol
// =====================================================================

// Helper: solve for hours given consumption, capital, and productivity
// From labor FOC: -u_h/u_c = (1-tau^h) * w where w = f_h
// For log utility: gamma/(1-h) / [(1-gamma)/c] = (1-tau^h) * f_h
// Simplifies to: gammac / [(1-gamma)(1-h)] = (1-tau^h) * f_h
//
// For the Ramsey planner, set tau_h = 0 to get efficient allocation
inline double solve_hours(const NonlinearModel &m, double c, double k, double z,
                          double tau_h, double tol = 1e-10,
                          int max_iter = 100) {
  // Newton iteration to solve for h
  double h = m.h_ss; // initial guess

  for (int iter = 0; iter < max_iter; ++iter) {
    double fh = m.f_h(k, z, h);

    // Residual: -u_h/u_c - (1-tau^h) * f_h = 0
    // For log utility: gammac/[(1-gamma)(1-h)] - (1-tau^h)f_h = 0
    double lhs = m.gamma * c / ((1.0 - m.gamma) * (1.0 - h));
    double rhs = (1.0 - tau_h) * fh;
    double res = lhs - rhs;

    if (std::abs(res) < tol)
      break;

    // Derivative w.r.t. h
    // d(lhs)/dh = gammac/[(1-gamma)(1-h)^2]
    // d(rhs)/dh = (1-tau^h) d(f_h)/dh (complicated, use finite diff)
    const double dh = 1e-6;
    double fh_p = m.f_h(k, z, h + dh);
    double lhs_p = m.gamma * c / ((1.0 - m.gamma) * (1.0 - h - dh));
    double rhs_p = (1.0 - tau_h) * fh_p;
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

// Helper: solve for consumption given hours and Euler equation RHS
// Euler: u_c(c,h) = rhs  =>  c = u_c^{-1}(rhs, h)
inline double solve_consumption(const NonlinearModel &m, double rhs, double h) {
  return m.c_from_uc(rhs, h);
}

// =====================================================================
//  time_iteration_step - Single iteration of time iteration
//
//  Solves competitive equilibrium with constant tau_h = tau_h_ss.
//  For log utility, this approximates the Ramsey allocation (tax smoothing).
//  Tax rates are backed out from allocations using tau_h_from_foc.
// =====================================================================
inline double time_iteration_step(const NonlinearModel &model,
                                  const GridSpec &grid, PolicyFunctions &pol,
                                  const DecisionRules &rules, double tau_h_ss) {
  const int n_k = grid.n_k;
  const int n_z = grid.n_z;
  const int n_g = grid.n_g;

  // Store new policies
  cube k_prime_new(n_k, n_z, n_g);
  cube c_new(n_k, n_z, n_g);
  cube h_new(n_k, n_z, n_g);
  cube tau_h_new(n_k, n_z, n_g);

  double max_diff = 0.0;

  for (int i_g = 0; i_g < n_g; ++i_g) {
    double g_hat = grid.g_grid(i_g);
    double g = std::exp(g_hat) * model.g_ss; // level of govt spending

    for (int i_z = 0; i_z < n_z; ++i_z) {
      double z_hat = grid.z_grid(i_z);
      double z = std::exp(z_hat); // technology level

      for (int i_k = 0; i_k < n_k; ++i_k) {
        double k = grid.k_grid(i_k);
        double k_hat = std::log(k / model.k_ss);

        // State-dependent labor tax from LQ formula (BW eq 2.17)
        double tau_h_hat = rules.gamma_tau_h(0) * k_hat +
                           rules.gamma_tau_h(1) * z_hat +
                           rules.gamma_tau_h(2) * g_hat;
        double tau_h = tau_h_ss + (1.0 - tau_h_ss) * tau_h_hat;
        tau_h_new(i_k, i_z, i_g) = tau_h;

        // Compute expectation E[u_c' * (1 + (1-tau^k')(f_k' - delta))]
        double E_term = 0.0;

        for (int j_g = 0; j_g < n_g; ++j_g) {
          double prob_g = grid.P_g(i_g, j_g);

          for (int j_z = 0; j_z < n_z; ++j_z) {
            double prob_z = grid.P_z(i_z, j_z);
            double prob = prob_g * prob_z;

            double z_hat_p = grid.z_grid(j_z);
            double z_p = std::exp(z_hat_p);

            // Tomorrow's capital (from today's policy)
            double k_p = pol.k_prime(i_k, i_z, i_g);

            // Interpolate tomorrow's consumption and hours at k'
            double c_p = pol.interp_c(grid.k_grid, j_z, j_g, k_p);
            double h_p = pol.interp_h(grid.k_grid, j_z, j_g, k_p);
            if (c_p <= 0)
              c_p = model.c_ss;
            if (h_p <= 0 || h_p >= 1)
              h_p = model.h_ss;

            // For Ramsey problem, assume tau^k' = 0 in expectation
            // (this is the "ex ante zero" result from BW)
            double tau_k_p = 0.0;

            double uc_p = model.u_c(c_p, h_p);
            double fk_p = model.f_k(k_p, z_p, h_p);
            double Rk_p = 1.0 + (1.0 - tau_k_p) * (fk_p - model.delta);

            E_term += prob * uc_p * Rk_p;
          }
        }

        // Euler equation: u_c(c,h) = beta * E_term
        double uc_target = model.beta * E_term;

        // Guess hours, solve system
        double h_guess = pol.h(i_k, i_z, i_g);
        if (h_guess <= 0 || h_guess >= 1)
          h_guess = 0.33;

        // Iterate to find consistent (c, h) using state-dependent tau_h from LQ
        double c_val, h_val;
        for (int inner = 0; inner < 20; ++inner) {
          c_val = solve_consumption(model, uc_target, h_guess);
          if (c_val <= 0)
            c_val = 0.1;

          // Solve for hours using STATE-DEPENDENT tau_h from LQ formula
          h_val = solve_hours(model, c_val, k, z, tau_h);
          if (std::abs(h_val - h_guess) < 1e-6)
            break;
          h_guess = 0.5 * h_guess + 0.5 * h_val;
        }

        // tau_h already computed from LQ formula above

        // Resource constraint: k' = (1-delta)k + y - c - g
        double y = model.f(k, z, h_val);
        double k_p_new = (1.0 - model.delta) * k + y - c_val - g;

        // Ensure k' stays in bounds
        k_p_new = std::max(grid.k_min, std::min(grid.k_max, k_p_new));

        k_prime_new(i_k, i_z, i_g) = k_p_new;
        c_new(i_k, i_z, i_g) = c_val;
        h_new(i_k, i_z, i_g) = h_val;

        double diff = std::abs(k_p_new - pol.k_prime(i_k, i_z, i_g));
        if (diff > max_diff)
          max_diff = diff;
      }
    }
  }

  // Update policies with damping
  const double damp = 0.3;
  pol.k_prime = (1.0 - damp) * pol.k_prime + damp * k_prime_new;
  pol.c = (1.0 - damp) * pol.c + damp * c_new;
  pol.h = (1.0 - damp) * pol.h + damp * h_new;
  pol.tau_h = tau_h_new; // tau_h from LQ formula - no damping needed

  return max_diff;
}

// =====================================================================
//  time_iteration_solve - Full time iteration solver
//
//  Uses LQ tax formula for state-dependent tau_h, then solves nonlinear
//  Euler for allocations. This matches BW Tables 4/5 methodology.
// =====================================================================
inline SolverStats time_iteration_solve(const Params &p, const GridSpec &grid,
                                        PolicyFunctions &pol,
                                        int max_iter = 1000,
                                        double tol = 1e-6) {
  auto start = std::chrono::high_resolution_clock::now();

  NonlinearModel model(p);

  // Compute LQ decision rules for state-dependent tau_h
  DecisionRules rules = compute_decision_rules(p);

  // Initialize policies at steady state
  pol.initialize(grid.n_k, grid.n_z, grid.n_g);

  for (int i_g = 0; i_g < grid.n_g; ++i_g) {
    for (int i_z = 0; i_z < grid.n_z; ++i_z) {
      for (int i_k = 0; i_k < grid.n_k; ++i_k) {
        pol.k_prime(i_k, i_z, i_g) = grid.k_grid(i_k); // k' = k (steady state)
        pol.c(i_k, i_z, i_g) = model.c_ss;
        pol.h(i_k, i_z, i_g) = model.h_ss;
        pol.tau_h(i_k, i_z, i_g) = p.tau_h_ss;
      }
    }
  }

  SolverStats stats;
  stats.converged = false;

  for (int iter = 0; iter < max_iter; ++iter) {
    double err = time_iteration_step(model, grid, pol, rules, p.tau_h_ss);

    if (err < tol) {
      stats.converged = true;
      stats.iterations = iter + 1;
      stats.error = err;
      break;
    }

    stats.iterations = iter + 1;
    stats.error = err;
  }

  // Set capital tax to zero (ex ante BW result)
  pol.tau_k.zeros();

  auto end = std::chrono::high_resolution_clock::now();
  stats.elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return stats;
}
