#pragma once
#include "01_parameters.h"
#include "04_nonlinear_common.h"
#include "05_time_iteration.h" // for solve_hours
#include <algorithm>
#include <armadillo4r.hpp>
#include <chrono>
#include <cmath>

using namespace arma;

// =====================================================================
//  Endogenous Grid Method (EGM) - Carroll (2005)
//
//  Key insight: Fix k' (future capital) as the grid variable,
//  and solve for k (current capital) as an endogenous outcome.
//
//  This avoids root-finding in the Euler equation.
//
//  Algorithm:
//    1. Create grid on k' (future endogenous state)
//    2. For each (k', z, g):
//       a. Compute E[u_c(k', z', g') * (1 + R_k')]
//       b. From Euler: u_c(c,h) = beta * E[...] => solve for c directly
//       c. From resource: y = c + g + k' - (1-delta)k => solve for k
//    3. Interpolate back to exogenous k grid
//    4. Iterate until convergence
// =====================================================================

// Helper: Given k' and (c, h), solve for implied current k
// From resource constraint: k^alpha(zh)^{1-alpha} = c + g + k' - (1-delta)k
// This is nonlinear in k, use Newton or bisection
inline double solve_k_from_kprime(const NonlinearModel &model, double k_prime,
                                  double c, double h, double z, double g,
                                  double k_min, double k_max,
                                  double tol = 1e-10, int max_iter = 50) {
  // Residual: f(k, zh) - c - g - k_prime + (1-delta)k = 0
  // Or: k^alpha(zh)^{1-alpha} + (1-delta)k - (c + g + k_prime) = 0

  auto residual = [&](double k) {
    return model.f(k, z, h) + (1.0 - model.delta) * k - (c + g + k_prime);
  };

  // Bisection search
  double k_lo = k_min;
  double k_hi = k_max;
  double res_lo = residual(k_lo);
  double res_hi = residual(k_hi);

  // Check if solution exists
  if (res_lo * res_hi > 0) {
    // No sign change - return midpoint
    return 0.5 * (k_lo + k_hi);
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    double k_mid = 0.5 * (k_lo + k_hi);
    double res_mid = residual(k_mid);

    if (std::abs(res_mid) < tol || (k_hi - k_lo) < tol) {
      return k_mid;
    }

    if (res_mid * res_lo < 0) {
      k_hi = k_mid;
      res_hi = res_mid;
    } else {
      k_lo = k_mid;
      res_lo = res_mid;
    }
  }

  return 0.5 * (k_lo + k_hi);
}

// =====================================================================
//  egm_iteration_step - Single EGM iteration
//
//  Uses LQ tax formula for state-dependent tau_h (BW eq 2.17).
// =====================================================================
inline double egm_iteration_step(const NonlinearModel &model,
                                 const GridSpec &grid, PolicyFunctions &pol,
                                 const DecisionRules &rules, double tau_h_ss) {
  const int n_k = grid.n_k;
  const int n_z = grid.n_z;
  const int n_g = grid.n_g;

  // We use k' grid as the fixed grid (same as k grid for simplicity)
  // Then solve for endogenous (k, c, h)

  // Temporary storage for endogenous grid results
  cube k_endo(n_k, n_z, n_g); // endogenous current capital
  cube c_endo(n_k, n_z, n_g); // consumption at (k_endo, z, g)
  cube h_endo(n_k, n_z, n_g); // hours at (k_endo, z, g)

  for (int i_g = 0; i_g < n_g; ++i_g) {
    double g_hat = grid.g_grid(i_g);
    double g = std::exp(g_hat) * model.g_ss;

    for (int i_z = 0; i_z < n_z; ++i_z) {
      double z_hat = grid.z_grid(i_z);
      double z = std::exp(z_hat);

      for (int i_kp = 0; i_kp < n_k; ++i_kp) {
        // k' is fixed at grid point
        double k_prime = grid.k_grid(i_kp);

        // State-dependent tau_h from LQ formula for current state estimate
        // Note: This is approximate since k is endogenous in EGM
        double k_hat_est = std::log(grid.k_grid(i_kp) / model.k_ss);
        double tau_h_hat = rules.gamma_tau_h(0) * k_hat_est +
                           rules.gamma_tau_h(1) * z_hat +
                           rules.gamma_tau_h(2) * g_hat;
        double tau_h = tau_h_ss + (1.0 - tau_h_ss) * tau_h_hat;

        // Compute E[u_c' * (1 + R_k')]
        double E_term = 0.0;

        for (int j_g = 0; j_g < n_g; ++j_g) {
          double prob_g = grid.P_g(i_g, j_g);

          for (int j_z = 0; j_z < n_z; ++j_z) {
            double prob_z = grid.P_z(i_z, j_z);
            double prob = prob_g * prob_z;

            double z_p = std::exp(grid.z_grid(j_z));

            // Tomorrow's variables at (k', z', g')
            double c_p = pol.c(i_kp, j_z, j_g);
            double h_p = pol.h(i_kp, j_z, j_g);

            if (c_p <= 0)
              c_p = model.c_ss;
            if (h_p <= 0 || h_p >= 1)
              h_p = model.h_ss;

            double uc_p = model.u_c(c_p, h_p);
            double fk_p = model.f_k(k_prime, z_p, h_p);

            // For Ramsey: tau^k' = 0 (ex ante zero)
            double Rk_p = 1.0 + (fk_p - model.delta);

            E_term += prob * uc_p * Rk_p;
          }
        }

        // From Euler: u_c = beta * E_term
        double uc_target = model.beta * E_term;

        // In EGM, k' is fixed and we solve for (k, c, h) simultaneously
        // Initial guesses
        double h_val = pol.h(i_kp, i_z, i_g);
        if (h_val <= 0 || h_val >= 1)
          h_val = model.h_ss;
        double c_val = model.c_from_uc(uc_target, h_val);
        if (c_val <= 0)
          c_val = model.c_ss;

        // Iterate to find consistent (k, c, h) given k'
        double k_val = grid.k_grid(i_kp); // initial guess
        for (int inner = 0; inner < 15; ++inner) {
          // Update h from household FOC with state-dependent tau_h
          double h_new = solve_hours(model, c_val, k_val, z, tau_h);

          // Update c from Euler given new h
          double c_new = model.c_from_uc(uc_target, h_new);
          if (c_new <= 0)
            c_new = model.c_ss;

          // Update k from resource constraint given new (c, h)
          double k_new = solve_k_from_kprime(model, k_prime, c_new, h_new, z, g,
                                             grid.k_min, grid.k_max);

          // Check convergence
          if (std::abs(k_new - k_val) < 1e-8 &&
              std::abs(h_new - h_val) < 1e-8) {
            k_val = k_new;
            c_val = c_new;
            h_val = h_new;
            break;
          }

          // Damped update
          k_val = 0.5 * k_val + 0.5 * k_new;
          c_val = 0.5 * c_val + 0.5 * c_new;
          h_val = 0.5 * h_val + 0.5 * h_new;
        }

        k_endo(i_kp, i_z, i_g) = k_val;
        c_endo(i_kp, i_z, i_g) = c_val;
        h_endo(i_kp, i_z, i_g) = h_val;
      }
    }
  }

  // Interpolate back to exogenous k grid
  cube k_prime_new(n_k, n_z, n_g);
  cube c_new(n_k, n_z, n_g);
  cube h_new(n_k, n_z, n_g);

  double max_diff = 0.0;

  for (int i_g = 0; i_g < n_g; ++i_g) {
    for (int i_z = 0; i_z < n_z; ++i_z) {
      // Get the endogenous k values for this (z, g)
      vec k_endo_vec = k_endo.slice(i_g).col(i_z);
      vec k_prime_vec = grid.k_grid; // k' grid values
      vec c_endo_vec = c_endo.slice(i_g).col(i_z);
      vec h_endo_vec = h_endo.slice(i_g).col(i_z);

      // Sort by k_endo for interpolation
      uvec sort_idx = sort_index(k_endo_vec);

      for (int i_k = 0; i_k < n_k; ++i_k) {
        double k = grid.k_grid(i_k);

        // Find where k fits in sorted k_endo
        int i_lo = 0;
        while (i_lo < n_k - 2 && k_endo_vec(sort_idx(i_lo + 1)) < k)
          ++i_lo;
        int i_hi = std::min(i_lo + 1, n_k - 1);

        // Linear interpolation
        double k_lo = k_endo_vec(sort_idx(i_lo));
        double k_hi = k_endo_vec(sort_idx(i_hi));

        double w = 0.5;
        if (std::abs(k_hi - k_lo) > 1e-10) {
          w = (k - k_lo) / (k_hi - k_lo);
          w = std::max(0.0, std::min(1.0, w));
        }

        double kp_interp = (1.0 - w) * k_prime_vec(sort_idx(i_lo)) +
                           w * k_prime_vec(sort_idx(i_hi));
        double c_interp = (1.0 - w) * c_endo_vec(sort_idx(i_lo)) +
                          w * c_endo_vec(sort_idx(i_hi));
        double h_interp = (1.0 - w) * h_endo_vec(sort_idx(i_lo)) +
                          w * h_endo_vec(sort_idx(i_hi));

        // Bound k'
        kp_interp = std::max(grid.k_min, std::min(grid.k_max, kp_interp));
        c_interp = std::max(0.01, c_interp);
        h_interp = std::max(0.01, std::min(0.99, h_interp));

        k_prime_new(i_k, i_z, i_g) = kp_interp;
        c_new(i_k, i_z, i_g) = c_interp;
        h_new(i_k, i_z, i_g) = h_interp;

        double diff = std::abs(kp_interp - pol.k_prime(i_k, i_z, i_g));
        if (diff > max_diff)
          max_diff = diff;
      }
    }
  }

  // Update with damping
  const double damp = 0.5;
  pol.k_prime = (1.0 - damp) * pol.k_prime + damp * k_prime_new;
  pol.c = (1.0 - damp) * pol.c + damp * c_new;
  pol.h = (1.0 - damp) * pol.h + damp * h_new;

  return max_diff;
}

// =====================================================================
//  egm_solve - Full EGM solver
//
//  Solves the Ramsey planner's problem using endogenous grid method.
//  Tax rates are implied by allocations, not computed from LQ formulas.
// =====================================================================
inline SolverStats egm_solve(const Params &p, const GridSpec &grid,
                             PolicyFunctions &pol, int max_iter = 500,
                             double tol = 1e-6) {
  auto start = std::chrono::high_resolution_clock::now();

  NonlinearModel model(p);

  // Initialize policies at steady state
  pol.initialize(grid.n_k, grid.n_z, grid.n_g);

  for (int i_g = 0; i_g < grid.n_g; ++i_g) {
    for (int i_z = 0; i_z < grid.n_z; ++i_z) {
      for (int i_k = 0; i_k < grid.n_k; ++i_k) {
        pol.k_prime(i_k, i_z, i_g) = grid.k_grid(i_k);
        pol.c(i_k, i_z, i_g) = model.c_ss;
        pol.h(i_k, i_z, i_g) = model.h_ss;
        pol.tau_h(i_k, i_z, i_g) = p.tau_h_ss;
      }
    }
  }

  // Compute LQ decision rules for state-dependent tau_h
  DecisionRules rules = compute_decision_rules(p);

  SolverStats stats;
  stats.converged = false;

  for (int iter = 0; iter < max_iter; ++iter) {
    double err = egm_iteration_step(model, grid, pol, rules, p.tau_h_ss);

    if (err < tol) {
      stats.converged = true;
      stats.iterations = iter + 1;
      stats.error = err;
      break;
    }

    stats.iterations = iter + 1;
    stats.error = err;
  }

  // Compute tax rates using LQ formula
  for (int i_g = 0; i_g < grid.n_g; ++i_g) {
    double g_hat = grid.g_grid(i_g);
    for (int i_z = 0; i_z < grid.n_z; ++i_z) {
      double z_hat = grid.z_grid(i_z);
      for (int i_k = 0; i_k < grid.n_k; ++i_k) {
        double k = grid.k_grid(i_k);
        double k_hat = std::log(k / model.k_ss);
        double tau_h_hat = rules.gamma_tau_h(0) * k_hat +
                           rules.gamma_tau_h(1) * z_hat +
                           rules.gamma_tau_h(2) * g_hat;
        pol.tau_h(i_k, i_z, i_g) = p.tau_h_ss + (1.0 - p.tau_h_ss) * tau_h_hat;
        pol.tau_k(i_k, i_z, i_g) = 0.0;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  stats.elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return stats;
}
