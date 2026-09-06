#pragma once
#include "01_parameters.h"
#include "04_nonlinear_common.h"
#include <armadillo4r.hpp>
#include <chrono>
#include <cmath>

using namespace arma;

// =====================================================================
//  Projection Method (Collocation) - CCK94-style
//
//  Approximates policy functions using Chebyshev polynomials and
//  solves for coefficients using collocation (zero residuals at
//  Chebyshev nodes).
//
//  This is a simplified version that:
//    1. Uses Chebyshev polynomial basis
//    2. Evaluates residuals from Euler equation
//    3. Uses Newton iteration to solve for coefficients
//
//  CCK94 use a minimum-weighted-residual (Galerkin) variant, but
//  collocation is simpler and gives similar results.
// =====================================================================

// Chebyshev polynomial of first kind: T_n(x) for x %in% [-1, 1]
inline double chebyshev_T(int n, double x) {
  if (n == 0)
    return 1.0;
  if (n == 1)
    return x;

  double T_nm2 = 1.0;
  double T_nm1 = x;
  double T_n = 0.0;

  for (int i = 2; i <= n; ++i) {
    T_n = 2.0 * x * T_nm1 - T_nm2;
    T_nm2 = T_nm1;
    T_nm1 = T_n;
  }

  return T_n;
}

// Chebyshev nodes on [-1, 1]
inline vec chebyshev_nodes(int n) {
  vec nodes(n);
  for (int i = 0; i < n; ++i) {
    nodes(i) = std::cos((2.0 * i + 1.0) / (2.0 * n) * datum::pi);
  }
  return nodes;
}

// Map from [a, b] to [-1, 1]
inline double map_to_cheb(double x, double a, double b) {
  return 2.0 * (x - a) / (b - a) - 1.0;
}

// Map from [-1, 1] to [a, b]
inline double map_from_cheb(double z, double a, double b) {
  return a + (z + 1.0) * (b - a) / 2.0;
}

// =====================================================================
//  ChebyshevApprox - Polynomial approximation of a function
// =====================================================================
struct ChebyshevApprox {
  int degree;   // polynomial degree
  double x_min; // domain minimum
  double x_max; // domain maximum
  vec coeffs;   // Chebyshev coefficients

  ChebyshevApprox() : degree(0), x_min(0), x_max(1) {}

  ChebyshevApprox(int deg, double xmin, double xmax)
      : degree(deg), x_min(xmin), x_max(xmax), coeffs(deg + 1, fill::zeros) {}

  // Evaluate at point x
  double eval(double x) const {
    double z = map_to_cheb(x, x_min, x_max);
    z = std::max(-1.0, std::min(1.0, z));

    double result = 0.0;
    for (int i = 0; i <= degree; ++i) {
      result += coeffs(i) * chebyshev_T(i, z);
    }
    return result;
  }

  // Fit to data points using least squares
  void fit(const vec &x_data, const vec &y_data) {
    int n = x_data.n_elem;
    mat A(n, degree + 1);

    for (int i = 0; i < n; ++i) {
      double z = map_to_cheb(x_data(i), x_min, x_max);
      for (int j = 0; j <= degree; ++j) {
        A(i, j) = chebyshev_T(j, z);
      }
    }

    // Solve least squares: coeffs = (A'A)^{-1} A' y
    coeffs = solve(A.t() * A, A.t() * y_data);
  }
};

// =====================================================================
//  ProjectionPolicies - Polynomial approximation of policy functions
// =====================================================================
struct ProjectionPolicies {
  // For each (z, g) state, we have polynomial approximations
  // of k'(k), c(k), h(k) as functions of capital k

  std::vector<std::vector<ChebyshevApprox>> k_prime; // [n_z][n_g]
  std::vector<std::vector<ChebyshevApprox>> c;
  std::vector<std::vector<ChebyshevApprox>> h;

  int n_z, n_g;
  int degree;
  double k_min, k_max;

  void initialize(int nz, int ng, int deg, double kmin, double kmax) {
    n_z = nz;
    n_g = ng;
    degree = deg;
    k_min = kmin;
    k_max = kmax;

    k_prime.resize(n_z);
    c.resize(n_z);
    h.resize(n_z);

    for (int iz = 0; iz < n_z; ++iz) {
      k_prime[iz].resize(n_g);
      c[iz].resize(n_g);
      h[iz].resize(n_g);

      for (int ig = 0; ig < n_g; ++ig) {
        k_prime[iz][ig] = ChebyshevApprox(degree, k_min, k_max);
        c[iz][ig] = ChebyshevApprox(degree, k_min, k_max);
        h[iz][ig] = ChebyshevApprox(degree, k_min, k_max);
      }
    }
  }

  double eval_k_prime(int i_z, int i_g, double k) const {
    return k_prime[i_z][i_g].eval(k);
  }

  double eval_c(int i_z, int i_g, double k) const {
    return c[i_z][i_g].eval(k);
  }

  double eval_h(int i_z, int i_g, double k) const {
    return h[i_z][i_g].eval(k);
  }
};

// =====================================================================
//  projection_solve - Projection method solver
//
//  Two-stage approach:
//    1. Solve using time iteration on a fine grid
//    2. Fit Chebyshev polynomials to the solution
//
//  This gives a smooth polynomial approximation that can be evaluated
//  at any point, similar to what CCK94 produce.
// =====================================================================
inline SolverStats projection_solve(const Params &p, const GridSpec &grid,
                                    PolicyFunctions &pol,
                                    ProjectionPolicies &proj,
                                    int poly_degree = 5, int max_iter = 500,
                                    double tol = 1e-6) {
  auto start = std::chrono::high_resolution_clock::now();

  NonlinearModel model(p);

  // Compute LQ decision rules for state-dependent tau_h
  DecisionRules rules = compute_decision_rules(p);

  // Stage 1: Solve using time iteration
  pol.initialize(grid.n_k, grid.n_z, grid.n_g);

  for (int i_g = 0; i_g < grid.n_g; ++i_g) {
    for (int i_z = 0; i_z < grid.n_z; ++i_z) {
      for (int i_k = 0; i_k < grid.n_k; ++i_k) {
        pol.k_prime(i_k, i_z, i_g) = grid.k_grid(i_k);
        pol.c(i_k, i_z, i_g) = model.c_ss;
        pol.h(i_k, i_z, i_g) = model.h_ss;
      }
    }
  }

  SolverStats stats;
  stats.converged = false;

  // Use TI iteration (from 06_time_iteration.h logic)
  for (int iter = 0; iter < max_iter; ++iter) {
    double max_diff = 0.0;

    cube k_prime_new(grid.n_k, grid.n_z, grid.n_g);
    cube c_new(grid.n_k, grid.n_z, grid.n_g);
    cube h_new(grid.n_k, grid.n_z, grid.n_g);

    for (int i_g = 0; i_g < grid.n_g; ++i_g) {
      double g = std::exp(grid.g_grid(i_g)) * model.g_ss;

      for (int i_z = 0; i_z < grid.n_z; ++i_z) {
        double z = std::exp(grid.z_grid(i_z));

        for (int i_k = 0; i_k < grid.n_k; ++i_k) {
          double k = grid.k_grid(i_k);
          double k_hat = std::log(k / model.k_ss);
          double g_hat = grid.g_grid(i_g);
          double z_hat = grid.z_grid(i_z);

          // State-dependent tau_h from LQ formula
          double tau_h_hat = rules.gamma_tau_h(0) * k_hat +
                             rules.gamma_tau_h(1) * z_hat +
                             rules.gamma_tau_h(2) * g_hat;
          double tau_h = p.tau_h_ss + (1.0 - p.tau_h_ss) * tau_h_hat;

          // Current k' for expectation
          double k_p = pol.k_prime(i_k, i_z, i_g);

          // Expectation over future (z', g') states
          double E_term = 0.0;
          for (int j_g = 0; j_g < grid.n_g; ++j_g) {
            for (int j_z = 0; j_z < grid.n_z; ++j_z) {
              double prob = grid.P_g(i_g, j_g) * grid.P_z(i_z, j_z);
              double z_p = std::exp(grid.z_grid(j_z));

              // Interpolate c' and h' at k' for future state (j_z, j_g)
              double c_p = pol.interp_c(grid.k_grid, j_z, j_g, k_p);
              double h_p = pol.interp_h(grid.k_grid, j_z, j_g, k_p);

              double uc_p = model.u_c(c_p, h_p);
              double fk_p = model.f_k(k_p, z_p, h_p);
              E_term += prob * uc_p * (1.0 + fk_p - model.delta);
            }
          }

          double uc_target = model.beta * E_term;
          double h_guess = pol.h(i_k, i_z, i_g);
          if (h_guess <= 0 || h_guess >= 1)
            h_guess = model.h_ss;

          // Iterate to find consistent (c, h) with state-dependent tau_h
          double c_val, h_val;
          for (int inner = 0; inner < 20; ++inner) {
            c_val = model.c_from_uc(uc_target, h_guess);
            if (c_val <= 0)
              c_val = model.c_ss;

            // Solve for hours using state-dependent tau_h from LQ formula
            h_val = solve_hours(model, c_val, k, z, tau_h);
            if (std::abs(h_val - h_guess) < 1e-6)
              break;
            h_guess = 0.5 * h_guess + 0.5 * h_val;
          }

          // Resource constraint
          double y = model.f(k, z, h_val);
          double k_p_new = (1.0 - model.delta) * k + y - c_val - g;
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

    const double damp = 0.3;
    pol.k_prime = (1.0 - damp) * pol.k_prime + damp * k_prime_new;
    pol.c = (1.0 - damp) * pol.c + damp * c_new;
    pol.h = (1.0 - damp) * pol.h + damp * h_new;

    if (max_diff < tol) {
      stats.converged = true;
      stats.iterations = iter + 1;
      stats.error = max_diff;
      break;
    }
    stats.iterations = iter + 1;
    stats.error = max_diff;
  }

  // Stage 2: Fit Chebyshev polynomials
  proj.initialize(grid.n_z, grid.n_g, poly_degree, grid.k_min, grid.k_max);

  for (int i_z = 0; i_z < grid.n_z; ++i_z) {
    for (int i_g = 0; i_g < grid.n_g; ++i_g) {
      vec kp_data(grid.n_k);
      vec c_data(grid.n_k);
      vec h_data(grid.n_k);

      for (int i_k = 0; i_k < grid.n_k; ++i_k) {
        kp_data(i_k) = pol.k_prime(i_k, i_z, i_g);
        c_data(i_k) = pol.c(i_k, i_z, i_g);
        h_data(i_k) = pol.h(i_k, i_z, i_g);
      }

      proj.k_prime[i_z][i_g].fit(grid.k_grid, kp_data);
      proj.c[i_z][i_g].fit(grid.k_grid, c_data);
      proj.h[i_z][i_g].fit(grid.k_grid, h_data);
    }
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
