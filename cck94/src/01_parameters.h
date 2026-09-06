#pragma once
#include <armadillo4r.hpp>
#include <cmath>
#include <cpp4r.hpp>
#include <stdexcept>

using namespace cpp4r;
using namespace arma;

// Helper: extract a named scalar from a cpp4r list with default
static inline double get_scalar(const list &L, const char *key, double def) {
  SEXP val = L[key];
  if (val == R_NilValue)
    return def;
  return as_cpp<doubles>(val)[0];
}

// =====================================================================
//  Params - BW (2005) calibration, LQ loss, and constraint coefficients
//
//  Implements BW NBER WP 11029 exactly:
//    - Table 1 calibration
//    - Eqs (2.4), (2.13) LQ problem formulation
//    - Eqs (2.14)-(2.17) tax formula coefficients
//
//  Utility: u(c,h) = [c^{1-gamma}(1-h)^gamma]^phi / phi if phi > 0
//           u(c,h) = (1-gamma) log c + gamma log(1-h) if phi = 0
//
//  Production: f(k, zh) = k^alpha (zh)^{1-alpha}  (detrended Cobb-Douglas)
// =====================================================================
struct Params {

  // ---- BW Table 1: directly calibrated ----
  double alpha;       // capital share
  double beta_tilde;  // detrended discount factor
  double gamma_bw;    // utility weight on leisure
  double delta_tilde; // detrended depreciation
  double mu;          // trend growth rate (rho = e^mu)
  double sg;          // barg/bary steady-state government spending share
  double tau_h_ss;    // bartau_h: steady-state labour tax (BW Table 1)
  double psi_u;       // psi = BW phi: utility curvature

  // ---- Shock parameters --------------------------------
  double rho_z, sigma_z;
  double rho_g, sigma_g;

  // ---- Derived NSSS quantities --------------------
  double sk;     // bark/bary
  double sc;     // barc/bary
  double h_ss;   // steady-state hours
  double exp_mu; // rho^{-1} = e^{-mu}

  // ---- BW utility coefficients (footnote 14) ------------------------------------------------
  double sigma_inv; // sigma^{-1} = 1 - (1-gamma)phi
  double nu;        // vee = (1-gammaphi) barh/(1-barh)
  double psi_bw;    // psi_BW = -gammaphi barh/(1-barh)
  double phi_bw;    // phi = (1-bartau^h)(1-alpha)/s_c

  // ---- BW implementability coefficients (eq 2.5) ------------------------------------
  double dc, dh; // d_c = 1-sigma^{-1}+psi,  d_h = psi-phi(1+vee)

  // ---- BW leverage coefficients (eq 2.16) ----------------------------------------------------
  double sb; // s_b = barb^s/bary
  double bc_bw, bh_bw, bk_bw, btau_bw;

  // ---- LQ loss function coefficients (BW eq 2.13) ------------------------------------
  double qc, theta_lq, qh, theta_z, qk;

  // ---- LQ constraint and cost matrices ----
  mat A_con;  // 3x3 state transition
  mat B_con;  // 3x2 control input
  mat Q_cost; // 3x3 state cost
  mat R_cost; // 2x2 control cost
  mat N_cost; // 3x2 cross cost

  // Constructor
  explicit Params(const list &args) {
    // ---- Step 1: Read BW Table 1 parameters ----------------------------------------------------
    alpha = 0.344;
    beta_tilde = 0.98;
    gamma_bw = get_scalar(args, "gamma", 0.75);
    delta_tilde = 0.095;
    mu = 0.016;
    sg = 0.167;
    tau_h_ss = get_scalar(args, "tau_h", 0.2387);
    psi_u = get_scalar(args, "psi", 0.0);

    rho_z = get_scalar(args, "rho_z", 0.81);
    sigma_z = get_scalar(args, "sigma_z", 0.041);
    rho_g = get_scalar(args, "rho_g", 0.89);
    sigma_g = get_scalar(args, "sigma_g", 0.070);

    exp_mu = std::exp(-mu);

    // ---- Step 2: NSSS shares (BW eqs 3.1-3.2) --------------------------------------------
    //  sk = alpha / (beta_inv - 1 + delta_tilde)  (same for all phi)
    //  sc + delta_tilde * sk + sg = 1  (resource constraint)
    //
    //  BW Table 1: barg = 0.069 is a fixed level, so sg = barg/bary varies
    //  with the steady state.  Allow optional overrides for sc (via
    //  "sigma_c") and sg (via "sg") to match BW Table 1 exactly.
    const double bt_inv = 1.0 / beta_tilde;
    sk = alpha / (bt_inv - 1.0 + delta_tilde);

    const double sc_override = get_scalar(args, "sigma_c", -1.0);
    if (sc_override > 0.0) {
      sc = sc_override;
      sg = 1.0 - sc - delta_tilde * sk;
    } else {
      sg = get_scalar(args, "sg", 0.167);
      sc = 1.0 - delta_tilde * sk - sg;
    }
    if (sc <= 0.0)
      throw std::runtime_error("Params: c/y ratio non-positive");

    // ---- Step 3: Steady-state hours (BW eqs 3.3-3.4) --------------------------------
    //  Allow optional override via "h_bar" for direct Table 1 match.
    const double h_override = get_scalar(args, "h_bar", -1.0);
    if (h_override > 0.0) {
      h_ss = h_override;
    } else {
      const double leis_ratio =
          gamma_bw * sc / ((1.0 - gamma_bw) * (1.0 - tau_h_ss) * (1.0 - alpha));
      h_ss = 1.0 / (1.0 + leis_ratio);
    }

    // ---- Step 4: BW utility coefficients (footnote 14) ----------------------------
    const double h_over_1mh = h_ss / (1.0 - h_ss);

    sigma_inv = 1.0 - (1.0 - gamma_bw) * psi_u;
    nu = (1.0 - gamma_bw * psi_u) * h_over_1mh;
    psi_bw = -gamma_bw * psi_u * h_over_1mh;
    phi_bw = (1.0 - tau_h_ss) * (1.0 - alpha) / sc;

    dc = 1.0 - sigma_inv + psi_bw;
    dh = psi_bw - phi_bw * (1.0 + nu);

    // ---- Step 5: Leverage coefficients (BW eq 2.16 definitions) ------------
    sb = (tau_h_ss * (1.0 - alpha) - sg) / (bt_inv - 1.0);
    bc_bw = (sb + bt_inv * sk) * sigma_inv;
    bh_bw = (sb + bt_inv * sk) * psi_bw + alpha * (1.0 - alpha);
    bk_bw = bt_inv * sk - alpha * (1.0 - alpha);
    btau_bw = sk * (bt_inv - exp_mu);

    // ---- Step 6: BW LQ loss coefficients (eqs 2.7-2.13) ------------------------
    compute_lq_coefficients();

    // ---- Step 7: Constraint and cost matrices ------------------------------------------------
    build_lq_matrices();
  }

private:
  void compute_lq_coefficients() {
    const double h_over_1mh = h_ss / (1.0 - h_ss);

    // Second-order utility coefficients for the correct LQ objective
    //   sigma^{-1}_1 = (1-gamma)phi - 2
    //   psi_2 = psi_BW
    //   psipsi_1 = -gammaphi(1-phigamma) barh^2 / (1-barh)^2
    //   vee_1 = (2-phigamma) barh/(1-barh)
    const double sigma_inv_1 = (1.0 - gamma_bw) * psi_u - 2.0;
    const double psi_2 = psi_bw;
    const double psi_psi_1 =
        -gamma_bw * psi_u * (1.0 - psi_u * gamma_bw) * h_over_1mh * h_over_1mh;
    const double nu_1 = (2.0 - psi_u * gamma_bw) * h_over_1mh;

    // A_x (BW eq 2.7)
    const double Ax_11 = -(1.0 - sigma_inv);
    const double Ax_12 = -psi_bw;
    const double Ax_22 = phi_bw * (1.0 + nu);

    // B_x (BW eq 2.9)
    const double Bx_11 = sc;
    const double Bx_22 = -(1.0 - alpha);

    // C_x (BW eq 2.11)
    const double Cx_11 = 1.0 - sigma_inv + psi_bw - sigma_inv * sigma_inv_1 -
                         sigma_inv * psi_2 - 2.0 * sigma_inv;
    const double Cx_12 = psi_psi_1 - sigma_inv * psi_2 + 2.0 * psi_bw;
    const double Cx_22 =
        psi_bw + psi_psi_1 - phi_bw * (1.0 + 3.0 * nu + nu * nu_1);

    // B_barxi: only (2,1) entry
    const double Bxi_21 = -(1.0 - alpha);

    // theta_1, theta_2 weights (BW page 20)
    const double denom = psi_bw * ((1.0 - alpha) + sc) +
                         tau_h_ss * (1.0 - alpha) * (1.0 + nu) -
                         (1.0 - alpha) * (sigma_inv + nu);

    if (std::abs(denom) < 1e-14)
      throw std::runtime_error("Params: vartheta denominator near zero");

    const double vartheta_1 =
        (-phi_bw * (sigma_inv + nu) + psi_bw * (phi_bw + 1.0)) / denom;
    const double vartheta_2 = tau_h_ss * (1.0 - alpha) / denom;

    // Q_x = A_x + theta_1 B_x + theta_2 C_x
    const double Qx_11 = Ax_11 + vartheta_1 * Bx_11 + vartheta_2 * Cx_11;
    const double Qx_12 = Ax_12 + vartheta_2 * Cx_12;
    const double Qx_22 = Ax_22 + vartheta_1 * Bx_22 + vartheta_2 * Cx_22;

    // Q_barxi: only (2,1) entry
    const double Qxi_21 = vartheta_1 * Bxi_21;

    // q_k = alpha(1-alpha) theta_1
    qk = alpha * (1.0 - alpha) * vartheta_1;

    // BW eq (2.13) loss coefficients
    if (std::abs(Qx_11) < 1e-14)
      throw std::runtime_error("Params: Q_{x,11} near zero");

    qc = Qx_11;
    theta_lq = -Qx_12 / Qx_11;
    qh = Qx_22 - Qx_12 * Qx_12 / Qx_11;
    theta_z = (std::abs(qh) > 1e-14) ? -Qxi_21 / qh : 0.0;
  }

  void build_lq_matrices() {
    const double bt_inv = 1.0 / beta_tilde;

    // Constraint: s_{t+1} = A s_t + B u_t  (BW eq 2.4)
    A_con.set_size(3, 3);
    A_con(0, 0) = bt_inv;
    A_con(0, 1) = (1.0 - alpha) / sk;
    A_con(0, 2) = -sg / sk;
    A_con(1, 0) = 0.0;
    A_con(1, 1) = rho_z;
    A_con(1, 2) = 0.0;
    A_con(2, 0) = 0.0;
    A_con(2, 1) = 0.0;
    A_con(2, 2) = rho_g;

    B_con.set_size(3, 2);
    B_con(0, 0) = -sc / sk;
    B_con(0, 1) = (1.0 - alpha) / sk;
    B_con(1, 0) = 0.0;
    B_con(1, 1) = 0.0;
    B_con(2, 0) = 0.0;
    B_con(2, 1) = 0.0;

    // Cost: L = 0.5*(s'Qs + 2s'Nu + u'Ru)
    R_cost.set_size(2, 2);
    R_cost(0, 0) = qc;
    R_cost(0, 1) = R_cost(1, 0) = -qc * theta_lq;
    R_cost(1, 1) = qc * theta_lq * theta_lq + qh + qk;

    Q_cost.set_size(3, 3);
    Q_cost.zeros();
    Q_cost(0, 0) = qk;
    Q_cost(0, 1) = Q_cost(1, 0) = -qk;
    Q_cost(1, 1) = qh * theta_z * theta_z + qk;

    N_cost.set_size(3, 2);
    N_cost.zeros();
    N_cost(0, 1) = -qk;
    N_cost(1, 1) = qk - qh * theta_z;
  }
};
