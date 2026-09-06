#ifndef TVP_STRUCTURAL_H
#define TVP_STRUCTURAL_H

/* ==========================================================================
   TVP-VAR Structural Analysis Functions
   Following Baumeister and Peersman (2013)

   Ported from MATLAB folder: mymatprog/structural/
   ========================================================================== */

// ============================================================================
// IMPACT MATRIX COMPUTATION
// ============================================================================

// impact: Compute contemporaneous impact matrix A0 (impact.m)
// Draws a random orthonormal rotation of the VAR covariance matrix
inline Mat<double> impact(const Mat<double> &VAR, int N) {
  // Eigenvalue decomposition of VAR covariance
  cx_mat P;
  cx_vec D;
  eig_gen(D, P, VAR);

  // Ensure positive eigenvalues for sqrt
  for (uword i = 0; i < D.n_elem; ++i) {
    if (std::real(D(i)) < 1e-10) {
      D(i) = std::complex<double>(1e-10, 0);
    }
  }

  // Random orthonormal matrix Q via QR decomposition of random matrix
  Mat<double> W = randn<Mat<double>>(N, N);
  Mat<double> Q, R;
  qr(Q, R, W);

  // Normalize signs for uniqueness (make diagonal of R positive)
  for (int i = 0; i < N; ++i) {
    if (R(i, i) < 0) {
      Q.col(i) = -Q.col(i);
    }
  }

  // A0 = P * D^0.5 * Q'
  cx_mat D_sqrt = diagmat(sqrt(D));
  Mat<double> A0 = real(P * D_sqrt * Q.t());

  return A0;
}

// Alternative impact using Cholesky with random rotation
inline Mat<double> impact_chol(const Mat<double> &VAR, int N) {
  // Cholesky decomposition
  Mat<double> VAR_reg = VAR + 1e-10 * eye<Mat<double>>(N, N);
  VAR_reg = 0.5 * (VAR_reg + VAR_reg.t());
  Mat<double> L = chol(VAR_reg, "lower");

  // Random orthonormal matrix
  Mat<double> W = randn<Mat<double>>(N, N);
  Mat<double> Q, R;
  qr(Q, R, W);

  for (int i = 0; i < N; ++i) {
    if (R(i, i) < 0) {
      Q.col(i) = -Q.col(i);
    }
  }

  return L * Q;
}

// ============================================================================
// SIGN RESTRICTION CHECKING
// ============================================================================

// Sign restriction codes:
// For oil market model (N=3):
//   Shock 1 (Supply): qo+, po-, y+  (negative supply shock)
//   Shock 2 (Oil-specific demand): qo+, po+, y-
//   Shock 3 (Global demand): qo+, po+, y+
// Returns: 0 = none, 1 = supply, 10 = demand, 100 = global

inline int check_sign_restrictions_h1(const Mat<double> &ir, int cors = 1,
                                      int FD = 1) {
  // ir is 3x3: columns are shocks, rows are variables
  // Check at impact (cors = 1 means contemporaneous)

  // Supply shock: qo >= 0, po <= 0, y >= 0
  bool supply = (ir(0, 0) >= 0) && (ir(1, 0) <= 0) && (ir(2, 0) >= 0);

  // Oil-specific demand shock: qo >= 0, po >= 0, y <= 0
  bool demand = (ir(0, 1) >= 0) && (ir(1, 1) >= 0) && (ir(2, 1) <= 0);

  // Global demand shock: qo >= 0, po >= 0, y >= 0
  bool global = (ir(0, 2) >= 0) && (ir(1, 2) >= 0) && (ir(2, 2) >= 0);

  int code = 0;
  if (supply)
    code += 100;
  if (demand)
    code += 10;
  if (global)
    code += 1;

  return code;
}

// Check for h periods (h_restrict in MATLAB)
inline bool check_sign_restrictions(const Cube<double> &ir_accum,
                                    int h_restrict, int cors = 1, int FD = 1) {
  int N = ir_accum.n_rows;
  if (N != 3)
    return false;

  bool all_ok = true;

  for (int h = 0; h < h_restrict && all_ok; ++h) {
    // ir_accum: (N x N x HOR) - accumulated IRFs
    // Check supply shock (column 0)
    if (ir_accum(0, 0, h) < 0)
      all_ok = false; // qo
    if (ir_accum(1, 0, h) > 0)
      all_ok = false; // po (should be negative)
    if (ir_accum(2, 0, h) < 0)
      all_ok = false; // y

    // Check demand shock (column 1)
    if (ir_accum(0, 1, h) < 0)
      all_ok = false; // qo
    if (ir_accum(1, 1, h) < 0)
      all_ok = false; // po
    if (ir_accum(2, 1, h) > 0)
      all_ok = false; // y (should be negative)

    // Check global shock (column 2)
    if (ir_accum(0, 2, h) < 0)
      all_ok = false; // qo
    if (ir_accum(1, 2, h) < 0)
      all_ok = false; // po
    if (ir_accum(2, 2, h) < 0)
      all_ok = false; // y
  }

  return all_ok;
}

// ============================================================================
// ELASTICITY COMPUTATION
// ============================================================================

// oilmarket_model: Compute elasticities and shock sizes (oilmarket_model.m)
// Input: impact responses for each shock (row vectors)
// Returns: elasticities and shock magnitudes
struct OilMarketElasticities {
  double slope_S_D; // Supply elasticity (with demand shock)
  double slope_S_A; // Supply elasticity (with global shock)
  double slope_D_D; // Demand elasticity
  double shock_S_D; // Supply shock magnitude (demand identification)
  double shock_S_A; // Supply shock magnitude (global identification)
  double shock_D_D; // Demand shock magnitude
  double shock_A_A; // Global shock magnitude
};

inline OilMarketElasticities
oilmarket_model(const rowvec &sirf, const rowvec &dirf, const rowvec &aggirf) {
  OilMarketElasticities elast;

  // Impact responses: sirf[0]=qo, sirf[1]=po, sirf[2]=y for supply shock
  // Slopes (elasticities) = (% change in quantity) / (% change in price)

  // Supply elasticity using demand shock
  if (std::abs(dirf(1)) > 1e-10) {
    elast.slope_S_D = dirf(0) / dirf(1);
  } else {
    elast.slope_S_D = 0.0;
  }

  // Supply elasticity using global shock
  if (std::abs(aggirf(1)) > 1e-10) {
    elast.slope_S_A = aggirf(0) / aggirf(1);
  } else {
    elast.slope_S_A = 0.0;
  }

  // Demand elasticity
  if (std::abs(sirf(1)) > 1e-10) {
    elast.slope_D_D = sirf(0) / sirf(1);
  } else {
    elast.slope_D_D = 0.0;
  }

  // Shock magnitudes (standard deviation of shock needed to cause unit
  // response)
  elast.shock_S_D = std::abs(sirf(1));   // Price response to supply shock
  elast.shock_S_A = std::abs(sirf(1));   // Same, using supply shock
  elast.shock_D_D = std::abs(dirf(1));   // Price response to demand shock
  elast.shock_A_A = std::abs(aggirf(1)); // Price response to global shock

  return elast;
}

// ============================================================================
// PERSISTENCE MEASURES
// ============================================================================

// persist: Contribution of structural shocks to persistence (persist.m)
// Computes normalized spectral density at frequency zero
inline double persist_contribution(const vec &irf_var, int h_cutoff = 40) {
  // Use first h_cutoff impulse responses
  int h = std::min((int)irf_var.n_elem, h_cutoff);
  vec irf = irf_var.head(h);

  // Normalized spectral density at frequency zero = sum(irf)^2 / sum(irf^2)
  double sum_irf = sum(irf);
  double sum_irf2 = dot(irf, irf);

  if (sum_irf2 < 1e-10)
    return 0.0;
  return (sum_irf * sum_irf) / sum_irf2;
}

// ============================================================================
// VARIANCE DECOMPOSITION
// ============================================================================

// Forecast error variance decomposition
inline Mat<double> variance_decomposition(const Cube<double> &IRFs, int h_max) {
  // IRFs: N x N x H (response of variable i to shock j at horizon h)
  int N = IRFs.n_rows;
  int H = std::min((int)IRFs.n_slices, h_max);

  Mat<double> FEVD(N, N,
                   fill::zeros); // variance share of shock j for variable i

  for (int i = 0; i < N; ++i) {
    double total_var = 0.0;
    vec shock_var(N, fill::zeros);

    for (int h = 0; h < H; ++h) {
      for (int j = 0; j < N; ++j) {
        double irf_sq = IRFs(i, j, h) * IRFs(i, j, h);
        shock_var(j) += irf_sq;
        total_var += irf_sq;
      }
    }

    if (total_var > 1e-10) {
      for (int j = 0; j < N; ++j) {
        FEVD(i, j) = shock_var(j) / total_var;
      }
    }
  }

  return FEVD;
}

#endif // TVP_STRUCTURAL_H
