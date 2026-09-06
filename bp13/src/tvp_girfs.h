#ifndef TVP_GIRFS_H
#define TVP_GIRFS_H

/* ==========================================================================
   TVP-VAR Generalized Impulse Response Functions with Sign Restrictions
   Following Baumeister and Peersman (2013)

   Ported from MATLAB: GIRFs_N3.m
   ========================================================================== */

// ============================================================================
// GIRF COMPUTATION
// ============================================================================

// GIRFs_N3: Compute GIRFs for 3-variable oil market model
// Following Baumeister-Peersman (2013) MATLAB code exactly
// Returns true if valid IRFs found, false otherwise
// Output: sirf (supply), dirf (demand), aggirf (global) - each HOR x N
//
// Key insight from MATLAB: Each of the 3 candidate impact columns is checked
// against ALL sign patterns, and codes are assigned:
//   Supply shock (code 100): qo >= 0, po <= 0, y >= 0
//   Demand shock (code 10): qo >= 0, po >= 0, y <= 0
//   Aggregate shock (code 1): qo >= 0, po >= 0, y > 0
// A draw is only accepted if identot = 111 (each pattern matched exactly once)
inline bool GIRFs_N3(const vec &sd, const vec &aa, const Mat<double> &qd,
                     double s1, const Mat<double> &s2, const vec &vd,
                     const vec &om, const Mat<double> &YY, int HOR, int N,
                     int L, int cors, int NMCInteg, int h_restrict, bool SC,
                     int FD, Mat<double> &sirf, Mat<double> &dirf,
                     Mat<double> &aggirf) {

  if (N != 3)
    return false;

  // Initialize output
  sirf.set_size(HOR, N);
  dirf.set_size(HOR, N);
  aggirf.set_size(HOR, N);
  sirf.zeros();
  dirf.zeros();
  aggirf.zeros();

  // =========================================================================
  // (1) Simulate covariance matrix elements into the future
  // =========================================================================

  // Build S matrix for A innovations (following MATLAB exactly)
  int n_aa = N * (N - 1) / 2; // = 3 for N=3
  Mat<double> S_full(n_aa, n_aa, fill::zeros);
  S_full(0, 0) = std::max(s1, 1e-10);
  if (n_aa > 1) {
    S_full.submat(1, 1, n_aa - 1, n_aa - 1) = s2;
  }
  S_full += 1e-10 * eye<Mat<double>>(n_aa, n_aa);
  Mat<double> S_sqrt = mysqrt_varm(S_full);

  // Simulate A elements: aa_e = sqrtm(S) * randn(3, HOR+L+1)
  Cube<double> aa_sim(n_aa, 1, HOR + L + 2);
  aa_sim.slice(0).col(0) = aa;
  for (int j = 0; j < HOR + L + 1; ++j) {
    vec aa_e = S_sqrt * randn<vec>(n_aa);
    aa_sim.slice(j + 1).col(0) = aa_sim.slice(j).col(0) + aa_e;
  }

  // Simulate volatilities: log random walk
  Cube<double> om_sim(N, 1, HOR + L + 2);
  om_sim.slice(0).col(0) = log(clamp(om, 1e-10, 1e10));
  for (int j = 0; j < HOR + L + 1; ++j) {
    vec om_e = diagmat(vd) * randn<vec>(N);
    om_sim.slice(j + 1).col(0) = om_sim.slice(j).col(0) + om_e;
  }
  // Convert back to levels
  for (uword j = 0; j < om_sim.n_slices; ++j) {
    om_sim.slice(j).col(0) = exp(clamp(om_sim.slice(j).col(0), -20.0, 20.0));
  }

  // Compute VAR covariance matrices: VAR = inv(A) * diag(om) * inv(A)'
  Cube<double> VAR_cov(N, N, HOR + L + 2);
  for (int j = 0; j < HOR + L + 2; ++j) {
    Mat<double> A_mat = chofac(N, aa_sim.slice(j).col(0));
    Mat<double> invA = inv(A_mat + 1e-10 * eye<Mat<double>>(N, N));
    VAR_cov.slice(j) = invA * diagmat(om_sim.slice(j).col(0)) * invA.t();
  }
  Mat<double> VARC = VAR_cov.slice(0); // Initial covariance for impact matrix

  // =========================================================================
  // (2) Simulate VAR coefficients
  // =========================================================================

  Cube<double> B_sim(N, N * L + 1, HOR + L + 2);

  // Reshape sd to matrix form: B = [c A1 A2 ... AL]
  Mat<double> b0(N, N * L + 1);
  for (int eq = 0; eq < N; ++eq) {
    b0.row(eq) = sd.subvec(eq * (1 + N * L), (eq + 1) * (1 + N * L) - 1).t();
  }
  B_sim.slice(0) = b0;

  Mat<double> qd_reg = qd + 1e-8 * eye<Mat<double>>(qd.n_rows, qd.n_cols);
  qd_reg = 0.5 * (qd_reg + qd_reg.t());
  Mat<double> Q_sqrt = mysqrt_varm(qd_reg);

  vec sd_current = sd;
  int max_trial = 50;

  for (int j = 1; j < HOR + L + 2; ++j) {
    int trial = 0;
    bool found = false;

    while (!found && trial < max_trial) {
      vec dd = sd_current + Q_sqrt * randn<vec>(sd.n_elem);

      Mat<double> b(N, N * L + 1);
      for (int eq = 0; eq < N; ++eq) {
        b.row(eq) = dd.subvec(eq * (1 + N * L), (eq + 1) * (1 + N * L) - 1).t();
      }

      if (!SC || varroots(L, N, b) < 1.0) {
        sd_current = dd;
        B_sim.slice(j) = b;
        found = true;
      }
      trial++;
    }

    if (!found) {
      return false;
    }
  }

  // =========================================================================
  // (3) Monte Carlo integration for GIRFs with proper sign identification
  // =========================================================================

  Mat<double> iroil1(NMCInteg, HOR, fill::zeros); // Supply shock responses
  Mat<double> iroil2(NMCInteg, HOR, fill::zeros);
  Mat<double> iroil3(NMCInteg, HOR, fill::zeros);

  Mat<double> irdem1(NMCInteg, HOR, fill::zeros); // Demand shock responses
  Mat<double> irdem2(NMCInteg, HOR, fill::zeros);
  Mat<double> irdem3(NMCInteg, HOR, fill::zeros);

  Mat<double> iragg1(NMCInteg, HOR, fill::zeros); // Aggregate shock responses
  Mat<double> iragg2(NMCInteg, HOR, fill::zeros);
  Mat<double> iragg3(NMCInteg, HOR, fill::zeros);

  int accepted = 0;
  int max_count = NMCInteg * 25;
  int count = 0;

  while (accepted < NMCInteg && count < max_count) {
    count++;

    // Get candidate impact matrix via random orthonormal rotation
    Mat<double> a0 = impact(VARC, N);

    // Simulate paths for baseline and 3 shock scenarios
    Mat<double> Y_BEN(N, HOR + L + 3, fill::zeros);
    Mat<double> Y_SH1(N, HOR + L + 3, fill::zeros); // +1 to shock 1
    Mat<double> Y_SH2(N, HOR + L + 3, fill::zeros); // +1 to shock 2
    Mat<double> Y_SH3(N, HOR + L + 3, fill::zeros); // +1 to shock 3

    // Initial conditions from YY (lagged values): YY is N x L
    Y_BEN.cols(0, L - 1) = YY;
    Y_SH1.cols(0, L - 1) = YY;
    Y_SH2.cols(0, L - 1) = YY;
    Y_SH3.cols(0, L - 1) = YY;

    // Common shocks for all paths
    Mat<double> SHOCKS = randn<Mat<double>>(N, HOR);

    // Impact period (MATLAB: Y_OILS(:,L+1)=a0*[1+SHOCKS(1,1) SHOCKS(2:3,1)']')
    Y_BEN.col(L) = a0 * SHOCKS.col(0);

    vec shock_1 = SHOCKS.col(0);
    shock_1(0) += 1.0;
    Y_SH1.col(L) = a0 * shock_1;

    vec shock_2 = SHOCKS.col(0);
    shock_2(1) += 1.0;
    Y_SH2.col(L) = a0 * shock_2;

    vec shock_3 = SHOCKS.col(0);
    shock_3(2) += 1.0;
    Y_SH3.col(L) = a0 * shock_3;

    // Propagate through VAR (following MATLAB myvec/myfliplr pattern)
    for (int t = L + 1; t < HOR + L; ++t) {
      vec shocks = mysqrt_varm(VAR_cov.slice(t - L)) * SHOCKS.col(t - L);

      // Build regressor [1, y(t-1)', y(t-2)', ..., y(t-L)'] with reversed lag order
      vec x_ben(1 + N * L), x_sh1(1 + N * L), x_sh2(1 + N * L), x_sh3(1 + N * L);
      x_ben(0) = x_sh1(0) = x_sh2(0) = x_sh3(0) = 1.0;

      for (int lag = 1; lag <= L; ++lag) {
        x_ben.subvec(1 + N * (lag - 1), N * lag) = Y_BEN.col(t - lag);
        x_sh1.subvec(1 + N * (lag - 1), N * lag) = Y_SH1.col(t - lag);
        x_sh2.subvec(1 + N * (lag - 1), N * lag) = Y_SH2.col(t - lag);
        x_sh3.subvec(1 + N * (lag - 1), N * lag) = Y_SH3.col(t - lag);
      }

      Mat<double> B_t = B_sim.slice(t - L);
      Y_BEN.col(t) = B_t * x_ben + shocks;
      Y_SH1.col(t) = B_t * x_sh1 + shocks;
      Y_SH2.col(t) = B_t * x_sh2 + shocks;
      Y_SH3.col(t) = B_t * x_sh3 + shocks;
    }

    // Trim to HOR periods after impact
    Mat<double> Y_BEN_trim = Y_BEN.cols(L, L + HOR - 1);
    Mat<double> Y_SH1_trim = Y_SH1.cols(L, L + HOR - 1);
    Mat<double> Y_SH2_trim = Y_SH2.cols(L, L + HOR - 1);
    Mat<double> Y_SH3_trim = Y_SH3.cols(L, L + HOR - 1);

    // Compute IRFs as difference from baseline
    Mat<double> irf1 = Y_SH1_trim - Y_BEN_trim;
    Mat<double> irf2 = Y_SH2_trim - Y_BEN_trim;
    Mat<double> irf3 = Y_SH3_trim - Y_BEN_trim;

    // Cumulate if data is in first differences
    Mat<double> ir1 = (FD == 1) ? cumsum(irf1, 1) : irf1;
    Mat<double> ir2 = (FD == 1) ? cumsum(irf2, 1) : irf2;
    Mat<double> ir3 = (FD == 1) ? cumsum(irf3, 1) : irf3;

    // =========================================================================
    // Check sign restrictions following MATLAB logic exactly
    // Each shock column is tested against ALL patterns
    // =========================================================================
    
    // Normalize by impact response on qo for each shock
    double norm1 = std::abs(ir1(0, cors - 1)) > 1e-10 ? ir1(0, cors - 1) : 1.0;
    double norm2 = std::abs(ir2(0, cors - 1)) > 1e-10 ? ir2(0, cors - 1) : 1.0;
    double norm3 = std::abs(ir3(0, cors - 1)) > 1e-10 ? ir3(0, cors - 1) : 1.0;

    // Normalized responses at impact (h_restrict horizons)
    // co(k, 1) = qo, co(k, 2) = po, co(k, 3) = y for shock column
    
    // Check patterns for h_restrict=1 (contemporaneous only)
    // s1is, s2is, s3is will hold which pattern each shock matches
    int s1is = 0, s2is = 0, s3is = 0;
    
    // Check shock 1 (column 1 of rotation)
    double qo1 = ir1(0, cors - 1) / norm1;
    double po1 = ir1(1, cors - 1) / norm1;
    double y1 = ir1(2, cors - 1) / norm1;
    
    // Supply: qo >= 0, po <= 0, y >= 0
    if (qo1 >= 0 && po1 <= 0 && y1 >= 0) s1is = 100;
    // Demand: qo >= 0, po >= 0, y <= 0
    else if (qo1 >= 0 && po1 >= 0 && y1 <= 0) s1is = 10;
    // Aggregate: qo >= 0, po >= 0, y > 0
    else if (qo1 >= 0 && po1 >= 0 && y1 > 0) s1is = 1;
    
    // Check shock 2 (column 2 of rotation)
    double qo2 = ir2(0, cors - 1) / norm2;
    double po2 = ir2(1, cors - 1) / norm2;
    double y2 = ir2(2, cors - 1) / norm2;
    
    if (qo2 >= 0 && po2 <= 0 && y2 >= 0) s2is = 100;
    else if (qo2 >= 0 && po2 >= 0 && y2 <= 0) s2is = 10;
    else if (qo2 >= 0 && po2 >= 0 && y2 > 0) s2is = 1;
    
    // Check shock 3 (column 3 of rotation)
    double qo3 = ir3(0, cors - 1) / norm3;
    double po3 = ir3(1, cors - 1) / norm3;
    double y3 = ir3(2, cors - 1) / norm3;
    
    if (qo3 >= 0 && po3 <= 0 && y3 >= 0) s3is = 100;
    else if (qo3 >= 0 && po3 >= 0 && y3 <= 0) s3is = 10;
    else if (qo3 >= 0 && po3 >= 0 && y3 > 0) s3is = 1;
    
    int identot = s1is + s2is + s3is;
    
    // Only accept if all three patterns are matched exactly once (identot = 111)
    if (identot == 111) {
      // Normalize signs to ensure consistent direction (MATLAB lines 373-398)
      // Supply shock: normalize so po > 0 (flip if needed)
      // Demand shock: normalize so po > 0
      // Aggregate shock: normalize so y > 0
      
      // Find which shock matches which pattern and store appropriately
      Mat<double>* supply_ir = nullptr;
      Mat<double>* demand_ir = nullptr;
      Mat<double>* agg_ir = nullptr;
      double supply_sign = 1.0, demand_sign = 1.0, agg_sign = 1.0;
      
      if (s1is == 100) { 
        supply_ir = &ir1; 
        supply_sign = (ir1(1, cors - 1) < 0) ? -1.0 : 1.0; // Normalize so po > 0
      }
      else if (s2is == 100) { 
        supply_ir = &ir2; 
        supply_sign = (ir2(1, cors - 1) < 0) ? -1.0 : 1.0;
      }
      else { 
        supply_ir = &ir3; 
        supply_sign = (ir3(1, cors - 1) < 0) ? -1.0 : 1.0;
      }
      
      if (s1is == 10) { 
        demand_ir = &ir1; 
        demand_sign = (ir1(1, cors - 1) < 0) ? -1.0 : 1.0;
      }
      else if (s2is == 10) { 
        demand_ir = &ir2; 
        demand_sign = (ir2(1, cors - 1) < 0) ? -1.0 : 1.0;
      }
      else { 
        demand_ir = &ir3; 
        demand_sign = (ir3(1, cors - 1) < 0) ? -1.0 : 1.0;
      }
      
      if (s1is == 1) { 
        agg_ir = &ir1; 
        agg_sign = (ir1(2, cors - 1) < 0) ? -1.0 : 1.0;
      }
      else if (s2is == 1) { 
        agg_ir = &ir2; 
        agg_sign = (ir2(2, cors - 1) < 0) ? -1.0 : 1.0;
      }
      else { 
        agg_ir = &ir3; 
        agg_sign = (ir3(2, cors - 1) < 0) ? -1.0 : 1.0;
      }
      
      // Store with sign normalization
      // Supply shock: qo-, po+, y- (negative supply shock raises price)
      iroil1.row(accepted) = supply_sign * supply_ir->row(0);
      iroil2.row(accepted) = supply_sign * supply_ir->row(1);
      iroil3.row(accepted) = supply_sign * supply_ir->row(2);
      
      // Demand shock: qo+, po+, y-
      irdem1.row(accepted) = demand_sign * demand_ir->row(0);
      irdem2.row(accepted) = demand_sign * demand_ir->row(1);
      irdem3.row(accepted) = demand_sign * demand_ir->row(2);
      
      // Aggregate shock: qo+, po+, y+
      iragg1.row(accepted) = agg_sign * agg_ir->row(0);
      iragg2.row(accepted) = agg_sign * agg_ir->row(1);
      iragg3.row(accepted) = agg_sign * agg_ir->row(2);

      accepted++;
    }
  }

  if (accepted == 0) {
    return false;
  }

  // =========================================================================
  // (4) Compute MEAN IRFs across accepted draws (MATLAB uses mean, not median)
  // =========================================================================

  for (int h = 0; h < HOR; ++h) {
    sirf(h, 0) = mean(iroil1.col(h).head(accepted));
    sirf(h, 1) = mean(iroil2.col(h).head(accepted));
    sirf(h, 2) = mean(iroil3.col(h).head(accepted));

    dirf(h, 0) = mean(irdem1.col(h).head(accepted));
    dirf(h, 1) = mean(irdem2.col(h).head(accepted));
    dirf(h, 2) = mean(irdem3.col(h).head(accepted));

    aggirf(h, 0) = mean(iragg1.col(h).head(accepted));
    aggirf(h, 1) = mean(iragg2.col(h).head(accepted));
    aggirf(h, 2) = mean(iragg3.col(h).head(accepted));
  }

  return true;
}

// Wrapper for getting elasticities along with IRFs
inline bool GIRFs_N3_with_elasticities(
    const vec &sd, const vec &aa, const Mat<double> &qd, double s1,
    const Mat<double> &s2, const vec &vd, const vec &om, const Mat<double> &YY,
    int HOR, int N, int L, int cors, int NMCInteg, int h_restrict, bool SC,
    int FD, Mat<double> &sirf, Mat<double> &dirf, Mat<double> &aggirf,
    OilMarketElasticities &elast) {

  bool success = GIRFs_N3(sd, aa, qd, s1, s2, vd, om, YY, HOR, N, L, cors,
                          NMCInteg, h_restrict, SC, FD, sirf, dirf, aggirf);

  if (success) {
    elast = oilmarket_model(sirf.row(0), dirf.row(0), aggirf.row(0));
  }

  return success;
}

#endif // TVP_GIRFS_H
