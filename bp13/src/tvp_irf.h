#ifndef TVP_IRF_H
#define TVP_IRF_H

/* ==========================================================================
   TVP-VAR Impulse Response Functions
   Following Baumeister and Peersman (2013)

   Ported from MATLAB: TVP_IRFS.m
   ========================================================================== */

// ============================================================================
// IRF COMPUTATION
// ============================================================================

// Storage for IRF results at a single time period
struct TimePointIRFs {
  Mat<double> supply; // HOR x N
  Mat<double> demand; // HOR x N
  Mat<double> global; // HOR x N
  OilMarketElasticities elast;
  bool valid;
};

// Compute IRFs at a single time point
inline TimePointIRFs
compute_irfs_at_time(const Cube<double> &SD, // States draws
                     const Cube<double> &QD, // Q draws
                     const Mat<double> &VD,  // Volatility SDs
                     const Cube<double> &OM, // Volatilities
                     const Cube<double> &AA, // A elements
                     const Mat<double> &S1,  // S1 draws
                     const Cube<double> &S2, // S2 draws
                     const Mat<double> &YY,  // Lagged Y (N x L)
                     int tt,                 // Time index
                     int n_saved, int HOR, int N, int L,
                     int n_reps, // Number of replications
                     int h_restrict, bool SC, int FD) {

  TimePointIRFs result;
  result.supply.set_size(HOR, N);
  result.demand.set_size(HOR, N);
  result.global.set_size(HOR, N);
  result.supply.zeros();
  result.demand.zeros();
  result.global.zeros();
  result.valid = false;

  // Storage for accepted IRFs
  Cube<double> accepted_sirf(HOR, N, n_reps);
  Cube<double> accepted_dirf(HOR, N, n_reps);
  Cube<double> accepted_aggirf(HOR, N, n_reps);
  vec accepted_slope_S_D(n_reps);
  vec accepted_slope_S_A(n_reps);
  vec accepted_slope_D_D(n_reps);

  int accepted = 0;
  int max_attempts = n_reps * 10;
  int attempts = 0;

  while (accepted < n_reps && attempts < max_attempts) {
    attempts++;

    // Random draw from Gibbs output
    int K = (int)(randu() * n_saved);
    if (K >= n_saved)
      K = n_saved - 1;

    // Extract parameters
    vec sd = SD.slice(K).col(tt);
    vec aa = AA.slice(K).col(tt);
    Mat<double> qd = QD.slice(K);
    double s1 = S1(0, K);
    Mat<double> s2 = S2.slice(K);
    vec vd = VD.col(K);
    vec om = OM.slice(K).row(tt).t();

    // Compute GIRFs
    Mat<double> sirf, dirf, aggirf;
    OilMarketElasticities elast;

    bool success = GIRFs_N3_with_elasticities(sd, aa, qd, s1, s2, vd, om, YY,
                                              HOR, N, L, 1, 100, h_restrict, SC,
                                              FD, sirf, dirf, aggirf, elast);

    if (success) {
      accepted_sirf.slice(accepted) = sirf;
      accepted_dirf.slice(accepted) = dirf;
      accepted_aggirf.slice(accepted) = aggirf;
      accepted_slope_S_D(accepted) = elast.slope_S_D;
      accepted_slope_S_A(accepted) = elast.slope_S_A;
      accepted_slope_D_D(accepted) = elast.slope_D_D;
      accepted++;
    }
  }

  if (accepted == 0) {
    return result;
  }

  // Compute medians
  for (int h = 0; h < HOR; ++h) {
    for (int v = 0; v < N; ++v) {
      vec s_vals(accepted), d_vals(accepted), a_vals(accepted);
      for (int i = 0; i < accepted; ++i) {
        s_vals(i) = accepted_sirf(h, v, i);
        d_vals(i) = accepted_dirf(h, v, i);
        a_vals(i) = accepted_aggirf(h, v, i);
      }
      result.supply(h, v) = median(s_vals);
      result.demand(h, v) = median(d_vals);
      result.global(h, v) = median(a_vals);
    }
  }

  result.elast.slope_S_D = median(accepted_slope_S_D.head(accepted));
  result.elast.slope_S_A = median(accepted_slope_S_A.head(accepted));
  result.elast.slope_D_D = median(accepted_slope_D_D.head(accepted));
  result.valid = true;

  return result;
}

// Full IRF results structure
struct TVPIRFResult {
  Cube<double> IRF_supply; // HOR x N x T_eff
  Cube<double> IRF_demand; // HOR x N x T_eff
  Cube<double> IRF_global; // HOR x N x T_eff
  Mat<double> slope_S_D;   // n_reps x T_eff
  Mat<double> slope_S_A;   // n_reps x T_eff
  Mat<double> slope_D_D;   // n_reps x T_eff
  int irf_horizon;
  int T_eff;
  int n_time_points;
};

// Compute IRFs for all time periods
inline TVPIRFResult tvp_var_irfs_cpp(const TVPVARResult &tvp_result,
                                     const Mat<double> &y, int L,
                                     int training_years,
                                     int irf_horizon, // e.g., 21 = 5 years + 1
                                     int n_reps,      // e.g., 500
                                     int h_restrict,  // e.g., 1
                                     int FD, bool verbose = true) {

  TVPIRFResult result;

  int T = y.n_rows;
  int N = y.n_cols;
  int T0 = 4 * training_years - L;
  int T1 = T - L;
  int T_eff = T1 - T0;

  // Time points to compute (start from L to have enough lags)
  std::vector<int> time_indices;
  for (int t = L; t < T_eff; ++t) {
    time_indices.push_back(t);
  }
  int n_times = time_indices.size();

  result.irf_horizon = irf_horizon;
  result.T_eff = T_eff;
  result.n_time_points = n_times;

  result.IRF_supply.set_size(irf_horizon, N, n_times);
  result.IRF_demand.set_size(irf_horizon, N, n_times);
  result.IRF_global.set_size(irf_horizon, N, n_times);
  result.slope_S_D.set_size(1, n_times);
  result.slope_S_A.set_size(1, n_times);
  result.slope_D_D.set_size(1, n_times);

  result.IRF_supply.zeros();
  result.IRF_demand.zeros();
  result.IRF_global.zeros();

  // Create lag structure to get YS
  Cube<double> X;
  Mat<double> Y_mat, X1;
  lagdep(y, N, L, X, Y_mat, X1);
  Mat<double> YS = Y_mat.cols(T0, T1 - 1);

  if (verbose) {
    Rprintf("TVP-IRF: Computing IRFs for %d time points\n", n_times);
  }

  for (int idx = 0; idx < n_times; ++idx) {
    int tt = time_indices[idx];

    if (verbose && idx % 10 == 0) {
      Rprintf("TVP-IRF: Time point %d/%d\n", idx + 1, n_times);
    }

    // Get lagged Y for this time point
    // YS is N x T_eff, we need Y at times (tt-L) to (tt-1) relative to
    // estimation sample But we also need the original y for proper lagging
    Mat<double> YY(N, L);
    for (int lag = 0; lag < L; ++lag) {
      // tt is index in estimation sample (T0 onwards in original)
      // We need y from the estimation sample
      int orig_idx = T0 + tt - (L - lag);
      if (orig_idx >= L && orig_idx < T) {
        YY.col(lag) = y.row(orig_idx).t();
      }
    }

    TimePointIRFs tp = compute_irfs_at_time(
        tvp_result.SD, tvp_result.QD, tvp_result.VD, tvp_result.OM,
        tvp_result.AA, tvp_result.S1, tvp_result.S2, YY, tt, tvp_result.n_saved,
        irf_horizon, N, L, n_reps, h_restrict, false, FD);

    if (tp.valid) {
      result.IRF_supply.slice(idx) = tp.supply;
      result.IRF_demand.slice(idx) = tp.demand;
      result.IRF_global.slice(idx) = tp.global;
      result.slope_S_D(0, idx) = tp.elast.slope_S_D;
      result.slope_S_A(0, idx) = tp.elast.slope_S_A;
      result.slope_D_D(0, idx) = tp.elast.slope_D_D;
    }
  }

  if (verbose) {
    Rprintf("TVP-IRF: Completed\n");
  }

  return result;
}

#endif // TVP_IRF_H
