#ifndef TVP_ESTIMATE_H
#define TVP_ESTIMATE_H

/* ==========================================================================
   TVP-VAR Estimation with Stochastic Volatility
   Following Baumeister and Peersman (2013) / Primiceri (2005)

   Ported from MATLAB: TVP_BURN.m, TVP_MCMC.m
   ========================================================================== */

// ============================================================================
// TVP-VAR ESTIMATION
// ============================================================================

// Main estimation structure to hold results
struct TVPVARResult {
  Cube<double> SD; // States (state_dim x T_eff x n_save)
  Cube<double> QD; // Q matrices (state_dim x state_dim x n_save)
  Mat<double> VD;  // Volatility SDs (N x n_save)
  Cube<double> OM; // Volatilities (T_eff x N x n_save)
  Cube<double> AA; // Off-diagonal A elements (n_aa x T_eff+1 x n_save)
  Mat<double> S1;  // S1 scalar (1 x n_save)
  Cube<double> S2; // S2 matrix (n_aa-1 x n_aa-1 x n_save)
  int n_saved;
  int T_eff;
  int N;
  int L;
};

// Single Gibbs iteration
// Returns true if iteration successful (stable if SC=1)
inline bool tvp_gibbs_iteration(
    // Input/output: current state
    Mat<double> &SA_current, // state_dim x T_eff
    Mat<double> &Q_current,  // state_dim x state_dim
    vec &SV_current,         // N
    Mat<double> &H_current,  // T_eff+1 x N
    Mat<double> &AA_current, // n_aa x T_eff+1
    double &S1_current, Mat<double> &S2_current,
    // Previous iteration (for MH steps)
    const Mat<double> &H_prev, const Mat<double> &AA_prev,
    // Data
    const Mat<double> &YS,  // N x T_eff
    const Cube<double> &XS, // state_dim x N x T_eff
    const Mat<double> &X1,  // T_eff x (1+N*L)
    // Priors
    const vec &SI, const Mat<double> &PI, const Mat<double> &TQ0, int df,
    const vec &mu0, double ss0, double v0, double d0, const vec &muA0,
    const Mat<double> &ssA0, double TS1_0, int dfS1, const Mat<double> &TS2_0,
    int dfS2,
    // Dimensions
    int T_eff, int N, int L,
    // Settings
    bool SC) {

  int n_aa = N * (N - 1) / 2;

  // =========================================================================
  // Step 1: Draw states via Kalman filter + backward sampling
  // =========================================================================
  Mat<double> S0;
  Cube<double> P0, P1;
  kfP(YS, XS, Q_current, AA_prev, H_prev, SI, PI, T_eff, N, L, S0, P0, P1);

  Mat<double> SA_draw = gibbs1(S0, P0, P1, T_eff, N, L);

  // Check stability if required
  if (SC) {
    for (int t = 0; t < T_eff; ++t) {
      Mat<double> b(N, N * L + 1);
      for (int eq = 0; eq < N; ++eq) {
        b.row(eq) = SA_draw.col(t)
                        .subvec(eq * (1 + N * L), (eq + 1) * (1 + N * L) - 1)
                        .t();
      }
      if (varroots(L, N, b) >= 1.0) {
        return false; // Unstable, reject
      }
    }
  }
  SA_current = SA_draw;

  // =========================================================================
  // Step 2: Draw Q from inverse Wishart
  // =========================================================================
  Mat<double> TQ_post;
  int DF_post;
  iwpQ(SA_draw, T_eff, TQ0, df, TQ_post, DF_post);
  Q_current = gibbs2Q(TQ_post, DF_post, N, L);

  // =========================================================================
  // Step 3: Compute VAR residuals
  // =========================================================================
  Mat<double> U_current = innovm(YS, X1, SA_draw, N, T_eff, L);

  // =========================================================================
  // Step 4: Draw S1 and S2 (covariances for A innovations)
  // =========================================================================

  // S1 (scalar, for alpha21)
  rowvec aa1 = AA_prev.row(0);
  rowvec aa1_diff = aa1.cols(1, T_eff) - aa1.cols(0, T_eff - 1);
  double TS1_post = TS1_0 + dot(aa1_diff, aa1_diff);
  int DFS1_post = dfS1 + T_eff;

  // Draw from inverse chi-square
  double chi_sq = 0.0;
  for (int j = 0; j < DFS1_post; ++j) {
    double z = randn();
    chi_sq += z * z;
  }
  chi_sq = std::max(chi_sq, 1e-10);
  S1_current = TS1_post / chi_sq;

  // S2 (matrix, for alpha31, alpha32)
  if (n_aa > 1) {
    Mat<double> aa2 = AA_prev.rows(1, n_aa - 1);
    Mat<double> aa2_diff = aa2.cols(1, T_eff) - aa2.cols(0, T_eff - 1);
    Mat<double> TS2_post = TS2_0 + aa2_diff * aa2_diff.t();
    int DFS2_post = dfS2 + T_eff;
    S2_current = iwishrnd_arma(TS2_post, DFS2_post);
  }

  // =========================================================================
  // Step 5: Draw off-diagonal A elements
  // =========================================================================
  AA_current =
      getalpha(U_current, S1_current, S2_current, muA0, ssA0, T_eff, H_prev);

  // =========================================================================
  // Step 6: Orthogonalize VAR residuals
  // =========================================================================
  Mat<double> f(T_eff, N);
  for (int t = 0; t < T_eff; ++t) {
    Mat<double> CF = chofac(N, AA_current.col(t + 1));
    f.row(t) = (CF * U_current.col(t)).t();
  }

  // =========================================================================
  // Step 7: Draw stochastic volatilities
  // =========================================================================
  Mat<double> lh_prev = log(clamp(H_prev, 1e-10, 1e10));

  for (int i = 0; i < N; ++i) {
    // Volatility innovations from previous H
    vec eh(T_eff);
    for (int t = 0; t < T_eff; ++t) {
      eh(t) = lh_prev(t + 1, i) - lh_prev(t, i);
    }

    // Update volatility innovation variance via inverse gamma
    double v_new = sqrt(ig2(v0, d0, eh));
    SV_current(i) = v_new;

    // MH step at t=0
    H_current(0, i) = svmh0(H_prev(1, i), 0.0, 1.0, SV_current(i), mu0(i), ss0);

    // MH steps at interior dates
    for (int t = 1; t < T_eff; ++t) {
      H_current(t, i) = svmh(H_prev(t + 1, i), H_current(t - 1, i), 0.0, 1.0,
                             SV_current(i), f(t - 1, i), H_prev(t, i));
    }

    // MH step at t=T
    H_current(T_eff, i) =
        svmhT(H_current(T_eff - 1, i), 0.0, 1.0, SV_current(i), f(T_eff - 1, i),
              H_prev(T_eff, i));
  }

  return true;
}

// Main TVP-VAR estimation function
// Combines burn-in and MCMC phases
inline TVPVARResult
tvp_var_estimate_cpp(const Mat<double> &y, // T x N
                     int L,                // Lag order
                     int training_years,   // Training sample in years
                     int n_burn,           // Burn-in iterations
                     int n_draws,          // Draws to keep
                     int thinning,         // Keep every D-th draw
                     bool SC,              // Stability constraint
                     double lambda_param,  // Time variation parameter
                     bool FD,              // First differences
                     bool verbose = true) {

  TVPVARResult result;

  int T = y.n_rows;
  int N = y.n_cols;

  // Training sample
  int T0 = 4 * training_years - L;
  int T1 = T - L;
  int T_eff = T1 - T0;

  result.T_eff = T_eff;
  result.N = N;
  result.L = L;

  if (verbose) {
    Rprintf("TVP-VAR: T=%d, N=%d, L=%d, T_eff=%d\n", T, N, L, T_eff);
  }

  // Create lag structure
  Cube<double> X;
  Mat<double> Y_mat, X1;
  lagdep(y, N, L, X, Y_mat, X1);

  // Partition data
  Mat<double> Y0 = Y_mat.cols(0, T0 - 1);
  Mat<double> YS = Y_mat.cols(T0, T1 - 1);
  Cube<double> X0 = X.slices(0, T0 - 1);
  Cube<double> XS = X.slices(T0, T1 - 1);
  Mat<double> XS1 = X1.rows(T0, T1 - 1);

  // =========================================================================
  // SET PRIORS
  // =========================================================================

  // Initial estimates via SUR
  vec SI;
  Mat<double> PI, RI;
  surreg(Y0, X0, T0, SI, PI, RI);

  int state_dim = N * (1 + N * L);
  int n_aa = N * (N - 1) / 2;

  // Prior for Q
  int df = (T0 < state_dim) ? state_dim + 1 : T0;
  double lambda_scale = (lambda_param < 0.001) ? 0.0001 : lambda_param;
  Mat<double> Q0 = lambda_scale * PI;
  Mat<double> TQ0 = df * Q0;
  PI = 4 * PI; // Inflate prior variance

  // Initial A and H from Cholesky of RI
  Mat<double> B = chol(RI + 1e-8 * eye<Mat<double>>(N, N), "lower");
  Mat<double> invA(N, N, fill::zeros);
  for (int j = 0; j < N; ++j) {
    invA.col(j) = B.col(j) / B(j, j);
  }
  Mat<double> A = inv(invA + 1e-8 * eye<Mat<double>>(N, N));
  vec H_diag = square(diagvec(B));

  vec mu0 = log(clamp(H_diag, 1e-10, 1e10));
  double ss0 = 10.0;

  double sv0 = 0.01;
  double v0 = 1.0;
  double d0 = sv0 * sv0;

  vec muA0 = stackA(A);
  Mat<double> ssA0 = diagmat(clamp(abs(muA0), 0.01, 1e10)) * 10;

  // Priors for S1, S2
  int dfS1 = 2;
  double S1_0 = std::max(std::abs(muA0(0)), 0.01) * 10 * 0.0001;
  double TS1_0 = dfS1 * S1_0;

  int dfS2 = 3;
  vec muA0_sub = clamp(abs(muA0.subvec(1, n_aa - 1)), 0.01, 1e10);
  Mat<double> S2_0 = diagmat(muA0_sub) * 10 * 0.0001;
  Mat<double> TS2_0 = dfS2 * S2_0;

  // =========================================================================
  // INITIALIZE STORAGE
  // =========================================================================

  int n_save = n_draws / thinning;
  result.SD.set_size(state_dim, T_eff, n_save);
  result.QD.set_size(state_dim, state_dim, n_save);
  result.VD.set_size(N, n_save);
  result.OM.set_size(T_eff, N, n_save);
  result.AA.set_size(n_aa, T_eff + 1, n_save);
  result.S1.set_size(1, n_save);
  result.S2.set_size(n_aa - 1, n_aa - 1, n_save);

  // =========================================================================
  // INITIALIZE MCMC
  // =========================================================================

  // Current state
  Mat<double> SA_current(state_dim, T_eff);
  for (int t = 0; t < T_eff; ++t) {
    SA_current.col(t) = SI;
  }

  Mat<double> Q_current = Q0;

  vec SV_current(N);
  SV_current.fill(sv0);

  Mat<double> H_current(T_eff + 1, N);
  H_current.row(0) = exp(mu0.t());
  // Initialize from residual variance
  Mat<double> e = YS;
  for (int t = 0; t < T_eff; ++t) {
    e.col(t) = YS.col(t) - XS.slice(t).t() * SI;
  }
  for (int t = 1; t <= T_eff; ++t) {
    H_current.row(t) = square(e.col(t - 1).t()) + 1e-6;
  }

  Mat<double> AA_current(n_aa, T_eff + 1);
  for (int t = 0; t <= T_eff; ++t) {
    AA_current.col(t) = muA0;
  }

  double S1_current = S1_0;
  Mat<double> S2_current = S2_0;

  // Previous iteration storage
  Mat<double> H_prev = H_current;
  Mat<double> AA_prev = AA_current;

  // =========================================================================
  // GIBBS SAMPLER
  // =========================================================================

  int total_iter = n_burn + n_draws * thinning;
  int progress_interval = std::max(1, total_iter / 20);
  int save_idx = 0;
  int unstable_count = 0;

  if (verbose) {
    Rprintf("TVP-VAR: Starting Gibbs sampler (%d burn-in + %d draws)\n", n_burn,
            n_draws * thinning);
  }

  for (int iter = 0; iter < total_iter; ++iter) {
    // Progress reporting
    if (verbose && (iter % progress_interval == 0 || iter == total_iter - 1)) {
      double pct = 100.0 * iter / total_iter;
      Rprintf("TVP-VAR: Iteration %d/%d (%.1f%%)\n", iter, total_iter, pct);
    }

    // Single Gibbs iteration
    bool success = tvp_gibbs_iteration(
        SA_current, Q_current, SV_current, H_current, AA_current, S1_current,
        S2_current, H_prev, AA_prev, YS, XS, XS1, SI, PI, TQ0, df, mu0, ss0, v0,
        d0, muA0, ssA0, TS1_0, dfS1, TS2_0, dfS2, T_eff, N, L, SC);

    if (!success) {
      unstable_count++;
      // Reset to previous stable draw
      continue;
    }

    // Update previous iteration storage
    H_prev = H_current;
    AA_prev = AA_current;

    // Save draws after burn-in with thinning
    if (iter >= n_burn && (iter - n_burn) % thinning == 0 &&
        save_idx < n_save) {
      result.SD.slice(save_idx) = SA_current;
      result.QD.slice(save_idx) = Q_current;
      result.VD.col(save_idx) = SV_current;
      result.OM.slice(save_idx) = H_current.rows(0, T_eff - 1);
      result.AA.slice(save_idx) = AA_current;
      result.S1(0, save_idx) = S1_current;
      result.S2.slice(save_idx) = S2_current;
      save_idx++;
    }
  }

  result.n_saved = save_idx;

  if (verbose) {
    Rprintf("TVP-VAR: Completed. Saved %d draws, rejected %d unstable.\n",
            save_idx, unstable_count);
  }

  return result;
}

#endif // TVP_ESTIMATE_H
