#ifndef TVP_REDUCED_H
#define TVP_REDUCED_H

/* ==========================================================================
   TVP-VAR Reduced Form Estimation Functions
   Following Baumeister and Peersman (2013) / Primiceri (2005)

   Ported from MATLAB folder: mymatprog/reduced/
   ========================================================================== */

// ============================================================================
// LAG STRUCTURE
// ============================================================================

// lagdep: Create VAR lag structure (lagdep.m)
// Creates lagged dependent variable structure for TVP-VAR
// X: tensor (N*(1+N*L) x N x T-L) - SUR format
// Y: matrix (N x T-L) - dependent variables
// X1: matrix (T-L x 1+N*L) - row format regressors
inline void lagdep(const Mat<double> &y, int N, int L, Cube<double> &X,
                   Mat<double> &Y, Mat<double> &X1) {
  int T = y.n_rows;
  int T_eff = T - L;

  // X1: [1, y(t-1), y(t-2), ..., y(t-L)] for each t
  X1.set_size(T_eff, 1 + N * L);
  X1.col(0).ones();

  for (int lag = 1; lag <= L; ++lag) {
    for (int t = 0; t < T_eff; ++t) {
      X1.submat(t, 1 + N * (lag - 1), t, N * lag) = y.row(L - lag + t);
    }
  }

  // Y: dependent variables (N x T_eff)
  Y = y.rows(L, T - 1).t();

  // X: tensor structure for SUR estimation
  int rx = N * (1 + N * L);
  X.set_size(rx, N, T_eff);
  X.zeros();

  for (int t = 0; t < T_eff; ++t) {
    for (int eq = 0; eq < N; ++eq) {
      int row_start = eq * (1 + N * L);
      int row_end = (eq + 1) * (1 + N * L) - 1;
      X.subcube(row_start, eq, t, row_end, eq, t) = X1.row(t).t();
    }
  }
}

// ============================================================================
// INITIAL ESTIMATION (SUR)
// ============================================================================

// surreg: Seemingly Unrelated Regression (surreg.m)
// Two-step GLS for initial OLS estimates
// Returns: SI (state prior mean), PI (state prior covariance), RI (residual
// covariance)
inline void surreg(const Mat<double> &Y, const Cube<double> &X, int T, vec &SI,
                   Mat<double> &PI, Mat<double> &RI) {
  int rx = X.n_rows;
  int N = Y.n_rows;

  // 1st stage: OLS
  Mat<double> Mxx(rx, rx, fill::zeros);
  vec Mxy(rx, fill::zeros);

  for (int t = 0; t < T; ++t) {
    Mxx += X.slice(t) * X.slice(t).t();
    Mxy += X.slice(t) * Y.col(t);
  }

  // Regularization for numerical stability
  Mxx += 1e-8 * eye<Mat<double>>(rx, rx);
  SI = solve(Mxx, Mxy);

  // 1st stage residuals
  Mat<double> e(N, T);
  for (int t = 0; t < T; ++t) {
    e.col(t) = Y.col(t) - X.slice(t).t() * SI;
  }

  // Residual covariance
  RI = (e * e.t()) / (T - 1);
  RI += 1e-8 * eye<Mat<double>>(N, N);

  // 2nd stage: GLS
  Mat<double> W = inv_sympd(RI);
  Mxx.zeros();
  Mxy.zeros();

  for (int t = 0; t < T; ++t) {
    Mxx += X.slice(t) * W * X.slice(t).t();
    Mxy += X.slice(t) * W * Y.col(t);
  }

  Mxx += 1e-8 * eye<Mat<double>>(rx, rx);
  SI = solve(Mxx, Mxy);

  // 2nd stage residuals
  for (int t = 0; t < T; ++t) {
    e.col(t) = Y.col(t) - X.slice(t).t() * SI;
  }

  RI = (e * e.t()) / (T - 1);
  RI += 1e-8 * eye<Mat<double>>(N, N);

  // Coefficient covariance
  W = inv_sympd(RI);
  Mxx.zeros();
  for (int t = 0; t < T; ++t) {
    Mxx += X.slice(t) * W * X.slice(t).t();
  }
  Mxx += 1e-8 * eye<Mat<double>>(rx, rx);
  PI = inv_sympd(Mxx);
}

// ============================================================================
// KALMAN FILTER
// ============================================================================

// kfP: Kalman filter for TVP-VAR (kfP.m)
// Forward filtering
// Uses time-varying R = inv(A)*diag(H)*inv(A)'
inline void kfP(const Mat<double> &YS, const Cube<double> &XS,
                const Mat<double> &Q, const Mat<double> &AA,
                const Mat<double> &H, const vec &SI, const Mat<double> &PI,
                int T, int N, int L, Mat<double> &S0, Cube<double> &P0,
                Cube<double> &P1) {
  int state_dim = N * (1 + N * L);

  S0.set_size(state_dim, T);
  P0.set_size(state_dim, state_dim, T);
  P1.set_size(state_dim, state_dim, T);

  // Date 1
  P1.slice(0) = PI;

  // A matrix from AA (AA has T+1 columns, use column 2 for t=1)
  Mat<double> A = chofac(N, AA.col(1));
  Mat<double> invA = inv(A + 1e-10 * eye<Mat<double>>(N, N));
  Mat<double> R = invA * diagmat(H.row(1).t()) * invA.t();

  Mat<double> X_t = XS.slice(0);
  Mat<double> KK = X_t.t() * P1.slice(0) * X_t + R;
  Mat<double> K = P1.slice(0) * X_t * inv(KK + 1e-10 * eye<Mat<double>>(N, N));
  P0.slice(0) = P1.slice(0) - K * X_t.t() * P1.slice(0);
  S0.col(0) = SI + K * (YS.col(0) - X_t.t() * SI);

  // Iterate through sample
  for (int i = 1; i < T; ++i) {
    P1.slice(i) = P0.slice(i - 1) + Q;

    A = chofac(N, AA.col(i + 1));
    invA = inv(A + 1e-10 * eye<Mat<double>>(N, N));
    R = invA * diagmat(H.row(i + 1).t()) * invA.t();

    X_t = XS.slice(i);
    KK = X_t.t() * P1.slice(i) * X_t + R;
    K = P1.slice(i) * X_t * inv(KK + 1e-10 * eye<Mat<double>>(N, N));
    P0.slice(i) = P1.slice(i) - K * X_t.t() * P1.slice(i);
    S0.col(i) = S0.col(i - 1) + K * (YS.col(i) - X_t.t() * S0.col(i - 1));
  }
}

// kfR: Simplified Kalman filter for initial states (kfR.m)
// Uses constant CF for A matrix
inline void kfR(const Mat<double> &YS, const Cube<double> &XS,
                const Mat<double> &Q, const Mat<double> &CF,
                const Mat<double> &H, const vec &SI, const Mat<double> &PI,
                int T, int N, int L, Mat<double> &S0, Cube<double> &P0,
                Cube<double> &P1) {
  int state_dim = N * (1 + N * L);

  S0.set_size(state_dim, T);
  P0.set_size(state_dim, state_dim, T);
  P1.set_size(state_dim, state_dim, T);

  Mat<double> invCF = inv(CF + 1e-10 * eye<Mat<double>>(N, N));

  // Date 1
  P1.slice(0) = PI;
  Mat<double> R = invCF * diagmat(H.row(1).t()) * invCF.t();

  Mat<double> X_t = XS.slice(0);
  Mat<double> KK = X_t.t() * P1.slice(0) * X_t + R;
  Mat<double> K = P1.slice(0) * X_t * inv(KK + 1e-10 * eye<Mat<double>>(N, N));
  P0.slice(0) = P1.slice(0) - K * X_t.t() * P1.slice(0);
  S0.col(0) = SI + K * (YS.col(0) - X_t.t() * SI);

  for (int i = 1; i < T; ++i) {
    P1.slice(i) = P0.slice(i - 1) + Q;
    R = invCF * diagmat(H.row(i + 1).t()) * invCF.t();

    X_t = XS.slice(i);
    KK = X_t.t() * P1.slice(i) * X_t + R;
    K = P1.slice(i) * X_t * inv(KK + 1e-10 * eye<Mat<double>>(N, N));
    P0.slice(i) = P1.slice(i) - K * X_t.t() * P1.slice(i);
    S0.col(i) = S0.col(i - 1) + K * (YS.col(i) - X_t.t() * S0.col(i - 1));
  }
}

// ============================================================================
// BACKWARD SAMPLING
// ============================================================================

// gibbs1: Carter-Kohn backward sampler (gibbs1.m)
// Draw states from p(theta | Q, R, Y)
inline Mat<double> gibbs1(const Mat<double> &S0, const Cube<double> &P0,
                          const Cube<double> &P1, int T, int N, int L) {
  int state_dim = N * (1 + N * L);
  Mat<double> SA(state_dim, T);

  // Terminal state: draw from N(S(T|T), P(T|T))
  Mat<double> P_term = P0.slice(T - 1);
  P_term = 0.5 * (P_term + P_term.t());
  P_term += 1e-8 * eye<Mat<double>>(state_dim, state_dim);
  Mat<double> P_sqrt = mysqrt_varm(P_term);
  SA.col(T - 1) = S0.col(T - 1) + P_sqrt * randn<vec>(state_dim);

  // Backward recursion
  for (int i = 1; i < T; ++i) {
    int t = T - 1 - i;

    Mat<double> P1_inv =
        inv(P1.slice(t + 1) + 1e-8 * eye<Mat<double>>(state_dim, state_dim));
    Mat<double> PM = P0.slice(t) * P1_inv;
    Mat<double> P = P0.slice(t) - PM * P0.slice(t);
    P = 0.5 * (P + P.t());
    P += 1e-8 * eye<Mat<double>>(state_dim, state_dim);

    vec SM = S0.col(t) + PM * (SA.col(t + 1) - S0.col(t));
    P_sqrt = mysqrt_varm(P);
    SA.col(t) = SM + P_sqrt * randn<vec>(state_dim);
  }

  return SA;
}

// ============================================================================
// Q MATRIX SAMPLING
// ============================================================================

// iwpQ: Compute posterior parameters for inverse Wishart (iwpQ.m)
inline void iwpQ(const Mat<double> &theta, int T, const Mat<double> &TQ0,
                 int df0, Mat<double> &TQ, int &DF) {
  // Innovations are observable: v(t) = theta(t) - theta(t-1)
  Mat<double> v = theta.cols(1, T - 1) - theta.cols(0, T - 2);
  TQ = TQ0 + v * v.t();
  DF = df0 + T - 1;
}

// gibbs2Q: Draw Q from inverse Wishart (gibbs2Q.m)
inline Mat<double> gibbs2Q(const Mat<double> &TQ, int DF, int N, int L) {
  int n = N * (1 + N * L);

  Mat<double> TQ_reg = TQ + 1e-8 * eye<Mat<double>>(n, n);
  TQ_reg = 0.5 * (TQ_reg + TQ_reg.t());

  // Draw from Wishart(TQ^{-1}, DF), then invert
  Mat<double> TQ_inv = inv_sympd(TQ_reg);
  Mat<double> W = wishrnd_arma(TQ_inv, DF);

  return inv_sympd(W + 1e-10 * eye<Mat<double>>(n, n));
}

// ============================================================================
// RESIDUALS AND INNOVATIONS
// ============================================================================

// innovm: Calculate innovations for measurement equation (innovm.m)
// Returns U (N x T) - VAR residuals
inline Mat<double> innovm(const Mat<double> &YS, const Mat<double> &X1,
                          const Mat<double> &SA, int N, int T, int L) {
  Mat<double> U(N, T);
  int k = 1 + N * L; // number of regressors per equation

  for (int i = 0; i < N; ++i) {
    for (int t = 0; t < T; ++t) {
      // Extract coefficients for equation i at time t
      vec theta_i = SA.col(t).subvec(i * k, (i + 1) * k - 1);
      U(i, t) = YS(i, t) - dot(X1.row(t).t(), theta_i);
    }
  }

  return U;
}

// ============================================================================
// STOCHASTIC VOLATILITY
// ============================================================================

// svmh: Metropolis-Hastings for stochastic volatility interior dates (svmh.m)
// h(t) | h(t+1), h(t-1), sv, f(t)
inline double svmh(double hlead, double hlag, double alpha, double delta,
                   double sv, double yt, double hlast) {
  // Ensure positive inputs
  hlead = std::max(hlead, 1e-10);
  hlag = std::max(hlag, 1e-10);
  hlast = std::max(hlast, 1e-10);

  // Conditional mean and variance for log(h)
  double mu = alpha * (1 - delta) +
              delta * (log(hlead) + log(hlag)) / (1 + delta * delta);
  double ss = (sv * sv) / (1 + delta * delta);
  ss = std::max(ss, 1e-10);

  // Candidate draw from lognormal
  double htrial = exp(mu + sqrt(ss) * randn());
  htrial = std::max(htrial, 1e-10);

  // Acceptance probability (MH step)
  double lp1 = -0.5 * log(htrial) - (yt * yt) / (2 * htrial);
  double lp0 = -0.5 * log(hlast) - (yt * yt) / (2 * hlast);
  double accept = std::min(1.0, exp(lp1 - lp0));

  if (randu() <= accept) {
    return htrial;
  }
  return hlast;
}

// svmh0: MH for stochastic volatility at t=0 (svmh0.m)
// h(0) | h(1)
inline double svmh0(double hlead, double alpha, double delta, double sv,
                    double mu0, double ss0) {
  hlead = std::max(hlead, 1e-10);

  double denom = ss0 + sv * sv / (delta * delta);
  double mu = (ss0 * (log(hlead) - alpha) / delta + sv * sv * mu0) / denom;
  double ss = (ss0 * sv * sv / (delta * delta)) / denom;
  ss = std::max(ss, 1e-10);

  return exp(mu + sqrt(ss) * randn());
}

// svmhT: MH for stochastic volatility at t=T (svmhT.m)
// h(T) | h(T-1), f(T)
inline double svmhT(double hlag, double alpha, double delta, double sv,
                    double yt, double hlast) {
  hlag = std::max(hlag, 1e-10);
  hlast = std::max(hlast, 1e-10);

  double mu = alpha + delta * log(hlag);
  double ss = sv * sv;
  ss = std::max(ss, 1e-10);

  double htrial = exp(mu + sqrt(ss) * randn());
  htrial = std::max(htrial, 1e-10);

  double lp1 = -0.5 * log(htrial) - (yt * yt) / (2 * htrial);
  double lp0 = -0.5 * log(hlast) - (yt * yt) / (2 * hlast);
  double accept = std::min(1.0, exp(lp1 - lp0));

  if (randu() <= accept) {
    return htrial;
  }
  return hlast;
}

// ============================================================================
// OFF-DIAGONAL A(t) ELEMENTS
// ============================================================================

// getalpha: Draw off-diagonal A(t) elements using Kalman filter (getalpha.m)
// For N=3: alpha21 (scalar), then [alpha31, alpha32]
inline Mat<double> getalpha(const Mat<double> &U, double s1,
                            const Mat<double> &s2, const vec &muA0,
                            const Mat<double> &ssA0, int T,
                            const Mat<double> &H) {
  int n_aa = muA0.n_elem; // Should be 3 for N=3
  Mat<double> AA(n_aa, T + 1);

  // Prepend zero to U (MATLAB: u = [zeros(1,3); U'])
  Mat<double> u(T + 1, U.n_rows, fill::zeros);
  u.rows(1, T) = U.t();

  // ======= Block 1: alpha21 (scalar Kalman filter) =======
  double Q1 = s1;

  vec st(T + 1), Pt(T + 1), s_pred(T + 1), P_pred(T + 1);
  vec sT(T + 1), PT(T + 1);

  st(0) = muA0(0);
  Pt(0) = ssA0(0, 0);
  s_pred(0) = muA0(0);
  P_pred(0) = ssA0(0, 0) + Q1;

  // Forward filter
  for (int t = 1; t <= T; ++t) {
    double H_coef = -u(t, 0); // -u1(t)
    double y_obs = u(t, 1);   // u2(t)
    double R_t = H(t, 1);     // h2(t)

    double uhat = y_obs - H_coef * s_pred(t - 1);
    double denom = R_t + H_coef * P_pred(t - 1) * H_coef;
    denom = std::max(std::abs(denom), 1e-10);
    double G = (P_pred(t - 1) * H_coef) / denom;

    st(t) = s_pred(t - 1) + G * uhat;
    Pt(t) = P_pred(t - 1) - G * H_coef * P_pred(t - 1);
    Pt(t) = std::max(Pt(t), 1e-10);

    s_pred(t) = st(t);
    P_pred(t) = Pt(t) + Q1;
    P_pred(t) = std::max(P_pred(t), 1e-10);
  }

  // Backward sampling
  PT(T) = Pt(T);
  for (int t = T - 1; t >= 0; --t) {
    double P_safe = std::max(P_pred(t), 1e-10);
    PT(t) = Pt(t) - Pt(t) * Pt(t) / P_safe;
    PT(t) = std::max(PT(t), 1e-10);
  }

  sT(T) = st(T) + randn() * sqrt(PT(T));
  for (int t = T - 1; t >= 0; --t) {
    double P_safe = std::max(P_pred(t), 1e-10);
    double ratio = Pt(t) / P_safe;
    sT(t) = st(t) + ratio * (sT(t + 1) - s_pred(t)) + randn() * sqrt(PT(t));
  }
  sT(0) = muA0(0); // Prior at t=0

  AA.row(0) = sT.t();

  // ======= Block 2: [alpha31, alpha32] (2D Kalman filter) =======
  if (n_aa >= 3) {
    Mat<double> Q2 = s2;

    Cube<double> st2(2, 1, T + 1), Pt2(2, 2, T + 1);
    Cube<double> s_pred2(2, 1, T + 1), P_pred2(2, 2, T + 1);
    Cube<double> sT2(2, 1, T + 1), PT2(2, 2, T + 1);

    st2.slice(0).col(0) = muA0.subvec(1, 2);
    Pt2.slice(0) = ssA0.submat(1, 1, 2, 2);
    s_pred2.slice(0).col(0) = muA0.subvec(1, 2);
    P_pred2.slice(0) = Pt2.slice(0) + Q2;

    // Forward filter
    for (int t = 1; t <= T; ++t) {
      vec H_vec = {-u(t, 0), -u(t, 1)}; // [-u1(t), -u2(t)]
      double y_obs = u(t, 2);           // u3(t)
      double R_t = H(t, 2);             // h3(t)

      double uhat = y_obs - dot(H_vec, s_pred2.slice(t - 1).col(0));
      double denom = R_t + dot(H_vec, P_pred2.slice(t - 1) * H_vec);
      denom = std::max(std::abs(denom), 1e-10);
      vec G = (P_pred2.slice(t - 1) * H_vec) / denom;

      st2.slice(t).col(0) = s_pred2.slice(t - 1).col(0) + G * uhat;
      Pt2.slice(t) =
          P_pred2.slice(t - 1) - G * H_vec.t() * P_pred2.slice(t - 1);
      Pt2.slice(t) = 0.5 * (Pt2.slice(t) + Pt2.slice(t).t());
      Pt2.slice(t) += 1e-10 * eye<Mat<double>>(2, 2);

      s_pred2.slice(t).col(0) = st2.slice(t).col(0);
      P_pred2.slice(t) = Pt2.slice(t) + Q2;
      P_pred2.slice(t) = 0.5 * (P_pred2.slice(t) + P_pred2.slice(t).t());
    }

    // Backward sampling
    PT2.slice(T) = Pt2.slice(T);
    for (int t = T - 1; t >= 0; --t) {
      Mat<double> P_inv = inv(P_pred2.slice(t) + 1e-8 * eye<Mat<double>>(2, 2));
      PT2.slice(t) = Pt2.slice(t) - Pt2.slice(t) * P_inv * Pt2.slice(t);
      PT2.slice(t) = 0.5 * (PT2.slice(t) + PT2.slice(t).t());
      PT2.slice(t) += 1e-10 * eye<Mat<double>>(2, 2);
    }

    Mat<double> PT_sqrt = mysqrt_varm(PT2.slice(T));
    sT2.slice(T).col(0) = st2.slice(T).col(0) + PT_sqrt * randn<vec>(2);

    for (int t = T - 1; t >= 0; --t) {
      Mat<double> P_inv = inv(P_pred2.slice(t) + 1e-8 * eye<Mat<double>>(2, 2));
      vec s_mean = st2.slice(t).col(0) +
                   Pt2.slice(t) * P_inv *
                       (sT2.slice(t + 1).col(0) - s_pred2.slice(t).col(0));
      PT_sqrt = mysqrt_varm(PT2.slice(t));
      sT2.slice(t).col(0) = s_mean + PT_sqrt * randn<vec>(2);
    }
    sT2.slice(0).col(0) = muA0.subvec(1, 2);

    for (int t = 0; t <= T; ++t) {
      AA(1, t) = sT2.slice(t)(0, 0);
      AA(2, t) = sT2.slice(t)(1, 0);
    }
  }

  return AA;
}

// bayesreg: Bayesian regression for A elements warmup (bayesreg.m)
inline vec bayesreg(const vec &b0, const Mat<double> &V0, double sig2,
                    const vec &y, const Mat<double> &X) {
  int k = X.n_cols;

  Mat<double> V0_inv = inv(V0 + 1e-8 * eye<Mat<double>>(k, k));
  Mat<double> V1 = inv(V0_inv + X.t() * X / sig2);
  V1 = 0.5 * (V1 + V1.t());

  vec b1 = V1 * (V0_inv * b0 + X.t() * y / sig2);

  Mat<double> V1_sqrt = mysqrt_varm(V1);
  return b1 + V1_sqrt * randn<vec>(k);
}

#endif // TVP_REDUCED_H
