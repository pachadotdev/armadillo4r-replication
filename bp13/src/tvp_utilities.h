#ifndef TVP_UTILITIES_H
#define TVP_UTILITIES_H

/* ==========================================================================
   TVP-VAR Utility Functions
   Following Baumeister and Peersman (2013) / Primiceri (2005)

   Ported from MATLAB folders: mymatprog/utilities/, mymatprog/reduced/
   ========================================================================== */

// ============================================================================
// MATRIX UTILITIES
// ============================================================================

// chofac: Transform vector to lower triangular matrix with 1s on diagonal
// Ported from mymatprog/reduced/chofac.m
inline Mat<double> chofac(int N, const vec &aa) {
  Mat<double> A(N, N, fill::eye);
  int idx = 0;
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      A(i, j) = aa(idx);
      idx++;
    }
  }
  return A;
}

// stackA: Extract lower triangular elements row by row (stackA.m)
// Inverse of chofac
inline vec stackA(const Mat<double> &A) {
  int N = A.n_rows;
  int n_elem = N * (N - 1) / 2;
  vec V(n_elem);
  int idx = 0;
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      V(idx) = A(i, j);
      idx++;
    }
  }
  return V;
}

// chovec: Extract Cholesky elements column by column (chovec.m)
inline vec chovec(const Mat<double> &A) {
  int N = A.n_rows;
  int n_elem = N * (N + 1) / 2;
  vec V(n_elem);
  int idx = 0;
  for (int j = 0; j < N; ++j) {
    for (int i = j; i < N; ++i) {
      V(idx) = A(i, j);
      idx++;
    }
  }
  return V;
}

// myvec: Vectorize matrix column by column
inline vec myvec(const Mat<double> &M) { return vectorise(M); }

// myfliplr: Flip matrix left-right
inline Mat<double> myfliplr(const Mat<double> &M) { return fliplr(M); }

// mysqrt_varm: Matrix square root of variance-covariance matrix
// Uses Cholesky for positive semi-definite matrices
inline Mat<double> mysqrt_varm(const Mat<double> &V) {
  Mat<double> V_reg = V;
  // Ensure symmetry
  V_reg = 0.5 * (V_reg + V_reg.t());
  // Add small regularization
  V_reg += 1e-10 * eye<Mat<double>>(V.n_rows, V.n_cols);

  Mat<double> L;
  bool ok = chol(L, V_reg, "lower");
  if (!ok) {
    // Fallback to eigenvalue decomposition
    vec eigval;
    Mat<double> eigvec;
    eig_sym(eigval, eigvec, V_reg);
    // Clamp negative eigenvalues
    for (uword i = 0; i < eigval.n_elem; ++i) {
      if (eigval(i) < 1e-10)
        eigval(i) = 1e-10;
    }
    L = eigvec * diagmat(sqrt(eigval)) * eigvec.t();
  }
  return L;
}

// ============================================================================
// VAR UTILITIES
// ============================================================================

// varroots: Check VAR stability
inline double varroots(int L, int N, const Mat<double> &B) {
  // B is N x (1 + N*L), first column is constant
  Mat<double> B_coef = B.cols(1, B.n_cols - 1); // N x (N*L)

  // Build companion matrix
  Mat<double> F(N * L, N * L, fill::zeros);
  F.submat(0, 0, N - 1, N * L - 1) = B_coef;

  if (L > 1) {
    F.submat(N, 0, N * L - 1, N * (L - 1) - 1) =
        eye<Mat<double>>(N * (L - 1), N * (L - 1));
  }

  cx_vec eigval = eig_gen(F);
  return abs(eigval).max();
}

// varcompanion: Create companion form matrix
inline Mat<double> varcompanion(int N, int L, const Mat<double> &B) {
  Mat<double> B_coef = B.cols(1, B.n_cols - 1);
  Mat<double> F(N * L, N * L, fill::zeros);
  F.submat(0, 0, N - 1, N * L - 1) = B_coef;
  if (L > 1) {
    F.submat(N, 0, N * L - 1, N * (L - 1) - 1) =
        eye<Mat<double>>(N * (L - 1), N * (L - 1));
  }
  return F;
}

// ============================================================================
// RANDOM NUMBER UTILITIES
// ============================================================================

// ig2: Draw from inverse gamma distribution (ig2.m)
// Prior: v0 degree of freedom, d0 scale
// Returns: draw from IG(v1/2, d1/2) where v1 = v0 + T, d1 = d0 + sum(x^2)
inline double ig2(double v0, double d0, const vec &x) {
  int T = x.n_elem;
  double v1 = v0 + T;
  double d1 = d0 + dot(x, x);

  // Draw from chi-square(v1), then compute d1/chi_sq
  double chi_sq = 0.0;
  for (int i = 0; i < (int)v1; ++i) {
    double z = randn();
    chi_sq += z * z;
  }
  if (chi_sq < 1e-10)
    chi_sq = 1e-10;

  return d1 / chi_sq;
}

// wishrnd_arma: Draw from Wishart distribution
inline Mat<double> wishrnd_arma(const Mat<double> &S, int df) {
  int n = S.n_rows;
  Mat<double> S_reg = S + 1e-10 * eye<Mat<double>>(n, n);
  S_reg = 0.5 * (S_reg + S_reg.t());

  Mat<double> W;
  bool ok = wishrnd(W, S_reg, (double)df);
  if (!ok) {
    // Fallback: use Bartlett decomposition manually
    Mat<double> L = mysqrt_varm(S_reg);
    Mat<double> A(n, n, fill::zeros);
    for (int i = 0; i < n; ++i) {
      A(i, i) = sqrt(chi2rnd(df - i));
      for (int j = 0; j < i; ++j) {
        A(i, j) = randn();
      }
    }
    W = L * A * A.t() * L.t();
  }
  return W;
}

// iwishrnd_arma: Draw from Inverse Wishart distribution
inline Mat<double> iwishrnd_arma(const Mat<double> &S, int df) {
  Mat<double> W = wishrnd_arma(S, df);
  return inv_sympd(W + 1e-10 * eye<Mat<double>>(W.n_rows, W.n_cols));
}

// ============================================================================
// DATA TRANSFORMATION UTILITIES
// ============================================================================

// lagn: Compute first differences (or n-th differences)
inline Mat<double> lagn(const Mat<double> &X, int n = 1) {
  if (n == 0)
    return X;
  int T = X.n_rows;
  Mat<double> result = X.rows(n, T - 1) - X.rows(0, T - n - 1);
  return result;
}

// databasics: Basic data transformations (databasics.m)
// Returns data after removing L lags and training period TP
inline void databasics(const Mat<double> &data, const vec &time_full, int L,
                       int TP, Mat<double> &YS, vec &time_out) {
  int T = data.n_rows;
  int T0 = 4 * TP - L; // Training sample end
  int T1 = T - L;      // Full sample end

  // YS is transposed (N x T_eff)
  YS = data.rows(L + T0, T1 - 1).t();
  time_out = time_full.subvec(L + T0, T1 - 1);
}

#endif // TVP_UTILITIES_H
