// Impulse Response Function computation
// Port of comp_irf.m from CMW26

// Compute IRFs from state-space representation
// ME: nx x ne matrix (impact of shocks on states)
// MX: nx x nx matrix (state transition)
// MY: ny x nx matrix (observation equation)
// nrep: number of periods
// kk: which shock (1-indexed)
//
// Returns list with YY (ny x nrep) and XX (nx x nrep)
[[cpp4r::register]] list comp_irf(const doubles_matrix<>& ME_r,
                                   const doubles_matrix<>& MX_r,
                                   const doubles_matrix<>& MY_r,
                                   int nrep,
                                   int kk) {
  Mat<double> ME = as_Mat(ME_r);
  Mat<double> MX = as_Mat(MX_r);
  Mat<double> MY = as_Mat(MY_r);

  int nx = ME.n_rows;
  int ne = ME.n_cols;

  // Create unit shock vector (1 at position kk-1, 0 elsewhere)
  // kk is 1-indexed from R
  vec shock(ne, fill::zeros);
  shock(kk - 1) = 1.0;

  // State response matrix
  Mat<double> XX(nx, nrep, fill::zeros);

  // Initial impact
  XX.col(0) = ME * shock;

  // Iterate forward
  for (int t = 1; t < nrep; t++) {
    XX.col(t) = MX * XX.col(t - 1);
  }

  // Observable response
  Mat<double> YY = MY * XX;

  // Return transposed (nrep x ny and nrep x nx) to match MATLAB output
  writable::list result;
  result.push_back({"YY"_nm = as_doubles_matrix(YY.t())});
  result.push_back({"XX"_nm = as_doubles_matrix(XX.t())});

  return result;
}

// Compute IRFs for all shocks at once
// Returns 3D array-like structure: list of shock responses
[[cpp4r::register]] list comp_irf_all(const doubles_matrix<>& ME_r,
                                       const doubles_matrix<>& MX_r,
                                       const doubles_matrix<>& MY_r,
                                       int nrep) {
  Mat<double> ME = as_Mat(ME_r);
  Mat<double> MX = as_Mat(MX_r);
  Mat<double> MY = as_Mat(MY_r);

  int nx = ME.n_rows;
  int ne = ME.n_cols;

  writable::list result;

  for (int k = 0; k < ne; k++) {
    // Unit shock
    vec shock(ne, fill::zeros);
    shock(k) = 1.0;

    // State response
    Mat<double> XX(nx, nrep, fill::zeros);
    XX.col(0) = ME * shock;
    for (int t = 1; t < nrep; t++) {
      XX.col(t) = MX * XX.col(t - 1);
    }

    // Observable response
    Mat<double> YY = MY * XX;

    writable::list shock_result;
    shock_result.push_back({"YY"_nm = as_doubles_matrix(YY.t())});
    shock_result.push_back({"XX"_nm = as_doubles_matrix(XX.t())});
    result.push_back(shock_result);
  }

  return result;
}

// Compute VAR IRFs from reduced form coefficients
// B: (n_var * n_lags + constant) x n_var coefficient matrix
// Sigma: n_var x n_var covariance matrix
// n_var: number of variables
// n_lags: number of lags
// n_hor: horizon for IRFs
// 
// Returns n_hor x n_var x n_var array of IRFs (as list of matrices)
[[cpp4r::register]] list var_irf(const doubles_matrix<>& B_r,
                                  const doubles_matrix<>& Sigma_r,
                                  int n_var,
                                  int n_lags,
                                  int n_hor) {
  Mat<double> B = as_Mat(B_r);
  Mat<double> Sigma = as_Mat(Sigma_r);

  // Extract VAR coefficients (exclude constant/trend)
  // B is (n_var * n_lags + constant) x n_var
  // We need the first n_var * n_lags rows
  Mat<double> A = B.head_rows(n_var * n_lags);

  // Build companion form matrix
  int n_state = n_var * n_lags;
  Mat<double> F(n_state, n_state, fill::zeros);
  
  // First n_var rows: VAR coefficients
  F.head_rows(n_var) = A.t();
  
  // Identity matrices for lag structure
  if (n_lags > 1) {
    F.submat(n_var, 0, n_state - 1, n_state - n_var - 1) = 
      eye<mat>(n_state - n_var, n_state - n_var);
  }

  // Cholesky of covariance for structural shocks
  Mat<double> P = chol(Sigma, "lower");

  // Selection matrix (first n_var states are current values)
  Mat<double> J(n_var, n_state, fill::zeros);
  J.head_cols(n_var) = eye<mat>(n_var, n_var);

  // Compute IRFs
  // Theta(:,:,h) = J * F^(h-1) * [P; 0; ...; 0]
  cube Theta(n_var, n_var, n_hor);
  
  Mat<double> F_power = eye<mat>(n_state, n_state);
  Mat<double> shock_impact(n_state, n_var, fill::zeros);
  shock_impact.head_rows(n_var) = P;

  for (int h = 0; h < n_hor; h++) {
    Theta.slice(h) = J * F_power * shock_impact;
    F_power = F_power * F;
  }

  // Convert to list of matrices
  writable::list result;
  for (int h = 0; h < n_hor; h++) {
    result.push_back(as_doubles_matrix(Mat<double>(Theta.slice(h))));
  }

  return result;
}
