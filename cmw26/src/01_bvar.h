// Bayesian VAR estimation with flat prior
// Port of bvar_fn.m from CMW26

// Draw from Inverse-Wishart distribution using Armadillo's RNG
// df = degrees of freedom, S = scale matrix
// Uses Bartlett decomposition: if W ~ Wishart(df, S^{-1}), then W^{-1} ~ InvWishart(df, S)
// MATLAB's iwishrnd(S, df) returns X ~ IW(df, S) where E[X] = S / (df - p - 1)
Mat<double> iwishrnd_arma_(const Mat<double>& S, int df) {
  int p = S.n_rows;
  // To get IW(df, S), we need Wishart(df, S^{-1}), so use chol of S^{-1}
  Mat<double> L = chol(inv_sympd(S), "lower");
  
  // Generate Bartlett decomposition using Armadillo RNG
  // This matches MATLAB's approach more closely
  Mat<double> A(p, p, fill::zeros);
  
  // Diagonal: sqrt of chi-squared draws
  // MATLAB: chi2rnd(df:-1:df-p+1) draws all at once
  for (int i = 0; i < p; i++) {
    // chi2(df-i) = sum of (df-i) squared standard normals
    vec z = randn<vec>(df - i);
    A(i, i) = std::sqrt(dot(z, z));
  }
  
  // Below diagonal: standard normals
  for (int i = 1; i < p; i++) {
    for (int j = 0; j < i; j++) {
      A(i, j) = randn<double>();
    }
  }
  
  // W ~ Wishart(df, S) = L * A * A' * L'
  Mat<double> W = L * A * A.t() * L.t();
  
  // Return inverse for Inverse-Wishart
  return inv_sympd(W);
}

// Internal function to set up VAR matrices
// Returns: {Y, X} where Y is T x n_var, X is T x m
std::pair<Mat<double>, Mat<double>> setup_var_matrices_(
    const Mat<double>& data, int n_lags, int constant) {
  
  int n_var = data.n_cols;
  int T_full = data.n_rows;
  int T = T_full - n_lags;
  int m = n_var * n_lags + constant;

  // Y: dependent variable (data from n_lags+1 to end)
  Mat<double> Y = data.rows(n_lags, T_full - 1);

  // X: lagged regressors
  Mat<double> X(T, m, fill::zeros);
  
  for (int i = 0; i < n_lags; i++) {
    // data from (n_lags - i) to (end - i - 1)
    int start_row = n_lags - i - 1;
    int end_row = T_full - i - 2;
    X.cols(n_var * i, n_var * (i + 1) - 1) = data.rows(start_row, end_row);
  }

  // Add constant/trend terms
  if (constant >= 1) {
    X.col(n_var * n_lags) = ones<vec>(T);
  }
  if (constant >= 2) {
    X.col(n_var * n_lags + 1) = linspace<vec>(1, T, T);
  }
  if (constant >= 3) {
    vec t = linspace<vec>(1, T, T);
    X.col(n_var * n_lags + 2) = t % t;  // element-wise square
  }

  return {Y, X};
}

// Bayesian VAR estimation with flat prior
// Returns: list with B_draws, Sigma_draws, B_OLS, Sigma_OLS
// seed: random seed for reproducibility (0 = use random seed from device)
// Uses Armadillo's RNG (Mersenne Twister) which can be seeded via arma_rng::set_seed()
[[cpp4r::register]] list bvar_fn(const doubles_matrix<>& data_r,
                                  int n_lags,
                                  int constant,
                                  int n_draws,
                                  int seed = 0) {
  Mat<double> data = as_Mat(data_r);
  int n_var = data.n_cols;
  int m = n_var * n_lags + constant;

  // Set Armadillo RNG seed for reproducibility
  // Use fully qualified name (::arma::arma_rng) to avoid ambiguity with armadillo4r wrapper
  if (seed != 0) {
    ::arma::arma_rng::set_seed(static_cast<arma::uword>(seed));
  } else {
    ::arma::arma_rng::set_seed_random();
  }

  // Set up VAR matrices
  auto [Y, X] = setup_var_matrices_(data, n_lags, constant);
  int T = Y.n_rows;

  // Flat prior (Jeffrey's prior)
  double nnuBar = 0;
  Mat<double> OomegaBarInverse(m, m, fill::zeros);
  Mat<double> PpsiBar(m, n_var, fill::zeros);
  Mat<double> PphiBar(n_var, n_var, fill::zeros);

  // Posterior parameters
  double nnuTilde = T + nnuBar;
  Mat<double> OomegaTildeInverse = X.t() * X + OomegaBarInverse;
  Mat<double> OomegaTilde = inv_sympd(OomegaTildeInverse);
  Mat<double> PpsiTilde = OomegaTilde * (X.t() * Y + OomegaBarInverse * PpsiBar);
  Mat<double> PphiTilde = Y.t() * Y + PphiBar +
                          PpsiBar.t() * OomegaBarInverse * PpsiBar -
                          PpsiTilde.t() * OomegaTildeInverse * PpsiTilde;
  PphiTilde = 0.5 * (PphiTilde + PphiTilde.t());  // Ensure symmetry

  // Storage for draws
  cube B_draws(m, n_var, n_draws);
  cube Sigma_draws(n_var, n_var, n_draws);

  // Cholesky for drawing B|Sigma
  Mat<double> cholOomegaTilde = chol(OomegaTilde, "lower");

  // Draw from posterior using Armadillo's RNG
  for (int i = 0; i < n_draws; i++) {
    // Draw Sigma from Inverse-Wishart
    Mat<double> Sigmadraw = iwishrnd_arma_(PphiTilde, static_cast<int>(nnuTilde));
    Mat<double> cholSigmadraw = chol(Sigmadraw, "lower");

    // Draw B|Sigma from matrix normal
    // vec(B) ~ N(vec(PpsiTilde), Sigma ⊗ OomegaTilde)
    vec z = randn<vec>(m * n_var);
    
    vec Bdraw_vec = kron(cholSigmadraw, cholOomegaTilde) * z + 
                    vectorise(PpsiTilde);
    Mat<double> Bdraw = reshape(Bdraw_vec, m, n_var);

    // Store draws
    B_draws.slice(i) = Bdraw;
    Sigma_draws.slice(i) = Sigmadraw;
  }

  // OLS estimates
  Mat<double> B_OLS = PpsiTilde;
  Mat<double> Sigma_OLS = PphiTilde / T;

  // Return results as list
  writable::list result;
  result.push_back({"B_OLS"_nm = as_doubles_matrix(B_OLS)});
  result.push_back({"Sigma_OLS"_nm = as_doubles_matrix(Sigma_OLS)});
  
  // Convert cubes to list of matrices for R
  writable::list B_list;
  writable::list Sigma_list;
  for (int i = 0; i < n_draws; i++) {
    B_list.push_back(as_doubles_matrix(Mat<double>(B_draws.slice(i))));
    Sigma_list.push_back(as_doubles_matrix(Mat<double>(Sigma_draws.slice(i))));
  }
  result.push_back({"B_draws"_nm = B_list});
  result.push_back({"Sigma_draws"_nm = Sigma_list});

  return result;
}
