// Counterfactual Analysis Functions
// Port of cnfctl_fn.m, cnfctl_pred_fn.m, optpol_fn.m from CMW26

// Counterfactual policy analysis
// A_pi, A_y, A_i: T x T policy rule matrices
// wedge: T x 1 vector
// Pi_m, Y_m, I_m: T x n_shock policy IRF matrices
// pi_z, y_z, i_z: T x 1 baseline sequences
//
// Returns list with pi_z_cnfctl, y_z_cnfctl, i_z_cnfctl, nu_z_cnfctl
[[cpp4r::register]] list cnfctl_fn(const doubles_matrix<>& A_pi_r,
                                    const doubles_matrix<>& A_y_r,
                                    const doubles_matrix<>& A_i_r,
                                    const doubles& wedge_r,
                                    const doubles_matrix<>& Pi_m_r,
                                    const doubles_matrix<>& Y_m_r,
                                    const doubles_matrix<>& I_m_r,
                                    const doubles& pi_z_r,
                                    const doubles& y_z_r,
                                    const doubles& i_z_r) {
  Mat<double> A_pi = as_Mat(A_pi_r);
  Mat<double> A_y = as_Mat(A_y_r);
  Mat<double> A_i = as_Mat(A_i_r);
  vec wedge = as_Col(wedge_r);
  Mat<double> Pi_m = as_Mat(Pi_m_r);
  Mat<double> Y_m = as_Mat(Y_m_r);
  Mat<double> I_m = as_Mat(I_m_r);
  vec pi_z = as_Col(pi_z_r);
  vec y_z = as_Col(y_z_r);
  vec i_z = as_Col(i_z_r);

  // Combined policy response matrix
  Mat<double> M_combined = A_pi * Pi_m + A_y * Y_m + A_i * I_m;

  // Target deviation from rule
  vec target = A_pi * pi_z + A_y * y_z + A_i * i_z - wedge;

  // Optimal shock sequence: nu = -(M'M)^{-1} M' target
  Mat<double> MtM = M_combined.t() * M_combined;
  vec nu_z_cnfctl = -solve(MtM, M_combined.t() * target);

  // Counterfactual outcomes
  vec pi_z_cnfctl = pi_z + Pi_m * nu_z_cnfctl;
  vec y_z_cnfctl = y_z + Y_m * nu_z_cnfctl;
  vec i_z_cnfctl = i_z + I_m * nu_z_cnfctl;

  writable::list result;
  result.push_back({"pi_z_cnfctl"_nm = as_doubles(pi_z_cnfctl)});
  result.push_back({"y_z_cnfctl"_nm = as_doubles(y_z_cnfctl)});
  result.push_back({"i_z_cnfctl"_nm = as_doubles(i_z_cnfctl)});
  result.push_back({"nu_z_cnfctl"_nm = as_doubles(nu_z_cnfctl)});

  return result;
}

// Counterfactual prediction function
// Same as cnfctl_fn but with optional first-period extraction
[[cpp4r::register]] list cnfctl_pred_fn(const doubles_matrix<>& A_pi_r,
                                         const doubles_matrix<>& A_y_r,
                                         const doubles_matrix<>& A_i_r,
                                         const doubles& wedge_r,
                                         const doubles_matrix<>& Pi_m_r,
                                         const doubles_matrix<>& Y_m_r,
                                         const doubles_matrix<>& I_m_r,
                                         const doubles& pi_x_r,
                                         const doubles& y_x_r,
                                         const doubles& i_x_r,
                                         int hist_indic) {
  Mat<double> A_pi = as_Mat(A_pi_r);
  Mat<double> A_y = as_Mat(A_y_r);
  Mat<double> A_i = as_Mat(A_i_r);
  vec wedge = as_Col(wedge_r);
  Mat<double> Pi_m = as_Mat(Pi_m_r);
  Mat<double> Y_m = as_Mat(Y_m_r);
  Mat<double> I_m = as_Mat(I_m_r);
  vec pi_x = as_Col(pi_x_r);
  vec y_x = as_Col(y_x_r);
  vec i_x = as_Col(i_x_r);

  // Combined policy response matrix
  Mat<double> M_combined = A_pi * Pi_m + A_y * Y_m + A_i * I_m;

  // Target deviation from rule
  vec target = A_pi * pi_x + A_y * y_x + A_i * i_x - wedge;

  // Optimal shock sequence
  Mat<double> MtM = M_combined.t() * M_combined;
  vec m_aux = -solve(MtM, M_combined.t() * target);

  // Counterfactual outcomes
  vec pi_cnfctl = pi_x + Pi_m * m_aux;
  vec y_cnfctl = y_x + Y_m * m_aux;
  vec i_cnfctl = i_x + I_m * m_aux;

  writable::list result;

  // hist_indic == 2: return only first period
  if (hist_indic == 2) {
    result.push_back({"pi_cnfctl"_nm = pi_cnfctl(0)});
    result.push_back({"y_cnfctl"_nm = y_cnfctl(0)});
    result.push_back({"i_cnfctl"_nm = i_cnfctl(0)});
  } else {
    result.push_back({"pi_cnfctl"_nm = as_doubles(pi_cnfctl)});
    result.push_back({"y_cnfctl"_nm = as_doubles(y_cnfctl)});
    result.push_back({"i_cnfctl"_nm = as_doubles(i_cnfctl)});
  }
  result.push_back({"m_aux"_nm = as_doubles(m_aux)});

  return result;
}

// Optimal policy function
// W_pi, W_y, W_i: T x T loss function weight matrices
// Pi_m, Y_m, I_m: T x n_shock policy IRF matrices
// pi_z, y_z, i_z: T x 1 baseline sequences
//
// Minimizes: sum_t [W_pi(t) * pi^2 + W_y(t) * y^2 + W_i(t) * i^2]
[[cpp4r::register]] list optpol_fn(const doubles_matrix<>& W_pi_r,
                                    const doubles_matrix<>& W_y_r,
                                    const doubles_matrix<>& W_i_r,
                                    const doubles_matrix<>& Pi_m_r,
                                    const doubles_matrix<>& Y_m_r,
                                    const doubles_matrix<>& I_m_r,
                                    const doubles& pi_z_r,
                                    const doubles& y_z_r,
                                    const doubles& i_z_r) {
  Mat<double> W_pi = as_Mat(W_pi_r);
  Mat<double> W_y = as_Mat(W_y_r);
  Mat<double> W_i = as_Mat(W_i_r);
  Mat<double> Pi_m = as_Mat(Pi_m_r);
  Mat<double> Y_m = as_Mat(Y_m_r);
  Mat<double> I_m = as_Mat(I_m_r);
  vec pi_z = as_Col(pi_z_r);
  vec y_z = as_Col(y_z_r);
  vec i_z = as_Col(i_z_r);

  // Quadratic terms: M' W M for each variable
  Mat<double> H = Pi_m.t() * W_pi * Pi_m + 
                  Y_m.t() * W_y * Y_m + 
                  I_m.t() * W_i * I_m;

  // Linear terms: M' W z for each variable
  vec g = Pi_m.t() * W_pi * pi_z + 
          Y_m.t() * W_y * y_z + 
          I_m.t() * W_i * i_z;

  // Optimal shock: nu = -H^{-1} g
  vec nu_z_optpol = -solve(H, g);

  // Counterfactual outcomes
  vec pi_z_optpol = pi_z + Pi_m * nu_z_optpol;
  vec y_z_optpol = y_z + Y_m * nu_z_optpol;
  vec i_z_optpol = i_z + I_m * nu_z_optpol;

  writable::list result;
  result.push_back({"pi_z_optpol"_nm = as_doubles(pi_z_optpol)});
  result.push_back({"y_z_optpol"_nm = as_doubles(y_z_optpol)});
  result.push_back({"i_z_optpol"_nm = as_doubles(i_z_optpol)});
  result.push_back({"nu_z_optpol"_nm = as_doubles(nu_z_optpol)});

  return result;
}

// Compute VMA-implied covariance from IRF array
// wold: n_y x n_shocks x T array of IRFs
// Returns n_y x n_y covariance matrix
[[cpp4r::register]] doubles_matrix<> vma_covariance(const list& wold_list) {
  // wold_list is a list of T matrices, each n_y x n_shocks
  int T = wold_list.size();
  
  if (T == 0) {
    cpp4r::stop("Empty wold_list in vma_covariance");
  }

  doubles_matrix<> first_mat = wold_list[0];
  Mat<double> first = as_Mat(first_mat);
  int n_y = first.n_rows;

  Mat<double> cov(n_y, n_y, fill::zeros);

  for (int h = 0; h < T; h++) {
    doubles_matrix<> mat_r = wold_list[h];
    Mat<double> Theta_h = as_Mat(mat_r);
    cov += Theta_h * Theta_h.t();
  }

  return as_doubles_matrix(cov);
}

// Compute correlation matrix from covariance
[[cpp4r::register]] doubles_matrix<> cov_to_corr(const doubles_matrix<>& cov_r) {
  Mat<double> cov = as_Mat(cov_r);
  int n = cov.n_rows;

  Mat<double> corr(n, n);
  vec diag_sqrt = sqrt(cov.diag());

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      corr(i, j) = cov(i, j) / (diag_sqrt(i) * diag_sqrt(j));
    }
  }

  return as_doubles_matrix(corr);
}

// Batch counterfactual computation for multiple draws
// For efficiency when running many draws
[[cpp4r::register]] list cnfctl_batch(const doubles_matrix<>& A_pi_r,
                                       const doubles_matrix<>& A_y_r,
                                       const doubles_matrix<>& A_i_r,
                                       const doubles& wedge_r,
                                       const list& Pi_m_list,
                                       const list& Y_m_list,
                                       const list& I_m_list,
                                       const doubles& pi_z_r,
                                       const doubles& y_z_r,
                                       const doubles& i_z_r) {
  Mat<double> A_pi = as_Mat(A_pi_r);
  Mat<double> A_y = as_Mat(A_y_r);
  Mat<double> A_i = as_Mat(A_i_r);
  vec wedge = as_Col(wedge_r);
  vec pi_z = as_Col(pi_z_r);
  vec y_z = as_Col(y_z_r);
  vec i_z = as_Col(i_z_r);

  int n_draws = Pi_m_list.size();
  int T = pi_z.n_elem;

  // Storage for results
  Mat<double> pi_cnfctl_all(T, n_draws);
  Mat<double> y_cnfctl_all(T, n_draws);
  Mat<double> i_cnfctl_all(T, n_draws);

  for (int d = 0; d < n_draws; d++) {
    doubles_matrix<> Pi_m_r = Pi_m_list[d];
    doubles_matrix<> Y_m_r = Y_m_list[d];
    doubles_matrix<> I_m_r = I_m_list[d];
    
    Mat<double> Pi_m = as_Mat(Pi_m_r);
    Mat<double> Y_m = as_Mat(Y_m_r);
    Mat<double> I_m = as_Mat(I_m_r);

    Mat<double> M_combined = A_pi * Pi_m + A_y * Y_m + A_i * I_m;
    vec target = A_pi * pi_z + A_y * y_z + A_i * i_z - wedge;
    Mat<double> MtM = M_combined.t() * M_combined;
    vec nu = -solve(MtM, M_combined.t() * target);

    pi_cnfctl_all.col(d) = pi_z + Pi_m * nu;
    y_cnfctl_all.col(d) = y_z + Y_m * nu;
    i_cnfctl_all.col(d) = i_z + I_m * nu;
  }

  writable::list result;
  result.push_back({"pi_cnfctl"_nm = as_doubles_matrix(pi_cnfctl_all)});
  result.push_back({"y_cnfctl"_nm = as_doubles_matrix(y_cnfctl_all)});
  result.push_back({"i_cnfctl"_nm = as_doubles_matrix(i_cnfctl_all)});

  return result;
}

// Batch optimal policy computation
[[cpp4r::register]] list optpol_batch(const doubles_matrix<>& W_pi_r,
                                       const doubles_matrix<>& W_y_r,
                                       const doubles_matrix<>& W_i_r,
                                       const list& Pi_m_list,
                                       const list& Y_m_list,
                                       const list& I_m_list,
                                       const doubles& pi_z_r,
                                       const doubles& y_z_r,
                                       const doubles& i_z_r) {
  Mat<double> W_pi = as_Mat(W_pi_r);
  Mat<double> W_y = as_Mat(W_y_r);
  Mat<double> W_i = as_Mat(W_i_r);
  vec pi_z = as_Col(pi_z_r);
  vec y_z = as_Col(y_z_r);
  vec i_z = as_Col(i_z_r);

  int n_draws = Pi_m_list.size();
  int T = pi_z.n_elem;

  Mat<double> pi_optpol_all(T, n_draws);
  Mat<double> y_optpol_all(T, n_draws);
  Mat<double> i_optpol_all(T, n_draws);

  for (int d = 0; d < n_draws; d++) {
    doubles_matrix<> Pi_m_r = Pi_m_list[d];
    doubles_matrix<> Y_m_r = Y_m_list[d];
    doubles_matrix<> I_m_r = I_m_list[d];
    
    Mat<double> Pi_m = as_Mat(Pi_m_r);
    Mat<double> Y_m = as_Mat(Y_m_r);
    Mat<double> I_m = as_Mat(I_m_r);

    Mat<double> H = Pi_m.t() * W_pi * Pi_m + 
                    Y_m.t() * W_y * Y_m + 
                    I_m.t() * W_i * I_m;
    vec g = Pi_m.t() * W_pi * pi_z + 
            Y_m.t() * W_y * y_z + 
            I_m.t() * W_i * i_z;
    vec nu = -solve(H, g);

    pi_optpol_all.col(d) = pi_z + Pi_m * nu;
    y_optpol_all.col(d) = y_z + Y_m * nu;
    i_optpol_all.col(d) = i_z + I_m * nu;
  }

  writable::list result;
  result.push_back({"pi_optpol"_nm = as_doubles_matrix(pi_optpol_all)});
  result.push_back({"y_optpol"_nm = as_doubles_matrix(y_optpol_all)});
  result.push_back({"i_optpol"_nm = as_doubles_matrix(i_optpol_all)});

  return result;
}
