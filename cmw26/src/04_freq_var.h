// Frequency domain variance functions
// Port of freq_var_fn.m from CMW26

// Compute frequency domain variance from IRFs
// Integrates the spectral density over [omega_1, omega_2]
//
// theta_list: list of n_hor matrices, each n_y x n_e (IRF at each horizon)
// omega_1: lower frequency bound
// omega_2: upper frequency bound
// n_omega: number of grid points for integration (default 100)
//
// Returns: n_y x n_y covariance matrix
[[cpp4r::register]] doubles_matrix<> freq_var_fn(const list& theta_list,
                                                  double omega_1,
                                                  double omega_2,
                                                  int n_omega) {
  // Convert list to cube
  int n_hor = theta_list.size();
  
  if (n_hor == 0) {
    cpp4r::stop("Empty theta_list in freq_var_fn");
  }

  // Get dimensions from first element
  doubles_matrix<> first_mat = theta_list[0];
  Mat<double> first = as_Mat(first_mat);
  int n_y = first.n_rows;
  int n_e = first.n_cols;

  // Build cube of IRFs
  cube Theta(n_y, n_e, n_hor);
  for (int h = 0; h < n_hor; h++) {
    doubles_matrix<> mat_r = theta_list[h];
    Theta.slice(h) = as_Mat(mat_r);
  }

  // Frequency grid
  vec omega_grid = linspace<vec>(omega_1, omega_2, n_omega);
  double domega = (omega_2 - omega_1) / (n_omega - 1);

  // Initialize variance matrix
  Mat<double> freq_var(n_y, n_y, fill::zeros);

  // Integrate over frequencies
  for (int i_omega = 0; i_omega < n_omega; i_omega++) {
    double omega = omega_grid(i_omega);

    // Compute transfer function: sum_h Theta(:,:,h) * exp(-i*omega*h)
    cx_mat transfer(n_y, n_e, fill::zeros);
    
    for (int h = 0; h < n_hor; h++) {
      // exp(-i*omega*h) = cos(omega*h) - i*sin(omega*h)
      double arg = -omega * h;
      std::complex<double> phase(std::cos(arg), std::sin(arg));
      transfer += Theta.slice(h) * phase;
    }

    // Spectral density contribution: transfer * transfer'
    // Note: ctranspose in MATLAB = .t() for complex (conjugate transpose) in Armadillo
    Mat<double> contribution = real(transfer * transfer.t());
    
    freq_var += (1.0 / (2.0 * datum::pi)) * contribution * domega;
  }

  return as_doubles_matrix(freq_var);
}

// Compute frequency domain variance from VAR IRF cube directly
// This version takes the IRF cube as a 3D array representation
[[cpp4r::register]] doubles_matrix<> freq_var_from_irf(const doubles_matrix<>& irf_r,
                                                        int n_y,
                                                        int n_e,
                                                        int n_hor,
                                                        double omega_1,
                                                        double omega_2,
                                                        int n_omega) {
  // irf_r is (n_y * n_hor) x n_e, stacked by horizon
  Mat<double> irf = as_Mat(irf_r);

  // Frequency grid
  vec omega_grid = linspace<vec>(omega_1, omega_2, n_omega);
  double domega = (omega_2 - omega_1) / (n_omega - 1);

  // Initialize variance matrix
  Mat<double> freq_var(n_y, n_y, fill::zeros);

  // Integrate over frequencies
  for (int i_omega = 0; i_omega < n_omega; i_omega++) {
    double omega = omega_grid(i_omega);

    // Compute transfer function
    cx_mat transfer(n_y, n_e, fill::zeros);
    
    for (int h = 0; h < n_hor; h++) {
      // Extract IRF at horizon h
      Mat<double> Theta_h = irf.rows(h * n_y, (h + 1) * n_y - 1);
      
      double arg = -omega * h;
      std::complex<double> phase(std::cos(arg), std::sin(arg));
      transfer += Theta_h * phase;
    }

    Mat<double> contribution = real(transfer * transfer.t());
    freq_var += (1.0 / (2.0 * datum::pi)) * contribution * domega;
  }

  return as_doubles_matrix(freq_var);
}

// Compute spectral density at a single frequency
[[cpp4r::register]] doubles_matrix<> spectral_density(const list& theta_list,
                                                       double omega) {
  int n_hor = theta_list.size();
  
  if (n_hor == 0) {
    cpp4r::stop("Empty theta_list in spectral_density");
  }

  doubles_matrix<> first_mat = theta_list[0];
  Mat<double> first = as_Mat(first_mat);
  int n_y = first.n_rows;
  int n_e = first.n_cols;

  // Compute transfer function
  cx_mat transfer(n_y, n_e, fill::zeros);
  
  for (int h = 0; h < n_hor; h++) {
    doubles_matrix<> mat_r = theta_list[h];
    Mat<double> Theta_h = as_Mat(mat_r);
    
    double arg = -omega * h;
    std::complex<double> phase(std::cos(arg), std::sin(arg));
    transfer += Theta_h * phase;
  }

  // Spectral density: (1/2pi) * transfer * transfer'
  Mat<double> S = (1.0 / (2.0 * datum::pi)) * real(transfer * transfer.t());

  return as_doubles_matrix(S);
}

// Business cycle frequency variance (6-32 quarters)
[[cpp4r::register]] doubles_matrix<> bc_freq_var(const list& theta_list) {
  // Business cycle: 6-32 quarters
  // omega = 2*pi/period, so:
  // omega_low = 2*pi/32 ≈ 0.196
  // omega_high = 2*pi/6 ≈ 1.047
  double omega_1 = 2.0 * datum::pi / 32.0;
  double omega_2 = 2.0 * datum::pi / 6.0;
  
  return freq_var_fn(theta_list, omega_1, omega_2, 100);
}
