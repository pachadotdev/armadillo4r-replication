// Utility functions
// Port of winsorize.m, ls_detrend.m, sample_from_models.m from CMW26

#include <random>

// Compute percentile of a vector
double percentile_(const vec& x, double p) {
  vec sorted_x = sort(x);
  int n = x.n_elem;
  double idx = (p / 100.0) * (n - 1);
  int lower_idx = static_cast<int>(std::floor(idx));
  int upper_idx = static_cast<int>(std::ceil(idx));
  
  if (lower_idx == upper_idx) {
    return sorted_x(lower_idx);
  }
  
  double frac = idx - lower_idx;
  return sorted_x(lower_idx) * (1 - frac) + sorted_x(upper_idx) * frac;
}

// Winsorize a vector
// x: input vector
// w: winsorization percentage (default 90, meaning 5% from each tail)
[[cpp4r::register]] doubles winsorize_vec(const doubles& x_r, double w = 90.0) {
  vec x = as_Col(x_r);
  
  double rw = (100.0 - w) / 2.0;
  double lb = percentile_(x, rw);
  double ub = percentile_(x, 100.0 - rw);
  
  vec y = x;
  for (uword i = 0; i < y.n_elem; i++) {
    if (y(i) < lb) y(i) = lb;
    if (y(i) > ub) y(i) = ub;
  }
  
  return as_doubles(y);
}

// Winsorize a matrix (column-wise)
[[cpp4r::register]] doubles_matrix<> winsorize_mat(const doubles_matrix<>& x_r, 
                                                    double w = 90.0) {
  Mat<double> x = as_Mat(x_r);
  Mat<double> y(x.n_rows, x.n_cols);
  
  double rw = (100.0 - w) / 2.0;
  
  for (uword j = 0; j < x.n_cols; j++) {
    vec col = x.col(j);
    double lb = percentile_(col, rw);
    double ub = percentile_(col, 100.0 - rw);
    
    for (uword i = 0; i < x.n_rows; i++) {
      double val = x(i, j);
      if (val < lb) val = lb;
      if (val > ub) val = ub;
      y(i, j) = val;
    }
  }
  
  return as_doubles_matrix(y);
}

// Least squares detrending
// Y: T x n matrix of data
// const_type: 1 = constant only, 2 = constant + linear trend
// Returns list with Beta (coefficients) and Res (residuals)
[[cpp4r::register]] list ls_detrend(const doubles_matrix<>& Y_r, int const_type = 1) {
  Mat<double> Y = as_Mat(Y_r);
  int T = Y.n_rows;
  
  Mat<double> X;
  if (const_type == 1) {
    X = ones<mat>(T, 1);
  } else if (const_type == 2) {
    X = join_horiz(ones<mat>(T, 1), linspace<vec>(1, T, T));
  } else {
    cpp4r::stop("const_type must be 1 or 2");
  }
  
  // OLS: Beta = (X'X)^{-1} X'Y
  Mat<double> Beta = solve(X, Y);
  Mat<double> Res = Y - X * Beta;
  
  writable::list result;
  result.push_back({"Beta"_nm = as_doubles_matrix(Beta)});
  result.push_back({"Res"_nm = as_doubles_matrix(Res)});
  
  return result;
}

// Hamilton filter (regcyc.m) - removes cyclical component
// Y: T x 1 vector
// h: forecast horizon (default 8 for quarterly data)
// p: number of lags (default 4)
// Returns: detrended series
[[cpp4r::register]] doubles hamilton_filter(const doubles& Y_r, int h = 8, int p = 4) {
  vec Y = as_Col(Y_r);
  int T = Y.n_elem;
  
  // Build design matrix with lags
  int T_eff = T - h - p + 1;
  
  if (T_eff <= 0) {
    cpp4r::stop("Not enough observations for Hamilton filter");
  }
  
  Mat<double> X(T_eff, p + 1, fill::ones);
  vec y(T_eff);
  
  for (int t = 0; t < T_eff; t++) {
    int idx = t + h + p - 1;
    y(t) = Y(idx);
    for (int j = 0; j < p; j++) {
      X(t, j + 1) = Y(idx - h - j);
    }
  }
  
  // OLS regression
  vec beta = solve(X, y);
  vec yhat = X * beta;
  vec resid = y - yhat;
  
  // Pad with NaNs
  vec result(T, fill::zeros);
  result.fill(datum::nan);
  result.subvec(h + p - 1, T - 1) = resid;
  
  return as_doubles(result);
}

// Kernel density estimation
// x: evaluation points
// xi: data points
// h: bandwidth (if 0, use Silverman's rule)
// ktype: 0 = Gaussian, 1 = Epanechnikov
[[cpp4r::register]] doubles kernel_density(const doubles& x_r,
                                            const doubles& xi_r,
                                            double h = 0,
                                            int ktype = 0) {
  vec x = as_Col(x_r);
  vec xi = as_Col(xi_r);
  
  int m = x.n_elem;
  int n = xi.n_elem;
  
  // Silverman's rule of thumb bandwidth
  if (h <= 0) {
    double sigma = stddev(xi);
    double iqr = percentile_(xi, 75) - percentile_(xi, 25);
    double scale = std::min(sigma, iqr / 1.34);
    h = 0.9 * scale * std::pow(static_cast<double>(n), -0.2);
  }
  
  vec f(m, fill::zeros);
  
  for (int i = 0; i < m; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      double u = (x(i) - xi(j)) / h;
      double k;
      
      if (ktype == 0) {
        // Gaussian kernel
        k = std::exp(-0.5 * u * u) / std::sqrt(2 * datum::pi);
      } else {
        // Epanechnikov kernel
        if (std::abs(u) <= 1) {
          k = 0.75 * (1 - u * u);
        } else {
          k = 0;
        }
      }
      sum += k;
    }
    f(i) = sum / (n * h);
  }
  
  return as_doubles(f);
}

// Quantile function
[[cpp4r::register]] doubles quantile_vec(const doubles& x_r, const doubles& probs_r) {
  vec x = as_Col(x_r);
  vec probs = as_Col(probs_r);
  
  vec q(probs.n_elem);
  for (uword i = 0; i < probs.n_elem; i++) {
    q(i) = percentile_(x, probs(i) * 100);
  }
  
  return as_doubles(q);
}

// Column-wise quantiles for a matrix
[[cpp4r::register]] doubles_matrix<> quantile_mat(const doubles_matrix<>& x_r, 
                                                   const doubles& probs_r) {
  Mat<double> x = as_Mat(x_r);
  vec probs = as_Col(probs_r);
  
  Mat<double> q(probs.n_elem, x.n_cols);
  
  for (uword j = 0; j < x.n_cols; j++) {
    vec col = x.col(j);
    for (uword i = 0; i < probs.n_elem; i++) {
      q(i, j) = percentile_(col, probs(i) * 100);
    }
  }
  
  return as_doubles_matrix(q);
}

// Sample from models with weights
// n_draws: number of samples
// model_probs: probability weights for each model
// Returns: indices of sampled models (1-indexed for R)
[[cpp4r::register]] integers sample_models(int n_draws, const doubles& model_probs_r) {
  vec model_probs = as_Col(model_probs_r);
  int n_models = model_probs.n_elem;
  
  // Normalize probabilities
  double sum_prob = accu(model_probs);
  vec probs = model_probs / sum_prob;
  
  // Cumulative probabilities
  vec cum_probs(n_models);
  cum_probs(0) = probs(0);
  for (int i = 1; i < n_models; i++) {
    cum_probs(i) = cum_probs(i - 1) + probs(i);
  }
  
  // Random sampling
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  
  writable::integers samples(n_draws);
  
  for (int d = 0; d < n_draws; d++) {
    double u = unif(gen);
    int idx = 0;
    while (idx < n_models - 1 && u > cum_probs(idx)) {
      idx++;
    }
    samples[d] = idx + 1;  // 1-indexed for R
  }
  
  return samples;
}

// Stack matrices by third dimension (for R arrays)
// Takes a list of matrices and returns a combined matrix
// where each input matrix is stacked as columns
[[cpp4r::register]] doubles_matrix<> stack_matrices(const list& mat_list) {
  int n = mat_list.size();
  
  if (n == 0) {
    cpp4r::stop("Empty list in stack_matrices");
  }
  
  doubles_matrix<> first = mat_list[0];
  Mat<double> m0 = as_Mat(first);
  int nrow = m0.n_rows;
  int ncol = m0.n_cols;
  
  // Combined matrix: (nrow * ncol) x n
  Mat<double> result(nrow * ncol, n);
  
  for (int i = 0; i < n; i++) {
    doubles_matrix<> mat_r = mat_list[i];
    Mat<double> mat = as_Mat(mat_r);
    result.col(i) = vectorise(mat);
  }
  
  return as_doubles_matrix(result);
}
