#' @title Bayesian VAR with flat prior
#' @description Estimate a Bayesian VAR using OLS and flat priors.
#' @param data A T x n_var matrix of data
#' @param n_lags Number of lags in the VAR
#' @param constant Type of deterministic terms: 0 = none, 1 = constant,
#'   2 = constant + trend, 3 = constant + quadratic trend
#' @param n_draws Number of posterior draws
#' @param seed Random seed for reproducibility (0 = use random seed)
#' @return A list with components:
#'   - B_OLS: OLS coefficient estimates
#'   - Sigma_OLS: OLS residual covariance
#'   - B_draws: List of posterior draws for coefficients
#'   - Sigma_draws: List of posterior draws for covariance
#' @examples
#' \dontrun{
#' # Simulate some data
#' set.seed(123)
#' T <- 200
#' data <- matrix(rnorm(T * 3), T, 3)
#'
#' # Estimate BVAR
#' result <- estimate_bvar(data, n_lags = 2, constant = 1, n_draws = 1000)
#' }
#' @export
estimate_bvar <- function(data, n_lags = 2, constant = 1, n_draws = 1000, 
                          seed = 0) {
  data <- as.matrix(data)
  result <- bvar_fn(data, n_lags, constant, n_draws, seed)
  
  # Add variable names if available
  if (!is.null(colnames(data))) {
    colnames(result$B_OLS) <- colnames(data)
    colnames(result$Sigma_OLS) <- colnames(data)
    rownames(result$Sigma_OLS) <- colnames(data)
  }
  
  class(result) <- "bvar"
  result
}

#' @title Impulse Response Functions from VAR
#' @description Compute IRFs from VAR coefficient matrices and covariance.
#' @param bvar_result Result from estimate_bvar()
#' @param n_var Number of variables
#' @param n_lags Number of lags
#' @param n_hor Horizon for IRFs
#' @param use_draws If TRUE, compute IRFs for all posterior draws
#' @return A list of IRF matrices or (if use_draws) a 4D array
#' @export
compute_var_irf <- function(bvar_result, n_var, n_lags, n_hor = 20,
                             use_draws = FALSE) {
  if (!use_draws) {
    # Use OLS point estimates
    irf_list <- var_irf(bvar_result$B_OLS, bvar_result$Sigma_OLS,
                         n_var, n_lags, n_hor)
    
    # Convert to 3D array
    irf_array <- array(0, dim = c(n_hor, n_var, n_var))
    for (h in seq_len(n_hor)) {
      irf_array[h, , ] <- irf_list[[h]]
    }
    return(irf_array)
  } else {
    # Compute for all draws
    n_draws <- length(bvar_result$B_draws)
    irf_draws <- array(0, dim = c(n_hor, n_var, n_var, n_draws))
    
    for (d in seq_len(n_draws)) {
      irf_list <- var_irf(bvar_result$B_draws[[d]],
                           bvar_result$Sigma_draws[[d]],
                           n_var, n_lags, n_hor)
      for (h in seq_len(n_hor)) {
        irf_draws[h, , , d] <- irf_list[[h]]
      }
    }
    return(irf_draws)
  }
}

#' @title IRF from State-Space Model
#' @description Compute IRFs from state-space representation of a model.
#' @param ME Impact matrix (nx x ne)
#' @param MX State transition matrix (nx x nx)
#' @param MY Observation matrix (ny x nx)
#' @param n_periods Number of periods
#' @param shock_idx Which shock to compute IRF for (1-indexed)
#' @return A list with YY (observations) and XX (states) IRFs
#' @export
compute_irf <- function(ME, MX, MY, n_periods = 20, shock_idx = 1) {
  ME <- as.matrix(ME)
  MX <- as.matrix(MX)
  MY <- as.matrix(MY)
  
  comp_irf(ME, MX, MY, n_periods, shock_idx)
}

#' @title VAR Forecasts
#' @description Generate forecasts from a VAR model.
#' @param data A T x n_var matrix of data
#' @param n_lags Number of lags
#' @param constant Deterministic terms (0, 1, 2, or 3)
#' @param B Coefficient matrix from VAR estimation
#' @param fcst_date Forecast origin (row index in original data)
#' @param fcst_length Number of periods to forecast
#' @param include_current If TRUE, include current period in forecast
#' @return A matrix of forecasts (fcst_length x n_var)
#' @export
var_fcst <- function(data, n_lags, constant, B, fcst_date, fcst_length,
                      include_current = FALSE) {
  data <- as.matrix(data)
  B <- as.matrix(B)
  hist_indic <- if (include_current) 2L else 1L
  
  var_forecast(data, n_lags, constant, B, fcst_date, fcst_length, hist_indic)
}

#' @title Frequency Domain Variance
#' @description Integrates the spectral density of a VAR over a frequency band.
#' @param irf_array 3D array of IRFs (n_hor x n_var x n_var)
#' @param omega_low Lower frequency bound
#' @param omega_high Upper frequency bound
#' @param n_grid Number of grid points for integration
#' @return Covariance matrix
#' @export
freq_variance <- function(irf_array, omega_low, omega_high, n_grid = 100) {
  # Convert 3D array to list of matrices
  n_hor <- dim(irf_array)[1]
  theta_list <- vector("list", n_hor)
  for (h in seq_len(n_hor)) {
    theta_list[[h]] <- irf_array[h, , ]
  }
  
  freq_var_fn(theta_list, omega_low, omega_high, n_grid)
}

#' @title Business Cycle Frequency Variance
#' @description Computes variance at business cycle frequencies (6-32 quarters).
#' @param irf_array 3D array of IRFs (n_hor x n_var x n_var)
#' @return Covariance matrix
#' @export
bc_variance <- function(irf_array) {
  n_hor <- dim(irf_array)[1]
  theta_list <- vector("list", n_hor)
  for (h in seq_len(n_hor)) {
    theta_list[[h]] <- irf_array[h, , ]
  }
  
  bc_freq_var(theta_list)
}

#' @title IRF Summary Statistics
#' @description Compute summary statistics across posterior IRF draws.
#' @param irf_draws 4D array (n_hor x n_var x n_var x n_draws)
#' @param probs Quantiles to compute
#' @return List with mean, median, and quantile bands
#' @export
irf_summary <- function(irf_draws, probs = c(0.16, 0.84)) {
  dims <- dim(irf_draws)
  n_hor <- dims[1]
  n_var <- dims[2]
  n_shocks <- dims[3]
  n_draws <- dims[4]
  
  # Compute statistics
  irf_mean <- apply(irf_draws, 1:3, mean)
  irf_median <- apply(irf_draws, 1:3, median)
  
  irf_lower <- apply(irf_draws, 1:3, quantile, probs = probs[1])
  irf_upper <- apply(irf_draws, 1:3, quantile, probs = probs[2])
  
  list(
    mean = irf_mean,
    median = irf_median,
    lower = irf_lower,
    upper = irf_upper,
    probs = probs
  )
}
