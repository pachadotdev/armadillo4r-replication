#' @title Winsorize a vector
#' @description Replace extreme values with specified percentiles.
#' @param x Numeric vector
#' @param w Winsorization percentage (default 90, meaning trim 5 percent from each tail)
#' @return Winsorized vector
#' @examples
#' x <- c(1, 2, 3, 100, 4, 5)
#' winsorize(x, 80)
#' @export
winsorize <- function(x, w = 90) {
  if (is.vector(x)) {
    winsorize_vec(x, w)
  } else if (is.matrix(x)) {
    winsorize_mat(x, w)
  } else {
    stop("x must be a vector or matrix")
  }
}

#' @title Detrend a time series
#' @description Remove deterministic trend using OLS.
#' @param y Numeric vector or matrix
#' @param const_type 1 = constant only, 2 = constant + linear trend
#' @return A list with Beta (trend coefficients) and Res (detrended series)
#' @export
detrend <- function(y, const_type = 1) {
  ls_detrend(as.matrix(y), const_type)
}

#' @title Hamilton filter
#' @description Remove business cycle component using Hamilton (2018) filter.
#' @param y Numeric vector
#' @param h Forecast horizon (default 8 for quarterly data)
#' @param p Number of lags (default 4)
#' @return Detrended series with leading NAs
#' @export
hamilton <- function(y, h = 8, p = 4) {
  hamilton_filter(y, h, p)
}

#' @title Kernel density estimation
#' @description Estimate probability density function using kernel methods.
#' @param x Evaluation points
#' @param xi Data points
#' @param h Bandwidth (0 for automatic selection)
#' @param kernel Type: "gaussian" or "epanechnikov"
#' @return Density estimates at x
#' @export
kde <- function(x, xi, h = 0, kernel = "gaussian") {
  ktype <- if (kernel == "gaussian") 0L else 1L
  kernel_density(x, xi, h, ktype)
}

#' @title Sample from models with probability weights
#' @description Draw model indices according to posterior probabilities.
#' @param n Number of samples
#' @param probs Probability weights (will be normalized)
#' @return Vector of model indices (1-indexed)
#' @export
sample_models_r <- function(n, probs) {
  sample_models(n, probs)
}

#' @title Extract draws from model collection
#' @description Sample policy IRF draws from a collection of models according to posterior
#'  probabilities.
#' @param n_draws Number of draws
#' @param model_probs Model probability weights
#' @param Pi_m_list List of lists: Pi_m_list[[model]][[draw]]
#' @param Y_m_list List of lists for output IRFs
#' @param I_m_list List of lists for interest rate IRFs
#' @return A list with sampled Pi_m, Y_m, I_m draws
#' @export
sample_from_models <- function(n_draws, model_probs,
                                Pi_m_list, Y_m_list, I_m_list) {
  n_models <- length(model_probs)

  # Sample model indices
  model_idx <- sample_models_r(n_draws, model_probs)

  # Sample draw indices within each model
  draws_per_model <- sapply(Pi_m_list, length)

  Pi_m_out <- vector("list", n_draws)
  Y_m_out <- vector("list", n_draws)
  I_m_out <- vector("list", n_draws)

  for (d in seq_len(n_draws)) {
    m <- model_idx[d]
    draw_idx <- sample.int(draws_per_model[m], 1)

    Pi_m_out[[d]] <- Pi_m_list[[m]][[draw_idx]]
    Y_m_out[[d]] <- Y_m_list[[m]][[draw_idx]]
    I_m_out[[d]] <- I_m_list[[m]][[draw_idx]]
  }

  list(Pi_m = Pi_m_out, Y_m = Y_m_out, I_m = I_m_out)
}

#' @title Quantile summary for arrays
#' @description Compute quantiles along a specified dimension.
#' @param x Array
#' @param probs Quantile probabilities
#' @param along Which dimension to summarize over
#' @return Array with quantiles
#' @export
quantile_along <- function(x, probs = c(0.16, 0.5, 0.84), along = length(dim(x))) {
  apply(x, setdiff(seq_along(dim(x)), along), quantile, probs = probs, na.rm = TRUE)
}

#' @title Annual average of quarterly data
#' @description Compute annual averages from quarterly observations.
#' @param x Numeric vector of quarterly data
#' @param start_quarter Starting quarter (1-4)
#' @return Vector of annual averages
#' @export
annual_avg <- function(x, start_quarter = 1) {
  n <- length(x)
  # Pad to align with calendar years
  pad_start <- (start_quarter - 1) %% 4
  if (pad_start > 0) {
    x <- c(rep(NA, pad_start), x)
  }

  n_padded <- length(x)
  n_years <- floor(n_padded / 4)

  if (n_years == 0) return(mean(x, na.rm = TRUE))

  result <- numeric(n_years)
  for (y in seq_len(n_years)) {
    idx <- ((y - 1) * 4 + 1):(y * 4)
    result[y] <- mean(x[idx], na.rm = TRUE)
  }

  result
}

#' @title Create quarterly date sequence
#' @description Generate a sequence of dates for quarterly data.
#' @param start_year Starting year
#' @param start_quarter Starting quarter
#' @param n Number of quarters
#' @return Vector of Date objects (first day of each quarter)
#' @export
quarterly_dates <- function(start_year, start_quarter, n) {
  years <- rep(start_year:((start_year + (n + start_quarter - 2) %/% 4)), each = 4)
  quarters <- rep(1:4, length.out = length(years))

  # Adjust to start from the right quarter
  idx <- which(years == start_year & quarters == start_quarter)[1]
  years <- years[idx:(idx + n - 1)]
  quarters <- quarters[idx:(idx + n - 1)]

  as.Date(paste(years, (quarters - 1) * 3 + 1, 1, sep = "-"))
}

#' @title Lag a vector
#' @description Create lagged values, filling with NA.
#' @param x Numeric vector
#' @param k Number of lags (positive) or leads (negative)
#' @return Lagged vector
#' @export
lag_vec <- function(x, k = 1) {
  n <- length(x)
  if (k >= 0) {
    c(rep(NA, k), x[1:(n - k)])
  } else {
    c(x[(-k + 1):n], rep(NA, -k))
  }
}

#' @title Difference a vector
#' @description Compute first difference.
#' @param x Numeric vector
#' @param k Difference order
#' @return Differenced vector (length n - k)
#' @export
diff_vec <- function(x, k = 1) {
  n <- length(x)
  x[(k + 1):n] - x[1:(n - k)]
}

#' @title Growth rate
#' @description Compute percentage growth rate.
#' @param x Numeric vector
#' @param k Period for growth rate
#' @param annualize If TRUE, annualize the growth rate
#' @return Growth rate vector
#' @export
growth_rate <- function(x, k = 1, annualize = FALSE) {
  n <- length(x)
  gr <- (x[(k + 1):n] / x[1:(n - k)] - 1) * 100

  if (annualize && k < 4) {
    gr <- gr * (4 / k)
  }

  c(rep(NA, k), gr)
}

#' @title HP filter
#' @description Hodrick-Prescott filter for trend extraction.
#' @param x Numeric vector
#' @param lambda Smoothing parameter (1600 for quarterly, 100 for annual)
#' @return A list with trend and cycle components
#' @export
hp_filter <- function(x, lambda = 1600) {
  n <- length(x)

  # Build D matrix (second difference)
  D <- matrix(0, n - 2, n)
  for (i in 1:(n - 2)) {
    D[i, i] <- 1
    D[i, i + 1] <- -2
    D[i, i + 2] <- 1
  }

  # Solve: (I + lambda * D'D) * trend = x
  I_n <- diag(n)
  A <- I_n + lambda * t(D) %*% D
  trend <- as.vector(solve(A, x))
  cycle <- x - trend

  list(trend = trend, cycle = cycle)
}
