#' @title Set up counterfactual policy rule matrices
#' @description Creates the A_pi, A_y, A_i matrices and wedge vector for different counterfactual
#'  policy rules.
#' @param T Number of periods
#' @param rule Type of policy rule. One of: "output_gap", "inflation",
#'   "rate_peg", "taylor", "ngdp", "rate_target"
#' @param rho_ib Interest rate smoothing parameter (for Taylor rule)
#' @param phi_pi Inflation response coefficient (for Taylor rule)
#' @param phi_y Output gap response coefficient (for Taylor rule)
#' @return A list with A_pi, A_y, A_i (T x T matrices) and wedge (T x 1 vector)
#' @examples
#' rule <- set_cnfctl_rule(20, "taylor")
#' @export
set_cnfctl_rule <- function(T, rule = "output_gap",
                             rho_ib = 0, phi_pi = 1.5, phi_y = 0.5) {
  A_pi <- matrix(0, T, T)
  A_y <- matrix(0, T, T)
  A_i <- matrix(0, T, T)
  wedge <- rep(0, T)

  if (rule == "output_gap") {
    # Stabilize output gap
    A_y <- diag(T)

  } else if (rule == "inflation") {
    # Stabilize inflation
    A_pi <- diag(T)

  } else if (rule == "rate_peg") {
    # Nominal rate peg
    A_i <- diag(T)

  } else if (rule == "taylor") {
    # Taylor rule: i_t = rho_ib * i_{t-1} + (1-rho_ib) * [phi_pi * pi_t + phi_y * y_t]
    A_pi <- (1 - rho_ib) * (-phi_pi) * diag(T)
    A_y <- (1 - rho_ib) * (-phi_y) * diag(T)
    A_i <- diag(T)
    for (t in 2:T) {
      A_i[t, t - 1] <- -rho_ib
    }

  } else if (rule == "ngdp") {
    # Nominal GDP targeting
    A_pi <- diag(T)
    A_y <- diag(T)
    for (t in 2:T) {
      A_y[t, t - 1] <- -1
    }

  } else if (rule == "rate_target") {
    # Interest rate target
    A_i <- diag(T)

  } else {
    stop("Unknown rule: ", rule)
  }

  list(A_pi = A_pi, A_y = A_y, A_i = A_i, wedge = wedge)
}

#' @title Set up optimal policy loss function weights
#' @description Creates W_pi, W_y, W_i weight matrices for optimal policy computation.
#' @param T Number of periods
#' @param lambda_pi Weight on inflation squared
#' @param lambda_y Weight on output gap squared
#' @param lambda_i Weight on interest rate squared
#' @param lambda_di Weight on interest rate change squared
#' @param beta Discount factor
#' @return A list with W_pi, W_y, W_i (T x T matrices)
#' @export
set_optpol_weights <- function(T, lambda_pi = 1, lambda_y = 1,
                                lambda_i = 0, lambda_di = 1,
                                beta = 1) {
  # Discount factor matrix
  W <- diag(beta^(0:(T - 1)))

  W_pi <- lambda_pi * W
  W_y <- lambda_y * W
  W_i <- lambda_i * W

  # Add interest rate change penalty
  W_i <- W_i + lambda_di * W * (1 + beta)
  for (t in 2:T) {
    W_i[t - 1, t] <- -lambda_di * beta^(t - 1)
    W_i[t, t - 1] <- -lambda_di * beta^(t - 1)
  }

  list(W_pi = W_pi, W_y = W_y, W_i = W_i)
}

#' Compute counterfactual outcomes
#'
#' Given baseline IRFs and policy IRFs, compute counterfactual outcomes
#' under a specified policy rule.
#'
#' @param pi_z Baseline inflation path (T x 1)
#' @param y_z Baseline output gap path (T x 1)
#' @param i_z Baseline interest rate path (T x 1)
#' @param Pi_m Inflation response to policy shock (T x n_shock)
#' @param Y_m Output response to policy shock (T x n_shock)
#' @param I_m Interest rate response to policy shock (T x n_shock)
#' @param rule Policy rule specification (from set_cnfctl_rule)
#' @return A list with counterfactual paths pi_z_cnfctl, y_z_cnfctl, i_z_cnfctl
#'   and optimal shock sequence nu_z_cnfctl
#' @export
compute_cnfctl <- function(pi_z, y_z, i_z, Pi_m, Y_m, I_m, rule) {
  cnfctl_fn(
    rule$A_pi, rule$A_y, rule$A_i, rule$wedge,
    as.matrix(Pi_m), as.matrix(Y_m), as.matrix(I_m),
    as.numeric(pi_z), as.numeric(y_z), as.numeric(i_z)
  )
}

#' @title Optimal policy outcomes
#' @description Given baseline IRFs and policy IRFs, compute optimal policy outcomes
#' that minimize a loss function.
#' @param pi_z Baseline inflation path (T x 1)
#' @param y_z Baseline output gap path (T x 1)
#' @param i_z Baseline interest rate path (T x 1)
#' @param Pi_m Inflation response to policy shock (T x n_shock)
#' @param Y_m Output response to policy shock (T x n_shock)
#' @param I_m Interest rate response to policy shock (T x n_shock)
#' @param weights Loss function weights (from set_optpol_weights)
#' @return A list with optimal paths and shock sequence
#' @export
compute_optpol <- function(pi_z, y_z, i_z, Pi_m, Y_m, I_m, weights) {
  optpol_fn(
    weights$W_pi, weights$W_y, weights$W_i,
    as.matrix(Pi_m), as.matrix(Y_m), as.matrix(I_m),
    as.numeric(pi_z), as.numeric(y_z), as.numeric(i_z)
  )
}

#' @title Counterfactual Wold IRFs for all shocks
#' @description Given baseline Wold IRFs and policy shock IRFs, compute counterfactual
#'  Wold IRFs under a policy rule.
#' @param wold_base 3D array (n_var x n_shocks x T) of baseline Wold IRFs
#' @param Pi_m_draws List of inflation policy IRFs, one per draw
#' @param Y_m_draws List of output policy IRFs
#' @param I_m_draws List of interest rate policy IRFs
#' @param rule Policy rule specification
#' @param use_optpol If TRUE, use optimal policy; otherwise use simple rule
#' @param weights Optimal policy weights (required if use_optpol = TRUE)
#' @return 4D array (n_var x n_shocks x T x n_draws) of counterfactual Wold IRFs
#' @export
compute_cnfctl_wold <- function(wold_base, Pi_m_draws, Y_m_draws, I_m_draws,
                                 rule = NULL, use_optpol = FALSE,
                                 weights = NULL) {
  n_var <- dim(wold_base)[1]
  n_shocks <- dim(wold_base)[2]
  T <- dim(wold_base)[3]
  n_draws <- length(Pi_m_draws)

  # Output array
  wold_cnfctl <- array(NA, dim = c(n_var, n_shocks, T, n_draws))

  for (d in seq_len(n_draws)) {
    Pi_m <- Pi_m_draws[[d]]
    Y_m <- Y_m_draws[[d]]
    I_m <- I_m_draws[[d]]

    for (s in seq_len(n_shocks)) {
      # Baseline sequences for this shock
      pi_z <- wold_base[1, s, ]
      y_z <- wold_base[2, s, ]
      i_z <- wold_base[3, s, ]

      if (use_optpol) {
        result <- compute_optpol(pi_z, y_z, i_z, Pi_m, Y_m, I_m, weights)
        wold_cnfctl[1, s, , d] <- result$pi_z_optpol
        wold_cnfctl[2, s, , d] <- result$y_z_optpol
        wold_cnfctl[3, s, , d] <- result$i_z_optpol
      } else {
        result <- compute_cnfctl(pi_z, y_z, i_z, Pi_m, Y_m, I_m, rule)
        wold_cnfctl[1, s, , d] <- result$pi_z_cnfctl
        wold_cnfctl[2, s, , d] <- result$y_z_cnfctl
        wold_cnfctl[3, s, , d] <- result$i_z_cnfctl
      }
    }
  }

  wold_cnfctl
}

#' @title VMA-implied covariance matrix from Wold IRFs
#' @description Computes the covariance matrix implied by the VMA representation of Wold IRFs.
#' @param wold 3D array (n_var x n_shocks x T) of Wold IRFs
#' @return n_var x n_var covariance matrix
#' @export
vma_cov <- function(wold) {
  n_var <- dim(wold)[1]
  n_shocks <- dim(wold)[2]
  T <- dim(wold)[3]

  cov_mat <- matrix(0, n_var, n_var)
  for (h in seq_len(T)) {
    Theta_h <- wold[, , h]
    cov_mat <- cov_mat + Theta_h %*% t(Theta_h)
  }

  cov_mat
}

#' @title Correlation matrix from covariance
#' @description Converts a covariance matrix to a correlation matrix.
#' @param cov Covariance matrix
#' @return Correlation matrix
#' @export
cov_to_corr_r <- function(cov) {
  cov_to_corr(cov)
}

#' @title Business cycle statistics from Wold IRFs
#' @description Computes variances, standard deviations, and correlations implied
#' by the VMA representation.
#' @param wold 3D array (n_var x n_shocks x T) or 4D array (n_var x n_shocks x T x n_draws)
#' @param var_names Optional variable names
#' @return A list with covariance, correlation, std, and quantiles across draws
#' @export
bc_stats <- function(wold, var_names = NULL) {
  dims <- dim(wold)

  if (length(dims) == 3) {
    # Single set of IRFs
    cov_mat <- vma_cov(wold)
    corr_mat <- cov_to_corr_r(cov_mat)
    std_vec <- sqrt(diag(cov_mat))

    if (!is.null(var_names)) {
      dimnames(cov_mat) <- list(var_names, var_names)
      dimnames(corr_mat) <- list(var_names, var_names)
      names(std_vec) <- var_names
    }

    return(list(
      cov = cov_mat,
      corr = corr_mat,
      std = std_vec
    ))
  } else if (length(dims) == 4) {
    # Multiple draws
    n_var <- dims[1]
    n_draws <- dims[4]

    cov_array <- array(NA, dim = c(n_var, n_var, n_draws))
    corr_array <- array(NA, dim = c(n_var, n_var, n_draws))
    std_mat <- matrix(NA, n_var, n_draws)

    for (d in seq_len(n_draws)) {
      cov_mat <- vma_cov(wold[, , , d])
      cov_array[, , d] <- cov_mat
      corr_array[, , d] <- cov_to_corr_r(cov_mat)
      std_mat[, d] <- sqrt(diag(cov_mat))
    }

    # Summary statistics
    cov_med <- apply(cov_array, 1:2, median)
    cov_lb <- apply(cov_array, 1:2, quantile, probs = 0.16)
    cov_ub <- apply(cov_array, 1:2, quantile, probs = 0.84)

    corr_med <- apply(corr_array, 1:2, median)
    corr_lb <- apply(corr_array, 1:2, quantile, probs = 0.16)
    corr_ub <- apply(corr_array, 1:2, quantile, probs = 0.84)

    std_med <- apply(std_mat, 1, median)
    std_lb <- apply(std_mat, 1, quantile, probs = 0.16)
    std_ub <- apply(std_mat, 1, quantile, probs = 0.84)

    if (!is.null(var_names)) {
      dn <- list(var_names, var_names)
      dimnames(cov_med) <- dn
      dimnames(corr_med) <- dn
      names(std_med) <- var_names
    }

    return(list(
      cov_med = cov_med, cov_lb = cov_lb, cov_ub = cov_ub,
      corr_med = corr_med, corr_lb = corr_lb, corr_ub = corr_ub,
      std_med = std_med, std_lb = std_lb, std_ub = std_ub,
      cov_array = cov_array,
      corr_array = corr_array,
      std_mat = std_mat
    ))
  }
}
