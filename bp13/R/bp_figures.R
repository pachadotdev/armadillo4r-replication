#' @title Prepare Data for Baumeister-Peersman (2013) Analysis
#' @description Prepares data matrix and estimates TVP-VAR model for replication.
#' @param data Data matrix with columns qo (oil production), po (real oil price),
#'   and y (global activity). If NULL, uses bp_bench dataset.
#' @param L Lag order (default 4 for quarterly data).
#' @param training_years Years for training sample (default 25).
#' @param n_burn Number of burn-in iterations (default 50000).
#' @param n_draws Number of draws from Gibbs sampler (default 5000).
#' @param thinning Thinning interval (default 10).
#' @param stability_constraint Whether to impose VAR stability (default 0).
#' @param lambda_param Time variation parameter (default 0.0001 = Primiceri "Small").
#' @param first_diff Whether data is in first differences (default 1).
#' @return A list containing:
#'   \describe{
#'     \item{y}{Prepared data matrix}
#'     \item{tvp_result}{TVP-VAR estimation results}
#'     \item{bp_theme}{Common ggplot theme for consistency}
#'   }
#' @export
bp_prepare_data <- function(data = NULL, L = 4, training_years = 25,
                            n_burn = 50000, n_draws = 5000, thinning = 10,
                            stability_constraint = 0, lambda_param = 0.0001,
                            first_diff = 1) {
  # Load default benchmark data if not provided
  if (is.null(data)) {
    data <- bp13::bp_bench
  }

  # Prepare data matrix (columns: qo, po, y)
  # Transform to match MATLAB preprocessing:
  # 1. Take 100 * log of each series
  # 2. If first_diff=1, compute first differences
  y_raw <- as.matrix(data[, c("qo", "po", "y")])
  y_log <- 100 * log(y_raw)
  
  if (first_diff == 1) {
    # First differences (like MATLAB lagn function with m=1)
    y <- diff(y_log)
  } else {
    y <- y_log
  }
  
  # Handle NaNs: remove rows with any NaNs (e.g. beginning of sample)
  y <- stats::na.omit(y)
  
  # Ensure clean numeric matrix
  y <- matrix(as.numeric(y), nrow = nrow(y), ncol = ncol(y))
  
  message("  Data prepared: ", nrow(y), " observations, ", ncol(y), " variables")
  message("  y range: [", paste(round(range(y), 2), collapse=", "), "]")

  # Common theme for plots
  bp_theme <- theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA,
        linewidth = 0.3),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_blank()
    )

  # Estimate TVP-VAR
  message("-----------------------------------")
  message("Step 1: Estimating TVP-VAR model...")
  tvp_result <- tvp_var_estimate(
    y = y,
    L = L,
    training_years = training_years,
    n_burn = n_burn,
    n_draws = n_draws,
    thinning = thinning,
    stability_constraint = stability_constraint,
    lambda_param = lambda_param,
    first_diff = first_diff
  )

  list(
    y = y,
    tvp_result = tvp_result,
    bp_theme = bp_theme
  )
}

#' @title Generate Figure 2 - Time-Varying Impact IRFs
#' @description Creates Figure 2 from Baumeister-Peersman (2013) showing time-varying impact IRFs.
#' @param y Prepared data matrix.
#' @param tvp_result TVP-VAR estimation results.
#' @param bp_theme Common ggplot theme.
#' @param L Lag order (default 4).
#' @param n_reps Number of sign-restriction draws per time period (default 500).
#' @param first_diff Whether data is in first differences (default 1).
#' @param start_year Starting year for plots (default 1974).
#' @param start_quarter Starting quarter for plots (default 1).
#' @return ggplot object for Figure 2.
#' @export
bp_figure2 <- function(y, tvp_result, bp_theme, L = 4, n_reps = 500,
                       first_diff = 1, start_year = 1974, start_quarter = 1) {
  message("-------------------------------------------")
  message("Step 2: Computing Figure 2 (Impact IRFs)...")

  fig2_data <- tvp_var_figure2(
    tvp_result = tvp_result,
    y = y,
    L = L,
    n_reps = n_reps,
    h_restrict = 1,
    first_diff = first_diff,
    upper_supply = 0.6,
    lower_demand = -0.8,
    start_period = 4
  )

  # Create date sequence for Figure 2
  start_date <- as.Date(paste0(start_year, "-", (start_quarter - 1) * 3 + 1, "-01"))
  dates_fig2 <- seq(start_date, by = "quarter", length.out = fig2_data$n_time)

  col_names_fig2 <- c("median", "p025", "p975", "p16", "p84", "max", "min")

  fig2_plot_data <- purrr::map_df(
    list(
      list(data = fig2_data$s_prod, shock = "Oil supply shock", variable = "World oil production"),
      list(data = fig2_data$s_price, shock = "Oil supply shock", variable = "Real oil price"),
      list(data = fig2_data$d_prod, shock = "Other oil demand shock", variable = "World oil production"),
      list(data = fig2_data$d_price, shock = "Other oil demand shock", variable = "Real oil price"),
      list(data = fig2_data$a_prod, shock = "Aggregate demand shock", variable = "World oil production"),
      list(data = fig2_data$a_price, shock = "Aggregate demand shock", variable = "Real oil price")
    ),
    function(item) {
      item$data |>
        as.data.frame() |>
        stats::setNames(col_names_fig2) |>
        mutate(
          date = dates_fig2,
          shock = item$shock,
          variable = item$variable
        )
    }
  ) |>
    mutate(
      shock = factor(!!sym("shock"), levels = c(
        "Oil supply shock", 
        "Other oil demand shock", 
        "Aggregate demand shock"
      )),
      variable = factor(!!sym("variable"), levels = c(
        "World oil production", 
        "Real oil price"
      ))
  )

  # colours taken from the article using GIMP
  # Figure 2. Time-varying median impact impulse responses (thick solid lines) of world oil production and the real
  # price of crude oil after oil supply shocks (1st row), other oil demand shocks (2nd row) and aggregate demand
  # shocks (3rd row), where the dark and light shaded areas indicate respectively 68% and 95% posterior credible sets
  # and the thin black lines indicate the range of admissible models
  ggplot(fig2_plot_data, aes(x = date)) +
    geom_ribbon(aes(ymin = !!rlang::sym("p025"), ymax = !!rlang::sym("p975")), fill = "#bfbfff") +
    geom_ribbon(aes(ymin = !!rlang::sym("p16"), ymax = !!rlang::sym("p84")), fill = "#408dff") +
    geom_line(aes(y = !!rlang::sym("median")), color = "#0000ff", linewidth = 1.2) +
    geom_line(aes(y = !!rlang::sym("max")), color = "black", linewidth = 0.3) +
    geom_line(aes(y = !!rlang::sym("min")), color = "black", linewidth = 0.3) +
    ggh4x::facet_grid2(rows = vars(!!rlang::sym("shock")), cols = vars(!!rlang::sym("variable")), scales = "free_y", independent = "y", switch = "y") +
    theme(
      strip.placement = "outside",
      strip.text.y.left = element_text(angle = 0, vjust = 0.5)
    ) +
    scale_x_date(date_breaks = "5 years", date_labels = "%Y") +
    labs(x = "", y = "") +
    bp_theme
}

#' @title Generate Figure 3 - Persistence of Real Oil Price Inflation
#' @description Creates Figure 3 from Baumeister-Peersman (2013) showing oil price persistence.
#' @param tvp_result TVP-VAR estimation results.
#' @param bp_theme Common ggplot theme.
#' @param start_year Starting year for plots (default 1974).
#' @param start_quarter Starting quarter for plots (default 1).
#' @return ggplot object for Figure 3.
#' @importFrom ggplot2 ggplot aes geom_ribbon geom_line
#'   labs scale_x_date scale_y_continuous
#' @export
bp_figure3 <- function(tvp_result, bp_theme, start_year = 1974, start_quarter = 1) {
  message("-------------------------------------------")
  message("Step 3: Computing Figure 3 (Persistence)...")

  fig3_data <- tvp_var_figure3(
    tvp_result = tvp_result,
    n_freq = 50,
    price_var = 2
  )

  start_date <- as.Date(paste0(start_year, "-", (start_quarter - 1) * 3 + 1, "-01"))
  dates_fig3 <- seq(start_date, by = "quarter", length.out = fig3_data$T_eff)

  fig3_plot_data <- data.frame(
    date = dates_fig3,
    median = fig3_data$persistence_median,
    lower = fig3_data$persistence_lower,
    upper = fig3_data$persistence_upper,
    p16 = fig3_data$persistence_16,
    p84 = fig3_data$persistence_84
  )

  # Figure 3. Time-varying normalized spectrum of real oil price inﬂation as a measure of persistence, where the
  # shaded area covers 95% of the posterior distribution and the solid line is the median
  ggplot(fig3_plot_data, aes(x = date)) +
    geom_ribbon(aes(ymin = !!rlang::sym("lower"), ymax = !!rlang::sym("upper")), fill = "#bfbfff") +
    geom_line(aes(y = !!rlang::sym("median")), color = "#0000ff", linewidth = 1.2) +
    scale_x_date(date_breaks = "5 years", date_labels = "%Y") +
    scale_y_continuous(limits = c(0, NA)) +
    labs(x = "", y = "", title = "Persistence of Real Oil Price Inflation") +
    bp_theme
}

#' @title Generate Figure 4 - Time-Varying Elasticities
#' @description Creates Figure 4 from Baumeister-Peersman (2013) showing time-varying elasticities.
#' @param y Prepared data matrix.
#' @param tvp_result TVP-VAR estimation results.
#' @param bp_theme Common ggplot theme.
#' @param L Lag order (default 4).
#' @param n_reps Number of sign-restriction draws per time period (default 500).
#' @param first_diff Whether data is in first differences (default 1).
#' @param start_year Starting year for plots (default 1974).
#' @param start_quarter Starting quarter for plots (default 1).
#' @return ggplot object for Figure 4.
#' @importFrom ggplot2 ggplot aes geom_ribbon geom_line facet_wrap
#'   labs scale_x_date
#' @importFrom dplyr mutate
#' @importFrom purrr map_df
#' @export
bp_figure4 <- function(y, tvp_result, bp_theme, L = 4, n_reps = 500,
                       first_diff = 1, start_year = 1974, start_quarter = 1) {
  message("--------------------------------------------")
  message("Step 4: Computing Figure 4 (Elasticities)...")

  fig4_data <- tvp_var_figure4(
    tvp_result = tvp_result,
    y = y,
    L = L,
    n_reps = n_reps,
    h_restrict = 1,
    first_diff = first_diff,
    upper_supply = 0.6,
    lower_demand = -0.8,
    start_period = 4
  )

  start_date <- as.Date(paste0(start_year, "-", (start_quarter - 1) * 3 + 1, "-01"))
  dates_fig4 <- seq(start_date, by = "quarter", length.out = fig4_data$n_time)

  col_names_fig4 <- c("median", "p025", "p975", "p16", "p84", "max", "min")

  fig4_plot_data <- purrr::map_df(
    list(
      list(data = fig4_data$elas_supply_agg, elasticity = "Oil supply elasticity with aggregate demand shock"),
      list(data = fig4_data$elas_supply_dem, elasticity = "Oil supply elasticity with other oil demand shock"),
      list(data = fig4_data$elas_demand, elasticity = "Oil demand elasticity")
    ),
    function(item) {
      item$data |>
        as.data.frame() |>
        stats::setNames(col_names_fig4) |>
        mutate(
          date = dates_fig4,
          elasticity = item$elasticity
        )
    }
  ) |>
    mutate(
      elasticity = factor(!!sym("elasticity"), levels = c(
        "Oil supply elasticity with aggregate demand shock",
        "Oil supply elasticity with other oil demand shock",
        "Oil demand elasticity"
      ))
    )

  # Figure 4. Median short-run price elasticities of oil supply and oil demand (bold solid lines) together with the 68%
  # and 95% posterior credible sets (dark and light shaded areas) and the range of admissible models (thin solid lines).
  # The slope of the oil supply curve is derived with aggregate demand shocks and other oil demand shocks
  ggplot(fig4_plot_data, aes(x = date)) +
    geom_ribbon(aes(ymin = !!rlang::sym("p025"), ymax = !!rlang::sym("p975")), fill = "#bfbfff") +
    geom_ribbon(aes(ymin = !!rlang::sym("p16"), ymax = !!rlang::sym("p84")), fill = "#408dff") +
    geom_line(aes(y = !!rlang::sym("median")), color = "#0000ff", linewidth = 1.2) +
    geom_line(aes(y = !!rlang::sym("max")), color = "black", linewidth = 0.3) +
    geom_line(aes(y = !!rlang::sym("min")), color = "black", linewidth = 0.3) +
    facet_wrap(vars(!!rlang::sym("elasticity")), ncol = 1, scales = "free_y") +
    scale_x_date(date_breaks = "5 years", date_labels = "%Y") +
    labs(x = "", y = "") +
    bp_theme
}
