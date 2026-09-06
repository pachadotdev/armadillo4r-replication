#' @useDynLib bp13, .registration = TRUE
#' @keywords internal
#' @import ggplot2
#' @importFrom ggh4x facet_grid2
#' @importFrom dplyr mutate
#' @importFrom purrr map_df
#' @importFrom rlang sym
"_PACKAGE"

# Silence R CMD check notes about NSE in ggplot2/dplyr
# utils::globalVariables(c(
#   "date", "shock", "variable", "elasticity",
#   "median", "p025", "p975", "p16", "p84", "max", "min",
#   "lower", "upper"
# ))

# Baumeister and Peersman (2013) datasets ----
# Imported from BP2oil.xls (the data file used by the MATLAB code)
# Note: The text files in bp-data/ are NOT used by the MATLAB programs

#' Benchmark oil market data
#'
#' Quarterly data on world oil production, the real price of crude oil, and
#' world industrial production from 1947Q1 to 2010Q4. This is the benchmark
#' specification used in Baumeister and Peersman (2013).
#'
#' @docType data
#' @usage bp_bench
#' @format A matrix with 256 rows and 3 columns:
#' \describe{
#'   \item{qo}{World oil production (QO)}
#'   \item{po}{Real price of crude oil (PO): imported RAC, not adjusted for price controls}
#'   \item{y}{World industrial production (Y) - UN monthly statistical bulletin, seasonally adjusted}
#' }
#' @references Christiane Baumeister & Gert Peersman, 2013. "The Role of
#'   Time-Varying Price Elasticities in Accounting for Volatility Changes in
#'   the Crude Oil Market," Journal of Applied Econometrics, vol. 28(7), pages
#'   1087-1109.
#' @source "BP2oil.xls" file (sheet "real_B") in the supplementary file "bp-codes.zip"
"bp_bench"

#' Oil market data with Kilian's global activity index
#'
#' Quarterly data on world oil production, the real price of crude oil, and a
#' global business cycle index from 1967Q4 to 2010Q4. This uses Kilian's
#' global activity index (JEEA specification) for robustness checks.
#'
#' @docType data
#' @usage bp_kilian
#' @format A matrix with 173 rows and 3 columns:
#' \describe{
#'   \item{qo}{World oil production}
#'   \item{po}{Real oil price index: imported RAC (not adjusted for price controls)}
#'   \item{y}{World industrial production index (UN monthly statistical bulletin)}
#' }
#' @references Christiane Baumeister & Gert Peersman, 2013. "The Role of
#'   Time-Varying Price Elasticities in Accounting for Volatility Changes in
#'   the Crude Oil Market," Journal of Applied Econometrics, vol. 28(7), pages
#'   1087-1109.
#' @source "BP2oil.xls" file (sheet "real_K") in the supplementary file "bp-codes.zip"
"bp_kilian"
