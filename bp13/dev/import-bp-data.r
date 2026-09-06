# Import data from Baumeister and Peersman (2013)
# "The Role of Time-Varying Price Elasticities in Accounting for Volatility
# Changes in the Crude Oil Market", Journal of Applied Econometrics

# Note: The datasets in bp-data/ are NOT the ones used by the MATLAB code.
# The actual d is in bp-codes/data/BP2oil.xls (see readme.bp.txt)

if (!require("readxl")) install.packages("readxl")
if (!require("dplyr")) install.packages("dplyr")
if (!require("janitor")) install.packages("janitor")

library(readxl)
library(dplyr)
library(janitor)

# Helper function to import BP data from Excel ----

import_bp_excel <- function(file = "dev/bp-codes/data/BP2oil.xls",
    sheet = "real_B", range = "A1:D257", freq = "quarter") {
  freq <- match.arg(freq)
  
  # Read from Excel
  d <- clean_names(read_excel(file, sheet = sheet, range = range, col_names = TRUE))

  d2 <- as.matrix(d[, colnames(d) != "x1"])

  # convert to numeric matrix
    mode(d2) <- "numeric"

  rownames(d2) <- d$x1

  d2
}

# bp_bench ----

# Quarterly data from 1947Q1 to 2010Q4 (256 observations)
# Source: BP2oil.xls sheet "real_B", range A1:D257

bp_bench <- import_bp_excel(
  file = "dev/bp-codes/data/BP2oil.xls",
  sheet = "real_B",
  range = "A1:D257"
)

colnames(bp_bench) <- c("qo", "po", "y")

usethis::use_data(bp_bench, overwrite = TRUE)

# bp_kilian ----

# Quarterly data from 1967Q4 to 2010Q4 (173 observations)
# Source: BP2oil.xls sheet "real_K", range A1:D174

bp_kilian <- import_bp_excel(
  file = "dev/bp-codes/data/BP2oil.xls",
  sheet = "real_K",
  range = "A1:D174"
)

colnames(bp_kilian) <- c("qo", "po", "y")

usethis::use_data(bp_kilian, overwrite = TRUE)
