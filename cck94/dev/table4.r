bw4 <- list()

bw4$tax_rate_labour_income <- matrix(
    c(
        23.87, 20.69, 23.80, 23.87, 23.84,
        0.1, 0.04, 0.08, 0.06, 0.15,
        0.8, 0.85, 0.71, 0.9, -0.04,
        0.65, -0.59, NA, 1, 0.1,
        0.55, -0.84, 0.64, NA, 0.95
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
)

colnames(bw4$tax_rate_labour_income) <- c("baseline", "high_ra", "only_z", "only_g", "iid")
rownames(bw4$tax_rate_labour_income) <- c("E", "sd", "rho", "corr_g", "corr_z")

bw4$ex_ante_tax_rate_capital <- matrix(
    c(
        0, -0.06, 0, 0, 0,
        0, 4.06, 0, 0, 0,
        NA, 0.83, NA, NA, NA,
        NA, 0.33, NA, NA, NA,
        NA, 0.95, NA, NA, NA
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
)

colnames(bw4$ex_ante_tax_rate_capital) <- colnames(bw4$tax_rate_labour_income)
rownames(bw4$ex_ante_tax_rate_capital) <- rownames(bw4$tax_rate_labour_income)

# the last 0.33 here may have been reported with a sign error
bw4$ex_post_tax_rate_capital <- matrix(
    c(
        0.55, -0.42, 1.19, -0.59, 0.23,
        40.93, 30.35, 17.67, 36.22, 12.03,
        -0.01, 0.02, 0.01, 0.01, -0.02,
        0.4, 0.47, NA, 0.46, 0.94,
        -0.24, -0.2, -0.56, NA, 0.33
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
)

colnames(bw4$ex_post_tax_rate_capital) <- colnames(bw4$tax_rate_labour_income)
rownames(bw4$ex_post_tax_rate_capital) <- rownames(bw4$tax_rate_labour_income)

usethis::use_data(bw4, overwrite = TRUE)
