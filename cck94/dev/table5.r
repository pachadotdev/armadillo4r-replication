bw5 <- list()

bw5$tax_rate_labour_income <- matrix(
    c(
        23.87, 20.69, 23.87, 23.87, 23.87,
        0.095, 0.034, 0.074, 0.059, 0.147,
        0.776, 0.811, 0.685, 0.895, -0.068,
        0.62, -0.55, NA, 0.999, 0.099,
        0.496, -0.802, 0.632, NA, 0.954
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
)

colnames(bw5$tax_rate_labour_income) <- c("baseline", "high_ra", "only_z", "only_g", "iid")
rownames(bw5$tax_rate_labour_income) <- c("E", "sd", "rho", "corr_g", "corr_z")

bw5$ex_ante_tax_rate_capital <- matrix(
    c(
        0, 0.002, 0, 0, 0,
        0, 3.289, 0, 0, 0,
        NA, 0.804, NA, NA, NA,
        NA, 0.252, NA, NA, NA,
        NA, 0.965, NA, NA, NA
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
)

colnames(bw5$ex_ante_tax_rate_capital) <- colnames(bw5$tax_rate_labour_income)
rownames(bw5$ex_ante_tax_rate_capital) <- rownames(bw5$tax_rate_labour_income)

bw5$ex_post_tax_rate_capital <- matrix(
    c(
        0.001, 0.003, -0.003, 0.004, 0.001,
        36.155, 30.581, 15.769, 32.512, 10.818,
        0, -0.003, -0.002, 0, 0,
        0.41, 0.444, NA, 0.456, 0.913,
        -0.255, -0.132, -0.586, NA, -0.409
    ),
    nrow = 5,
    ncol = 5,
    byrow = T
)

colnames(bw5$ex_post_tax_rate_capital) <- colnames(bw5$tax_rate_labour_income)
rownames(bw5$ex_post_tax_rate_capital) <- rownames(bw5$tax_rate_labour_income)

usethis::use_data(bw5, overwrite = TRUE)
