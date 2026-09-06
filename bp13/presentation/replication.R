# Baumeister-Peersman (2013) Figure Replication

testing <- FALSE

cpp4r::register(getwd())
devtools::document()
devtools::install()

library(bp13)

# Step 1: Prepare data and estimate TVP-VAR
# For testing: n_burn = 5000, n_draws = 500
# For full replication: n_burn = 50000, n_draws = 5000

n_burn <- if (testing) 5000 else 50000
n_draws <- if (testing) 500 else 5000

if (testing) {
    prep_data <- bp_prepare_data(n_burn = 5000, n_draws = 500)
} else {
    prep_data <- bp_prepare_data(n_burn = 50000, n_draws = 5000)
}

# Step 2: Generate individual figures
# n_reps controls sign-restriction draws per time period (MATLAB uses R=500)
figure2 <- bp_figure2(
    y = prep_data$y,
    tvp_result = prep_data$tvp_result,
    bp_theme = prep_data$bp_theme,
    n_reps = 500
)

figure3 <- bp_figure3(
    tvp_result = prep_data$tvp_result,
    bp_theme = prep_data$bp_theme
)

figure4 <- bp_figure4(
    y = prep_data$y,
    tvp_result = prep_data$tvp_result,
    bp_theme = prep_data$bp_theme,
    n_reps = 500
)

# View figures
figure2
figure3
figure4

fout <- sprintf("presentation/results-burns%d-draws%d.rds", n_burn, n_draws)

results <- list(
    figure2 = figure2,
    figure3 = figure3,
    figure4 = figure4,
    prep_data = prep_data
)

saveRDS(results, fout, compress = "xz")
