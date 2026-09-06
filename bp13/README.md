# Replication of 'The Role of Time-Varying Price Elasticities in Accounting for Volatility Changes in the Crude Oil Market' (Baumeister and Peersman, 2013)

Replication of figures 2, and 4 using [armadillo4r](https://github.com/pachadotdev/armadillo4r).

Reference:

```
@article{baumeister_role_2013,
	title = {The {Role} of {Time}-{Varying} {Price} {Elasticities} in {Accounting} for {Volatility} {Changes} in the {Crude} {Oil} {Market}},
	volume = {28},
	copyright = {Copyright © 2012 John Wiley \& Sons, Ltd.},
	issn = {1099-1255},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.2283},
	doi = {10.1002/jae.2283},
	abstract = {There has been a systematic increase in the volatility of the real price of crude oil since 1986, followed by a decline in the volatility of oil production since the early 1990s. We explore reasons for this evolution. We show that a likely explanation of this empirical fact is that both the short-run price elasticities of oil demand and of oil supply have declined considerably since the second half of the 1980s. This implies that small disturbances on either side of the oil market can generate large price responses without large quantity movements, which helps explain the latest run-up and subsequent collapse in the price of oil. Our analysis suggests that the variability of oil demand and supply shocks actually has decreased in the more recent past, preventing even larger oil price fluctuations than observed in the data.},
	language = {en},
	number = {7},
	urldate = {2025-12-04},
	journal = {Journal of Applied Econometrics},
	author = {Baumeister, Christiane and Peersman, Gert},
	year = {2013},
	note = {\_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/jae.2283},
	pages = {1087--1109}
}
```

# Structure of this repository

This provides an R package with C++ code to use Armadillo.

Run the following code to install the package:

```r
if (!require("remotes")) install.packages("remotes", repos = "http://cran.us.r-project.org")
remotes::install_github("pachadotdev/armadillo4r-replication", subdir = "bp13")
```

The package uses two dependencies:

1. armadillo4r for the computation part
2. ggplot2 for plotting

The fitted models were added to [releases](https://github.com/pachadotdev/armadillo4r-replication/releases/tag/models) because those are too heavy for a commit to the repository.

Here is the code I used, the package provides some user-callable function to fit the models and plot. If you want to
just test the code *please* set `testing <- TRUE` to reduce computation time.

```r
# Baumeister-Peersman (2013) Figure Replication

testing <- FALSE

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
```

# How to use armadillo4r?

## Read the documentation

The [basic examples](https://pacha.dev/armadillo4r/articles/basic-usage.html)
should be enough to get you started. 

## Test the package

1. Open `./dev/01_load_or_install.R` and run it.
2. Open `./src/01_ols.cpp`, inspect the OLS functions.
3. Create your own functions in `./02_your_functions.cpp`.
4. Run `devtools::load_all()` before testing your functions, and then add tests
   to `./dev/02_test.R` and run it.
5. Run `./dev/03_readme_and_license.R` to add a README and license.
6. Run `devtools::install()` in `./dev/01_load_or_install.R` to install the
   package locally when you are ready.
7. The package template includes a configuration with a predefined number of
   cores to use. You can change this value by doing this:

   Unix: Edit `./src/Makevars.in` to set `DARMA_OPENMP_THREADS` to another
         value or edit `./configure` to change `PKG_NCORES` to another value.
   
   Windows: Edit `./src/Makevars.win` to set `DARMA_OPENMP_THREADS` to another
            value.

## Additional documentation

1. [Matrix, vector, cube and field classes](https://pacha.dev/armadillo4r/articles/v06-matrix-vector-cube-and-field-classes.html)
2. [Member functions and variables](https://pacha.dev/armadillo4r/articles/v07-member-functions-and-variables.html)
3. [Functions of vectors, matrices, and cubes](https://pacha.dev/armadillo4r/articles/v08-functions-of-vector-matrices-cubes.html)
4. [Generated vectors, matrices, and cubes](https://pacha.dev/armadillo4r/articles/v09-generated-vectors-matrices-cubes.html)
5. [Fitting regressions with Armadillo](https://pacha.dev/armadillo4r/articles/v16-linear-model.html)

For specific Econometrics examples, see the [hansen package](https://pacha.dev/hansen/) documentation.
