# Evaluating Monetary Policy  Counterfactuals: (When) Do We Need Structural Models?

R port of the replication files for "Evaluating Monetary Policy  Counterfactuals: (When) Do We
Need Structural Models?" by Caravello, McKay and Wolf. Implements Bayesian VAR estimation, impulse
response analysis, and frequency domain variance decomposition using Armadillo for efficient matrix
computations.

```r
if (!require("remotes")) install.packages("remotes", repos = "http://cran.us.r-project.org")
remotes::install_github("pachadotdev/armadillo4r-replication", subdir = "cmw26")
```

The full replication is in `presentation/replication.r` and the results are summarised in
`presentation/presentation.qmd`

To run the replication code (and not just the presentation), you need to extract:

1. These into into `presentation/suff_stats`

https://github.com/pachadotdev/armadillo4r-replication/releases/download/v0.1/cmw26_suff_stats_part1.zip

https://github.com/pachadotdev/armadillo4r-replication/releases/download/v0.1/cmw26_suff_stats_part2.zip

https://github.com/pachadotdev/armadillo4r-replication/releases/download/v0.1/cmw26_suff_stats_part3.zip

https://github.com/pachadotdev/armadillo4r-replication/releases/download/v0.1/cmw26_suff_stats_part4.zip

2. These `presentation/var_inputs`

https://github.com/pachadotdev/armadillo4r-replication/releases/download/v0.1/cmw26_var_inputs.zip

and then run `presentation/replication.r` directly from the `presentation` directory.

These files are too heavy to be pushed directly to the repository.

# How to use armadillo4r?

## Read the documentation

The [basic examples](https://pacha.dev/armadillo4r/articles/v01-basic-usage.html)
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
