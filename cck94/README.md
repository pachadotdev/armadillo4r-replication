# Replication of ‘Optimal Fiscal Policy in a Business Cycle Model’ by Chari, Christiano, and Kehoe (1994)

The goal of cck94 is to replicate the results from Chari et al. (1994)
using time iteration, the Endogenous Grid Method (ECM) by Maliar and
Maliar (2013), and projections (original method).

For the time iteration approach I used a Monte Carlo 1st-order Linear
Quadratic (LQ) approximation and a 2nd-order perturbation following
Benigno and Woodford (2005).

I do not have a MATLAB license, so I looked at Grohe-Uribe solvers to
adapt a bit an write a small program using Armadillo.

The files `dev/replicate-cck94-table4.R` and
`dev/replicate-cck94-table5.R` contain the code to compute the tables
below. `dev/table4.r` and `dev/table5.r` contain the code to organize
the original results.

## Installation

You can install the development version of cck94 from
[GitHub](https://github.com/) with:

```r
if (!require("remotes")) install.packages("remotes", repos = "http://cran.us.r-project.org")
remotes::install_github("pachadotdev/armadillo4r-replication", subdir = "cck94")
```

## Replication result (following BW05 presentation)

### Table 4: Statistics on optimal tax rates reported by Chari et al. (1994, Table 2).

| Scenario | Variable | Statistic | Published | Time.Iter |     ECM | Projection |
|:---------|:---------|:----------|----------:|----------:|--------:|-----------:|
| baseline | tau_h    | E         |     23.87 |   23.8661 | 23.8659 |    23.8661 |
| baseline | tau_h    | sd        |      0.10 |    0.0934 |  0.0934 |     0.0934 |
| baseline | tau_h    | rho       |      0.80 |    0.7580 |  0.7580 |     0.7580 |
| baseline | tau_h    | corr_g    |      0.65 |    0.6069 |  0.6069 |     0.6069 |
| baseline | tau_h    | corr_z    |      0.55 |    0.5076 |  0.5077 |     0.5076 |
| baseline | theta_e  | E         |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| baseline | theta_e  | sd        |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| baseline | theta_e  | rho       |        NA |    0.8773 |  0.8773 |     0.8773 |
| baseline | theta_e  | corr_g    |        NA |    0.9708 |  0.9708 |     0.9708 |
| baseline | theta_e  | corr_z    |        NA |    0.1512 |  0.1512 |     0.1512 |
| baseline | tau_k    | E         |      0.55 |    0.0061 |  0.0088 |     0.0061 |
| baseline | tau_k    | sd        |     40.93 |   36.2321 | 36.2321 |    36.2321 |
| baseline | tau_k    | rho       |     -0.01 |    0.0000 |  0.0000 |     0.0000 |
| baseline | tau_k    | corr_g    |      0.40 |    0.4109 |  0.4109 |     0.4109 |
| baseline | tau_k    | corr_z    |     -0.24 |   -0.2609 | -0.2609 |    -0.2609 |
| high_ra  | tau_h    | E         |     20.69 |   20.6914 | 20.6914 |    20.6914 |
| high_ra  | tau_h    | sd        |      0.04 |    0.0339 |  0.0339 |     0.0339 |
| high_ra  | tau_h    | rho       |      0.85 |    0.8094 |  0.8094 |     0.8094 |
| high_ra  | tau_h    | corr_g    |     -0.59 |   -0.5568 | -0.5568 |    -0.5568 |
| high_ra  | tau_h    | corr_z    |     -0.84 |   -0.7936 | -0.7937 |    -0.7936 |
| high_ra  | theta_e  | E         |     -0.06 |   -0.0303 | -0.0314 |    -0.0303 |
| high_ra  | theta_e  | sd        |      4.06 |    3.2808 |  3.2808 |     3.2808 |
| high_ra  | theta_e  | rho       |      0.83 |    0.8033 |  0.8033 |     0.8033 |
| high_ra  | theta_e  | corr_g    |      0.33 |    0.2521 |  0.2521 |     0.2521 |
| high_ra  | theta_e  | corr_z    |      0.95 |    0.9643 |  0.9643 |     0.9643 |
| high_ra  | tau_k    | E         |     -0.42 |    0.3883 |  0.4010 |     0.3883 |
| high_ra  | tau_k    | sd        |     30.35 |   29.6167 | 29.6165 |    29.6167 |
| high_ra  | tau_k    | rho       |      0.02 |    0.0048 |  0.0048 |     0.0048 |
| high_ra  | tau_k    | corr_g    |      0.47 |    0.4660 |  0.4660 |     0.4660 |
| high_ra  | tau_k    | corr_z    |     -0.20 |   -0.0913 | -0.0913 |    -0.0913 |
| only_z   | tau_h    | E         |     23.80 |   23.8679 | 23.8677 |    23.8679 |
| only_z   | tau_h    | sd        |      0.08 |    0.0742 |  0.0742 |     0.0742 |
| only_z   | tau_h    | rho       |      0.71 |    0.6827 |  0.6827 |     0.6827 |
| only_z   | tau_h    | corr_g    |        NA |   -0.0015 | -0.0015 |    -0.0015 |
| only_z   | tau_h    | corr_z    |      0.64 |    0.6414 |  0.6415 |     0.6414 |
| only_z   | theta_e  | E         |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| only_z   | theta_e  | sd        |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| only_z   | theta_e  | rho       |        NA |    0.6759 |  0.6758 |     0.6759 |
| only_z   | theta_e  | corr_g    |        NA |    0.0018 |  0.0018 |     0.0018 |
| only_z   | theta_e  | corr_z    |        NA |   -0.7569 | -0.7570 |    -0.7569 |
| only_z   | tau_k    | E         |      1.19 |    0.0099 |  0.0126 |     0.0099 |
| only_z   | tau_k    | sd        |     17.67 |   15.9630 | 15.9630 |    15.9630 |
| only_z   | tau_k    | rho       |      0.01 |   -0.0009 | -0.0009 |    -0.0009 |
| only_z   | tau_k    | corr_g    |        NA |    0.0043 |  0.0043 |     0.0043 |
| only_z   | tau_k    | corr_z    |     -0.56 |   -0.5875 | -0.5875 |    -0.5875 |
| only_g   | tau_h    | E         |     23.87 |   23.8682 | 23.8681 |    23.8682 |
| only_g   | tau_h    | sd        |      0.06 |    0.0566 |  0.0566 |     0.0566 |
| only_g   | tau_h    | rho       |      0.90 |    0.8845 |  0.8845 |     0.8845 |
| only_g   | tau_h    | corr_g    |      1.00 |    0.9997 |  0.9997 |     0.9997 |
| only_g   | tau_h    | corr_z    |        NA |   -0.0019 | -0.0019 |    -0.0019 |
| only_g   | theta_e  | E         |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| only_g   | theta_e  | sd        |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| only_g   | theta_e  | rho       |        NA |    0.8887 |  0.8887 |     0.8887 |
| only_g   | theta_e  | corr_g    |        NA |    1.0000 |  1.0000 |     1.0000 |
| only_g   | theta_e  | corr_z    |        NA |   -0.0019 | -0.0019 |    -0.0019 |
| only_g   | tau_k    | E         |     -0.59 |   -0.0036 | -0.0026 |    -0.0036 |
| only_g   | tau_k    | sd        |     36.22 |   32.4576 | 32.4576 |    32.4576 |
| only_g   | tau_k    | rho       |      0.01 |    0.0016 |  0.0016 |     0.0016 |
| only_g   | tau_k    | corr_g    |      0.46 |    0.4578 |  0.4578 |     0.4578 |
| only_g   | tau_k    | corr_z    |        NA |   -0.0042 | -0.0042 |    -0.0042 |
| iid      | tau_h    | E         |     23.84 |   23.8662 | 23.8660 |    23.8662 |
| iid      | tau_h    | sd        |      0.15 |    0.1545 |  0.1545 |     0.1545 |
| iid      | tau_h    | rho       |     -0.04 |   -0.0445 | -0.0445 |    -0.0445 |
| iid      | tau_h    | corr_g    |      0.10 |    0.0923 |  0.0923 |     0.0923 |
| iid      | tau_h    | corr_z    |      0.95 |    0.9091 |  0.9091 |     0.9091 |
| iid      | theta_e  | E         |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| iid      | theta_e  | sd        |      0.00 |    0.0000 |  0.0000 |     0.0000 |
| iid      | theta_e  | rho       |        NA |    0.9251 |  0.9251 |     0.9251 |
| iid      | theta_e  | corr_g    |        NA |    0.0758 |  0.0758 |     0.0758 |
| iid      | theta_e  | corr_z    |        NA |   -0.2992 | -0.2992 |    -0.2992 |
| iid      | tau_k    | E         |      0.23 |    0.0395 |  0.0423 |     0.0395 |
| iid      | tau_k    | sd        |     12.03 |   10.9509 | 10.9509 |    10.9509 |
| iid      | tau_k    | rho       |     -0.02 |   -0.0596 | -0.0596 |    -0.0596 |
| iid      | tau_k    | corr_g    |      0.94 |    0.9034 |  0.9034 |     0.9034 |
| iid      | tau_k    | corr_z    |      0.33 |   -0.4216 | -0.4216 |    -0.4216 |

### Table 5: Statistics on optimal tax rates from Monte Carlo simulation of log-linearized optimal policy rules

| Scenario | Variable | Statistic | Published | 1st Order LQ | 2nd.Order |
|:---------|:---------|:----------|----------:|-------------:|----------:|
| baseline | tau_h    | E         |    23.870 |      23.8697 |   23.8697 |
| baseline | tau_h    | sd        |     0.095 |       0.0948 |    0.0948 |
| baseline | tau_h    | rho       |     0.776 |       0.7677 |    0.7677 |
| baseline | tau_h    | corr_g    |     0.620 |       0.6243 |    0.6243 |
| baseline | tau_h    | corr_z    |     0.496 |       0.4913 |    0.4913 |
| baseline | theta_e  | E         |     0.000 |       0.0000 |    0.0000 |
| baseline | theta_e  | sd        |     0.000 |       0.0000 |    0.0000 |
| baseline | theta_e  | rho       |        NA |       0.8795 |    0.8795 |
| baseline | theta_e  | corr_g    |        NA |       0.9714 |    0.9714 |
| baseline | theta_e  | corr_z    |        NA |       0.1474 |    0.1474 |
| baseline | tau_k    | E         |     0.000 |      -0.0458 |   -0.0458 |
| baseline | tau_k    | sd        |    36.155 |      36.2248 |   36.2248 |
| baseline | tau_k    | rho       |     0.000 |      -0.0008 |   -0.0008 |
| baseline | tau_k    | corr_g    |     0.410 |       0.4094 |    0.4094 |
| baseline | tau_k    | corr_z    |    -0.255 |      -0.2603 |   -0.2603 |
| high_ra  | tau_h    | E         |    20.690 |      20.6900 |   20.6900 |
| high_ra  | tau_h    | sd        |     0.034 |       0.0341 |    0.0341 |
| high_ra  | tau_h    | rho       |     0.811 |       0.8127 |    0.8127 |
| high_ra  | tau_h    | corr_g    |    -0.550 |      -0.5638 |   -0.5638 |
| high_ra  | tau_h    | corr_z    |    -0.802 |      -0.7892 |   -0.7892 |
| high_ra  | theta_e  | E         |     0.002 |       0.0073 |    0.0074 |
| high_ra  | theta_e  | sd        |     3.289 |       3.2855 |    3.2855 |
| high_ra  | theta_e  | rho       |     0.804 |       0.8039 |    0.8039 |
| high_ra  | theta_e  | corr_g    |     0.252 |       0.2551 |    0.2551 |
| high_ra  | theta_e  | corr_z    |     0.965 |       0.9635 |    0.9635 |
| high_ra  | tau_k    | E         |     0.000 |      -0.0263 |   -0.0288 |
| high_ra  | tau_k    | sd        |    30.581 |      29.4564 |   29.4564 |
| high_ra  | tau_k    | rho       |    -0.003 |       0.0012 |    0.0012 |
| high_ra  | tau_k    | corr_g    |     0.444 |       0.4564 |    0.4564 |
| high_ra  | tau_k    | corr_z    |    -0.132 |      -0.0948 |   -0.0948 |
| only_z   | tau_h    | E         |    23.870 |      23.8699 |   23.8699 |
| only_z   | tau_h    | sd        |     0.074 |       0.0741 |    0.0741 |
| only_z   | tau_h    | rho       |     0.685 |       0.6850 |    0.6850 |
| only_z   | tau_h    | corr_g    |        NA |      -0.0011 |   -0.0011 |
| only_z   | tau_h    | corr_z    |     0.632 |       0.6320 |    0.6320 |
| only_z   | theta_e  | E         |     0.000 |       0.0000 |    0.0000 |
| only_z   | theta_e  | sd        |     0.000 |       0.0000 |    0.0000 |
| only_z   | theta_e  | rho       |        NA |       0.6761 |    0.6761 |
| only_z   | theta_e  | corr_g    |        NA |       0.0014 |    0.0014 |
| only_z   | theta_e  | corr_z    |        NA |      -0.7504 |   -0.7504 |
| only_z   | tau_k    | E         |     0.000 |      -0.0195 |   -0.0195 |
| only_z   | tau_k    | sd        |    15.769 |      15.9500 |   15.9500 |
| only_z   | tau_k    | rho       |    -0.002 |      -0.0016 |   -0.0016 |
| only_z   | tau_k    | corr_g    |        NA |       0.0042 |    0.0042 |
| only_z   | tau_k    | corr_z    |    -0.586 |      -0.5864 |   -0.5864 |
| only_g   | tau_h    | E         |    23.870 |      23.8698 |   23.8698 |
| only_g   | tau_h    | sd        |     0.059 |       0.0591 |    0.0591 |
| only_g   | tau_h    | rho       |     0.895 |       0.8947 |    0.8947 |
| only_g   | tau_h    | corr_g    |     0.999 |       0.9998 |    0.9998 |
| only_g   | tau_h    | corr_z    |        NA |      -0.0019 |   -0.0019 |
| only_g   | theta_e  | E         |     0.000 |       0.0000 |    0.0000 |
| only_g   | theta_e  | sd        |     0.000 |       0.0000 |    0.0000 |
| only_g   | theta_e  | rho       |        NA |       0.8907 |    0.8907 |
| only_g   | theta_e  | corr_g    |        NA |       1.0000 |    1.0000 |
| only_g   | theta_e  | corr_z    |        NA |      -0.0019 |   -0.0019 |
| only_g   | tau_k    | E         |     0.000 |      -0.0263 |   -0.0263 |
| only_g   | tau_k    | sd        |    32.512 |      32.4576 |   32.4576 |
| only_g   | tau_k    | rho       |     0.000 |       0.0006 |    0.0006 |
| only_g   | tau_k    | corr_g    |     0.456 |       0.4560 |    0.4560 |
| only_g   | tau_k    | corr_z    |        NA |      -0.0042 |   -0.0042 |
| iid      | tau_h    | E         |    23.870 |      23.8700 |   23.8700 |
| iid      | tau_h    | sd        |     0.147 |       0.1473 |    0.1473 |
| iid      | tau_h    | rho       |    -0.068 |      -0.0674 |   -0.0674 |
| iid      | tau_h    | corr_g    |     0.099 |       0.0968 |    0.0968 |
| iid      | tau_h    | corr_z    |     0.954 |       0.9536 |    0.9536 |
| iid      | theta_e  | E         |     0.000 |       0.0000 |    0.0000 |
| iid      | theta_e  | sd        |     0.000 |       0.0000 |    0.0000 |
| iid      | theta_e  | rho       |        NA |       0.8953 |    0.8953 |
| iid      | theta_e  | corr_g    |        NA |       0.1074 |    0.1074 |
| iid      | theta_e  | corr_z    |        NA |      -0.4241 |   -0.4241 |
| iid      | tau_k    | E         |     0.000 |      -0.0136 |   -0.0136 |
| iid      | tau_k    | sd        |    10.818 |      10.8983 |   10.8983 |
| iid      | tau_k    | rho       |     0.000 |      -0.0008 |   -0.0008 |
| iid      | tau_k    | corr_g    |     0.913 |       0.9077 |    0.9077 |
| iid      | tau_k    | corr_z    |    -0.409 |      -0.4235 |   -0.4235 |

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
