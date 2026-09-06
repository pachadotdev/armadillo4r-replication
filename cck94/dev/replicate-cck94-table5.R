# ==========================================================================
# Ramsey Optimal Taxation: Solver Comparison vs BW Table 5 Reference
# ==========================================================================
# Compares package solvers against the published BW Table 5 values
# stored in bw5 package data:
#   - 1st-Order LQ (cck94_stats)
#   - 2nd-Order Perturbation (cck94_stats_2nd)
# ==========================================================================

pkgpath <- "~/Documents/cck94/cck94"

devtools::clean_dll(pkgpath)
cpp4r::register(pkgpath)
devtools::document(pkgpath)
devtools::install(pkgpath)

library(cck94)

# ==========================================================================
# Setup
# ==========================================================================

# Grid and simulation parameters
N_K     <- 50L
N_Z     <- 5L
N_G     <- 5L
T_TOTAL <- 200000L
T_BURN  <- 20000L
SEED    <- 42L

# Nonlinear solver parameters
MAX_ITER   <- 500L
TOL        <- 1e-6
POLY_DEG   <- 3L

# Scenarios (matching bw5 data)
scenarios <- list(
  baseline = list(),
  high_ra  = list(psi = -8.0, tau_h = 0.2069),
  only_z   = list(sigma_g = 1e-8, rho_g = 0.0),
  only_g   = list(sigma_z = 1e-8, rho_z = 0.0),
  iid      = list(rho_z = 0.0, rho_g = 0.0)
)

# Variable and statistic names (matching bw5 structure)
var_names   <- c("tau_h", "theta_e", "tau_k")
bw5_vars    <- c("tax_rate_labour_income", "ex_ante_tax_rate_capital", 
                 "ex_post_tax_rate_capital")
stat_labels <- c("E", "sd", "rho", "corr_g", "corr_z")

# ==========================================================================
# Reference Data: BW Table 5 (from package data)
# ==========================================================================

data(bw5, package = "cck94")

# ==========================================================================
# Run All Solvers
# ==========================================================================

results <- list()

# --- 1st-Order LQ ---
cat("\n=== Running 1st-Order LQ ===\n")
results[["1st-Order LQ"]] <- lapply(names(scenarios), function(sc_name) {
  cat(sprintf("  %s...\n", sc_name))
  cck94_stats(scenarios[[sc_name]], n_k = N_K, T_total = T_TOTAL, 
             T_burn = T_BURN, seed = SEED)
})
names(results[["1st-Order LQ"]]) <- names(scenarios)

# --- 2nd-Order Perturbation ---
cat("\n=== Running 2nd-Order Perturbation ===\n")
results[["2nd-Order"]] <- lapply(names(scenarios), function(sc_name) {
  cat(sprintf("  %s...\n", sc_name))
  cck94_stats_2nd(scenarios[[sc_name]], n_k = N_K, T_total = T_TOTAL, 
                 T_burn = T_BURN, seed = SEED)
})
names(results[["2nd-Order"]]) <- names(scenarios)


# ==========================================================================
# Build Comparison Table
# ==========================================================================

solver_names <- names(results)

build_comparison <- function() {
  rows <- list()
  
  for (sc_name in names(scenarios)) {
    for (var_idx in seq_along(var_names)) {
      var_name <- var_names[var_idx]
      bw5_var  <- bw5_vars[var_idx]
      
      for (stat_idx in seq_along(stat_labels)) {
        stat_name <- stat_labels[stat_idx]
        
        # Reference value from bw5
        ref_val <- bw5[[bw5_var]][stat_idx, sc_name]
        
        # Create row with reference + all solvers
        row <- data.frame(
          Scenario  = sc_name,
          Variable  = var_name,
          Statistic = stat_name,
          Published = round(ref_val, 4),
          stringsAsFactors = FALSE
        )
        
        # Add each solver's result
        for (solver in solver_names) {
          solver_res <- results[[solver]][[sc_name]]
          # Nonlinear solvers return list with $stats, perturbation returns matrix
          if (is.list(solver_res)) {
            solver_val <- solver_res$stats[stat_idx, var_idx]
          } else {
            solver_val <- solver_res[stat_idx, var_idx]
          }
          row[[solver]] <- round(solver_val, 4)
        }
        
        rows[[length(rows) + 1L]] <- row
      }
    }
  }
  
  do.call(rbind, rows)
}

# ==========================================================================
# Output
# ==========================================================================

full_table <- build_comparison()

# CSV
write.csv(full_table, "dev/replicate-cck94-table5.csv", row.names = FALSE)

# Text
sink("dev/replicate-cck94-table5.txt")

cat("==========================================================================\n")
cat("Ramsey Optimal Taxation: Solver Comparison vs BW Table 5 Reference\n")
cat("==========================================================================\n")
cat("Solvers:\n")
cat("  Published    : BW Table 5\n")
cat("  1st-Order LQ : 1st-order LQ approximation\n")
cat("  2nd-Order    : 2nd-order perturbation (SGU 2004)\n")
cat("--------------------------------------------------------------------------\n")
cat(sprintf("Grid: n_k=%d, n_z=%d, n_g=%d\n", N_K, N_Z, N_G))
cat(sprintf("Simulation: T=%d, burn-in=%d, seed=%d\n", T_TOTAL, T_BURN, SEED))
cat(sprintf("Nonlinear: max_iter=%d, tol=%.0e, poly_degree=%d\n", MAX_ITER, TOL, POLY_DEG))
cat("==========================================================================\n\n")

print(full_table, row.names = FALSE)

sink()

cat("\nResults written to:\n")
cat("  dev/replicate-cck94-table5.csv\n")
cat("  dev/replicate-cck94-table5.txt\n")
