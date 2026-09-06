#ifndef TVP_EXPORTS_H
#define TVP_EXPORTS_H

/* ==========================================================================
   TVP-VAR Exported R Functions
   Following Baumeister and Peersman (2013)
   ========================================================================== */

#include "tvp_estimate.h"
#include "tvp_figures.h"
#include "tvp_irf.h"

// ============================================================================
// EXPORTED R FUNCTIONS
// ============================================================================

/* roxygen
@title TVP-VAR Estimation with Stochastic Volatility (Baumeister-Peersman)
@description Estimates a Time-Varying Parameter VAR with stochastic volatility
  following Baumeister and Peersman (2013) / Primiceri (2005). Uses MCMC
  (Gibbs sampling with Metropolis-Hastings steps) for estimation.
@param y Time series matrix (T x N). For benchmark specification, use log
  differences of oil production, real oil price, and world industrial
production.
@param L Lag order (default 4 for quarterly data).
@param training_years Years for training sample (25 for benchmark, 5 for Kilian
spec).
@param n_burn Number of burn-in iterations (default 50000).
@param n_draws Number of draws to keep from Gibbs sampler (default 500).
@param thinning Thinning interval - save every D-th draw (default 10).
@param stability_constraint Whether to impose VAR stability (0 = no, 1 = yes).
@param lambda_param Time variation parameter. Use 0.0001 for "Small"
(Primiceri), 0.01 for "Large" (most variation), or 0.00035 for Cogley-Sargent.
@param first_diff Whether data is in first differences (1 = yes for benchmark).
@return A list containing:
  \describe{
    \item{SD}{List of state draws (time-varying VAR coefficients)}
    \item{QD}{List of state innovation covariance matrix draws}
    \item{VD}{Matrix of volatility innovation standard deviations}
    \item{OM}{List of stochastic volatility draws}
    \item{AA}{List of off-diagonal A(t) element draws}
    \item{S1}{Scalar covariance for first block of A innovations}
    \item{S2}{List of covariance matrices for second block of A innovations}
    \item{n_saved}{Number of draws saved}
    \item{T_eff}{Effective sample size}
    \item{N}{Number of variables}
    \item{L}{Lag order}
  }
@export
*/
[[cpp4r::register]] list tvp_var_estimate(const doubles_matrix<> &y, int L,
                                          int training_years, int n_burn,
                                          int n_draws, int thinning,
                                          int stability_constraint,
                                          double lambda_param, int first_diff) {

  // Input validation
  if (L <= 0) {
    stop("L (lag order) must be positive");
  }
  if (training_years <= 0) {
    stop("training_years must be positive");
  }
  if (n_burn < 0 || n_draws <= 0 || thinning <= 0) {
    stop("Invalid MCMC parameters: check n_burn, n_draws, thinning");
  }

  Mat<double> Y = as_Mat(y);

  // Check for NaN/Inf
  if (!Y.is_finite()) {
    stop("Input data contains NaN or Inf values.");
  }

  // Run estimation
  TVPVARResult res = tvp_var_estimate_cpp(Y, L, training_years, n_burn, n_draws,
                                          thinning, stability_constraint == 1,
                                          lambda_param, first_diff == 1, true);

  // Convert results to R list
  writable::list result;

  int n_saved = res.n_saved;

  writable::list SD_list(n_saved);
  writable::list QD_list(n_saved);
  writable::list OM_list(n_saved);
  writable::list AA_list(n_saved);
  writable::list S2_list(n_saved);

  for (int i = 0; i < n_saved; ++i) {
    SD_list[i] = as_doubles_matrix(res.SD.slice(i));
    QD_list[i] = as_doubles_matrix(res.QD.slice(i));
    OM_list[i] = as_doubles_matrix(res.OM.slice(i));
    AA_list[i] = as_doubles_matrix(res.AA.slice(i));
    S2_list[i] = as_doubles_matrix(res.S2.slice(i));
  }

  result.push_back({"SD"_nm = SD_list});
  result.push_back({"QD"_nm = QD_list});
  result.push_back({"VD"_nm = as_doubles_matrix(res.VD)});
  result.push_back({"OM"_nm = OM_list});
  result.push_back({"AA"_nm = AA_list});
  result.push_back({"S1"_nm = as_doubles_matrix(res.S1)});
  result.push_back({"S2"_nm = S2_list});
  result.push_back({"n_saved"_nm = n_saved});
  result.push_back({"T_eff"_nm = res.T_eff});
  result.push_back({"N"_nm = res.N});
  result.push_back({"L"_nm = res.L});

  return result;
}

/* roxygen
@title Compute TVP-VAR Impulse Responses with Sign Restrictions
@description Computes generalized impulse response functions for a TVP-VAR
  with sign restrictions, following Baumeister and Peersman (2013).
  Identifies three structural shocks: oil supply, oil-specific demand, and
  global demand (aggregate activity) shocks.
@param tvp_result Output from tvp_var_estimate.
@param y Original data matrix (T x N).
@param L Lag order (must match tvp_var_estimate).
@param training_years Years for training sample (must match tvp_var_estimate).
@param irf_horizon IRF horizon in periods (default 21 = 5 years + 1).
@param n_reps Number of sign-restriction draws per time period (default 500).
@param h_restrict Number of periods sign restrictions must hold (default 1).
@param first_diff Whether data is in first differences (1 = yes).
@return A list containing:
  \describe{
    \item{IRF_supply}{Array of supply shock IRFs (HOR x N x T)}
    \item{IRF_demand}{Array of oil-specific demand shock IRFs}
    \item{IRF_global}{Array of global demand shock IRFs}
    \item{slope_S_D}{Supply elasticities over time}
    \item{slope_S_A}{Supply elasticities (global shock identification)}
    \item{slope_D_D}{Demand elasticities over time}
    \item{irf_horizon}{IRF horizon used}
    \item{T_eff}{Effective sample size}
  }
@export
*/
[[cpp4r::register]] list tvp_var_irfs(const list &tvp_result,
                                      const doubles_matrix<> &y, int L,
                                      int training_years, int irf_horizon,
                                      int n_reps, int h_restrict,
                                      int first_diff) {

  Mat<double> Y = as_Mat(y);

  // Extract TVP result
  TVPVARResult tvp_res;

  list SD_list = tvp_result["SD"];
  list QD_list = tvp_result["QD"];
  list OM_list = tvp_result["OM"];
  list AA_list = tvp_result["AA"];
  list S2_list = tvp_result["S2"];

  integers n_saved_vec = tvp_result["n_saved"];
  integers T_eff_vec = tvp_result["T_eff"];
  integers N_vec = tvp_result["N"];
  integers L_vec = tvp_result["L"];

  int n_saved = n_saved_vec[0];
  int T_eff = T_eff_vec[0];
  tvp_res.N = N_vec[0];
  tvp_res.L = L_vec[0];
  tvp_res.T_eff = T_eff;
  tvp_res.n_saved = n_saved;

  int state_dim = tvp_res.N * (1 + tvp_res.N * tvp_res.L);
  int n_aa = tvp_res.N * (tvp_res.N - 1) / 2;

  // Reconstruct cubes
  tvp_res.SD.set_size(state_dim, T_eff, n_saved);
  tvp_res.QD.set_size(state_dim, state_dim, n_saved);
  tvp_res.OM.set_size(T_eff, tvp_res.N, n_saved);
  tvp_res.AA.set_size(n_aa, T_eff + 1, n_saved);
  tvp_res.S2.set_size(n_aa - 1, n_aa - 1, n_saved);

  for (int i = 0; i < n_saved; ++i) {
    doubles_matrix<> SD_i = SD_list[i];
    doubles_matrix<> QD_i = QD_list[i];
    doubles_matrix<> OM_i = OM_list[i];
    doubles_matrix<> AA_i = AA_list[i];
    doubles_matrix<> S2_i = S2_list[i];

    tvp_res.SD.slice(i) = as_Mat(SD_i);
    tvp_res.QD.slice(i) = as_Mat(QD_i);
    tvp_res.OM.slice(i) = as_Mat(OM_i);
    tvp_res.AA.slice(i) = as_Mat(AA_i);
    tvp_res.S2.slice(i) = as_Mat(S2_i);
  }

  doubles_matrix<> VD_mat = tvp_result["VD"];
  doubles_matrix<> S1_mat = tvp_result["S1"];
  tvp_res.VD = as_Mat(VD_mat);
  tvp_res.S1 = as_Mat(S1_mat);

  // Compute IRFs
  TVPIRFResult irf_res =
      tvp_var_irfs_cpp(tvp_res, Y, L, training_years, irf_horizon, n_reps,
                       h_restrict, first_diff, true);

  // Convert to R list
  writable::list result;

  // Convert cubes to list of matrices
  int n_times = irf_res.n_time_points;

  writable::list IRF_supply_list(n_times);
  writable::list IRF_demand_list(n_times);
  writable::list IRF_global_list(n_times);

  for (int i = 0; i < n_times; ++i) {
    IRF_supply_list[i] = as_doubles_matrix(irf_res.IRF_supply.slice(i));
    IRF_demand_list[i] = as_doubles_matrix(irf_res.IRF_demand.slice(i));
    IRF_global_list[i] = as_doubles_matrix(irf_res.IRF_global.slice(i));
  }

  result.push_back({"IRF_supply"_nm = IRF_supply_list});
  result.push_back({"IRF_demand"_nm = IRF_demand_list});
  result.push_back({"IRF_global"_nm = IRF_global_list});
  result.push_back({"slope_S_D"_nm = as_doubles_matrix(irf_res.slope_S_D)});
  result.push_back({"slope_S_A"_nm = as_doubles_matrix(irf_res.slope_S_A)});
  result.push_back({"slope_D_D"_nm = as_doubles_matrix(irf_res.slope_D_D)});
  result.push_back({"irf_horizon"_nm = irf_res.irf_horizon});
  result.push_back({"T_eff"_nm = irf_res.T_eff});
  result.push_back({"n_time_points"_nm = n_times});

  return result;
}

/* roxygen
@title Compute Time-Varying Elasticities
@description Computes time-varying oil supply and demand elasticities
  following Baumeister and Peersman (2013), Figure 2.
@param tvp_result Output from tvp_var_estimate.
@param y Original data matrix (T x N).
@param L Lag order.
@param training_years Years for training sample.
@param n_reps Number of draws per time period (default 500).
@param h_restrict Number of periods sign restrictions must hold (default 1).
@param first_diff Whether data is in first differences (1 = yes).
@param upper_supply Upper bound on supply elasticity (default 0.6).
@param lower_demand Lower bound on demand elasticity (default -0.8).
@return A list containing elasticity distributions over time.
@export
*/
[[cpp4r::register]] list
tvp_var_elasticities(const list &tvp_result, const doubles_matrix<> &y, int L,
                     int training_years, int n_reps, int h_restrict,
                     int first_diff, double upper_supply, double lower_demand) {

  // Compute IRFs and extract elasticities
  list irf_result = tvp_var_irfs(tvp_result, y, L, training_years, 21, n_reps,
                                 h_restrict, first_diff);

  writable::list result;
  result.push_back({"slope_S_D"_nm = irf_result["slope_S_D"]});
  result.push_back({"slope_S_A"_nm = irf_result["slope_S_A"]});
  result.push_back({"slope_D_D"_nm = irf_result["slope_D_D"]});
  result.push_back({"upper_supply"_nm = upper_supply});
  result.push_back({"lower_demand"_nm = lower_demand});

  return result;
}

// ============================================================================
// HELPER: Extract TVP result from R list
// ============================================================================

inline TVPVARResult extract_tvp_result(const list &tvp_result) {
  TVPVARResult tvp_res;

  list SD_list = tvp_result["SD"];
  list QD_list = tvp_result["QD"];
  list OM_list = tvp_result["OM"];
  list AA_list = tvp_result["AA"];
  list S2_list = tvp_result["S2"];

  integers n_saved_vec = tvp_result["n_saved"];
  integers T_eff_vec = tvp_result["T_eff"];
  integers N_vec = tvp_result["N"];
  integers L_vec = tvp_result["L"];

  int n_saved = n_saved_vec[0];
  int T_eff = T_eff_vec[0];
  tvp_res.N = N_vec[0];
  tvp_res.L = L_vec[0];
  tvp_res.T_eff = T_eff;
  tvp_res.n_saved = n_saved;

  int state_dim = tvp_res.N * (1 + tvp_res.N * tvp_res.L);
  int n_aa = tvp_res.N * (tvp_res.N - 1) / 2;

  tvp_res.SD.set_size(state_dim, T_eff, n_saved);
  tvp_res.QD.set_size(state_dim, state_dim, n_saved);
  tvp_res.OM.set_size(T_eff, tvp_res.N, n_saved);
  tvp_res.AA.set_size(n_aa, T_eff + 1, n_saved);
  tvp_res.S2.set_size(n_aa - 1, n_aa - 1, n_saved);

  for (int i = 0; i < n_saved; ++i) {
    doubles_matrix<> SD_i = SD_list[i];
    doubles_matrix<> QD_i = QD_list[i];
    doubles_matrix<> OM_i = OM_list[i];
    doubles_matrix<> AA_i = AA_list[i];
    doubles_matrix<> S2_i = S2_list[i];

    tvp_res.SD.slice(i) = as_Mat(SD_i);
    tvp_res.QD.slice(i) = as_Mat(QD_i);
    tvp_res.OM.slice(i) = as_Mat(OM_i);
    tvp_res.AA.slice(i) = as_Mat(AA_i);
    tvp_res.S2.slice(i) = as_Mat(S2_i);
  }

  doubles_matrix<> VD_mat = tvp_result["VD"];
  doubles_matrix<> S1_mat = tvp_result["S1"];
  tvp_res.VD = as_Mat(VD_mat);
  tvp_res.S1 = as_Mat(S1_mat);

  return tvp_res;
}

// ============================================================================
// FIGURE 2: TIME-VARYING IMPACT RESPONSES
// ============================================================================

/* roxygen
@title Compute Figure 2: Time-Varying Impact IRFs
@description Computes time-varying impact impulse responses for oil supply,
  oil-specific demand, and aggregate demand shocks, with elasticity bounds.
  Following Baumeister and Peersman (2013), Figure 2.
@param tvp_result Output from tvp_var_estimate.
@param y Original data matrix (T x N).
@param L Lag order.
@param n_reps Number of sign-restriction draws per time period (default 500).
@param h_restrict Number of periods sign restrictions must hold (default 1).
@param first_diff Whether data is in first differences (1 = yes).
@param upper_supply Upper bound on supply elasticity (default 0.6).
@param lower_demand Lower bound on demand elasticity (default -0.8).
@param start_period Starting period for output (default 4 for 1974Q1).
@return A list containing matrices for each shock-variable combination.
  Each matrix is n_time x 7 with columns: median, p025, p975, p16, p84, max,
min.
@export
*/
[[cpp4r::register]] list
tvp_var_figure2(const list &tvp_result, const doubles_matrix<> &y, int L,
                int n_reps, int h_restrict, int first_diff, double upper_supply,
                double lower_demand, int start_period) {

  Mat<double> Y = as_Mat(y);
  TVPVARResult tvp_res = extract_tvp_result(tvp_result);

  Figure2Result fig2 = compute_figure2(
      tvp_res.SD, tvp_res.AA, tvp_res.QD, tvp_res.S1, tvp_res.S2, tvp_res.VD,
      tvp_res.OM, Y, L, tvp_res.N, tvp_res.T_eff, tvp_res.n_saved, n_reps,
      h_restrict, first_diff, upper_supply, lower_demand, start_period);

  writable::list result;
  result.push_back({"s_prod"_nm = as_doubles_matrix(fig2.s_prod)});
  result.push_back({"s_price"_nm = as_doubles_matrix(fig2.s_price)});
  result.push_back({"d_prod"_nm = as_doubles_matrix(fig2.d_prod)});
  result.push_back({"d_price"_nm = as_doubles_matrix(fig2.d_price)});
  result.push_back({"a_prod"_nm = as_doubles_matrix(fig2.a_prod)});
  result.push_back({"a_price"_nm = as_doubles_matrix(fig2.a_price)});
  result.push_back({"n_time"_nm = fig2.n_time});

  return result;
}

// ============================================================================
// FIGURE 3: PERSISTENCE OF REAL OIL PRICE INFLATION
// ============================================================================

/* roxygen
@title Compute Figure 3: Persistence of Oil Price Inflation
@description Computes time-varying persistence of real oil price inflation
  using spectral density at frequency zero. Following Baumeister and
  Peersman (2013), Figure 3.
@param tvp_result Output from tvp_var_estimate.
@param n_freq Not used (for compatibility).
@param price_var Index of price variable (default 2, 1-indexed for R).
@return A list containing persistence measures over time with percentiles.
@export
*/
[[cpp4r::register]] list tvp_var_figure3(const list &tvp_result, int n_freq,
                                         int price_var) {

  TVPVARResult tvp_res = extract_tvp_result(tvp_result);

  // Convert 1-indexed R to 0-indexed C++
  int price_var_cpp = price_var - 1;

  Figure3Result fig3 =
      compute_figure3(tvp_res.SD, tvp_res.AA, tvp_res.OM, tvp_res.N,
                      tvp_res.L, tvp_res.T_eff, tvp_res.n_saved,
                      price_var_cpp);

  writable::list result;
  result.push_back(
      {"persistence_median"_nm = as_doubles(fig3.persistence_median)});
  result.push_back(
      {"persistence_lower"_nm = as_doubles(fig3.persistence_lower)});
  result.push_back(
      {"persistence_upper"_nm = as_doubles(fig3.persistence_upper)});
  result.push_back({"persistence_16"_nm = as_doubles(fig3.persistence_16)});
  result.push_back({"persistence_84"_nm = as_doubles(fig3.persistence_84)});
  result.push_back({"T_eff"_nm = fig3.T_eff});

  return result;
}

// ============================================================================
// FIGURE 4: TIME-VARYING ELASTICITIES
// ============================================================================

/* roxygen
@title Compute Figure 4: Time-Varying Elasticities
@description Computes time-varying oil supply and demand elasticities
  with elasticity bounds. Following Baumeister and Peersman (2013), Figure 4.
@param tvp_result Output from tvp_var_estimate.
@param y Original data matrix (T x N).
@param L Lag order.
@param n_reps Number of sign-restriction draws per time period (default 500).
@param h_restrict Number of periods sign restrictions must hold (default 1).
@param first_diff Whether data is in first differences (1 = yes).
@param upper_supply Upper bound on supply elasticity (default 0.6).
@param lower_demand Lower bound on demand elasticity (default -0.8).
@param start_period Starting period for output (default 4 for 1974Q1).
@return A list containing elasticity matrices over time.
  Each matrix is n_time x 7 with columns: median, p025, p975, p16, p84, max, min.
@export
*/
[[cpp4r::register]] list
tvp_var_figure4(const list &tvp_result, const doubles_matrix<> &y, int L,
                int n_reps, int h_restrict, int first_diff, double upper_supply,
                double lower_demand, int start_period) {

  Mat<double> Y = as_Mat(y);
  TVPVARResult tvp_res = extract_tvp_result(tvp_result);

  Figure4Result fig4 = compute_figure4(
      tvp_res.SD, tvp_res.AA, tvp_res.QD, tvp_res.S1, tvp_res.S2, tvp_res.VD,
      tvp_res.OM, Y, L, tvp_res.N, tvp_res.T_eff, tvp_res.n_saved, n_reps,
      h_restrict, first_diff, upper_supply, lower_demand, start_period);

  writable::list result;
  result.push_back(
      {"elas_supply_agg"_nm = as_doubles_matrix(fig4.elas_supply_agg)});
  result.push_back(
      {"elas_supply_dem"_nm = as_doubles_matrix(fig4.elas_supply_dem)});
  result.push_back({"elas_demand"_nm = as_doubles_matrix(fig4.elas_demand)});
  result.push_back({"n_time"_nm = fig4.n_time});

  return result;
}

#endif // TVP_EXPORTS_H
