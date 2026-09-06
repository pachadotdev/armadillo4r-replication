#ifndef TVP_FIGURES_H
#define TVP_FIGURES_H

/* ==========================================================================
   TVP-VAR Figure Computation Functions
   Following Baumeister and Peersman (2013)

   Ported from MATLAB: figure2_perc2.m, figure3.m, figures4and6_perc2.m
   ========================================================================== */

#include "tvp_girfs.h"

// ============================================================================
// CHRISTIANO-FITZGERALD BAND-PASS FILTER
// Ported from MATLAB cffilter.m
// ============================================================================

// CF filter: returns the cycle component (yC)
// freq_type: 'A' for annual, 'Q' for quarterly, 'M' for monthly
// pl, pu: lower and upper bounds of frequency band (in years for annual data)
inline vec cffilter_cycle(const vec &y, char freq_type, double pl, double pu) {
  int T = y.n_elem;
  
  // Need at least 4 observations for the filter to work
  if (T < 4) {
    return y;  // Return original if too short
  }
  
  vec yC(T, fill::zeros);
  
  // Convert pl, pu to radians
  double A = (2.0 * datum::pi) / pu;
  double B = (2.0 * datum::pi) / pl;
  
  // Compute business-cycle component
  double B0 = (B - A) / datum::pi;
  
  // Compute the Bj's
  vec Bj(T);
  for (int j = 1; j <= T; ++j) {
    Bj(j-1) = (sin(B * j) - sin(A * j)) / (j * datum::pi);
  }
  
  // t = 0 (first observation)
  {
    double B_last = -0.5 * B0 - sum(Bj.subvec(0, T-3));
    vec W(T, fill::zeros);
    W(0) = 0.5 * B0;
    W.subvec(1, T-2) = Bj.subvec(0, T-3);
    W(T-1) = B_last;
    yC(0) = dot(W, y);
  }
  
  // t = 1 to T-2 (middle observations)
  for (int t = 1; t < T-1; ++t) {
    vec W(T, fill::zeros);
    
    // Indices for subvec operations - ensure they're valid
    int idx_end_T = T - t - 2;
    if (idx_end_T < 0) idx_end_T = 0;
    
    double B_last = -(0.5 * B0 + sum(Bj.subvec(0, idx_end_T)));
    double B_first = -B_last - B0;
    if (t >= 2) {
      B_first -= sum(Bj.subvec(0, t-2));
    }
    B_first -= sum(Bj.subvec(0, idx_end_T));
    
    // Build weight vector W
    int idx = 0;
    W(idx++) = B_first;
    
    // flipud(Bj(1:t-2)) - elements from index 0 to t-2, reversed
    if (t >= 2) {
      for (int j = t-2; j >= 0; --j) {
        if (idx < T) W(idx++) = Bj(j);
      }
    }
    
    if (idx < T) W(idx++) = B0;
    
    // Bj(1:T-t-1) - elements from index 0 to T-t-2
    for (int j = 0; j <= idx_end_T && idx < T; ++j) {
      W(idx++) = Bj(j);
    }
    
    if (idx < T) W(idx++) = B_last;
    
    yC(t) = dot(W, y);
  }
  
  // t = T-1 (last observation)
  {
    double B_first = -0.5 * B0 - sum(Bj.subvec(0, T-3));
    vec W(T, fill::zeros);
    int idx = 0;
    W(idx++) = B_first;
    
    // flipud(Bj(1:t-2)) where t = T (MATLAB indexing)
    for (int j = T-3; j >= 0 && idx < T; --j) {
      W(idx++) = Bj(j);
    }
    
    if (idx < T) W(idx++) = 0.5 * B0;
    
    yC(T-1) = dot(W, y);
  }
  
  return yC;
}

// Apply CF filter to remove high-frequency noise (keep trend + cycle)
// This matches MATLAB: yN+yC from cffilter(x,'A',4,12)
inline vec cffilter_smooth(const vec &y, double pl = 4.0, double pu = 12.0) {
  int T = y.n_elem;
  
  // Need at least 4 observations for the filter to work
  if (T < 4) {
    return y;  // Return original if too short
  }
  
  vec yC = cffilter_cycle(y, 'A', pl, pu);
  
  // Compute irregular component
  double A_irr = (2.0 * datum::pi) / pu;
  double B_irr = datum::pi;  // For irregular
  
  vec yI(T, fill::zeros);
  
  double B0_irr = (B_irr - A_irr) / datum::pi;
  
  vec Bj_irr(T);
  for (int j = 1; j <= T; ++j) {
    Bj_irr(j-1) = (sin(datum::pi * j) - sin(A_irr * j)) / (j * datum::pi);
  }
  
  // t = 0
  {
    double B_last = -0.5 * B0_irr - sum(Bj_irr.subvec(0, T-3));
    vec W(T, fill::zeros);
    W(0) = 0.5 * B0_irr;
    W.subvec(1, T-2) = Bj_irr.subvec(0, T-3);
    W(T-1) = B_last;
    yI(0) = dot(W, y);
  }
  
  for (int t = 1; t < T-1; ++t) {
    vec W(T, fill::zeros);
    
    int idx_end_T = T - t - 2;
    if (idx_end_T < 0) idx_end_T = 0;
    
    double B_last = -(0.5 * B0_irr + sum(Bj_irr.subvec(0, idx_end_T)));
    double B_first = -B_last - B0_irr;
    if (t >= 2) B_first -= sum(Bj_irr.subvec(0, t-2));
    B_first -= sum(Bj_irr.subvec(0, idx_end_T));
    
    int idx = 0;
    W(idx++) = B_first;
    if (t >= 2) {
      for (int j = t-2; j >= 0 && idx < T; --j) W(idx++) = Bj_irr(j);
    }
    if (idx < T) W(idx++) = B0_irr;
    for (int j = 0; j <= idx_end_T && idx < T; ++j) W(idx++) = Bj_irr(j);
    if (idx < T) W(idx++) = B_last;
    
    yI(t) = dot(W, y);
  }
  
  // t = T-1
  {
    double B_first = -0.5 * B0_irr - sum(Bj_irr.subvec(0, T-3));
    vec W(T, fill::zeros);
    int idx = 0;
    W(idx++) = B_first;
    for (int j = T-3; j >= 0 && idx < T; --j) W(idx++) = Bj_irr(j);
    if (idx < T) W(idx++) = 0.5 * B0_irr;
    
    yI(T-1) = dot(W, y);
  }
  
  // yN = y - yC - yI (trend component)
  vec yN = y - yC - yI;
  
  // Return yN + yC (removes irregular, keeps trend + cycle)
  return yN + yC;
}

// Apply CF filter to columns 1-6 (0-indexed) of a matrix, skipping column 0 (median)
// This matches MATLAB figure2_perc2.m lines 175-188
inline void apply_cffilter_to_bands(Mat<double> &M) {
  // Need enough time periods for the filter
  if (M.n_rows < 4) {
    return;  // Skip filtering if too few observations
  }
  
  for (int col = 1; col <= 6; ++col) {
    vec v = M.col(col);
    M.col(col) = cffilter_smooth(v, 4.0, 12.0);
  }
}

// ============================================================================
// FIGURE 2: TIME-VARYING IMPACT RESPONSES
// ============================================================================

// Structure to hold Figure 2 results
struct Figure2Result {
  // Each matrix is n_time x 7 (median, p025, p975, p16, p84, max, min)
  Mat<double> s_prod;  // Supply shock -> oil production
  Mat<double> s_price; // Supply shock -> oil price
  Mat<double> d_prod;  // Demand shock -> oil production
  Mat<double> d_price; // Demand shock -> oil price
  Mat<double> a_prod;  // Aggregate shock -> oil production
  Mat<double> a_price; // Aggregate shock -> oil price
  int n_time;
};

// Compute Figure 2 data: time-varying impact IRFs with elasticity bounds
inline Figure2Result compute_figure2(
    const Cube<double> &SD, const Cube<double> &AA, const Cube<double> &QD,
    const Mat<double> &S1, const Cube<double> &S2, const Mat<double> &VD,
    const Cube<double> &OM, const Mat<double> &Y, int L, int N, int T_eff,
    int n_saved, int n_reps, int h_restrict, int first_diff,
    double upper_supply, double lower_demand, int start_period) {

  Figure2Result result;
  
  // Validate start_period
  if (start_period < 1) start_period = 1;
  if (start_period > T_eff) start_period = T_eff;
  
  int n_time = T_eff - start_period + 1;
  result.n_time = n_time;

  // Initialize output matrices (n_time x 7)
  result.s_prod.set_size(n_time, 7);
  result.s_price.set_size(n_time, 7);
  result.d_prod.set_size(n_time, 7);
  result.d_price.set_size(n_time, 7);
  result.a_prod.set_size(n_time, 7);
  result.a_price.set_size(n_time, 7);

  result.s_prod.fill(datum::nan);
  result.s_price.fill(datum::nan);
  result.d_prod.fill(datum::nan);
  result.d_price.fill(datum::nan);
  result.a_prod.fill(datum::nan);
  result.a_price.fill(datum::nan);

  int HOR = 21; // IRF horizon
  int cors = 1; // Normalization period

  // Loop over time periods
  for (int t_idx = 0; t_idx < n_time; ++t_idx) {
    int t = start_period - 1 + t_idx; // 0-indexed time in T_eff
    
    // Bounds check for t
    if (t < 0 || t >= T_eff) continue;

    // Storage for accepted draws at this time period
    std::vector<double> s1_vec, s2_vec, d1_vec, d2_vec, a1_vec, a2_vec;
    std::vector<double> elast_sd_vec, elast_sa_vec, elast_dd_vec;

    // Loop over MCMC draws
    for (int draw = 0; draw < n_saved; ++draw) {
      // Bounds check for draw
      if (draw >= (int)SD.n_slices) continue;
      
      // Bounds check for matrix dimensions
      if (t >= (int)SD.slice(draw).n_cols) continue;
      if (t >= (int)AA.slice(draw).n_cols) continue;
      if (t >= (int)OM.slice(draw).n_rows) continue;
      
      // Get parameters for this draw and time
      vec sd = SD.slice(draw).col(t);
      vec aa = AA.slice(draw).col(t);
      Mat<double> qd = QD.slice(draw);
      double s1 = S1(0, draw);
      Mat<double> s2 = S2.slice(draw);
      vec vd = VD.col(draw);
      vec om = OM.slice(draw).row(t).t();

      // Get lagged Y values for this time
      // YY should be N x L matrix of lagged values
      Mat<double> YY(N, L);
      int T_total = Y.n_rows;
      int TP = (T_total - T_eff) / 4; // Training periods in years
      int data_start = 4 * TP;        // Where effective sample starts

      for (int lag = 0; lag < L; ++lag) {
        int y_idx = data_start + t - lag;
        if (y_idx >= 0 && y_idx < (int)Y.n_rows) {
          YY.col(lag) = Y.row(y_idx).t();
        }
      }

      // Run multiple sign-restriction draws
      for (int rep = 0; rep < n_reps; ++rep) {
        Mat<double> sirf, dirf, aggirf;

        bool success =
            GIRFs_N3(sd, aa, qd, s1, s2, vd, om, YY, HOR, N, L, cors, 1,
                     h_restrict, false, first_diff, sirf, dirf, aggirf);

        if (success) {
          // Compute elasticities
          OilMarketElasticities elast =
              oilmarket_model(sirf.row(0), dirf.row(0), aggirf.row(0));

          // Check elasticity bounds
          if (elast.slope_S_D <= upper_supply &&
              elast.slope_S_A <= upper_supply &&
              elast.slope_D_D >= lower_demand) {

            // Store impact responses (horizon 0)
            s1_vec.push_back(sirf(0, 0));   // qo response to supply
            s2_vec.push_back(sirf(0, 1));   // po response to supply
            d1_vec.push_back(dirf(0, 0));   // qo response to demand
            d2_vec.push_back(dirf(0, 1));   // po response to demand
            a1_vec.push_back(aggirf(0, 0)); // qo response to aggregate
            a2_vec.push_back(aggirf(0, 1)); // po response to aggregate

            elast_sd_vec.push_back(elast.slope_S_D);
            elast_sa_vec.push_back(elast.slope_S_A);
            elast_dd_vec.push_back(elast.slope_D_D);
          }
        }
      }
    }

    // Compute percentiles if we have accepted draws
    if (s1_vec.size() > 0) {
      vec s1_v = conv_to<vec>::from(s1_vec);
      vec s2_v = conv_to<vec>::from(s2_vec);
      vec d1_v = conv_to<vec>::from(d1_vec);
      vec d2_v = conv_to<vec>::from(d2_vec);
      vec a1_v = conv_to<vec>::from(a1_vec);
      vec a2_v = conv_to<vec>::from(a2_vec);

      // Sort for percentile computation
      vec s1_s = sort(s1_v);
      vec s2_s = sort(s2_v);
      vec d1_s = sort(d1_v);
      vec d2_s = sort(d2_v);
      vec a1_s = sort(a1_v);
      vec a2_s = sort(a2_v);

      int n = s1_s.n_elem;

      // Percentile indices
      int i_med = std::max(0, (int)(n * 0.5) - 1);
      int i_025 = std::max(0, (int)(n * 0.025) - 1);
      int i_975 = std::min(n - 1, (int)(n * 0.975));
      int i_16 = std::max(0, (int)(n * 0.16) - 1);
      int i_84 = std::min(n - 1, (int)(n * 0.84));

      // Store: median, p025, p975, p16, p84, max, min
      result.s_prod(t_idx, 0) = s1_s(i_med);
      result.s_prod(t_idx, 1) = s1_s(i_025);
      result.s_prod(t_idx, 2) = s1_s(i_975);
      result.s_prod(t_idx, 3) = s1_s(i_16);
      result.s_prod(t_idx, 4) = s1_s(i_84);
      result.s_prod(t_idx, 5) = s1_s(n - 1); // max
      result.s_prod(t_idx, 6) = s1_s(0);     // min

      result.s_price(t_idx, 0) = s2_s(i_med);
      result.s_price(t_idx, 1) = s2_s(i_025);
      result.s_price(t_idx, 2) = s2_s(i_975);
      result.s_price(t_idx, 3) = s2_s(i_16);
      result.s_price(t_idx, 4) = s2_s(i_84);
      result.s_price(t_idx, 5) = s2_s(n - 1);
      result.s_price(t_idx, 6) = s2_s(0);

      result.d_prod(t_idx, 0) = d1_s(i_med);
      result.d_prod(t_idx, 1) = d1_s(i_025);
      result.d_prod(t_idx, 2) = d1_s(i_975);
      result.d_prod(t_idx, 3) = d1_s(i_16);
      result.d_prod(t_idx, 4) = d1_s(i_84);
      result.d_prod(t_idx, 5) = d1_s(n - 1);
      result.d_prod(t_idx, 6) = d1_s(0);

      result.d_price(t_idx, 0) = d2_s(i_med);
      result.d_price(t_idx, 1) = d2_s(i_025);
      result.d_price(t_idx, 2) = d2_s(i_975);
      result.d_price(t_idx, 3) = d2_s(i_16);
      result.d_price(t_idx, 4) = d2_s(i_84);
      result.d_price(t_idx, 5) = d2_s(n - 1);
      result.d_price(t_idx, 6) = d2_s(0);

      result.a_prod(t_idx, 0) = a1_s(i_med);
      result.a_prod(t_idx, 1) = a1_s(i_025);
      result.a_prod(t_idx, 2) = a1_s(i_975);
      result.a_prod(t_idx, 3) = a1_s(i_16);
      result.a_prod(t_idx, 4) = a1_s(i_84);
      result.a_prod(t_idx, 5) = a1_s(n - 1);
      result.a_prod(t_idx, 6) = a1_s(0);

      result.a_price(t_idx, 0) = a2_s(i_med);
      result.a_price(t_idx, 1) = a2_s(i_025);
      result.a_price(t_idx, 2) = a2_s(i_975);
      result.a_price(t_idx, 3) = a2_s(i_16);
      result.a_price(t_idx, 4) = a2_s(i_84);
      result.a_price(t_idx, 5) = a2_s(n - 1);
      result.a_price(t_idx, 6) = a2_s(0);
    }
  }

  // NOTE: CF filter disabled for now - causes issues with NaN handling
  // The MATLAB code applies cffilter to smooth bands (lines 175-188 in figure2_perc2.m)
  // but this is optional smoothing that can be added later if needed
  // apply_cffilter_to_bands(result.s_prod);
  // apply_cffilter_to_bands(result.s_price);
  // apply_cffilter_to_bands(result.d_prod);
  // apply_cffilter_to_bands(result.d_price);
  // apply_cffilter_to_bands(result.a_prod);
  // apply_cffilter_to_bands(result.a_price);

  return result;
}

// ============================================================================
// FIGURE 3: PERSISTENCE OF REAL OIL PRICE INFLATION
// ============================================================================

struct Figure3Result {
  vec persistence_median;
  vec persistence_lower;
  vec persistence_upper;
  vec persistence_16;
  vec persistence_84;
  int T_eff;
};

// Compute spectral density at zero for AR representation
// This measures persistence: high value = high persistence
inline double compute_persistence(const rowvec &ar_coeffs) {
  // Sum of AR coefficients
  double sum_ar = sum(ar_coeffs);

  // Spectral density at zero (normalized)
  // For AR(p): S(0) = sigma^2 / (1 - sum(phi_i))^2
  double denom = 1.0 - sum_ar;
  if (std::abs(denom) < 1e-10) {
    return 1.0; // Unit root case
  }

  return 1.0 / (denom * denom);
}

inline Figure3Result compute_figure3(const Cube<double> &SD, const Cube<double> &AA,
                                     const Cube<double> &OM, int N, int L,
                                     int T_eff, int n_saved, int price_var) {
  Figure3Result result;
  result.T_eff = T_eff;

  result.persistence_median.set_size(T_eff);
  result.persistence_lower.set_size(T_eff);
  result.persistence_upper.set_size(T_eff);
  result.persistence_16.set_size(T_eff);
  result.persistence_84.set_size(T_eff);

  // int K = 1 + N * L; // Number of coefficients per equation


  // Frequency grid used in MATLAB: Nw = 50 (0..pi)
  const int Nw = 50;
  vec w = linspace<vec>(0.0, datum::pi, Nw);

  for (int t = 0; t < T_eff; ++t) {
    std::vector<double> pers_vec;

    for (int draw = 0; draw < n_saved; ++draw) {
      // Bounds / size checks
      if (draw >= (int)SD.n_slices) continue;
      if (draw >= (int)AA.n_slices) continue;
      if (draw >= (int)OM.n_slices) continue;
      if (t >= (int)SD.slice(draw).n_cols) continue;
      if (t >= (int)AA.slice(draw).n_cols) continue; // AA has T_eff+1 cols
      if (t >= (int)OM.slice(draw).n_rows) continue;

      // Get state vector for this draw/time and reconstruct A matrices
      vec SI = SD.slice(draw).col(t);

      // Build matrix 'a' of size N x (1+N*L) as in MATLAB: reshape(SI',1+N*L,N)'
      Mat<double> a(N, 1 + N * L);
      for (int eq = 0; eq < N; ++eq) {
        int start = eq * (1 + N * L);
        a.row(eq) = SI.subvec(start, start + (1 + N * L) - 1).t();
      }

      // Remove constant column -> N x (N*L)
      Mat<double> a_sub = a.cols(1, a.n_cols - 1);

      // Build A(:,:,jj)
      std::vector<Mat<double>> A_mat(L, Mat<double>(N, N, fill::zeros));
      for (int jj = 0; jj < L; ++jj) {
        int cstart = jj * N;
        A_mat[jj] = a_sub.cols(cstart, cstart + N - 1);
      }

      // Get contemporaneous AA vector (off-diagonal elements) and build invA
      vec AA_col = AA.slice(draw).col(t + 1); // AA stored with T_eff+1 cols
      Mat<double> A0 = chofac(N, AA_col);
      Mat<double> invA;
      bool ok_inv = inv(invA, A0);
      if (!ok_inv) continue; // skip if non-invertible

      // Get OM row (variances) for this draw/time
      rowvec OM_row = OM.slice(draw).row(t);
      Mat<double> VAR = invA * diagmat(OM_row.t()) * invA.t();

      // Compute spectral density across frequencies for the price variable
      std::vector<double> Spec(Nw, 0.0);
      for (int wi = 0; wi < Nw; ++wi) {
        std::complex<double> cw = std::exp(std::complex<double>(0, -w(wi)));
        cx_mat G = eye<cx_mat>(N, N);
        for (int jj = 0; jj < L; ++jj) {
          cx_mat Aj = conv_to<cx_mat>::from(A_mat[jj]);
          G -= Aj * std::pow(cw, jj + 1);
        }
        // Invert G
        cx_mat Ginv = inv(G);
        cx_mat VARcx = conv_to<cx_mat>::from(VAR);
        cx_mat S = Ginv * VARcx * Ginv.t();
        std::complex<double> val = S(price_var, price_var);
        Spec[wi] = std::real(val);
      }

      double sumSpec = 0.0;
      for (double v : Spec) sumSpec += v;
      if (sumSpec <= 0) continue;

      double NormSpecAtZero = Spec[0] / sumSpec; // normalized spectral density at omega=0
      pers_vec.push_back(NormSpecAtZero);
    }

    if (pers_vec.size() > 0) {
      vec pers_v = conv_to<vec>::from(pers_vec);
      vec pers_s = sort(pers_v);
      int n = pers_s.n_elem;

      int i_med = std::max(0, (int)(n * 0.5) - 1);
      int i_025 = std::max(0, (int)(n * 0.025) - 1);
      int i_975 = std::min(n - 1, (int)(n * 0.975));
      int i_16 = std::max(0, (int)(n * 0.16) - 1);
      int i_84 = std::min(n - 1, (int)(n * 0.84));

      result.persistence_median(t) = pers_s(i_med);
      result.persistence_lower(t) = pers_s(i_025);
      result.persistence_upper(t) = pers_s(i_975);
      result.persistence_16(t) = pers_s(i_16);
      result.persistence_84(t) = pers_s(i_84);
    }
  }

  return result;
}

// ============================================================================
// FIGURE 4: TIME-VARYING ELASTICITIES
// ============================================================================

struct Figure4Result {
  // Each matrix is n_time x 7 (median, p025, p975, p16, p84, max, min)
  Mat<double> elas_supply_agg;  // Supply elasticity (from aggregate shock)
  Mat<double> elas_supply_dem;  // Supply elasticity (from demand shock)
  Mat<double> elas_demand;      // Demand elasticity
  int n_time;
};

inline Figure4Result compute_figure4(
    const Cube<double> &SD, const Cube<double> &AA, const Cube<double> &QD,
    const Mat<double> &S1, const Cube<double> &S2, const Mat<double> &VD,
    const Cube<double> &OM, const Mat<double> &Y, int L, int N, int T_eff,
    int n_saved, int n_reps, int h_restrict, int first_diff,
    double upper_supply, double lower_demand, int start_period) {

  Figure4Result result;
  
  // Validate start_period
  if (start_period < 1) start_period = 1;
  if (start_period > T_eff) start_period = T_eff;
  
  int n_time = T_eff - start_period + 1;
  result.n_time = n_time;
  
  result.elas_supply_agg.set_size(n_time, 7);
  result.elas_supply_dem.set_size(n_time, 7);
  result.elas_demand.set_size(n_time, 7);
  result.elas_supply_agg.fill(datum::nan);
  result.elas_supply_dem.fill(datum::nan);
  result.elas_demand.fill(datum::nan);

  int HOR = 21;
  int cors = 1;

  for (int t_idx = 0; t_idx < n_time; ++t_idx) {
    int t = start_period - 1 + t_idx;
    
    // Bounds check for t
    if (t < 0 || t >= T_eff) continue;

    std::vector<double> elast_sa_vec, elast_sd_vec, elast_dd_vec;

    for (int draw = 0; draw < n_saved; ++draw) {
      // Bounds check for draw
      if (draw >= (int)SD.n_slices) continue;
      
      // Bounds check for matrix dimensions
      if (t >= (int)SD.slice(draw).n_cols) continue;
      if (t >= (int)AA.slice(draw).n_cols) continue;
      if (t >= (int)OM.slice(draw).n_rows) continue;
      
      vec sd = SD.slice(draw).col(t);
      vec aa = AA.slice(draw).col(t);
      Mat<double> qd = QD.slice(draw);
      double s1 = S1(0, draw);
      Mat<double> s2 = S2.slice(draw);
      vec vd = VD.col(draw);
      vec om = OM.slice(draw).row(t).t();

      Mat<double> YY(N, L);
      int T_total = Y.n_rows;
      int TP = (T_total - T_eff) / 4;
      int data_start = 4 * TP;

      for (int lag = 0; lag < L; ++lag) {
        int y_idx = data_start + t - lag;
        if (y_idx >= 0 && y_idx < (int)Y.n_rows) {
          YY.col(lag) = Y.row(y_idx).t();
        }
      }

      for (int rep = 0; rep < n_reps; ++rep) {
        Mat<double> sirf, dirf, aggirf;

        bool success =
            GIRFs_N3(sd, aa, qd, s1, s2, vd, om, YY, HOR, N, L, cors, 1,
                     h_restrict, false, first_diff, sirf, dirf, aggirf);

        if (success) {
          OilMarketElasticities elast =
              oilmarket_model(sirf.row(0), dirf.row(0), aggirf.row(0));

          if (elast.slope_S_D <= upper_supply &&
              elast.slope_S_A <= upper_supply &&
              elast.slope_D_D >= lower_demand) {

            elast_sa_vec.push_back(elast.slope_S_A);
            elast_sd_vec.push_back(elast.slope_S_D);
            elast_dd_vec.push_back(elast.slope_D_D);
          }
        }
      }
    }

    if (elast_sa_vec.size() > 0) {
      vec sa_v = conv_to<vec>::from(elast_sa_vec);
      vec sd_v = conv_to<vec>::from(elast_sd_vec);
      vec dd_v = conv_to<vec>::from(elast_dd_vec);

      vec sa_s = sort(sa_v);
      vec sd_s = sort(sd_v);
      vec dd_s = sort(dd_v);

      int n = sa_s.n_elem;
      
      // Percentile indices (matching Figure 2 and MATLAB)
      int i_med = std::max(0, (int)(n * 0.5) - 1);
      int i_025 = std::max(0, (int)(n * 0.025) - 1);
      int i_975 = std::min(n - 1, (int)(n * 0.975));
      int i_16 = std::max(0, (int)(n * 0.16) - 1);
      int i_84 = std::min(n - 1, (int)(n * 0.84));

      // Store: median, p025, p975, p16, p84, max, min
      result.elas_supply_agg(t_idx, 0) = sa_s(i_med);
      result.elas_supply_agg(t_idx, 1) = sa_s(i_025);
      result.elas_supply_agg(t_idx, 2) = sa_s(i_975);
      result.elas_supply_agg(t_idx, 3) = sa_s(i_16);
      result.elas_supply_agg(t_idx, 4) = sa_s(i_84);
      result.elas_supply_agg(t_idx, 5) = sa_s(n - 1);  // max
      result.elas_supply_agg(t_idx, 6) = sa_s(0);      // min

      result.elas_supply_dem(t_idx, 0) = sd_s(i_med);
      result.elas_supply_dem(t_idx, 1) = sd_s(i_025);
      result.elas_supply_dem(t_idx, 2) = sd_s(i_975);
      result.elas_supply_dem(t_idx, 3) = sd_s(i_16);
      result.elas_supply_dem(t_idx, 4) = sd_s(i_84);
      result.elas_supply_dem(t_idx, 5) = sd_s(n - 1);  // max
      result.elas_supply_dem(t_idx, 6) = sd_s(0);      // min

      result.elas_demand(t_idx, 0) = dd_s(i_med);
      result.elas_demand(t_idx, 1) = dd_s(i_025);
      result.elas_demand(t_idx, 2) = dd_s(i_975);
      result.elas_demand(t_idx, 3) = dd_s(i_16);
      result.elas_demand(t_idx, 4) = dd_s(i_84);
      result.elas_demand(t_idx, 5) = dd_s(n - 1);  // max
      result.elas_demand(t_idx, 6) = dd_s(0);      // min
    }
  }

  return result;
}

#endif // TVP_FIGURES_H
