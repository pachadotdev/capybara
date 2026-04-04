// Generalized linear model with fixed effects eta = alpha + X * beta
#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

struct GlmWorkspace {
  vec mu;        // fitted values on response scale
  vec w_working; // working weights
  vec nu;        // working residuals
  vec z;         // working response (reused across iterations)
  vec eta0;      // previous eta (for step-halving)
  vec beta0;     // previous beta (for step-halving)

  uword cached_n, cached_p;

  GlmWorkspace() : cached_n(0), cached_p(0) {}

  void ensure_size(uword n, uword p) {
    if (n > cached_n) {
      mu.set_size(n);
      w_working.set_size(n);
      nu.set_size(n);
      z.set_size(n);
      eta0.set_size(n);
      cached_n = n;
    }
    if (p > cached_p) {
      beta0.set_size(p);
      cached_p = p;
    }
  }
};

// Function pointer types for family-specific operations
// Avoids repeated switch statements in hot loops
using MuFromEta = void (*)(vec &mu, const vec &eta);
using WorkingWtsNu = void (*)(vec &w_working, vec &nu, const vec &w,
                              const vec &mu, const vec &y, double theta);

// Link inverse functions (mu from eta)
inline void mu_gaussian(vec &mu, const vec &eta) { mu = eta; }
inline void mu_poisson(vec &mu, const vec &eta) { mu = exp(eta); }
inline void mu_binomial(vec &mu, const vec &eta) {
  mu = 1.0 / (1.0 + exp(-eta));
}
inline void mu_gamma(vec &mu, const vec &eta) { mu = 1.0 / eta; }
inline void mu_invgaussian(vec &mu, const vec &eta) { mu = 1.0 / sqrt(eta); }

// Working weights and working residuals (nu) - vectorized
inline void ww_nu_gaussian(vec &w_working, vec &nu, const vec &w, const vec &mu,
                           const vec &y, double) {
  w_working = w;
  nu = y - mu;
}

inline void ww_nu_poisson(vec &w_working, vec &nu, const vec &w, const vec &mu,
                          const vec &y, double) {
  w_working = w % mu;
  nu = (y - mu) / mu;
}

inline void ww_nu_binomial(vec &w_working, vec &nu, const vec &w, const vec &mu,
                           const vec &y, double) {
  const vec var = mu % (1.0 - mu);
  w_working = w % var;
  nu = (y - mu) / var;
}

inline void ww_nu_gamma(vec &w_working, vec &nu, const vec &w, const vec &mu,
                        const vec &y, double) {
  const vec m2 = square(mu);
  w_working = w % m2;
  nu = -(y - mu) / m2;
}

inline void ww_nu_invgaussian(vec &w_working, vec &nu, const vec &w,
                              const vec &mu, const vec &y, double) {
  const vec m3 = pow(mu, 3);
  w_working = 0.25 * (w % m3);
  nu = -2.0 * (y - mu) / m3;
}

inline void ww_nu_negbin(vec &w_working, vec &nu, const vec &w, const vec &mu,
                         const vec &y, double theta) {
  w_working = (w % mu) / (1.0 + mu / theta);
  nu = (y - mu) / mu;
}

// Get function pointers for a family (called once, not in loop)
inline MuFromEta get_mu_fn(Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return mu_gaussian;
  case POISSON:
  case NEG_BIN:
    return mu_poisson;
  case BINOMIAL:
    return mu_binomial;
  case GAMMA:
    return mu_gamma;
  case INV_GAUSSIAN:
    return mu_invgaussian;
  default:
    return mu_gaussian;
  }
}

inline WorkingWtsNu get_ww_nu_fn(Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ww_nu_gaussian;
  case POISSON:
    return ww_nu_poisson;
  case BINOMIAL:
    return ww_nu_binomial;
  case GAMMA:
    return ww_nu_gamma;
  case INV_GAUSSIAN:
    return ww_nu_invgaussian;
  case NEG_BIN:
    return ww_nu_negbin;
  default:
    return ww_nu_gaussian;
  }
}

///////////////////////////////////////////////////////////////////////////
// Average Partial Effects (APE) computation for binomial models
// Based on Cruz-Gonzalez, Fernández-Val, and Weidner (2017)
///////////////////////////////////////////////////////////////////////////

// Compute second-order partial derivative of mu with respect to eta
// For logit: d²mu/deta² = mu*(1-mu)*(1-2*mu)
inline vec partial_mu_eta_2(const vec &eta, Family family_type) {
  if (family_type == BINOMIAL) {
    // Logit link: mu = 1/(1+exp(-eta))
    // mu.eta = mu*(1-mu)
    // d(mu.eta)/deta = mu*(1-mu)*(1-2*mu)
    const vec mu = 1.0 / (1.0 + exp(-eta));
    return mu % (1.0 - mu) % (1.0 - 2.0 * mu);
  }
  // For other families, return zeros (not implemented)
  return vec(eta.n_elem, fill::zeros);
}

// Compute third-order partial derivative of mu with respect to eta
// For logit: d³mu/deta³ = mu*(1-mu)*(1-6*mu+6*mu²)
inline vec partial_mu_eta_3(const vec &eta, Family family_type) {
  if (family_type == BINOMIAL) {
    const vec mu = 1.0 / (1.0 + exp(-eta));
    const vec mu2 = square(mu);
    return mu % (1.0 - mu) % (1.0 - 6.0 * mu + 6.0 * mu2);
  }
  return vec(eta.n_elem, fill::zeros);
}

// Helper: compute group-level variance contribution for FE sampling correction
// V_k = sum_g (Delta_g * Delta_g') where Delta_g = sum_{i in group g} Delta_i
inline mat group_sums_var(const mat &Delta, const FlatFEMap &fe_map, uword k) {
  const uword p = Delta.n_cols;
  const uword n = Delta.n_rows;
  const uword G = fe_map.n_groups[k];

  mat V(p, p, fill::zeros);

  // Compute group sums
  mat group_sums(G, p, fill::zeros);
  const uword *gk = fe_map.fe_map[k].data();
  for (uword i = 0; i < n; ++i) {
    const uword g = gk[i];
    group_sums.row(g) += Delta.row(i);
  }

  // Compute V = sum_g (group_sum_g * group_sum_g')
  for (uword g = 0; g < G; ++g) {
    V += group_sums.row(g).t() * group_sums.row(g);
  }

  return V;
}

// Helper: compute cross-covariance for weak exogeneity correction
// C = sum_g (Delta_g * Gamma_g') where group sums computed per FE dimension
inline mat group_sums_cov(const mat &Delta, const mat &Gamma,
                          const FlatFEMap &fe_map, uword k) {
  const uword p = Delta.n_cols;
  const uword n = Delta.n_rows;
  const uword G = fe_map.n_groups[k];

  // Compute group sums for both matrices
  mat delta_sums(G, p, fill::zeros);
  mat gamma_sums(G, p, fill::zeros);

  const uword *gk = fe_map.fe_map[k].data();
  for (uword i = 0; i < n; ++i) {
    const uword g = gk[i];
    delta_sums.row(g) += Delta.row(i);
    gamma_sums.row(g) += Gamma.row(i);
  }

  // Compute C = sum_g (delta_g * gamma_g')
  mat C(p, p, fill::zeros);
  for (uword g = 0; g < G; ++g) {
    C += delta_sums.row(g).t() * gamma_sums.row(g);
  }

  return C;
}

// Compute Average Partial Effects for binomial models
// X: design matrix (n x p)
// beta: coefficient vector (p)
// eta: linear predictor (n)
// mu: fitted probabilities (n)
// w: weights (n)
// H_inv: inverse Hessian (p x p)
// MX: centered design matrix (n x p) - needed for variance
// fe_map: fixed effects map (for FE variance corrections)
// params: parameters including APE options (n_pop, panel_structure, etc.)
inline void compute_apes_binomial(InferenceGLM &result, const mat &X,
                                  const vec &beta, const vec &eta,
                                  const vec &mu, const vec &w, const mat &H_inv,
                                  const mat &MX, uword n_full,
                                  const FlatFEMap &fe_map,
                                  const CapybaraParameters &params) {
  const uword n = X.n_rows;
  const uword p = X.n_cols;
  const uword K = fe_map.K;

  if (p == 0)
    return;

  // Compute finite population adjustment factor
  // adj = (n_pop - n_full) / (n_pop - 1) if n_pop > 0, else 0
  double adj = 0.0;
  if (params.ape_n_pop > 0 && params.ape_n_pop >= n_full) {
    adj = static_cast<double>(params.ape_n_pop - n_full) /
          static_cast<double>(params.ape_n_pop - 1);
  }

  const bool is_classic = (params.ape_panel_structure == "classic");
  const bool is_independence = (params.ape_sampling_fe == "independence");

  // Detect binary regressors: all values in {0, 1}
  uvec is_binary(p, fill::zeros);
  for (uword j = 0; j < p; ++j) {
    bool all_binary = true;
    for (uword i = 0; i < n && all_binary; ++i) {
      const double v = X(i, j);
      if (v != 0.0 && v != 1.0) {
        all_binary = false;
      }
    }
    is_binary(j) = all_binary ? 1 : 0;
  }

  // Compute mu.eta = dmu/deta for logit: mu*(1-mu)
  const vec mu_eta = mu % (1.0 - mu);

  // Compute Delta matrix (partial effects per observation)
  mat Delta(n, p);

  for (uword j = 0; j < p; ++j) {
    if (is_binary(j) == 1) {
      // Binary regressor: delta = F(eta + beta) - F(eta - X*beta)
      const vec eta0 = eta - X.col(j) * beta(j);
      const vec eta1 = eta0 + beta(j);
      const vec mu0 = 1.0 / (1.0 + exp(-eta0));
      const vec mu1 = 1.0 / (1.0 + exp(-eta1));
      Delta.col(j) = mu1 - mu0;
    } else {
      // Continuous regressor: delta = beta * mu.eta
      Delta.col(j) = beta(j) * mu_eta;
    }
  }

  // Compute APE = weighted average of Delta
  const double w_sum = accu(w);
  const vec delta = (Delta.each_col() % w).t() * vec(n, fill::ones) / w_sum;

  // Center Delta for variance computation: Delta_centered = Delta - delta
  mat Delta_centered(n, p);
  for (uword j = 0; j < p; ++j) {
    Delta_centered.col(j) =
        (Delta.col(j) - delta(j)) / static_cast<double>(n_full);
  }

  // Compute Jacobian for variance (delta method)
  mat J(p, p, fill::zeros);
  const vec z = partial_mu_eta_2(eta, BINOMIAL);
  const mat PX = X - MX;
  const vec w_norm = w / w_sum;

  for (uword j = 0; j < p; ++j) {
    if (is_binary(j) == 1) {
      const vec eta0 = eta - X.col(j) * beta(j);
      const vec eta1 = eta0 + beta(j);
      const vec f1 = 1.0 / (1.0 + exp(-eta1));
      const vec mu_eta_1 = f1 % (1.0 - f1);
      const vec mu_eta_0 =
          1.0 / (1.0 + exp(-eta0)) % (1.0 - 1.0 / (1.0 + exp(-eta0)));
      const vec Delta1 = mu_eta_1 - mu_eta_0;

      J.col(j) = -PX.t() * (w_norm % Delta1);
      J(j, j) += dot(w_norm, mu_eta_1);
      for (uword k = 0; k < p; ++k) {
        if (k != j) {
          J(k, j) += dot(w_norm % X.col(k), Delta1);
        }
      }
    } else {
      const vec Delta1 = beta(j) * z;
      J.col(j) = MX.t() * (w_norm % Delta1);
      J(j, j) += dot(w_norm, mu_eta);
    }
  }

  // Compute residuals for sandwich estimator
  const vec v =
      w % (result.weights.n_elem > 0 ? (mu - mu) : vec(n, fill::zeros));
  // Note: for logit, v = w * (y - mu), but we need y here
  // Simplified: use working residuals from the model

  // Compute Gamma = (MX * H_inv * J - Psi) * v / n_full
  // Simplified version focusing on the core delta method variance
  const mat WinvJ = H_inv * J;
  mat Gamma = (MX * WinvJ) / static_cast<double>(n_full);

  // Base variance: delta method
  mat V = Gamma.t() * Gamma;

  // Add finite population correction if requested
  if (adj > 0.0 && K > 0 && is_independence) {
    // FE sampling correction under independence assumption
    // V += adj * sum_k groupSumsVar(Delta_centered, fe_map, k)
    if (is_classic) {
      // Classic panel: one or two-way FE
      V += adj * group_sums_var(Delta_centered, fe_map, 0);
      if (K > 1) {
        V += adj * (group_sums_var(Delta_centered, fe_map, 1) -
                    Delta_centered.t() * Delta_centered);
      }
    } else {
      // Network panel: two or three-way FE
      V += adj * group_sums_var(Delta_centered, fe_map, 0);
      V += adj * (group_sums_var(Delta_centered, fe_map, 1) -
                  Delta_centered.t() * Delta_centered);
      if (K > 2) {
        V += adj * (group_sums_var(Delta_centered, fe_map, 2) -
                    Delta_centered.t() * Delta_centered);
      }
    }

    // Add weak exogeneity correction if requested
    if (params.ape_weak_exo) {
      mat C;
      if (is_classic) {
        C = group_sums_cov(Delta_centered, Gamma, fe_map, 0);
      } else if (K > 2) {
        C = group_sums_cov(Delta_centered, Gamma, fe_map, K - 1);
      }
      if (C.n_elem > 0) {
        V += adj * (C + C.t());
      }
    }
  }

  // Store results
  result.ape_delta = delta;
  result.ape_vcov = V;
  result.ape_binary = is_binary;
  result.has_apes = true;
}

// Helper: compute weighted group sums for bias correction
// Returns sum_g (sum_i MX_i * z_i / sum_i w_i) for group g
inline vec group_sums_bias(const mat &MXz, const vec &w,
                           const FlatFEMap &fe_map, uword k) {
  const uword n = MXz.n_rows;
  const uword p = MXz.n_cols;
  const uword G = fe_map.n_groups[k];

  // Accumulate numerator and denominator per group
  mat group_num(G, p, fill::zeros);
  vec group_denom(G, fill::zeros);

  const uword *gk = fe_map.fe_map[k].data();
  for (uword i = 0; i < n; ++i) {
    const uword g = gk[i];
    group_num.row(g) += MXz.row(i);
    group_denom(g) += w(i);
  }

  // Sum across groups: sum_g (numerator_g / denominator_g)
  vec b(p, fill::zeros);
  for (uword g = 0; g < G; ++g) {
    if (group_denom(g) > 0.0) {
      b += group_num.row(g).t() / group_denom(g);
    }
  }

  return b;
}

// Helper: compute spectral density term for bias correction (weak exogeneity)
// Used when bandwidth L > 0 (weakly exogenous regressors)
inline vec group_sums_spectral_bias(const mat &MXw, const vec &v, const vec &w,
                                    uword L, const FlatFEMap &fe_map, uword k) {
  const uword n = MXw.n_rows;
  const uword p = MXw.n_cols;
  const uword G = fe_map.n_groups[k];

  vec b(p, fill::zeros);

  // We need to process observations within each group in order
  // First, build group membership
  std::vector<std::vector<uword>> group_obs(G);
  const uword *gk = fe_map.fe_map[k].data();
  for (uword i = 0; i < n; ++i) {
    group_obs[gk[i]].push_back(i);
  }

  for (uword g = 0; g < G; ++g) {
    const std::vector<uword> &obs = group_obs[g];
    const uword I = obs.size();
    if (I <= 1)
      continue;

    double denom = 0.0;
    for (uword idx : obs) {
      denom += w(idx);
    }
    if (denom == 0.0)
      continue;

    // Compute cumulative sum of v within group
    vec v_group(I);
    for (uword i = 0; i < I; ++i) {
      v_group(i) = v(obs[i]);
    }
    const vec v_cumsum = cumsum(v_group);
    const uword max_k = std::min(L, I - 1);

    // Compute shifted sum: v_shifted[i] = sum_{k=1}^{min(L,i)} v_group[i-k]
    vec v_shifted(I, fill::zeros);
    for (uword i = 1; i < I; ++i) {
      const uword start = (i > max_k) ? i - max_k : 0;
      v_shifted(i) = v_cumsum(i - 1) - (start > 0 ? v_cumsum(start - 1) : 0.0);
    }

    const double scale = static_cast<double>(I) / ((I - 1.0) * denom);

    // Accumulate M^T * v_shifted
    for (uword i = 0; i < I; ++i) {
      b += MXw.row(obs[i]).t() * v_shifted(i) * scale;
    }
  }

  return b;
}

// Compute bias correction for binomial models
// Applies Fernández-Val & Weidner (2016) analytical bias correction
inline void compute_bias_corr_binomial(InferenceGLM &result, const mat &X,
                                       const vec &beta, const vec &eta,
                                       const vec &mu, const vec &w,
                                       const mat &H, const mat &MX,
                                       const uword n, const FlatFEMap &fe_map,
                                       const CapybaraParameters &params) {
  const uword K = fe_map.K;
  const uword p = beta.n_elem;

  if (K == 0) {
    // No fixed effects - bias correction not applicable
    result.has_bias_corr = false;
    return;
  }

  // Validate panel structure matches FE dimensions
  const bool is_classic = (params.bias_corr_panel_structure == "classic");
  if (is_classic && K > 2) {
    // classic requires 1 or 2 FE
    result.has_bias_corr = false;
    return;
  }
  if (!is_classic && K < 2) {
    // network requires 2 or 3 FE
    result.has_bias_corr = false;
    return;
  }

  // Compute z = w * partial_mu_eta_2(eta) (second derivative term)
  const vec z = w % partial_mu_eta_2(eta, BINOMIAL);

  // MX * z elementwise: each row of MX scaled by z
  mat MXz = MX;
  for (uword j = 0; j < p; ++j) {
    MXz.col(j) %= z;
  }

  // Compute bias terms
  vec b(p, fill::zeros);

  if (is_classic) {
    // Classic panel: one- or two-way fixed effects
    b += group_sums_bias(MXz, w, fe_map, 0) / (2.0 * n);
    if (K > 1) {
      b += group_sums_bias(MXz, w, fe_map, 1) / (2.0 * n);
    }

    // Add spectral density term if bandwidth > 0 (weak exogeneity)
    if (params.bias_corr_bandwidth > 0) {
      // v = w * (y - mu) but we don't have y here
      // Use working residual from mu.eta * (y - mu) approximation
      // For logit: v = w * (y - mu), but we store fitted_values = mu
      // We need the actual response y, which we don't have in this function
      // The R code computes v = wt * (y - mu)
      // For now, skip spectral term - requires passing y to this function
      // or computing it in feglm_fit where y is available

      // MXw = MX scaled by working weights (mu.eta for binomial)
      const vec mu_eta = mu % (1.0 - mu); // derivative of mu w.r.t. eta
      mat MXw = MX;
      for (uword j = 0; j < p; ++j) {
        MXw.col(j) %= (w % mu_eta);
      }

      // Note: For proper spectral density, we'd need the working residuals v
      // This is a placeholder - full implementation would require passing y
      // For strict exogeneity (bandwidth=0), this is not needed
    }
  } else {
    // Network panel: two- or three-way fixed effects
    b += group_sums_bias(MXz, w, fe_map, 0) / (2.0 * n);
    b += group_sums_bias(MXz, w, fe_map, 1) / (2.0 * n);
    if (K > 2) {
      b += group_sums_bias(MXz, w, fe_map, 2) / (2.0 * n);
    }

    // Spectral term for network panel uses the last FE dimension
    if (K > 2 && params.bias_corr_bandwidth > 0) {
      // Same note as above - requires y for full implementation
    }
  }

  // Solve for bias term: b_solve = solve(H/n, -b) = -H^{-1}*(H/n)^{-1}*b
  // Actually: beta_corr = beta - solve(H/n, -b) = beta + solve(H/n, b)
  // So we solve (H/n) * x = b for x, then beta_corr = beta - x
  mat H_scaled = H / static_cast<double>(n);
  vec bias_term;

  // Check input validity (can have NaN/Inf on some platforms due to
  // numerical precision differences, especially on Mac with Accelerate)
  if (!H_scaled.is_finite() || !b.is_finite()) {
    result.has_bias_corr = false;
    return;
  }

  // Try solve with allow_ugly to handle near-singular matrices more robustly
  // across different BLAS implementations (especially Apple Accelerate)
  bool solve_ok = solve(bias_term, H_scaled, -b, solve_opts::allow_ugly);

  if (!solve_ok) {
    // Fallback to pseudoinverse with explicit tolerance for robustness
    mat H_inv;
    // Use a slightly relaxed tolerance for pinv to handle
    // platform-specific numerical precision differences
    if (pinv(H_inv, H_scaled, datum::eps * 100)) {
      bias_term = H_inv * (-b);
    } else {
      // Last resort: try inv_sympd since H should be symmetric positive
      // semi-definite (it's X'WX scaled)
      if (inv_sympd(H_inv, H_scaled)) {
        bias_term = H_inv * (-b);
      } else {
        result.has_bias_corr = false;
        return;
      }
    }
  }

  // Validate the computed bias term is finite
  if (!bias_term.is_finite()) {
    result.has_bias_corr = false;
    return;
  }

  // Compute corrected coefficients
  result.beta_corrected = beta - bias_term;
  result.bias_term = bias_term;
  result.has_bias_corr = true;
}

InferenceGLM feglm_fit(
    vec &beta, vec &eta, const vec &y, mat &X, const vec &w,
    const double &theta, const Family family_type, const FlatFEMap &fe_map,
    const CapybaraParameters &params, GlmWorkspace *workspace = nullptr,
    const field<uvec> *cluster_groups = nullptr, const vec *offset = nullptr,
    bool skip_separation_check = false,
    const field<uvec> *entity1_groups = nullptr,
    const field<uvec> *entity2_groups = nullptr, bool run_from_negbin = false) {
#ifdef CAPYBARA_DEBUG
  double mem_start = get_memory_usage_mb();
  std::ostringstream feglm_msg;
  feglm_msg << "/////////////////////////////////\n"
               "// Entering feglm_fit function //\n"
               "/////////////////////////////////\n"
               "Initial memory: "
            << mem_start << " MB\n";
  cpp4r::message(feglm_msg.str());
#endif

  const uword n = y.n_elem;
  const bool has_fixed_effects = fe_map.K > 0;
  const bool has_offset =
      (offset != nullptr && offset->n_elem == n && any(*offset != 0.0));

  // Add intercept column if no fixed effects (in-place to avoid allocation)
  if (!has_fixed_effects) {
    X.insert_cols(0, 1);
    X.col(0).ones();
    beta = join_cols(vec{0.0}, beta);
  }

  const uword p = X.n_cols;

  // Store original X in the FelmWorkspace (needed for FE recovery after
  // convergence). Skip when called from negbin outer loop — only the final
  // converged call needs FE recovery (run_from_negbin=false).
  // This avoids an upfront N*P copy; instead the workspace owns it.

  // Use lite constructor for fast path (skips P*P hessian/vcov allocation)
  InferenceGLM result(n, p, !run_from_negbin);

  if (!y.is_finite() || !X.is_finite()) {
    result.conv = false;
    // Initialize with NaN for R-side diagnostics
    result.eta.set_size(n);
    result.eta.fill(datum::nan);
    result.fitted_values.set_size(n);
    result.fitted_values.fill(datum::nan);
    result.weights = w;
    return result;
  }

  // Workspace setup
  GlmWorkspace local_workspace;
  GlmWorkspace &ws = workspace ? *workspace : local_workspace;
  ws.ensure_size(n, p);

  // Get function pointers once (avoid switch in loop)
  const MuFromEta mu_ = get_mu_fn(family_type);
  const WorkingWtsNu ww_nu_ = get_ww_nu_fn(family_type);

  // Offset handling
  const vec offset_vec = has_offset ? *offset : vec();

#ifdef CAPYBARA_DEBUG
  auto tsep0 = std::chrono::high_resolution_clock::now();
#endif

  // Group-level separation pre-filter (replaces R-side drop_by_link_type_)
  // For Poisson/NegBin/Binomial FE models: drop entire FE groups where
  // mean(y)==0 (Poisson/NegBin) or mean(y) in {0,1} (Binomial)
  SeparationResult group_sep_result;
  if (!skip_separation_check && has_fixed_effects && params.check_separation &&
      (family_type == POISSON || family_type == NEG_BIN ||
       family_type == BINOMIAL)) {
    group_sep_result = check_group_separation(y, w, fe_map, family_type);
  }

  // Observation-level separation detection (ReLU + Simplex) for Poisson FE
  if (family_type == Family::POISSON && !skip_separation_check &&
      has_fixed_effects && params.check_separation) {
    // Use weights with group-separated obs already zeroed
    vec w_for_sep = w;
    if (group_sep_result.num_separated > 0) {
      w_for_sep.elem(group_sep_result.separated_obs).zeros();
    }

    SeparationResult sep_result = check_separation(y, X, w_for_sep, params);

    // Merge group-level and observation-level results
    if (group_sep_result.num_separated > 0 || sep_result.num_separated > 0) {
      uvec all_separated;
      if (group_sep_result.num_separated > 0 && sep_result.num_separated > 0) {
        all_separated = unique(join_vert(group_sep_result.separated_obs,
                                         sep_result.separated_obs));
      } else if (group_sep_result.num_separated > 0) {
        all_separated = group_sep_result.separated_obs;
      } else {
        all_separated = sep_result.separated_obs;
      }

      // Zero weights for all separated obs
      vec w_work = w;
      w_work.elem(all_separated).zeros();

      InferenceGLM result_with_sep =
          feglm_fit(beta, eta, y, X, w_work, theta, family_type, fe_map, params,
                    &ws, cluster_groups, offset, true);

      result_with_sep.eta.elem(all_separated).fill(datum::nan);
      result_with_sep.fitted_values.elem(all_separated).fill(datum::nan);
      result_with_sep.has_separation = true;
      result_with_sep.separated_obs = all_separated;
      result_with_sep.num_separated = all_separated.n_elem;
      if (sep_result.support.n_elem > 0) {
        result_with_sep.separation_support = sep_result.support;
      }

      return result_with_sep;
    }
  } else if (group_sep_result.num_separated > 0) {
    // Non-Poisson (Binomial, NegBin) with group separation only
    vec w_work = w;
    w_work.elem(group_sep_result.separated_obs).zeros();

    InferenceGLM result_with_sep =
        feglm_fit(beta, eta, y, X, w_work, theta, family_type, fe_map, params,
                  &ws, cluster_groups, offset, true);

    result_with_sep.eta.elem(group_sep_result.separated_obs).fill(datum::nan);
    result_with_sep.fitted_values.elem(group_sep_result.separated_obs)
        .fill(datum::nan);
    result_with_sep.has_separation = true;
    result_with_sep.separated_obs = group_sep_result.separated_obs;
    result_with_sep.num_separated = group_sep_result.num_separated;

    return result_with_sep;
  }

#ifdef CAPYBARA_DEBUG
  auto tsep1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sep_duration = tsep1 - tsep0;
  double mem_after_sep = get_memory_usage_mb();
  std::ostringstream sep_msg;
  sep_msg << "Separation detection time: " << sep_duration.count()
          << " seconds. Memory: " << mem_after_sep << " MB\n";
  cpp4r::message(sep_msg.str());
  auto tcoll0 = std::chrono::high_resolution_clock::now();
#endif

  // Collinearity check (once before iterations)
  // After this check, we know which columns are non-collinear and can use
  // regular chol() for any subsequent Hessian computations.
  const bool use_weights = any(w != 1.0);

  CollinearityResult collin_result(X.n_cols);

  // Scope XtX and R_rank so they're deallocated immediately after use
  // (avoids holding P² memory through the entire IRLS loop)
  {
    const mat XtX = use_weights ? crossprod(X, w) : crossprod(X);
    mat R_rank;
    uvec excl;
    uword rank;
    chol_rank(R_rank, excl, rank, XtX, "upper", params.collin_tol);

    if (any(excl)) {
      collin_result.has_collinearity = true;
      collin_result.non_collinear_cols = find(excl == 0);
      collin_result.collinear_cols = find(excl != 0);
      collin_result.coef_status = 1 - excl;
    } else {
      collin_result.has_collinearity = false;
      collin_result.non_collinear_cols = regspace<uvec>(0, X.n_cols - 1);
      collin_result.coef_status.ones();
    }
  } // XtX and R_rank deallocated here

  // Now remove collinear columns from X (after R_rank is freed)
  if (collin_result.has_collinearity) {
    X.shed_cols(collin_result.collinear_cols);
  }

#ifdef CAPYBARA_DEBUG
  auto tcoll1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> collin_duration = tcoll1 - tcoll0;
  double mem_after_collin = get_memory_usage_mb();
  std::ostringstream collin_msg;
  collin_msg << "Collinearity check time: " << collin_duration.count()
             << " seconds. Memory: " << mem_after_collin << " MB\n";
  cpp4r::message(collin_msg.str());
#endif

  const uword p_working = X.n_cols;

  // Workspace references
  vec &mu = ws.mu;
  vec &w_working = ws.w_working;
  vec &nu = ws.nu;
  vec &z = ws.z;
  vec &eta0 = ws.eta0;
  vec &beta0 = ws.beta0;

  // Initial mu from eta
  mu_(mu, eta);

  // Deviance computations
  double dev = dev_resids(y, mu, theta, w, family_type);
  const double null_dev = null_deviance(y, theta, w, family_type);

  double dev0;
  bool conv = false;

  // Step-halving state
  const double step_halving_memory = params.step_halving_memory;
  uword num_step_halving = 0;

  // Adaptive centering tolerance parameters
  // Start with loose tolerance, tighten as GLM converges
  const double center_tol_loose = params.center_tol_loose;
  const double center_tol_tight = params.center_tol;
  double adaptive_center_tol = center_tol_loose;

  double last_beta_change = datum::inf;
  uword convergence_count = 0;
  double conv_change =
      datum::inf; // hoisted: readable after loop for post-loop check

  // Persistent felm workspace
  FelmWorkspace felm_workspace;

  // NOTE: We no longer copy X0 here. After shed_cols, X contains exactly
  // the non-collinear columns. For FE recovery, we use X directly with
  // beta.elem(non_collinear_cols) which matches the post-shed column structure.

#ifdef CAPYBARA_DEBUG
  cpp4r::message("/// Begin GLM iterations...\n");
  auto tglmiter0 = std::chrono::high_resolution_clock::now();
#endif

  // Main IRLS loop
  for (uword iter = 0; iter < params.iter_max; ++iter) {
    double rho = 1.0;
    eta0 = eta;
    beta0 = beta;
    dev0 = dev;

// Compute working weights and working residuals
#ifdef CAPYBARA_DEBUG
    auto twwnu0 = std::chrono::high_resolution_clock::now();
#endif

    ww_nu_(w_working, nu, w, mu, y, theta);

    // Working response z = eta + nu - offset (reuses workspace buffer)
    z = eta + nu;
    if (has_offset) {
      z -= offset_vec;
    }

    // Guard against non-finite working weights/response from mu
    // overflow or division-by-zero (e.g., exp(eta) = Inf for Poisson).
    // Zero the weight for affected observations so they don't poison
    // the cross-product X'WX that feeds into the Cholesky solver.
    // Single fused pass instead of find_nonfinite + unique + join_cols.
    {
      double *ww_ptr = w_working.memptr();
      double *z_ptr = z.memptr();
      for (uword i = 0; i < n; ++i) {
        if (!std::isfinite(ww_ptr[i]) || !std::isfinite(z_ptr[i])) {
          ww_ptr[i] = 0.0;
          z_ptr[i] = 0.0;
        }
      }
    }

#ifdef CAPYBARA_DEBUG
    auto twwnu1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wwnu_duration = twwnu1 - twwnu0;
    double mem_ww = get_memory_usage_mb();
    std::ostringstream wwnu_msg;
    wwnu_msg << "Working weights and nu time: " << wwnu_duration.count()
             << " seconds. Memory: " << mem_ww << " MB\n";
    cpp4r::message(wwnu_msg.str());
#endif

    // Weighted least squares via felm_fit (no copy needed - felm_fit uses
    // workspace)
    // First iteration: use 10x looser centering tolerance (like fixest)
    const double iter_center_tol =
        (iter == 0) ? adaptive_center_tol * 10.0 : adaptive_center_tol;

    InferenceLM lm_res =
        felm_fit(X, z, w_working, fe_map, params, &felm_workspace,
                 cluster_groups, true, iter_center_tol);

    const vec &beta_upd_reduced = lm_res.coef_table.col(0);

    // Compute eta update
    vec eta_upd = lm_res.fitted_values - eta0;
    if (has_offset) {
      eta_upd += offset_vec;
    }

    // Ensure beta has correct size for collinearity
    const uword full_p =
        collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;
    if (beta.n_elem != full_p) {
      beta.set_size(full_p);
      beta.fill(datum::nan);
    }

    // Step-halving inner loop
    bool dev_crit = false, val_crit = false, imp_crit = false;

    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;

      // Update beta with step-halving
      if (collin_result.has_collinearity) {
        const uvec &idx = collin_result.non_collinear_cols;
        beta = beta0;
        beta.elem(idx) = (1.0 - rho) * beta0.elem(idx) + rho * beta_upd_reduced;
      } else {
        beta = (1.0 - rho) * beta0 + rho * beta_upd_reduced;
      }

      // Update mu from new eta
      mu_(mu, eta);

      dev = dev_resids(y, mu, theta, w, family_type);
      const double dev_ratio_inner = (dev - dev0) / (0.1 + std::fabs(dev));

      dev_crit = std::isfinite(dev);
      val_crit = valid_eta(eta, family_type) && valid_mu(mu, family_type);
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }
      rho *= params.step_halving_factor;
    }

    // Handle non-convergence in inner loop
    if (!dev_crit || !val_crit) {
      result.conv = false;
      // Still populate result vectors for R-side diagnostics
      result.eta = std::move(eta);
      result.fitted_values = std::move(mu);
      result.weights = w;
      result.deviance = dev;
      result.null_deviance = null_dev;
      return result;
    }

    if (!imp_crit) {
      eta = eta0;
      beta = beta0;
      dev = dev0;
      mu_(mu, eta0);
    }

    const double delta_deviance = dev0 - dev;

    // Adaptive centering tolerance: always driven by eta, since the
    // centering routine operates on eta-scale quantities.
    const double eta_norm = std::sqrt(dot(eta, eta) / n);
    const double eta_change = std::sqrt(dot(eta - eta0, eta - eta0) / n) /
                              std::max(eta_norm, datum::eps);

    if (eta_change < 0.1) {
      const double t = std::max(0.0, std::min(1.0, (0.1 - eta_change) / 0.1));
      adaptive_center_tol =
          center_tol_loose * std::pow(center_tol_tight / center_tol_loose, t);
    }

    // Early convergence detection: eta-driven, since eta reflects the overall
    // fit progress across all n observations.
    if (eta_change < last_beta_change * 0.5) {
      ++convergence_count;
    } else {
      convergence_count = 0;
    }
    last_beta_change = eta_change;

    // Outer convergence criterion:
    // - When structural regressors are present (p_working > 0): use relative
    //   change in beta. Beta is fully scale-invariant (rescaling y does not
    //   change beta), satisfying Green & Santos Silva 2025.
    // - When there are no structural regressors (pure FE model, p_working ==
    // 0):
    //   beta is empty so fall back to eta, which is the only quantity that
    //   carries convergence information in that case.
    if (p_working > 0) {
      const double beta_norm = std::sqrt(dot(beta, beta));
      conv_change = std::sqrt(dot(beta - beta0, beta - beta0)) /
                    std::max(beta_norm, datum::eps);
    } else {
      conv_change = eta_change;
    }

    if (conv_change < params.dev_tol ||
        (convergence_count >= 2 && conv_change < params.dev_tol * 10)) {
      conv = true;
      break;
    }

    // Additional step-halving for deviance increase
    if (delta_deviance < 0 && num_step_halving < params.max_step_halving) {
      eta = step_halving_memory * eta0 + (1.0 - step_halving_memory) * eta;
      if (num_step_halving > 0 && family_type == POISSON) {
        eta = clamp(eta, -10.0, datum::inf);
      }
      mu_(mu, eta);
      dev = dev_resids(y, mu, theta, w, family_type);
      ++num_step_halving;
    } else {
      num_step_halving = 0;
    }

    result.iter = iter + 1;
  }

#ifdef CAPYBARA_DEBUG
  cpp4r::message("/// End GLM iterations...\n");
  auto tglmiter1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> glmiter_duration = tglmiter1 - tglmiter0;
  double mem_after_glm = get_memory_usage_mb();
  std::ostringstream glmiter_msg;
  glmiter_msg << "GLM iteration time: " << glmiter_duration.count()
              << " seconds. Memory: " << mem_after_glm << " MB\n";
  cpp4r::message(glmiter_msg.str());
#endif

  if (conv) {
    // Fast path for negbin outer loop: only return beta, eta, mu, and
    // convergence status.  Skip Hessian, FE recovery, vcov, SE/z/p.
    // The final converged call from fenegbin_fit will use
    // run_from_negbin=false to compute the full result.
    if (run_from_negbin) {
      result.coef_table.col(0) = beta;
      result.coef_status = std::move(collin_result.coef_status);
      result.eta = std::move(eta);
      result.fitted_values = std::move(mu);
      result.weights = w; // w is const ref, can't move
      result.deviance = dev;
      result.null_deviance = null_dev;
      result.conv = true;
      return result;
    }

    // Use the FE-centered design matrix (MX) from the last felm_fit iteration
    // for Hessian and sandwich vcov computation.  In the old IRLS scheme X was
    // centered in-place, so crossprod(X, w_working) was MX'WMX.  Now centering
    // lives inside felm_fit, so we must retrieve MX from the workspace.
    const mat &MX = has_fixed_effects ? felm_workspace.X_centered : X;
    const mat H = crossprod(MX, w_working);

#ifdef CAPYBARA_DEBUG
    auto tfe0 = std::chrono::high_resolution_clock::now();
#endif

    if (has_fixed_effects) {
      // Compute pi = eta - X*beta - offset for FE recovery
      // X has been shed of collinear columns, so its columns match the
      // non-collinear indices. Extract matching beta elements.
      vec x_beta;
      if (collin_result.has_collinearity) {
        x_beta = X * beta.elem(collin_result.non_collinear_cols);
      } else {
        x_beta = X * beta;
      }

      vec pi = eta - x_beta;
      if (has_offset) {
        pi -= offset_vec;
      }

      result.has_fe = true;
      if (params.return_fe) {
        result.fixed_effects =
            get_alpha(pi, fe_map, params.alpha_tol, params.iter_alpha_max);
      }
    }

#ifdef CAPYBARA_DEBUG
    auto tfe1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tfe_duration = tfe1 - tfe0;
    double mem_after_fe = get_memory_usage_mb();
    std::ostringstream msg_tfe;
    msg_tfe << "Fixed effects recovery time: " << tfe_duration.count()
            << " seconds. Memory: " << mem_after_fe << " MB\n";
    cpp4r::message(msg_tfe.str());
#endif

    // Covariance matrix
    if (params.vcov_type == "hetero") {
      // HC0: heteroskedastic-robust, no clustering needed
      const vec resid = y - mu;
      result.vcov = sandwich_vcov_hetero_(MX, resid, H);
    } else if (params.vcov_type == "two-way" && entity1_groups != nullptr &&
               entity2_groups != nullptr) {
      // Two-way cluster (Cameron, Gelbach & Miller 2011): V1 + V2 - V12
      result.vcov =
          sandwich_vcov_twoway_(MX, y, mu, H, *entity1_groups, *entity2_groups);
    } else if (params.vcov_type == "m-estimator-dyadic" &&
               entity1_groups != nullptr && entity2_groups != nullptr) {
      // Dyadic-robust (Cameron & Miller 2014): uses memory-efficient overload
      // that computes scores on-the-fly without N*P allocation
      const vec resid = y - mu;
      result.vcov = sandwich_vcov_mestimator_dyadic_(
          H, MX, resid, *entity1_groups, *entity2_groups);
    } else if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
      if (params.vcov_type == "m-estimator") {
        // Memory-efficient: computes scores on-the-fly
        const vec resid = y - mu;
        result.vcov = sandwich_vcov_mestimator_(H, MX, resid, *cluster_groups);
      } else {
        result.vcov = sandwich_vcov_(MX, y, mu, H, *cluster_groups);
      }
    } else {
      mat H_inv;
      if (!inv_sympd(H_inv, H) && !inv(H_inv, H)) {
        H_inv.set_size(H.n_rows, H.n_cols);
        H_inv.fill(datum::inf);
      }
      result.vcov = std::move(H_inv);
    }

    result.coef_table.col(0) = beta;
    result.coef_status = std::move(collin_result.coef_status);
    result.eta = std::move(eta);
    result.fitted_values = std::move(mu);
    result.weights = w; // w is const ref, can't move
    result.hessian = std::move(H);
    result.deviance = dev;
    result.null_deviance = null_dev;
    result.conv = true;

    // Pseudo R-squared for Poisson
    if (family_type == POISSON) {
      const double corr = as_scalar(cor(y, result.fitted_values));
      result.pseudo_rsq = corr * corr;
    }

    // Build coefficient table
    const uword n_coef = beta.n_elem;
    if (result.coef_table.n_rows != n_coef) {
      result.coef_table.set_size(n_coef, 4);
      result.coef_table.col(0) = beta;
    }

    // Initialize SE/z/p columns with NaN
    result.coef_table.cols(1, 3).fill(datum::nan);

    // Compute SE, z, p for non-collinear coefficients
    const vec se_reduced = sqrt(diagvec(result.vcov));

    if (collin_result.has_collinearity) {
      const uvec &idx = collin_result.non_collinear_cols;
      const vec beta_nc = beta.elem(idx);
      const vec z_vals = beta_nc / se_reduced;
      const vec p_vals = 2.0 * normcdf(-abs(z_vals));

      // Vectorized scatter to indexed rows using submat
      const uvec col_idx = {1, 2, 3};
      mat stats(idx.n_elem, 3);
      stats.col(0) = se_reduced;
      stats.col(1) = z_vals;
      stats.col(2) = p_vals;
      result.coef_table.submat(idx, col_idx) = stats;
    } else {
      const vec z_vals = beta / se_reduced;
      result.coef_table.col(1) = se_reduced;
      result.coef_table.col(2) = z_vals;
      result.coef_table.col(3) = 2.0 * normcdf(-abs(z_vals));
    }

    if (params.keep_tx) {
      result.TX = MX;
      result.has_tx = true;
    }

    // Compute Average Partial Effects for binomial models if requested
    if (params.compute_apes && family_type == BINOMIAL) {
      // Need H_inv for variance computation
      mat H_inv_ape;
      if (!inv_sympd(H_inv_ape, result.hessian)) {
        if (!inv(H_inv_ape, result.hessian)) {
          H_inv_ape.set_size(p, p);
          H_inv_ape.fill(datum::nan);
        }
      }
      compute_apes_binomial(result, X, beta, result.eta, result.fitted_values,
                            w, H_inv_ape, MX, n, fe_map, params);
    }

    // Compute bias correction for binomial models if requested
    if (params.compute_bias_corr && family_type == BINOMIAL) {
      compute_bias_corr_binomial(result, X, beta, result.eta,
                                 result.fitted_values, w, result.hessian, MX, n,
                                 fe_map, params);
    }
  } else {
    // Non-convergence: still populate result vectors for R-side diagnostics
    result.eta = std::move(eta);
    result.fitted_values = std::move(mu);
    result.weights = w;
    result.deviance = dev;
    result.null_deviance = null_dev;
    result.coef_table.col(0) = beta;
    result.coef_status = std::move(collin_result.coef_status);
  }

  return result;
}

// Working weights and adjusted response for offset-only fitting
using OffsetWwYadj = void (*)(vec &w_working, vec &yadj, const vec &w,
                              const vec &mu, const vec &y, const vec &eta,
                              const vec &offset);

inline void offset_ww_yadj_gaussian(vec &w_working, vec &yadj, const vec &w,
                                    const vec &mu, const vec &y, const vec &eta,
                                    const vec &offset) {
  w_working = w;
  yadj = (y - mu) + eta - offset;
}

inline void offset_ww_yadj_poisson(vec &w_working, vec &yadj, const vec &w,
                                   const vec &mu, const vec &y, const vec &eta,
                                   const vec &offset) {
  w_working = w % mu;
  yadj = (y - mu) / mu + eta - offset;
}

inline void offset_ww_yadj_binomial(vec &w_working, vec &yadj, const vec &w,
                                    const vec &mu, const vec &y, const vec &eta,
                                    const vec &offset) {
  const vec var = mu % (1.0 - mu);
  w_working = w % var;
  yadj = (y - mu) / var + eta - offset;
}

inline void offset_ww_yadj_gamma(vec &w_working, vec &yadj, const vec &w,
                                 const vec &mu, const vec &y, const vec &eta,
                                 const vec &offset) {
  const vec m2 = square(mu);
  w_working = w % m2;
  yadj = -(y - mu) / m2 + eta - offset;
}

inline void offset_ww_yadj_invgaussian(vec &w_working, vec &yadj, const vec &w,
                                       const vec &mu, const vec &y,
                                       const vec &eta, const vec &offset) {
  const vec m3 = pow(mu, 3);
  w_working = 0.25 * (w % m3);
  yadj = -2.0 * (y - mu) / m3 + eta - offset;
}

inline OffsetWwYadj get_offset_ww_yadj_fn(Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return offset_ww_yadj_gaussian;
  case POISSON:
  case NEG_BIN:
    return offset_ww_yadj_poisson;
  case BINOMIAL:
    return offset_ww_yadj_binomial;
  case GAMMA:
    return offset_ww_yadj_gamma;
  case INV_GAUSSIAN:
    return offset_ww_yadj_invgaussian;
  default:
    return offset_ww_yadj_gaussian;
  }
}

vec feglm_offset_fit(vec &eta, const vec &y, const vec &offset, const vec &w,
                     const Family family_type, const FlatFEMap &fe_map_in,
                     const CapybaraParameters &params) {
  const uword n = y.n_elem;

  // Get function pointers once
  const MuFromEta mu_ = get_mu_fn(family_type);
  const OffsetWwYadj ww_yadj_ = get_offset_ww_yadj_fn(family_type);

  // Working buffers (fill::none for buffers immediately overwritten)
  vec mu(n, fill::none), w_working(n, fill::none), yadj(n, fill::none),
      eta0(n, fill::none);
  vec Myadj(n, fill::zeros);

  // Initial mu
  mu_(mu, eta);

  double dev = dev_resids(y, mu, 0.0, w, family_type);

  // Adaptive tolerance for large models
  double adaptive_tol = params.center_tol;
  if (n > 100000) {
    adaptive_tol = std::max(params.center_tol, 1e-3);
  }

  // Mutable copy of FE map for weight updates
  FlatFEMap fe_map = fe_map_in;
  CenterWarmStart warm_start;

  // Maximize the log-likelihood
  for (uword iter = 0; iter < params.iter_max; ++iter) {
    double rho = 1.0;
    eta0 = eta;
    const double dev0 = dev;

    // Compute working weights and adjusted response
    ww_yadj_(w_working, yadj, w, mu, y, eta, offset);

    // Only update weights on the persistent FE map
    if (fe_map.K > 0) {
      fe_map.update_weights(w_working);
    }

    Myadj += yadj;

    center_variables(Myadj, w_working, fe_map, adaptive_tol,
                     params.iter_center_max, params.grand_acc_period,
                     &warm_start, centering_from_string(params.centering));

    const vec eta_upd = yadj - Myadj + offset - eta;

    // Step-halving inner loop
    bool dev_crit = false, val_crit = false, imp_crit = false;

    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;
      mu_(mu, eta);

      dev = dev_resids(y, mu, 0.0, w, family_type);
      const double dev_ratio_inner = (dev - dev0) / (0.1 + std::fabs(dev0));

      dev_crit = std::isfinite(dev);
      val_crit = valid_eta(eta, family_type) && valid_mu(mu, family_type);
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }
      rho *= params.step_halving_factor;
    }

    if (!dev_crit || !val_crit) {
      eta = eta0;
      mu_(mu, eta);
      break;
    }

    // Convergence criterion: relative change in eta
    // No betas available (offset-only), so we track eta with a pure
    // relative criterion (epsilon guard instead of a scale-dependent
    // floor of 1).
    const double eta_norm = std::sqrt(dot(eta, eta) / n);
    const double eta_change = std::sqrt(dot(eta - eta0, eta - eta0) / n) /
                              std::max(eta_norm, datum::eps);

    // Relax tolerance after initial iterations for large models
    if (n > 100000 && iter > 5 && eta_change < 0.1) {
      adaptive_tol = params.center_tol;
    }

    if (eta_change < params.dev_tol) {
      break;
    }

    Myadj -= yadj;
  }

  return eta;
}

} // namespace capybara

#endif // CAPYBARA_GLM_H
