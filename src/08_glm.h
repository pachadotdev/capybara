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
                              const vec &mu, const vec &y, const vec &eta,
                              double theta);

// Link inverse functions (mu from eta)
inline void mu_gaussian(vec &mu, const vec &eta) { mu = eta; }

inline void mu_poisson(vec &mu, const vec &eta) { mu = exp(eta); }

// Working weights and working residuals (nu) - vectorized
inline void ww_nu_gaussian(vec &w_working, vec &nu, const vec &w, const vec &mu,
                           const vec &y, const vec &, double) {
  w_working = w;
  nu = y - mu;
}

inline void ww_nu_poisson(vec &w_working, vec &nu, const vec &w, const vec &mu,
                          const vec &y, const vec &, double) {
  w_working = w % mu;
  nu = (y - mu) / mu;
}

inline void ww_nu_negbin(vec &w_working, vec &nu, const vec &w, const vec &mu,
                         const vec &y, const vec &, double theta) {
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
  case NEG_BIN:
    return ww_nu_negbin;
  default:
    return ww_nu_gaussian;
  }
}

///////////////////////////////////////////////////////////////////////////
// Separation subsetting: holds subsetted data for separated observations
///////////////////////////////////////////////////////////////////////////

struct SeparationSubset {
  vec y_sub;
  mat X_sub;
  vec w_sub;
  vec eta_sub;
  vec beta_sub;
  vec offset_sub;
  FlatFEMap fe_map_sub;
  field<uvec> cluster_groups_sub;
  uvec keep_idx;
  uword n_orig;
  uword n_kept;
  bool has_offset;
  bool has_cluster_groups;
};

// Subset data for separation handling (no recursive call)
inline SeparationSubset
subset_for_separation(const vec &beta, const vec &eta, const vec &y,
                      const mat &X, const vec &w, const FlatFEMap &fe_map,
                      const field<uvec> *cluster_groups, const vec *offset,
                      const uvec &separated_obs) {

  SeparationSubset sub;
  sub.n_orig = y.n_elem;

  // Validate separated indices are in bounds
  if (separated_obs.n_elem > 0) {
    const uword max_sep = separated_obs.max();
    if (max_sep >= sub.n_orig) {
      cpp4r::stop("Internal error: separated index %u >= n_orig %u",
                  (unsigned)max_sep, (unsigned)sub.n_orig);
    }
  }

  uvec sep_flags(sub.n_orig, fill::zeros);
  sep_flags.elem(separated_obs).ones();
  sub.keep_idx = find(sep_flags == 0);
  sub.n_kept = sub.keep_idx.n_elem;

  if (sub.n_kept == 0) {
    cpp4r::stop("All observations are separated - cannot fit model");
  }

  // Subset all data to non-separated observations
  sub.y_sub = y.elem(sub.keep_idx);
  sub.X_sub = X.rows(sub.keep_idx);
  sub.w_sub = w.elem(sub.keep_idx);
  if (eta.n_elem > 0) {
    sub.eta_sub = eta.elem(sub.keep_idx);
  }
  sub.beta_sub = beta;

  // Subset offset if present
  sub.has_offset = (offset != nullptr && offset->n_elem == sub.n_orig);
  if (sub.has_offset) {
    sub.offset_sub = offset->elem(sub.keep_idx);
  }

  // Subset FE map
  sub.fe_map_sub = fe_map.subset(sub.keep_idx);

  // Remap cluster groups if present
  sub.has_cluster_groups =
      (cluster_groups != nullptr && cluster_groups->n_elem > 0);
  if (sub.has_cluster_groups) {
    // Build old-to-new index mapping
    uvec idx_map(sub.n_orig);
    idx_map.fill(sub.n_orig); // invalid marker
    for (uword i = 0; i < sub.n_kept; ++i) {
      idx_map(sub.keep_idx(i)) = i;
    }

    sub.cluster_groups_sub.set_size(cluster_groups->n_elem);
    for (uword c = 0; c < cluster_groups->n_elem; ++c) {
      const uvec &orig_cluster = (*cluster_groups)(c);
      std::vector<uword> new_idx;
      for (uword j = 0; j < orig_cluster.n_elem; ++j) {
        uword old_i = orig_cluster(j);
        if (idx_map(old_i) < sub.n_orig) {
          new_idx.push_back(idx_map(old_i));
        }
      }
      sub.cluster_groups_sub(c) = uvec(new_idx);
    }
  }

  return sub;
}

// Expand result from subsetted fit back to original indices
inline void expand_separation_result(InferenceGLM &result,
                                     const InferenceGLM &result_sub,
                                     const SeparationSubset &sub,
                                     const uvec &separated_obs,
                                     const vec *separation_support,
                                     const uvec *separated_coefs = nullptr) {
  // Validate result dimensions before mapping back
  if (result_sub.eta.n_elem != sub.n_kept) {
    cpp4r::stop("Internal error: result_sub.eta has %u elements, expected %u",
                (unsigned)result_sub.eta.n_elem, (unsigned)sub.n_kept);
  }
  if (result_sub.fitted_values.n_elem != sub.n_kept) {
    cpp4r::stop("Internal error: result_sub.fitted_values has %u elements, "
                "expected %u",
                (unsigned)result_sub.fitted_values.n_elem,
                (unsigned)sub.n_kept);
  }

  // Copy scalar and matrix results directly
  result.coef_table = result_sub.coef_table;
  result.vcov = result_sub.vcov;
  result.hessian = result_sub.hessian;
  result.deviance = result_sub.deviance;
  result.null_deviance = result_sub.null_deviance;
  result.conv = result_sub.conv;
  result.iter = result_sub.iter;
  result.coef_status = result_sub.coef_status;
  result.r_squared = result_sub.r_squared;

  // Set separated coefficients to -Inf (perfectly predict y=0)
  if (separated_coefs != nullptr && separated_coefs->n_elem > 0) {
    for (uword i = 0; i < separated_coefs->n_elem; ++i) {
      const uword coef_idx = (*separated_coefs)(i);
      if (coef_idx < result.coef_table.n_rows) {
        result.coef_table(coef_idx, 0) = -datum::inf; // Estimate
        result.coef_table(coef_idx, 1) = datum::nan;  // Std. Error
        result.coef_table(coef_idx, 2) = datum::nan;  // z value
        result.coef_table(coef_idx, 3) = datum::nan;  // Pr(>|z|)
      }
    }
  }

  // Expand eta and fitted_values back to original size
  result.eta.set_size(sub.n_orig);
  result.eta.fill(datum::nan);
  result.eta.elem(sub.keep_idx) = result_sub.eta;

  result.fitted_values.set_size(sub.n_orig);
  result.fitted_values.fill(datum::nan);
  result.fitted_values.elem(sub.keep_idx) = result_sub.fitted_values;

  // Copy fixed effects if present
  if (result_sub.fixed_effects.n_elem > 0) {
    result.fixed_effects = result_sub.fixed_effects;
  }
  result.has_fe = result_sub.has_fe;

  // Copy APE results if computed
  if (result_sub.has_apes) {
    result.ape_delta = result_sub.ape_delta;
    result.ape_vcov = result_sub.ape_vcov;
    result.ape_binary = result_sub.ape_binary;
    result.has_apes = true;
  }

  // Copy bias correction results if computed
  if (result_sub.has_bias_corr) {
    result.beta_corrected = result_sub.beta_corrected;
    result.bias_term = result_sub.bias_term;
    result.has_bias_corr = true;
  }

  result.has_separation = true;
  result.separated_obs = separated_obs;
  result.num_separated = separated_obs.n_elem;
  if (separation_support != nullptr && separation_support->n_elem > 0) {
    result.separation_support = *separation_support;
  }
  if (separated_coefs != nullptr && separated_coefs->n_elem > 0) {
    result.separated_coefs = *separated_coefs;
  }
}

InferenceGLM feglm_fit(vec &beta, vec &eta, const vec &y, mat &X, const vec &w,
                       const double &theta, const Family family_type,
                       const FlatFEMap &fe_map,
                       const CapybaraParameters &params,
                       GlmWorkspace *workspace,
                       const field<uvec> *cluster_groups, const vec *offset,
                       bool skip_separation_check,
                       const field<uvec> *entity1_groups,
                       const field<uvec> *entity2_groups, bool run_from_negbin,
                       bool suppress_intercept, bool has_intercept_column) {
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

  // Track whether intercept is now in X (for recursive calls)
  bool intercept_in_X = has_intercept_column;

  // Add intercept column if no fixed effects and intercept not suppressed
  // Skip if intercept was already pre-allocated in X
  // (has_intercept_column=true)
  if (!has_fixed_effects && !suppress_intercept && !has_intercept_column) {
    X.insert_cols(0, 1);
    X.col(0).ones();
    intercept_in_X = true;
    // Ensure beta matches X.n_cols after intercept insertion
    // R may have passed beta with different size due to poly(), factor(), etc.
    if (beta.n_elem != X.n_cols - 1) {
      beta.set_size(X.n_cols - 1);
      beta.zeros();
    }
    beta = join_cols(vec{0.0}, beta);
  } else {
    // For models with FE, suppressed intercept, or pre-allocated intercept:
    // ensure beta matches X.n_cols
    if (beta.n_elem != X.n_cols) {
      beta.set_size(X.n_cols);
      beta.zeros();
    }
  }

  const uword p = X.n_cols;

  // Store original X in the FelmWorkspace (needed for FE recovery after
  // convergence). Skip when called from negbin outer loop - only the final
  // converged call needs FE recovery (run_from_negbin=false).
  // This avoids an upfront N*P copy; instead the workspace owns it.

  // Use lite constructor for fast path (skips P*P hessian/vcov allocation)
  InferenceGLM result(n, p, !run_from_negbin);

  // Workspace setup
  GlmWorkspace local_workspace;
  GlmWorkspace &ws = workspace ? *workspace : local_workspace;
  ws.ensure_size(n, p);

  // Get function pointers once (avoid switch in loop)
  const MuFromEta mu_ = get_mu_fn(family_type);
  const WorkingWtsNu ww_nu_ = get_ww_nu_fn(family_type);

  // Offset handling: use empty static vec to avoid allocation when no offset
  static const vec empty_offset;
  const vec &offset_vec = has_offset ? *offset : empty_offset;

#ifdef CAPYBARA_DEBUG
  auto tsep0 = std::chrono::high_resolution_clock::now();
#endif

  // Group-level separation pre-filter (requires fixed effects)
  // For Poisson/NegBin/Binomial/Probit FE models: drop entire FE groups where
  // mean(y)==0 (Poisson/NegBin) or mean(y) in {0,1} (Binomial/Probit)
  SeparationResult group_sep_result;
#ifdef CAPYBARA_DEBUG
  cpp4r::message("Before separation check - n=%u, X.n_cols=%u, fe_map.K=%u, "
                 "check_sep=%d\n",
                 (unsigned)n, (unsigned)X.n_cols, (unsigned)fe_map.K,
                 (int)params.check_separation);
#endif
  if (!skip_separation_check && has_fixed_effects && params.check_separation &&
      (family_type == POISSON || family_type == NEG_BIN ||
       family_type == BINOMIAL || family_type == PROBIT)) {
#ifdef CAPYBARA_DEBUG
    cpp4r::message("Running group separation check\n");
#endif
    group_sep_result = check_group_separation(y, w, fe_map, family_type);
#ifdef CAPYBARA_DEBUG
    cpp4r::message("group_sep found %u separated obs\n",
                   (unsigned)group_sep_result.num_separated);
#endif
  }

  // Observation-level separation detection (ReLU + Simplex) for Poisson
  // Works with or without fixed effects
  if (family_type == Family::POISSON && !skip_separation_check &&
      params.check_separation) {
#ifdef CAPYBARA_DEBUG
    cpp4r::message("Running Poisson separation check\n");
#endif
    // Use weights with group-separated obs already zeroed
    vec w_for_sep = w;
    if (group_sep_result.num_separated > 0) {
#ifdef CAPYBARA_DEBUG
      cpp4r::message("Zeroing weights for %u group-separated obs, max_idx=%u, "
                     "w.n_elem=%u\n",
                     (unsigned)group_sep_result.num_separated,
                     (unsigned)group_sep_result.separated_obs.max(),
                     (unsigned)w_for_sep.n_elem);
#endif
      w_for_sep.elem(group_sep_result.separated_obs).zeros();
    }

    // Create a copy of fe_map for separation detection (modifies weights
    // internally)
    FlatFEMap fe_map_sep = fe_map;
    SeparationResult sep_result =
        check_separation(y, X, w_for_sep, fe_map_sep, params);

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

      // Subset data, run recursive fit, expand results
      SeparationSubset sub = subset_for_separation(
          beta, eta, y, X, w, fe_map, cluster_groups, offset, all_separated);

      const vec *offset_sub_ptr = sub.has_offset ? &sub.offset_sub : nullptr;
      const field<uvec> *cluster_sub_ptr =
          sub.has_cluster_groups ? &sub.cluster_groups_sub : nullptr;

      InferenceGLM result_sub =
          feglm_fit(sub.beta_sub, sub.eta_sub, sub.y_sub, sub.X_sub, sub.w_sub,
                    theta, family_type, sub.fe_map_sub, params, nullptr,
                    cluster_sub_ptr, offset_sub_ptr, true, nullptr, nullptr,
                    run_from_negbin, suppress_intercept, intercept_in_X);

      InferenceGLM result_with_sep(sub.n_orig, result_sub.coef_table.n_rows,
                                   true);
      const vec *support_ptr =
          sep_result.support.n_elem > 0 ? &sep_result.support : nullptr;
      const uvec *sep_coefs_ptr = sep_result.separated_coefs.n_elem > 0
                                      ? &sep_result.separated_coefs
                                      : nullptr;
      expand_separation_result(result_with_sep, result_sub, sub, all_separated,
                               support_ptr, sep_coefs_ptr);
      return result_with_sep;
    }
  } else if (group_sep_result.num_separated > 0) {
    // Non-Poisson (Binomial, NegBin) with group separation only
    SeparationSubset sub =
        subset_for_separation(beta, eta, y, X, w, fe_map, cluster_groups,
                              offset, group_sep_result.separated_obs);

    const vec *offset_sub_ptr = sub.has_offset ? &sub.offset_sub : nullptr;
    const field<uvec> *cluster_sub_ptr =
        sub.has_cluster_groups ? &sub.cluster_groups_sub : nullptr;

    InferenceGLM result_sub =
        feglm_fit(sub.beta_sub, sub.eta_sub, sub.y_sub, sub.X_sub, sub.w_sub,
                  theta, family_type, sub.fe_map_sub, params, nullptr,
                  cluster_sub_ptr, offset_sub_ptr, true, nullptr, nullptr,
                  run_from_negbin, suppress_intercept, intercept_in_X);

    InferenceGLM result_with_sep(sub.n_orig, result_sub.coef_table.n_rows,
                                 true);
    expand_separation_result(result_with_sep, result_sub, sub,
                             group_sep_result.separated_obs, nullptr);
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
  // (avoids holding P^2 memory through the entire IRLS loop)
  {
    const mat XtX = use_weights ? crossprod(X, w) : crossprod(X);
    mat R_rank;
    uvec excl;
    uword rank; // required output parameter; value not used by caller
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

  // Initialize eta from y if empty (when no eta_start was provided)
  if (eta.n_elem == 0) {
    eta.set_size(n);
    // Use family-appropriate initialization based on y
    switch (family_type) {
    case POISSON:
    case NEG_BIN: {
      // For Poisson/NegBin, use log(mean(y) + 0.1) to handle zeros
      double y_mean = mean(y) + 0.1;
      eta.fill(std::log(y_mean));
      break;
    }
    case TOBIT: {
      // For Tobit, use mean of uncensored observations if available
      double y_mean = mean(y);
      eta.fill(y_mean);
      break;
    }
    case GAUSSIAN:
    default:
      // For gaussian, eta = y mean
      eta.fill(mean(y));
      break;
    }
    // Add offset if present
    if (has_offset) {
      eta += offset_vec;
    }
  }

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
  const double center_tol_loose = params.center_tol * 10.0;
  double adaptive_center_tol = center_tol_loose;

  double conv_change = datum::inf;

  // Persistent felm workspace
  FelmWorkspace felm_workspace;

  // NOTE: We no longer copy X0 here. After shed_cols, X contains exactly
  // the non-collinear columns. For FE recovery, we use X directly with
  // beta.elem(non_collinear_cols) which matches the post-shed column structure.

  // Mu-based separation detection during IRLS (ppmlhdfe style)
  // Only for Poisson-family models when separation checking is enabled
  const bool do_mu_sep = (family_type == POISSON || family_type == NEG_BIN) &&
                         params.check_separation && params.sep_use_mu &&
                         !skip_separation_check;
  const double log_septol =
      do_mu_sep ? std::log(params.sep_mu_tol) : -datum::inf;
  Col<unsigned char> mu_sep_mask; // byte mask for memory efficiency
  uvec zero_sample;               // indices where y == 0
  if (do_mu_sep) {
    zero_sample = find(y == 0);
    mu_sep_mask.zeros(n);
  }

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

    ww_nu_(w_working, nu, w, mu, y, eta, theta);

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

    // Zero working weights for observations already marked as separated
    // This ensures separated obs don't contribute to coefficient estimation
    // (following ppmlhdfe: mu=0 for separated obs means w=0 in next iteration)
    if (do_mu_sep && iter > 0) {
      double *ww_ptr = w_working.memptr();
      const unsigned char *mask_ptr = mu_sep_mask.memptr();
      for (uword i = 0; i < n; ++i) {
        if (mask_ptr[i]) {
          ww_ptr[i] = 0.0;
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

// Check dimensions before step-halving
#ifdef CAPYBARA_DEBUG
    cpp4r::message("iteration=%u, beta.n_elem=%u, beta0.n_elem=%u, "
                   "beta_upd_reduced.n_elem=%u, "
                   "has_collinearity=%d, non_collinear_cols.n_elem=%u, "
                   "X.n_cols=%u, p=%u, full_p=%u\n",
                   (unsigned)iter, (unsigned)beta.n_elem,
                   (unsigned)beta0.n_elem, (unsigned)beta_upd_reduced.n_elem,
                   (int)collin_result.has_collinearity,
                   (unsigned)collin_result.non_collinear_cols.n_elem,
                   (unsigned)X.n_cols, (unsigned)p, (unsigned)full_p);
#endif
    if (collin_result.has_collinearity &&
        collin_result.non_collinear_cols.n_elem > 0) {
#ifdef CAPYBARA_DEBUG
      cpp4r::message("non_collinear_cols max=%u\n",
                     (unsigned)collin_result.non_collinear_cols.max());
#endif
    }

    // Step-halving inner loop
    bool dev_crit = false, val_crit = false, imp_crit = false;

    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;

      // Update beta with step-halving
      if (collin_result.has_collinearity) {
        const uvec &idx = collin_result.non_collinear_cols;
        // Validate indices before elem() calls
        if (idx.n_elem > 0) {
          uword max_idx = idx.max();
          if (max_idx >= beta.n_elem) {
            cpp4r::stop("Index out of bounds: max_idx=%u >= beta.n_elem=%u",
                        (unsigned)max_idx, (unsigned)beta.n_elem);
          }
          if (max_idx >= beta0.n_elem) {
            cpp4r::stop("Index out of bounds: max_idx=%u >= beta0.n_elem=%u",
                        (unsigned)max_idx, (unsigned)beta0.n_elem);
          }
          if (idx.n_elem != beta_upd_reduced.n_elem) {
            cpp4r::stop(
                "Size mismatch: idx.n_elem=%u != beta_upd_reduced.n_elem=%u",
                (unsigned)idx.n_elem, (unsigned)beta_upd_reduced.n_elem);
          }
        }
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

    // Mu-based separation detection during IRLS (ppmlhdfe style)
    // Check if eta is below threshold for observations with y=0
    if (do_mu_sep && zero_sample.n_elem > 0) {
      // Adjusted tolerance based on minimum eta when y > 0
      // Following ppmlhdfe: adjusted_log_septol = log_septol +
      // min(min(eta[y>0])
      // + 5, 0)
      const uvec pos_sample = find(y > 0);
      double adjusted_log_septol = log_septol;
      if (pos_sample.n_elem > 0) {
        const double min_eta_positive = min(eta.elem(pos_sample));
        adjusted_log_septol += std::min(min_eta_positive + 5.0, 0.0);
      }

      // Mark separated observations: eta <= adjusted_log_septol AND y == 0
      // Use OR to accumulate across iterations
      const double *eta_ptr = eta.memptr();
      double *mu_ptr = mu.memptr();
      unsigned char *mask_ptr = mu_sep_mask.memptr();
      for (uword i = 0; i < zero_sample.n_elem; ++i) {
        const uword idx = zero_sample(i);
        if (eta_ptr[idx] <= adjusted_log_septol) {
          mask_ptr[idx] = 1;
          mu_ptr[idx] = 0.0; // Set mu to 0 for separated observations
        }
      }
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
          center_tol_loose * std::pow(params.center_tol / center_tol_loose, t);
    }

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

    // Hybrid convergence criterion for cross-platform stability.
    // On Mac/ARM with FMA, dot() and sqrt() can round differently, causing
    // convergence to oscillate at the threshold. We use a hybrid criterion:
    // threshold = max(absolute_floor, relative_to_tolerance)
    // Since conv_change is already relative (normalized by ||beta||),
    // we compare it against dev_tol * (1 + platform_buffer)
    const double abs_tol_floor_glm = 1e-12;
    const double platform_buffer =
        1e-9; // Tolerance for FMA rounding differences
    const double conv_threshold =
        std::max(abs_tol_floor_glm, params.dev_tol * (1.0 + platform_buffer));
    if (conv_change < conv_threshold) {
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

  // Post-IRLS handling of mu-based separated observations
  // Following ppmlhdfe: if separation found during IRLS, subset data and re-fit
  if (do_mu_sep) {
    const uvec irls_sep_obs = find(mu_sep_mask);
    if (irls_sep_obs.n_elem > 0) {
#ifdef CAPYBARA_DEBUG
      cpp4r::message("IRLS detected %u mu-based separated observations\n",
                     (unsigned)irls_sep_obs.n_elem);
#endif
      // Subset data and recursively fit without separated observations
      SeparationSubset sub = subset_for_separation(
          beta, eta, y, X, w, fe_map, cluster_groups, offset, irls_sep_obs);

      const vec *offset_sub_ptr = sub.has_offset ? &sub.offset_sub : nullptr;
      const field<uvec> *cluster_sub_ptr =
          sub.has_cluster_groups ? &sub.cluster_groups_sub : nullptr;

      // Re-fit with subsetted data, skipping separation check (we already
      // handled it)
      InferenceGLM result_sub =
          feglm_fit(sub.beta_sub, sub.eta_sub, sub.y_sub, sub.X_sub, sub.w_sub,
                    theta, family_type, sub.fe_map_sub, params, nullptr,
                    cluster_sub_ptr, offset_sub_ptr, true, nullptr, nullptr,
                    run_from_negbin, suppress_intercept, intercept_in_X);

      InferenceGLM result_with_sep(sub.n_orig, result_sub.coef_table.n_rows,
                                   true);
      expand_separation_result(result_with_sep, result_sub, sub, irls_sep_obs,
                               nullptr);
      return result_with_sep;
    }
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
      result.r_squared = corr * corr;
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

inline OffsetWwYadj get_offset_ww_yadj_fn(Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return offset_ww_yadj_gaussian;
  case POISSON:
  case NEG_BIN:
    return offset_ww_yadj_poisson;
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

    // Hybrid convergence criterion for offset-only fitting (cross-platform).
    // Same approach as main GLM loop: hybrid absolute + relative tolerance
    // for robust handling of FMA rounding differences on Mac/ARM.
    const double abs_tol_floor_offset = 1e-12;
    const double platform_buffer_offset = 1e-9;
    const double conv_threshold_offset = std::max(
        abs_tol_floor_offset, params.dev_tol * (1.0 + platform_buffer_offset));
    if (eta_change < conv_threshold_offset) {
      break;
    }

    Myadj -= yadj;
  }

  return eta;
}

} // namespace capybara

#endif // CAPYBARA_GLM_H
