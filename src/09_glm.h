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
    // Only reallocate if needed
    // Use zeros() instead of set_size() to ensure deterministic initialization
    // (prevents non-determinism on Mac due to uninitialized memory)
    if (n > cached_n) {
      mu.zeros(n);
      w_working.zeros(n);
      nu.zeros(n);
      z.zeros(n);
      eta0.zeros(n);
      cached_n = n;
    }
    if (p > cached_p) {
      beta0.zeros(p);
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

    // APES and bias correction for binomial models
    if (family_type == BINOMIAL && has_fixed_effects &&
        (params.compute_bias_corr || params.compute_apes)) {

      const uword K = fe_map.K;
      const bool valid_classic =
          (params.panel_structure == "classic" && (K == 1 || K == 2));
      const bool valid_network =
          (params.panel_structure == "network" && (K == 2 || K == 3));

      if (valid_classic || valid_network) {
        // Compute finite population adjustment
        double adj = 0.0;
        if (params.apes_n_pop > 0 && params.apes_n_pop > n) {
          adj = static_cast<double>(params.apes_n_pop - n) /
                static_cast<double>(params.apes_n_pop - 1);
        }

        // Get original X matrix for APES (need non-centered version)
        // MX is the centered version; for binary variable detection we need X
        // But X has been modified (intercept added, cols shed), so we use MX
        // and note that binary detection works on MX too

        bool weak_exo = (params.bias_corr_l > 0);

        // Bias correction first (if requested)
        if (params.compute_bias_corr) {
          BiasResult bias_res = compute_bias_corr(
              y, MX, MX, result.eta, w, H, "logit", fe_map,
              params.panel_structure, params.bias_corr_l, params.center_tol,
              params.iter_center_max, params.grand_acc_period);

          if (bias_res.success) {
            // Store uncorrected coefficients
            result.beta_uncorrected = beta;

            // Apply bias correction: beta_corr = beta - H^{-1} * bias_term
            vec beta_correction;
            if (solve(beta_correction, H / static_cast<double>(n),
                      bias_res.bias_term)) {
              vec beta_corrected = beta - beta_correction;

              // Update coefficient table with corrected values
              result.coef_table.col(0) = beta_corrected;

              // Recompute eta with corrected beta
              vec eta_corrected = MX * beta_corrected;
              if (has_offset) {
                eta_corrected += offset_vec;
              }
              result.eta = std::move(eta_corrected);

              // Update fitted values
              mu_(result.fitted_values, result.eta);

              result.bias_corr_term = bias_res.bias_term;
              result.has_bias_corr = true;
              result.bias_corr_panel_structure = params.panel_structure;
              result.bias_corr_bandwidth = params.bias_corr_l;
            }
          }
        }

        // APES computation
        if (params.compute_apes) {
          // Use potentially bias-corrected coefficients
          vec beta_for_apes = result.coef_table.col(0);

          APESResult apes_res = compute_apes(
              y, MX, MX, result.eta, w, beta_for_apes, H, "logit", fe_map,
              n, // n_full = n (after separation)
              params.panel_structure, params.apes_sampling_fe, weak_exo, adj,
              params.bias_corr_l, params.compute_bias_corr, params.center_tol,
              params.iter_center_max, params.grand_acc_period);

          if (apes_res.success) {
            result.apes_delta = apes_res.delta;
            result.apes_vcov = apes_res.vcov;
            result.apes_bias_term = apes_res.bias_term;
            result.apes_panel_structure = apes_res.panel_structure;
            result.apes_sampling_fe = apes_res.sampling_fe;
            result.apes_weak_exo = apes_res.weak_exo;
            result.apes_bandwidth = apes_res.bandwidth;
            result.has_apes = true;
          }
        }
      }
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
