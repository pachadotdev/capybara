// Generalized linear model with fixed effects eta = alpha + X * beta
#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

struct GlmWorkspace {
  vec mu;        // fitted values on response scale
  vec w_working; // working weights
  vec nu;        // working residuals
  vec eta0;      // previous eta (for step-halving)
  vec beta0;     // previous beta (for step-halving)

  uword cached_n, cached_p;

  GlmWorkspace() : cached_n(0), cached_p(0) {}

  void ensure_size(uword n, uword p) {
    // Only reallocate if needed (Armadillo handles this efficiently)
    if (n > cached_n) {
      mu.set_size(n);
      w_working.set_size(n);
      nu.set_size(n);
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
using MuFromEtaFn = void (*)(vec &mu, const vec &eta);
using WorkingWtsNuFn = void (*)(vec &w_working, vec &nu, const vec &w,
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
inline MuFromEtaFn get_mu_fn(Family family_type) {
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

inline WorkingWtsNuFn get_ww_nu_fn(Family family_type) {
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

InferenceGLM feglm_fit(vec &beta, vec &eta, const vec &y, mat &X, const vec &w,
                       const double &theta, const Family family_type,
                       const FlatFEMap &fe_map,
                       const CapybaraParameters &params,
                       GlmWorkspace *workspace = nullptr,
                       const field<uvec> *cluster_groups = nullptr,
                       const vec *offset = nullptr,
                       bool skip_separation_check = false) {
#ifdef CAPYBARA_DEBUG
  std::ostringstream feglm_msg;
  feglm_msg << "/////////////////////////////////\n"
               "// Entering feglm_fit function //\n"
               "/////////////////////////////////\n";
  cpp4r::message(feglm_msg.str());
#endif

  const uword n = y.n_elem;
  const uword p_original = X.n_cols;
  const bool has_fixed_effects = fe_map.K > 0;
  const bool has_offset =
      (offset != nullptr && offset->n_elem == n && any(*offset != 0.0));

  // Add intercept column if no fixed effects
  if (!has_fixed_effects) {
    mat X_with_intercept(n, p_original + 1);
    X_with_intercept.col(0).ones();
    if (p_original > 0) {
      X_with_intercept.cols(1, p_original) = X;
    }
    X = std::move(X_with_intercept);
    beta = join_cols(vec{0.0}, beta);
  }

  const uword p = X.n_cols;

  // Store original X once (needed for FE recovery)
  const mat X0 = X;

  InferenceGLM result(n, p);

  if (!y.is_finite() || !X.is_finite()) {
    result.conv = false;
    return result;
  }

  // Workspace setup
  GlmWorkspace local_workspace;
  GlmWorkspace &ws = workspace ? *workspace : local_workspace;
  ws.ensure_size(n, p);

  // Get function pointers once (avoid switch in loop)
  const MuFromEtaFn compute_mu = get_mu_fn(family_type);
  const WorkingWtsNuFn compute_ww_nu = get_ww_nu_fn(family_type);

  // Offset handling
  const vec offset_vec = has_offset ? *offset : vec();

#ifdef CAPYBARA_DEBUG
  auto tsep0 = std::chrono::high_resolution_clock::now();
#endif

  // Separation detection for Poisson FE models
  if (family_type == Family::POISSON && !skip_separation_check &&
      has_fixed_effects && params.check_separation) {
    SeparationResult sep_result = check_separation(y, X, w, params);

    if (sep_result.num_separated > 0) {
      // Zero weights for separated obs (keeps dimensions consistent)
      vec w_work = w;
      w_work.elem(sep_result.separated_obs).zeros();

      InferenceGLM result_with_sep =
          feglm_fit(beta, eta, y, X, w_work, theta, family_type, fe_map, params,
                    &ws, cluster_groups, offset, true);

      // Mark separated observations
      result_with_sep.eta.elem(sep_result.separated_obs).fill(datum::nan);
      result_with_sep.fitted_values.elem(sep_result.separated_obs)
          .fill(datum::nan);
      result_with_sep.has_separation = true;
      result_with_sep.separated_obs = sep_result.separated_obs;
      result_with_sep.num_separated = sep_result.num_separated;
      result_with_sep.separation_support = sep_result.support;

      return result_with_sep;
    }
  }

#ifdef CAPYBARA_DEBUG
  auto tsep1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sep_duration = tsep1 - tsep0;
  std::ostringstream sep_msg;
  sep_msg << "Separation detection time: " << sep_duration.count()
          << " seconds.\n";
  cpp4r::message(sep_msg.str());
  auto tcoll0 = std::chrono::high_resolution_clock::now();
#endif

  // Collinearity check (once before iterations)
  const bool use_weights = !all(w == 1.0);
  const mat XtX = use_weights ? crossprod(X, w) : crossprod(X);

  mat R_rank;
  uvec excl;
  uword rank;
  chol_rank(R_rank, excl, rank, XtX, "upper", params.collin_tol);

  CollinearityResult collin_result(X.n_cols);
  if (any(excl)) {
    collin_result.has_collinearity = true;
    collin_result.non_collinear_cols = find(excl == 0);
    collin_result.collinear_cols = find(excl != 0);
    collin_result.coef_status = 1 - excl;
    X.shed_cols(collin_result.collinear_cols);
  } else {
    collin_result.has_collinearity = false;
    collin_result.non_collinear_cols = regspace<uvec>(0, X.n_cols - 1);
    collin_result.coef_status.ones();
  }

#ifdef CAPYBARA_DEBUG
  auto tcoll1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> collin_duration = tcoll1 - tcoll0;
  std::ostringstream collin_msg;
  collin_msg << "Collinearity check time: " << collin_duration.count()
             << " seconds.\n";
  cpp4r::message(collin_msg.str());
#endif

  const uword p_working = X.n_cols;

  // Workspace references
  vec &mu = ws.mu;
  vec &w_working = ws.w_working;
  vec &nu = ws.nu;
  vec &eta0 = ws.eta0;
  vec &beta0 = ws.beta0;

  // Initial mu from eta
  compute_mu(mu, eta);

  // Deviance computations
  const double y_mean_scalar = mean(y);
  const vec ymean(n, fill::value(y_mean_scalar));
  double dev = dev_resids(y, mu, theta, w, family_type);
  const double null_dev = dev_resids(y, ymean, theta, w, family_type);

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

  // Convergence acceleration for large models
  const bool is_large_model =
      (n > 100000) || (p_working > 1000) || (has_fixed_effects && fe_map.K > 1);
  double last_eta_change = datum::inf;
  uword convergence_count = 0;

  // Persistent felm workspace
  FelmWorkspace felm_workspace;

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

    compute_ww_nu(w_working, nu, w, mu, y, theta);

    // Working response z = eta + nu - offset
    vec z = eta + nu;
    if (has_offset) {
      z -= offset_vec;
    }

    // Guard against non-finite working weights/response from mu
    // overflow or division-by-zero (e.g., exp(eta) = Inf for Poisson).
    // Zero the weight for affected observations so they don't poison
    // the cross-product X'WX that feeds into the Cholesky solver.
    {
      uvec bad_w = find_nonfinite(w_working);
      uvec bad_z = find_nonfinite(z);
      uvec bad = (bad_w.n_elem > 0 && bad_z.n_elem > 0)
                     ? unique(join_cols(bad_w, bad_z))
                     : (bad_w.n_elem > 0 ? bad_w : bad_z);
      if (bad.n_elem > 0) {
        w_working.elem(bad).zeros();
        z.elem(bad).zeros();
      }
    }

#ifdef CAPYBARA_DEBUG
    auto twwnu1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wwnu_duration = twwnu1 - twwnu0;
    std::ostringstream wwnu_msg;
    wwnu_msg << "Working weights and nu time: " << wwnu_duration.count()
             << " seconds.\n";
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
      compute_mu(mu, eta);

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
      return result;
    }

    if (!imp_crit) {
      eta = eta0;
      beta = beta0;
      dev = dev0;
      compute_mu(mu, eta0);
    }

    const double delta_deviance = dev0 - dev;

    // Scale-invariant convergence criterion: relative change in eta
    // Green & Santos Silva 2025: deviance is scale-dependent for PPML,
    // so convergence should be based on the linear predictor)
    const double eta_norm = std::sqrt(dot(eta, eta) / n);
    const double eta_change =
        std::sqrt(dot(eta - eta0, eta - eta0) / n) / std::max(eta_norm, 1.0);

    // Update adaptive centering tolerance based on convergence
    if (eta_change < 0.1) {
      const double t = std::max(0.0, std::min(1.0, (0.1 - eta_change) / 0.1));
      adaptive_center_tol =
          center_tol_loose * std::pow(center_tol_tight / center_tol_loose, t);
    }

    // Early convergence detection for large models
    if (is_large_model && eta_change < last_eta_change * 0.5) {
      ++convergence_count;
    } else {
      convergence_count = 0;
    }
    last_eta_change = eta_change;

    if (eta_change < params.dev_tol ||
        (convergence_count >= 2 && eta_change < params.dev_tol * 10)) {
      conv = true;
      break;
    }

    // Additional step-halving for deviance increase
    if (delta_deviance < 0 && num_step_halving < params.max_step_halving) {
      eta = step_halving_memory * eta0 + (1.0 - step_halving_memory) * eta;
      if (num_step_halving > 0 && family_type == POISSON) {
        eta = clamp(eta, -10.0, datum::inf);
      }
      compute_mu(mu, eta);
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
  std::ostringstream glmiter_msg;
  glmiter_msg << "GLM iteration time: " << glmiter_duration.count()
              << " seconds.\n";
  cpp4r::message(glmiter_msg.str());
#endif

  if (conv) {
    const mat H = crossprod(X, w_working);

#ifdef CAPYBARA_DEBUG
    auto tfe0 = std::chrono::high_resolution_clock::now();
#endif

    if (has_fixed_effects) {
      // Compute pi = eta - X*beta - offset for FE recovery
      vec x_beta;
      if (collin_result.has_collinearity) {
        x_beta = X0.cols(collin_result.non_collinear_cols) *
                 beta.elem(collin_result.non_collinear_cols);
      } else {
        x_beta = X0 * beta;
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
    std::ostringstream msg_tfe;
    msg_tfe << "Fixed effects recovery time: " << tfe_duration.count()
            << " seconds.\n";
    cpp4r::message(msg_tfe.str());
#endif

    // Covariance matrix
    if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
      result.vcov = compute_sandwich_vcov(X, y, mu, H, *cluster_groups);
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
    result.eta = eta;
    result.fitted_values = mu;
    result.weights = w;
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
      result.TX = X;
      result.has_tx = true;
    }
  }

  return result;
}

// Working weights and adjusted response for offset-only fitting
using OffsetWwYadjFn = void (*)(vec &w_working, vec &yadj, const vec &w,
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

inline OffsetWwYadjFn get_offset_ww_yadj_fn(Family family_type) {
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
  const MuFromEtaFn compute_mu = get_mu_fn(family_type);
  const OffsetWwYadjFn compute_ww_yadj = get_offset_ww_yadj_fn(family_type);

  // Working buffers
  vec mu(n), w_working(n), yadj(n), eta0(n);
  vec Myadj(n, fill::zeros);

  // Initial mu
  compute_mu(mu, eta);

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
    compute_ww_yadj(w_working, yadj, w, mu, y, eta, offset);

    // Only update weights on the persistent FE map
    if (fe_map.K > 0) {
      fe_map.update_weights(w_working);
    }

    Myadj += yadj;

    center_variables(Myadj, w_working, fe_map, adaptive_tol,
                     params.iter_center_max, params.grand_acc_period,
                     &warm_start);

    const vec eta_upd = yadj - Myadj + offset - eta;

    // Step-halving inner loop
    bool dev_crit = false, val_crit = false, imp_crit = false;

    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;
      compute_mu(mu, eta);

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
      compute_mu(mu, eta);
      break;
    }

    // Scale-invariant convergence criterion: relative change in eta
    // (Green & Santos Silva 2025)
    const double eta_norm = std::sqrt(dot(eta, eta) / n);
    const double eta_change =
        std::sqrt(dot(eta - eta0, eta - eta0) / n) / std::max(eta_norm, 1.0);

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
