// Generalized linear model with fixed effects eta = alpha + X * beta
#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

struct GlmWorkspace {
  mat XtX;
  mat XtX_cache; // Cache for reuse across iterations
  vec XtY;
  mat L;
  vec beta_work;
  vec z_solve;

  // Reusable iteration buffers to avoid reallocations
  vec mu;
  vec mu_eta;
  vec w_working;
  vec nu;
  vec MNU;
  vec MNU_increment;
  vec eta_upd;
  vec eta0;
  vec beta0;
  vec nu0;

  vec y;
  vec alpha0;
  vec group_sums;

  uword cached_n, cached_p;
  bool is_initialized;

  GlmWorkspace() : cached_n(0), cached_p(0), is_initialized(false) {}

  GlmWorkspace(uword n, uword p)
      : cached_n(n), cached_p(p), is_initialized(true) {
    uword safe_n = std::max(n, uword(1));
    uword safe_p = std::max(p, uword(1));

    XtX.set_size(safe_p, safe_p);
    XtX_cache.set_size(safe_p, safe_p);
    XtY.set_size(safe_p);
    L.set_size(safe_p, safe_p);
    beta_work.set_size(safe_p);
    z_solve.set_size(safe_p);

    mu.set_size(safe_n);
    mu_eta.set_size(safe_n);
    w_working.set_size(safe_n);
    nu.set_size(safe_n);
    MNU.set_size(safe_n);
    MNU_increment.set_size(safe_n);
    eta_upd.set_size(safe_n);
    eta0.set_size(safe_n);

    beta0.set_size(safe_p);
    nu0.set_size(safe_n);

    y.set_size(safe_n);
    alpha0.set_size(safe_n);
    group_sums.set_size(safe_n);
  }

  void ensure_size(uword n, uword p) {
    if (!is_initialized || n > cached_n || p > cached_p || mu.n_elem < n ||
        XtX.n_rows < p) {
      uword new_n = std::max(n, cached_n);
      uword new_p = std::max(p, cached_p);

      // Update cached dimensions
      if (n > cached_n)
        new_n = n;
      if (p > cached_p)
        new_p = p;

      if (XtX.n_rows < new_p || XtX.n_cols < new_p)
        XtX.set_size(new_p, new_p);
      if (XtX_cache.n_rows < new_p || XtX_cache.n_cols < new_p)
        XtX_cache.set_size(new_p, new_p);
      if (XtY.n_elem < new_p)
        XtY.set_size(new_p);
      if (L.n_rows < new_p || L.n_cols < new_p)
        L.set_size(new_p, new_p);
      if (beta_work.n_elem < new_p)
        beta_work.set_size(new_p);
      if (z_solve.n_elem < new_p)
        z_solve.set_size(new_p);

      if (mu.n_elem < new_n)
        mu.set_size(new_n);
      if (mu_eta.n_elem < new_n)
        mu_eta.set_size(new_n);
      if (w_working.n_elem < new_n)
        w_working.set_size(new_n);
      if (nu.n_elem < new_n)
        nu.set_size(new_n);
      if (MNU.n_elem < new_n)
        MNU.set_size(new_n);
      if (MNU_increment.n_elem < new_n)
        MNU_increment.set_size(new_n);
      if (eta_upd.n_elem < new_n)
        eta_upd.set_size(new_n);
      if (eta0.n_elem < new_n)
        eta0.set_size(new_n);

      if (beta0.n_elem < new_p)
        beta0.set_size(new_p);
      if (nu0.n_elem < new_n)
        nu0.set_size(new_n);

      if (y.n_elem < new_n)
        y.set_size(new_n);
      if (alpha0.n_elem < new_n)
        alpha0.set_size(new_n);
      if (group_sums.n_elem < new_n)
        group_sums.set_size(new_n);

      cached_n = new_n;
      cached_p = new_p;
      is_initialized = true;
    }
  }

  ~GlmWorkspace() { clear(); }

  void clear() {
    XtX.reset();
    XtX_cache.reset();
    XtY.reset();
    L.reset();
    beta_work.reset();
    z_solve.reset();
    y.reset();
    alpha0.reset();
    group_sums.reset();
    mu.reset();
    mu_eta.reset();
    w_working.reset();
    nu.reset();
    MNU.reset();
    MNU_increment.reset();
    eta_upd.reset();
    eta0.reset();
    beta0.reset();
    nu0.reset();
    cached_n = 0;
    cached_p = 0;
    is_initialized = false;
  }
};

InferenceGLM feglm_fit(vec &beta, vec &eta, const vec &y, mat &X, const vec &w,
                       const double &theta, const Family family_type,
                       const field<field<uvec>> &fe_groups,
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
  const bool has_fixed_effects = fe_groups.n_elem > 0;
  const bool has_offset =
      (offset != nullptr && offset->n_elem == n && any(*offset != 0.0));

  // Add intercept column if no fixed effects
  if (!has_fixed_effects) {
    mat X_with_intercept(n, p_original + 1);
    X_with_intercept.col(0).ones();
    if (p_original > 0) {
      X_with_intercept.cols(1, p_original) = X;
    }
    X = X_with_intercept;

    // Expand beta to include intercept
    beta = join_cols(vec{0.0}, beta);
  }

  const uword p = X.n_cols;
  const uword k = beta.n_elem;

  // Always store original X for later use (needed for both FE and non-FE cases)
  const mat X0 = X;

  InferenceGLM result(n, p);

  if (!y.is_finite() || !X.is_finite()) {
    result.conv = false;
    return result;
  }

  GlmWorkspace local_workspace;
  if (!workspace) {
    workspace = &local_workspace;
  }
  workspace->ensure_size(n, p);

  // Store offset separately if present (for use in separation detection)
  vec offset_vec;
  if (has_offset) {
    offset_vec = *offset;
  }

  // For FE models with Poisson, detect separation early

#ifdef CAPYBARA_DEBUG
  auto tsep0 = std::chrono::high_resolution_clock::now();
#endif

  CenteringWorkspace centering_workspace;

  if (family_type == Family::POISSON && !skip_separation_check &&
      has_fixed_effects && params.check_separation) {
    // Use separation detection on original data
    SeparationParameters sep_params;
    sep_params.tol = params.sep_tol;
    sep_params.max_iter = params.sep_max_iter;
    sep_params.use_relu = true;
    sep_params.use_simplex = true;

    // Use check_separation which handles both simplex and ReLU
    SeparationResult sep_result = check_separation(y, X, w, sep_params);

    if (sep_result.num_separated > 0) {
      // Instead of filtering, set weights to zero for separated observations
      // This keeps dimensions consistent while excluding them from estimation
      vec w_work = w;
      for (uword i = 0; i < sep_result.separated_obs.n_elem; ++i) {
        w_work(sep_result.separated_obs(i)) = 0.0;
      }

      // Call feglm_fit with zero weights for separated obs
      InferenceGLM result_with_sep =
          feglm_fit(beta, eta, y, X, w_work, theta, family_type, fe_groups,
                    params, workspace, cluster_groups, offset,
                    true // skip_separation_check = true
          );

      // Set NA for separated observations in result vectors
      for (uword i = 0; i < sep_result.separated_obs.n_elem; ++i) {
        uword idx = sep_result.separated_obs(i);
        result_with_sep.eta(idx) = datum::nan;
        result_with_sep.fitted_values(idx) = datum::nan;
      }

      // Store separation info
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
#endif

  // Check collinearity once before iterations

#ifdef CAPYBARA_DEBUG
  auto tcoll0 = std::chrono::high_resolution_clock::now();
#endif

  const bool use_weights = !all(w == 1.0);
  CollinearityResult collin_result =
      check_collinearity(X, w, use_weights, params.collin_tol);

#ifdef CAPYBARA_DEBUG
  auto tcoll1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> collin_duration = tcoll1 - tcoll0;
  std::ostringstream collin_msg;
  collin_msg << "Collinearity check time: " << collin_duration.count()
             << " seconds.\n";
  cpp4r::message(collin_msg.str());
#endif

  // After collinearity check, X may have been filtered
  // Update p to reflect the actual working dimension
  const uword p_working = X.n_cols;

  vec MNU(n, fill::zeros);
  vec &mu = workspace->mu;
  vec &w_working = workspace->w_working;
  vec &nu = workspace->nu;
  vec &MNU_increment = workspace->MNU_increment;
  vec &eta_upd = workspace->eta_upd;
  vec &eta0 = workspace->eta0;
  vec &beta0 = workspace->beta0;
  vec &nu0 = workspace->nu0;

  // Initial mu computation
  switch (family_type) {
  case GAUSSIAN:
    mu = eta;
    break;
  case POISSON:
  case NEG_BIN:
    mu = exp(eta);
    break;
  case BINOMIAL:
    mu = 1.0 / (1.0 + exp(-eta));
    break;
  case GAMMA:
    mu = 1.0 / eta;
    break;
  case INV_GAUSSIAN:
    mu = 1.0 / sqrt(eta);
    break;
  default:
    mu = link_inv(eta, family_type);
  }

  const double y_mean_scalar = mean(y);
  vec beta_upd(k, fill::none);

  double dev = dev_resids(y, mu, theta, w, family_type);
  // For null deviance, create ymean vector only once
  vec ymean(n);
  ymean.fill(y_mean_scalar);
  double null_dev = dev_resids(y, ymean, theta, w, family_type);
  double dev0, dev_ratio = datum::inf, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit, conv = false;

  bool is_large_model = (n > 100000) || (p_working > 1000) ||
                        (has_fixed_effects && fe_groups.n_elem > 1);

  double step_halving_memory = params.step_halving_memory;
  uword num_step_halving = 0;
  uword max_step_halving = params.max_step_halving;

  MNU_increment.zeros();
  nu0.zeros();
  workspace->XtX_cache.reset();
  double last_dev_ratio = datum::inf;
  uword convergence_count = 0;

  // Persistent mapping to avoid rebuilding indices
  ObsToGroupMapping group_mapping;
  FelmWorkspace felm_workspace;

  for (uword iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    beta0 = beta;
    dev0 = dev;

    // Compute w_working and nu efficienty
    switch (family_type) {
    case GAUSSIAN: // mu=eta, mu'=1, V=1
      w_working = w;
      nu = y - mu;
      break;
    case POISSON: // mu=exp(eta), mu'=mu, V=mu
      w_working = w % mu;
      nu = (y - mu) / mu;
      break;
    case BINOMIAL: { // mu=ilogit(eta), mu'=mu(1-mu), V=mu(1-mu)
      vec var = mu % (1.0 - mu);
      w_working = w % var;
      nu = (y - mu) / var;
      break;
    }
    case GAMMA: { // mu=1/eta, mu'=-mu^2, V=mu^2
      // W = w * (-mu^2)^2 / mu^2 = w * mu^4 / mu^2 = w * mu^2
      // nu = (y-mu) / (-mu^2)
      vec m2 = square(mu);
      w_working = w % m2;
      nu = -(y - mu) / m2;
      break;
    }
    case INV_GAUSSIAN: { // mu=1/sqrt(eta), mu'=-mu^3/2, V=mu^3
      // W = w * (-mu^3/2)^2 / mu^3 = w * (mu^6/4) / mu^3 = w * mu^3 / 4
      // nu = (y-mu) / (-mu^3/2)
      vec m3 = pow(mu, 3);
      w_working = w % m3 * 0.25;
      nu = -2.0 * (y - mu) / m3;
      break;
    }
    case NEG_BIN: // mu=exp(eta), mu'=mu, V=mu+mu^2/theta
      // W = w * mu^2 / (mu + mu^2/theta) = w * mu / (1 + mu/theta)
      // nu = (y-mu)/mu
      w_working = (w % mu) / (1.0 + mu / theta);
      nu = (y - mu) / mu;
      break;
    default:
      // Fallback to vector ops if unknown family (shouldn't happen)
      {
        vec mu_eta = inverse_link_derivative(eta, family_type);
        vec var_mu = variance(mu, theta, family_type);
        w_working = (w % square(mu_eta)) / var_mu;
        nu = (y - mu) / mu_eta;
      }
    }

    // Compute z = eta + nu
    vec z = eta + nu;

    if (has_offset) {
      z -= offset_vec;
    }

    // Use felm_fit for the weighted least squares step
    // We need a copy of X because felm_fit modifies it (centering)
    // and we need the original X for the next iteration.
    mat X_iter = X;

    InferenceLM lm_res =
        felm_fit(X_iter, z, w_working, fe_groups, params, &felm_workspace,
                 cluster_groups, true);

    // Get new beta (absolute, reduced dimension).
    // Note: felm_fit returns coefficients for columns in X_iter.
    // Since X is already filtered (p_working cols), this matches
    // beta_upd_reduced size.
    vec beta_upd_reduced = lm_res.coef_table.col(0);

    // Compute eta update
    // eta_upd = eta_new - eta0
    // felm_fit returns fitted values = X*beta + alpha
    eta_upd = lm_res.fitted_values - eta0;
    if (has_offset) {
      eta_upd += offset_vec;
    }

    const uword full_p =
        collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;
    if (beta.n_elem != full_p) {
      beta.resize(full_p);
      beta.fill(datum::nan); // Initialize with NaN
    }



    // Step-halving with checks

    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;

      // Update beta with step-halving
      if (collin_result.has_collinearity &&
          collin_result.non_collinear_cols.n_elem > 0) {
        vec beta0_reduced = beta0.elem(collin_result.non_collinear_cols);
        vec beta_step = (1.0 - rho) * beta0_reduced + rho * beta_upd_reduced;

        beta = beta0;
        beta.elem(collin_result.non_collinear_cols) = beta_step;
      } else {
        beta = (1.0 - rho) * beta0 + rho * beta_upd_reduced;
      }

      // Mu update
      switch (family_type) {
      case GAUSSIAN:
        mu = eta;
        break;
      case POISSON:
        mu = exp(eta);
        break;
      case BINOMIAL:
        mu = 1.0 / (1.0 + exp(-eta));
        break;
      case GAMMA:
        mu = 1.0 / eta;
        break;
      case INV_GAUSSIAN:
        mu = 1.0 / sqrt(eta);
        break;
      case NEG_BIN:
        mu = exp(eta);
        break;
      default:
        mu = link_inv(eta, family_type); // Only allocate if unknown family
      }

      dev = dev_resids(y, mu, theta, w, family_type);
      dev_ratio_inner = (dev - dev0) / (0.1 + fabs(dev));

      dev_crit = std::isfinite(dev);
      val_crit = valid_eta(eta, family_type) && valid_mu(mu, family_type);
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= params.step_halving_factor;
    }

    if (!dev_crit || !val_crit) {
      result.conv = false;
      return result;
    }

    if (!imp_crit) {
      eta = eta0;
      beta = beta0;
      dev = dev0;
      // Mu update to be consistent with eta0
      switch (family_type) {
      case GAUSSIAN:
        mu = eta0;
        break;
      case POISSON:
      case NEG_BIN:
        mu = exp(eta0);
        break;
      case BINOMIAL:
        mu = 1.0 / (1.0 + exp(-eta0));
        break;
      case GAMMA:
        mu = 1.0 / eta0;
        break;
      case INV_GAUSSIAN:
        mu = 1.0 / sqrt(eta0);
        break;
      default:
        mu = link_inv(eta0, family_type);
      }
    }

    dev_ratio = fabs(dev - dev0) / (0.1 + fabs(dev));
    double delta_deviance = dev0 - dev;

    if (is_large_model && dev_ratio < last_dev_ratio * 0.5) {
      convergence_count++;
    } else {
      convergence_count = 0;
    }
    last_dev_ratio = dev_ratio;

    // Early convergence check
    if (dev_ratio < params.dev_tol) {
      conv = true;
      break;
    }

    // Additional early stopping: if convergence is very good for 2+ iterations
    if (convergence_count >= 2 && dev_ratio < params.dev_tol * 10) {
      conv = true;
      break;
    }

    if (delta_deviance < 0 && num_step_halving < max_step_halving) {
      eta = step_halving_memory * eta0 + (1.0 - step_halving_memory) * eta;

      if (num_step_halving > 0 && family_type == POISSON) {
        eta.clamp(-10.0, datum::inf);
      }

      // Update mu in-place after blending
      switch (family_type) {
      case GAUSSIAN:
        mu = eta;
        break;
      case POISSON:
        mu = exp(eta);
        break;
      case BINOMIAL:
        mu = 1.0 / (1.0 + exp(-eta));
        break;
      case GAMMA:
        mu = 1.0 / eta;
        break;
      case INV_GAUSSIAN:
        mu = 1.0 / sqrt(eta);
        break;
      case NEG_BIN:
        mu = exp(eta);
        break;
      default:
        mu = link_inv(eta, family_type);
      }

      dev = dev_resids(y, mu, theta, w, family_type);
      num_step_halving++;

      result.iter = iter + 1;
      continue;
    } else {
      num_step_halving = 0;
    }

    result.iter = iter + 1;
  }

  if (conv) {
    mat H = crossprod(X, w_working);

#ifdef CAPYBARA_DEBUG
    auto tfe0 = std::chrono::high_resolution_clock::now();
#endif

    if (has_fixed_effects) {
      // Following alpaca's getFE approach
      vec x_beta(n, fill::zeros);
      if (X0.n_cols > 0) {
        if (collin_result.has_collinearity &&
            collin_result.non_collinear_cols.n_elem > 0) {
          x_beta = X0.cols(collin_result.non_collinear_cols) *
                   beta.elem(collin_result.non_collinear_cols);
        } else {
          x_beta = X0 * beta;
        }
      } else {
        x_beta.zeros(n);
      }

      // Compute pi = eta - X*beta - offset
      // eta includes offset from R (added in capybara.cpp)
      // so we need to subtract offset to get just the fixed effects
      vec pi = eta - x_beta;
      if (has_offset) {
        pi -= offset_vec;
      }

      result.has_fe = true;
      if (params.return_fe) {
        result.fixed_effects =
            get_alpha(pi, fe_groups, params.alpha_tol, params.iter_alpha_max);
      }
    }

#ifdef CAPYBARA_DEBUG
    auto tfe1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tfe_duration = tfe1 - tfe0;
    std::ostringstream msg_tfe;
    msg_tfe << "\nFixed effects recovery time: " << tfe_duration.count()
            << " seconds.\n";
    cpp4r::message(msg_tfe.str());
#endif

    // Compute covariance matrix:
    // - If cluster groups provided: sandwich covariance
    // - Otherwise: inverse Hessian
    // vcov stays at reduced size (excluding collinear variables), same as
    // hessian
    if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
      // Sandwich covariance for clustered standard errors
      result.vcov = compute_sandwich_vcov(X, y, mu, H, *cluster_groups);
    } else {
      // Standard inverse Hessian covariance
      mat H_inv;
      bool success = inv_sympd(H_inv, H);
      if (!success) {
        success = inv(H_inv, H);
        if (!success) {
          H_inv = mat(H.n_rows, H.n_cols, fill::value(datum::inf));
        }
      }
      result.vcov = std::move(H_inv);
    }

    result.coef_table.col(0) = beta; // Store coefficients in first column
    result.coef_status = std::move(collin_result.coef_status);
    // Don't move eta/mu to avoid emptying the workspace buffers or input/output
    // refs
    result.eta = eta;
    result.fitted_values = mu;
    result.weights = w;
    result.hessian = std::move(H);
    result.deviance = dev;
    result.null_deviance = null_dev;
    result.conv = true;

    // Compute pseudo R-squared for Poisson models
    // http://personal.lse.ac.uk/tenreyro/r2.do
    // Pseudo-R^2 = (cor(y, yhat))^2
    if (family_type == POISSON) {
      double corr = as_scalar(cor(y, result.fitted_values));
      result.pseudo_rsq = corr * corr;
    }

    // Coefficients table: [estimate, std.error, z, p-value]
    // Resize coef_table if needed (for collinearity handling)
    uword n_coef = beta.n_elem;
    if (result.coef_table.n_rows != n_coef) {
      result.coef_table.set_size(n_coef, 4);
      result.coef_table.col(0) = beta;
    }

    // Initialize columns with NaN
    result.coef_table.col(1).fill(datum::nan); // Std. Error
    result.coef_table.col(2).fill(datum::nan); // z value
    result.coef_table.col(3).fill(datum::nan); // p-value

    // Compute SE, z, p only for non-collinear coefficients
    // vcov and hessian have reduced dimensions (only non-collinear)
    vec se_reduced = sqrt(diagvec(result.vcov));

    if (collin_result.has_collinearity &&
        collin_result.non_collinear_cols.n_elem > 0) {
      // Map reduced vcov diagonal to full coefficient vector
      uvec idx = collin_result.non_collinear_cols;
      for (uword i = 0; i < idx.n_elem; ++i) {
        uword full_idx = idx(i);
        double se_i = se_reduced(i);
        double z_i = beta(full_idx) / se_i;
        result.coef_table(full_idx, 1) = se_i;
        result.coef_table(full_idx, 2) = z_i;
        result.coef_table(full_idx, 3) = 2.0 * normcdf(-fabs(z_i));
      }
      // Mark collinear coefficients as NaN in estimate column too
      uvec collinear_idx = find(result.coef_status == 0);
      for (uword i = 0; i < collinear_idx.n_elem; ++i) {
        result.coef_table(collinear_idx(i), 0) = datum::nan;
      }
    } else {
      // No collinearity
      vec z_values = beta / se_reduced;
      result.coef_table.col(1) = se_reduced;
      result.coef_table.col(2) = z_values;
      result.coef_table.col(3) = 2.0 * normcdf(-abs(z_values));
    }

    if (params.keep_tx) {
      result.TX = X;
      result.has_tx = true;
    }
  }

  return result;
}

vec feglm_offset_fit(vec &eta, const vec &y, const vec &offset, const vec &w,
                     const Family family_type,
                     const field<field<uvec>> &fe_groups,
                     const CapybaraParameters &params) {

  const uword n = y.n_elem;

  vec Myadj = vec(n, fill::zeros);
  vec mu(n);

  // Initial mu
  switch (family_type) {
  case GAUSSIAN:
    mu = eta;
    break;
  case POISSON:
  case NEG_BIN:
    mu = exp(eta);
    break;
  case BINOMIAL:
    mu = 1.0 / (1.0 + exp(-eta));
    break;
  case GAMMA:
    mu = 1.0 / eta;
    break;
  case INV_GAUSSIAN:
    mu = 1.0 / sqrt(eta);
    break;
  default:
    mu = link_inv(eta, family_type);
  }

  vec yadj(n, fill::none);
  vec w_working(n, fill::none), eta_upd(n, fill::none), eta0(n, fill::none);

  CenteringWorkspace centering_workspace;

  double dev = dev_resids(y, mu, 0.0, w, family_type);
  double dev0, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  double adaptive_tol = params.center_tol;
  if (n > 100000) {
    adaptive_tol = std::max(params.center_tol, 1e-3);
  }

  // Maximize the log-likelihood
  for (uword iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    dev0 = dev;

    // Compute w_working and yadj
    // W = w * (d_mu/d_eta)^2 / V
    // y_adj = (y - mu) / (d_mu/d_eta) + eta - offset
    switch (family_type) {
    case GAUSSIAN: // mu'=1, V=1
      w_working = w;
      yadj = (y - mu) + eta - offset;
      break;
    case POISSON: // mu'=mu, V=mu
      w_working = w % mu;
      yadj = (y - mu) / mu + eta - offset;
      break;
    case BINOMIAL: { // mu'=mu(1-mu), V=mu(1-mu)
      vec var = mu % (1.0 - mu);
      w_working = w % var;
      yadj = (y - mu) / var + eta - offset;
      break;
    }
    case GAMMA: { // mu'=-mu^2, V=mu^2
      vec m2 = square(mu);
      w_working = w % m2;
      yadj = -(y - mu) / m2 + eta - offset;
      break;
    }
    case INV_GAUSSIAN: { // mu'=-mu^3/2, V=mu^3
      vec m3 = pow(mu, 3);
      w_working = w % m3 * 0.25;
      yadj = -2.0 * (y - mu) / m3 + eta - offset;
      break;
    }
    case NEG_BIN: // For offset fit without theta, treat as Poisson-like
      w_working = w % mu;
      yadj = (y - mu) / mu + eta - offset;
      break;
    default: {
      vec mu_eta = inverse_link_derivative(eta, family_type);
      vec var_mu = variance(mu, 0.0, family_type);
      w_working = (w % square(mu_eta)) / var_mu;
      yadj = (y - mu) / mu_eta + eta - offset;
    }
    }

    const ObsToGroupMapping *group_info_ptr = nullptr;
    ObsToGroupMapping group_info;
    if (fe_groups.n_elem > 0) {
      group_info = precompute_group_info(fe_groups, w_working);
      group_info_ptr = &group_info;
    }

    Myadj += yadj;

    center_variables(Myadj, w_working, fe_groups, adaptive_tol,
                     params.iter_center_max, params.iter_interrupt,
                     group_info_ptr, &centering_workspace);

    eta_upd = yadj - Myadj + offset - eta;

    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;

      // Inline mu update
      switch (family_type) {
      case GAUSSIAN:
        mu = eta;
        break;
      case POISSON:
      case NEG_BIN:
        mu = exp(eta);
        break;
      case BINOMIAL:
        mu = 1.0 / (1.0 + exp(-eta));
        break;
      case GAMMA:
        mu = 1.0 / eta;
        break;
      case INV_GAUSSIAN:
        mu = 1.0 / sqrt(eta);
        break;
      default:
        mu = link_inv(eta, family_type);
      }

      dev = dev_resids(y, mu, 0.0, w, family_type);
      dev_ratio_inner = (dev - dev0) / (0.1 + fabs(dev0));

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
      // Restore mu to match eta0
      switch (family_type) {
      case GAUSSIAN:
        mu = eta;
        break;
      case POISSON:
      case NEG_BIN:
        mu = exp(eta);
        break;
      case BINOMIAL:
        mu = 1.0 / (1.0 + exp(-eta));
        break;
      case GAMMA:
        mu = 1.0 / eta;
        break;
      case INV_GAUSSIAN:
        mu = 1.0 / sqrt(eta);
        break;
      default:
        mu = link_inv(eta, family_type);
      }
      break;
    }

    dev_ratio = fabs(dev - dev0) / (0.1 + fabs(dev));

    if (n > 100000 && iter > 5 && dev_ratio < 0.1) {
      adaptive_tol = params.center_tol;
    }

    if (dev_ratio < params.dev_tol) {
      break;
    }

    Myadj = Myadj - yadj;
  }

  return eta;
}

} // namespace capybara

#endif // CAPYBARA_GLM_H
