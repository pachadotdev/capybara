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
    if (!is_initialized || n > cached_n || p > cached_p) {
      uword new_n = std::max(n, cached_n);
      uword new_p = std::max(p, cached_p);

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
                       const field<uvec> *cluster_groups = nullptr) {
  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const uword k = beta.n_elem;
  const bool has_fixed_effects = fe_groups.n_elem > 0;

  mat X0;
  if (has_fixed_effects) {
    X0 = X; // Keep original design only when fixed effects are present
  }

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

  CenteringWorkspace centering_workspace;

  vec MNU = vec(n, fill::zeros);
  vec &mu = workspace->mu;
  vec &mu_eta = workspace->mu_eta;
  vec &w_working = workspace->w_working;
  vec &nu = workspace->nu;
  vec &MNU_increment = workspace->MNU_increment;
  vec &eta_upd = workspace->eta_upd;
  vec &eta0 = workspace->eta0;
  vec &beta0 = workspace->beta0;
  vec &nu0 = workspace->nu0;

  mu = link_inv(eta, family_type);
  vec ymean = mean(y) * vec(n, fill::ones);
  vec beta_upd(k, fill::none);
  mat H(p, p, fill::none);

  double dev = dev_resids(y, mu, theta, w, family_type);
  double null_dev = dev_resids(y, ymean, theta, w, family_type);
  double dev0, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit, conv = false;

  CollinearityResult collin_result =
      check_collinearity(X, w, /*use_weights =*/true, params.collin_tol);

  // Adaptive tolerance variables -> more aggressive for large models
  double hdfe_tolerance = params.center_tol;
  double highest_inner_tol =
      std::max(1e-12, std::min(params.center_tol, 0.1 * params.dev_tol));
  double start_inner_tol = params.start_inner_tol;

  bool is_large_model =
      (n > 100000) || (p > 1000) || (has_fixed_effects && fe_groups.n_elem > 1);

  if (has_fixed_effects) {
    if (is_large_model) {
      hdfe_tolerance =
          std::max(1e-3, std::max(start_inner_tol, params.dev_tol * 10));
    } else {
      hdfe_tolerance = std::max(start_inner_tol, params.dev_tol);
    }
  }

  bool use_partial = false;

  double step_halving_memory = params.step_halving_memory;
  uword num_step_halving = 0;
  uword max_step_halving = params.max_step_halving;

  vec eps_history(3, fill::value(datum::inf));
  double predicted_eps = datum::inf;

  MNU_increment.zeros();
  nu0.zeros();

  uword convergence_count = 0;
  double last_dev_ratio = datum::inf;

  for (uword iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    beta0 = beta;
    dev0 = dev;

    mu_eta = inverse_link_derivative(eta, family_type);
    w_working = (w % square(mu_eta)) / variance(mu, theta, family_type);
    nu = (y - mu) / mu_eta;

    const field<field<GroupInfo>> *group_info_ptr = nullptr;
    field<field<GroupInfo>> group_info;
    if (has_fixed_effects) {
      group_info = precompute_group_info(fe_groups, w_working);
      group_info_ptr = &group_info;
    }

    bool iter_solver = has_fixed_effects &&
                       (hdfe_tolerance > params.dev_tol * 11) &&
                       (predicted_eps > params.dev_tol);

    double current_hdfe_tol;
    uword current_max_iter;

    // Lower limits: Irons-Tuck + Grand accel converges faster than CG
    if (is_large_model) {
      if (iter < 5) {
        current_hdfe_tol = std::max(hdfe_tolerance, 1e-2);
        current_max_iter = std::min((uword)params.iter_center_max, uword(100));
      } else if (iter_solver) {
        current_hdfe_tol = std::min(hdfe_tolerance, 1e-3);
        current_max_iter = std::min((uword)params.iter_center_max, uword(200));
      } else {
        current_hdfe_tol = std::min(hdfe_tolerance, highest_inner_tol);
        current_max_iter = std::min((uword)params.iter_center_max, uword(300));
      }
    } else {
      if (iter < 3) {
        current_max_iter = std::min((uword)params.iter_center_max, uword(50));
      } else if (dev_ratio > 0.001) {
        current_max_iter = std::min((uword)params.iter_center_max, uword(150));
      } else {
        current_max_iter = std::min((uword)params.iter_center_max, uword(200));
      }
      current_hdfe_tol = iter_solver
                             ? hdfe_tolerance
                             : std::min(hdfe_tolerance, highest_inner_tol);
    }

    if (use_partial && iter > 1) {
      MNU_increment = nu - nu0;
      MNU += MNU_increment;

      if (has_fixed_effects) {
        center_variables(MNU, w_working, fe_groups, current_hdfe_tol,
                         current_max_iter, params.iter_interrupt,
                         group_info_ptr, &centering_workspace);
        center_variables(X, w_working, fe_groups, current_hdfe_tol,
                         current_max_iter, params.iter_interrupt,
                         group_info_ptr, &centering_workspace);

        // invalidate cache
        workspace->XtX_cache.reset();
      }
      nu0 = nu;
    } else {
      MNU += (nu - nu0);
      nu0 = nu;

      if (has_fixed_effects) {
        center_variables(MNU, w_working, fe_groups, current_hdfe_tol,
                         current_max_iter, params.iter_interrupt,
                         group_info_ptr, &centering_workspace);
        center_variables(X, w_working, fe_groups, current_hdfe_tol,
                         current_max_iter, params.iter_interrupt,
                         group_info_ptr, &centering_workspace);

        // invalidate cache
        workspace->XtX_cache.reset();
      }
    }

    // Enable partial out after first iteration
    use_partial = true;

    InferenceBeta beta_result = get_beta(X, MNU, MNU, w_working, collin_result,
                                         false, false, &workspace->XtX_cache);

    // Handle collinearity
    vec beta_upd_reduced;
    if (collin_result.has_collinearity &&
        collin_result.non_collinear_cols.n_elem > 0) {
      beta_upd_reduced =
          beta_result.coefficients.elem(collin_result.non_collinear_cols);
    } else {
      beta_upd_reduced = beta_result.coefficients;
    }

    const uword full_p =
        collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;
    if (beta.n_elem != full_p) {
      beta.resize(full_p);
    }

    if (X.n_cols > 0) {
      eta_upd = X * beta_upd_reduced + nu - MNU;
    } else {
      eta_upd = nu - MNU;
    }

    // Step-halving with checks
    for (uword iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;

      vec beta_new = beta0;
      if (collin_result.has_collinearity &&
          collin_result.non_collinear_cols.n_elem > 0) {
        vec beta0_reduced = beta0.elem(collin_result.non_collinear_cols);
        vec beta_upd_step = beta0_reduced + rho * beta_upd_reduced;
        beta_new.elem(collin_result.non_collinear_cols) = beta_upd_step;
      } else {
        beta_new = beta0 + rho * beta_upd_reduced;
      }
      beta = beta_new;

      mu = link_inv(eta, family_type);
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
      mu = link_inv(eta, family_type);
    }

    dev_ratio = fabs(dev - dev0) / (0.1 + fabs(dev));
    double delta_deviance = dev0 - dev;

    if (is_large_model && dev_ratio < last_dev_ratio * 0.5) {
      convergence_count++;
    } else {
      convergence_count = 0;
    }
    last_dev_ratio = dev_ratio;

    eps_history(0) = eps_history(1);
    eps_history(1) = eps_history(2);
    eps_history(2) = dev_ratio;

    predicted_eps = predict_convergence(eps_history, dev_ratio);

    if (dev_ratio < params.dev_tol) {
      conv = true;
      break;
    }

    if (delta_deviance < 0 && num_step_halving < max_step_halving) {
      eta = step_halving_memory * eta0 + (1.0 - step_halving_memory) * eta;

      if (num_step_halving > 0 && family_type == POISSON) {
        eta = arma::max(eta, vec(n, fill::value(-10.0)));
      }

      mu = link_inv(eta, family_type);
      dev = dev_resids(y, mu, theta, w, family_type);
      num_step_halving++;

      result.iter = iter + 1;
      continue;
    } else {
      num_step_halving = 0;
    }

    // Adaptive HDFE tolerance update with model size awareness
    if (has_fixed_effects) {
      if (is_large_model) {
        if (convergence_count >= 3 || dev_ratio < hdfe_tolerance * 0.1) {
          hdfe_tolerance = std::max(highest_inner_tol, hdfe_tolerance * 0.01);
        } else if (dev_ratio < hdfe_tolerance) {
          double alt_tol = std::pow(
              10.0,
              -std::ceil(std::log10(1.0 / std::max(0.1 * dev_ratio, 1e-16))));
          hdfe_tolerance = std::max(std::min(0.1 * hdfe_tolerance, alt_tol),
                                    highest_inner_tol);
        }
      } else {
        if (dev_ratio < hdfe_tolerance) {
          double alt_tol = std::pow(
              10.0,
              -std::ceil(std::log10(1.0 / std::max(0.1 * dev_ratio, 1e-16))));
          hdfe_tolerance = std::max(std::min(0.1 * hdfe_tolerance, alt_tol),
                                    highest_inner_tol);
        }
      }
    }

    result.iter = iter + 1;
  }

  if (conv) {
    H = crossprod(X, w_working);

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

      vec pi = eta - x_beta;

      result.has_fe = true;
      if (params.return_fe) {
        result.fixed_effects =
            get_alpha(pi, fe_groups, params.alpha_tol, params.iter_alpha_max);
      }
    }

    // Compute covariance matrix:
    // - If cluster groups provided: sandwich covariance
    // - Otherwise: inverse Hessian
    // X here is the centered design matrix (MX), H is MX'WMX
    mat vcov_reduced;
    if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
      // Sandwich covariance for clustered standard errors
      vcov_reduced = compute_sandwich_vcov(X, y, mu, H, *cluster_groups);
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
      vcov_reduced = std::move(H_inv);
    }
    
    // Expand vcov to full size if there's collinearity
    const uword full_p = beta.n_elem;
    if (collin_result.has_collinearity && collin_result.non_collinear_cols.n_elem > 0) {
      result.vcov.set_size(full_p, full_p);
      result.vcov.fill(datum::nan);
      // Fill in the non-collinear part
      uvec idx = collin_result.non_collinear_cols;
      for (uword i = 0; i < idx.n_elem; ++i) {
        for (uword j = 0; j < idx.n_elem; ++j) {
          result.vcov(idx(i), idx(j)) = vcov_reduced(i, j);
        }
      }
    } else {
      result.vcov = std::move(vcov_reduced);
    }

    result.coef_table.col(0) = beta;  // Store coefficients in first column
    result.coef_status = std::move(collin_result.coef_status);
    result.eta = std::move(eta);
    result.fitted_values = std::move(mu);
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

    // Compute coefficient table: [estimate, std.error, z, p-value]
    // This computation is done here so R only needs to format the results
    // Resize coef_table if needed (for collinearity handling)
    uword n_coef = beta.n_elem;
    if (result.coef_table.n_rows != n_coef) {
      result.coef_table.set_size(n_coef, 4);
      result.coef_table.col(0) = beta;
    }
    
    vec se = sqrt(diagvec(result.vcov));
    vec z_values = beta / se;
    
    result.coef_table.col(1) = se;         // Std. Error
    result.coef_table.col(2) = z_values;   // z value
    
    // Compute two-tailed p-values: 2 * Φ(-|z|)
    for (uword i = 0; i < n_coef; ++i) {
      result.coef_table(i, 3) = 2.0 * normcdf(-fabs(z_values(i)));
    }
    
    // Mark collinear coefficients as NaN
    if (collin_result.has_collinearity) {
      uvec collinear_idx = find(result.coef_status == 0);
      for (uword i = 0; i < collinear_idx.n_elem; ++i) {
        uword idx = collinear_idx(i);
        result.coef_table(idx, 0) = datum::nan;  // Estimate
        result.coef_table(idx, 1) = datum::nan;  // Std. Error
        result.coef_table(idx, 2) = datum::nan;  // z value
        result.coef_table(idx, 3) = datum::nan;  // p-value
      }
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
  vec mu = link_inv(eta, family_type);
  vec mu_eta(n, fill::none), yadj(n, fill::none);
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

    mu_eta = inverse_link_derivative(eta, family_type);
    w_working = (w % square(mu_eta)) / variance(mu, 0.0, family_type);
    yadj = (y - mu) / mu_eta + eta - offset;

    const field<field<GroupInfo>> *group_info_ptr = nullptr;
    field<field<GroupInfo>> group_info;
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
      eta = eta0 + (rho * eta_upd);
      mu = link_inv(eta, family_type);
      dev = dev_resids(y, mu, 0.0, w, family_type);
      dev_ratio_inner = (dev - dev0) / (0.1 + fabs(dev0));

      dev_crit = std::isfinite(dev);
      val_crit = (valid_eta(eta, family_type) && valid_mu(mu, family_type));
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= params.step_halving_factor;
    }

    if (!dev_crit || !val_crit) {
      eta = eta0;
      mu = link_inv(eta, family_type);
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
