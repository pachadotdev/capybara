// Generalized linear model with fixed effects eta = alpha + X * beta

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

struct GlmWorkspace {
  mat XtX;
  vec XtY;
  mat L;
  vec beta_work;
  vec z_solve;

  vec y;
  vec alpha0;
  vec group_sums;

  size_t cached_n, cached_p;
  bool is_initialized;

  GlmWorkspace() : cached_n(0), cached_p(0), is_initialized(false) {}

  GlmWorkspace(size_t n, size_t p)
      : cached_n(n), cached_p(p), is_initialized(true) {
    size_t safe_n = std::max(n, size_t(1));
    size_t safe_p = std::max(p, size_t(1));

    XtX.set_size(safe_p, safe_p);
    XtY.set_size(safe_p);
    L.set_size(safe_p, safe_p);
    beta_work.set_size(safe_p);
    z_solve.set_size(safe_p);

    y.set_size(safe_n);
    alpha0.set_size(safe_n);
    group_sums.set_size(safe_n);
  }

  void ensure_size(size_t n, size_t p) {
    if (!is_initialized || n > cached_n || p > cached_p) {
      size_t new_n = std::max(n, cached_n);
      size_t new_p = std::max(p, cached_p);

      if (XtX.n_rows < new_p || XtX.n_cols < new_p)
        XtX.set_size(new_p, new_p);
      if (XtY.n_elem < new_p)
        XtY.set_size(new_p);
      if (L.n_rows < new_p || L.n_cols < new_p)
        L.set_size(new_p, new_p);
      if (beta_work.n_elem < new_p)
        beta_work.set_size(new_p);
      if (z_solve.n_elem < new_p)
        z_solve.set_size(new_p);

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
    XtY.reset();
    L.reset();
    beta_work.reset();
    z_solve.reset();
    y.reset();
    alpha0.reset();
    group_sums.reset();
    cached_n = 0;
    cached_p = 0;
    is_initialized = false;
  }
};

InferenceGLM feglm_fit(vec &beta, vec &eta, const vec &y, mat &X, const vec &w,
                       const double &theta, const Family family_type,
                       const field<field<uvec>> &fe_groups,
                       const CapybaraParameters &params,
                       GlmWorkspace *workspace = nullptr) {
  const size_t n = y.n_elem;
  const size_t p = X.n_cols;
  const size_t k = beta.n_elem;
  const bool has_fixed_effects = fe_groups.n_elem > 0;

  mat X0 = X;

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

  vec MNU = vec(n, fill::zeros);
  vec mu = link_inv(eta, family_type);
  vec ymean = mean(y) * vec(n, fill::ones);
  vec mu_eta(n, fill::none), w_working(n, fill::none);
  vec nu(n, fill::none), beta_upd(k, fill::none);
  vec eta_upd(n, fill::none), eta0(n, fill::none);
  vec beta0(k, fill::none), nu0 = vec(n, fill::zeros);
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

  // Model size-based initial tolerance
  bool is_large_model =
      (n > 100000) || (p > 1000) || (has_fixed_effects && fe_groups.n_elem > 1);

  if (has_fixed_effects) {
    if (is_large_model) {
      // Much looser initial tolerance for large models
      hdfe_tolerance =
          std::max(1e-3, std::max(start_inner_tol, params.dev_tol * 10));
    } else {
      hdfe_tolerance = std::max(start_inner_tol, params.dev_tol);
    }
  }

  vec MNU_increment = vec(n, fill::zeros);
  bool use_partial = false; // Partial out variables

  // Step halving with memory variables
  double step_halving_memory = params.step_halving_memory;
  size_t num_step_halving = 0;
  size_t max_step_halving = params.max_step_halving;

  // Convergence history prediction
  vec eps_history(3, fill::value(datum::inf));
  double predicted_eps = datum::inf;

  // Adaptive tolerance scheduling
  size_t convergence_count = 0;
  double last_dev_ratio = datum::inf;

  // Maximize the log-likelihood
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    beta0 = beta;
    dev0 = dev;

    mu_eta = inverse_link_derivative(eta, family_type);
    w_working = (w % square(mu_eta)) / variance(mu, theta, family_type);
    nu = (y - mu) / mu_eta;

    bool iter_solver = has_fixed_effects &&
                       (hdfe_tolerance > params.dev_tol * 11) &&
                       (predicted_eps > params.dev_tol);

    double current_hdfe_tol;
    if (is_large_model) {
      if (iter < 5) {
        current_hdfe_tol = std::max(hdfe_tolerance, 1e-2);
      } else if (iter_solver) {
        current_hdfe_tol = std::min(hdfe_tolerance, 1e-3);
      } else {
        current_hdfe_tol = std::min(hdfe_tolerance, highest_inner_tol);
      }
    } else {
      current_hdfe_tol = iter_solver
                             ? hdfe_tolerance
                             : std::min(hdfe_tolerance, highest_inner_tol);
    }

    // Partial out -> only update the increment after first iteration (as in
    // ppmlhdfe)
    if (use_partial && iter > 1) {
      MNU_increment = nu - nu0;
      MNU += MNU_increment;

      if (has_fixed_effects) {
        center_variables(MNU, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start,
                         params.project_tol_factor, params.grand_accel_tol,
                         params.project_group_tol, params.irons_tuck_tol,
                         params.grand_accel_interval,
                         params.irons_tuck_interval, params.ssr_check_interval,
                         params.convergence_factor, params.tol_multiplier);
        center_variables(X, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start,
                         params.project_tol_factor, params.grand_accel_tol,
                         params.project_group_tol, params.irons_tuck_tol,
                         params.grand_accel_interval,
                         params.irons_tuck_interval, params.ssr_check_interval,
                         params.convergence_factor, params.tol_multiplier);
      }
      nu0 = nu;
    } else {
      MNU += (nu - nu0);
      nu0 = nu;

      if (has_fixed_effects) {
        center_variables(MNU, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start,
                         params.project_tol_factor, params.grand_accel_tol,
                         params.project_group_tol, params.irons_tuck_tol,
                         params.grand_accel_interval,
                         params.irons_tuck_interval, params.ssr_check_interval,
                         params.convergence_factor, params.tol_multiplier);
        center_variables(X, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start,
                         params.project_tol_factor, params.grand_accel_tol,
                         params.project_group_tol, params.irons_tuck_tol,
                         params.grand_accel_interval,
                         params.irons_tuck_interval, params.ssr_check_interval,
                         params.convergence_factor, params.tol_multiplier);
      }
    }

    // Enable partial out after first iteration
    use_partial = true;

    InferenceBeta beta_result =
        get_beta(X, MNU, MNU, w_working, collin_result, false, false);

    // Handle collinearity
    vec beta_upd_reduced;
    if (collin_result.has_collinearity &&
        collin_result.non_collinear_cols.n_elem > 0) {
      beta_upd_reduced =
          beta_result.coefficients.elem(collin_result.non_collinear_cols);
    } else {
      beta_upd_reduced = beta_result.coefficients;
    }

    const size_t full_p =
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
    for (size_t iter_inner = 0; iter_inner < params.iter_inner_max;
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

    result.coefficients = std::move(beta);
    result.coef_status = std::move(collin_result.coef_status);
    result.eta = std::move(eta);
    result.fitted_values = std::move(mu);
    result.weights = w;
    result.hessian = std::move(H);
    result.deviance = dev;
    result.null_deviance = null_dev;
    result.conv = true;

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

  const size_t n = y.n_elem;

  vec Myadj = vec(n, fill::zeros);
  vec mu = link_inv(eta, family_type);
  vec mu_eta(n, fill::none), yadj(n, fill::none);
  vec w_working(n, fill::none), eta_upd(n, fill::none), eta0(n, fill::none);

  double dev = dev_resids(y, mu, 0.0, w, family_type);
  double dev0, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  double adaptive_tol = params.center_tol;
  if (n > 100000) {
    adaptive_tol = std::max(params.center_tol, 1e-3);
  }

  // Maximize the log-likelihood
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    dev0 = dev;

    mu_eta = inverse_link_derivative(eta, family_type);
    w_working = (w % square(mu_eta)) / variance(mu, 0.0, family_type);
    yadj = (y - mu) / mu_eta + eta - offset;

    Myadj += yadj;

    center_variables(Myadj, w_working, fe_groups, adaptive_tol,
                     params.iter_center_max, params.iter_interrupt,
                     params.iter_ssr, params.accel_start,
                     params.project_tol_factor, params.grand_accel_tol,
                     params.project_group_tol, params.irons_tuck_tol,
                     params.grand_accel_interval, params.irons_tuck_interval,
                     params.ssr_check_interval, params.convergence_factor,
                     params.tol_multiplier);

    eta_upd = yadj - Myadj + offset - eta;

    for (size_t iter_inner = 0; iter_inner < params.iter_inner_max;
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
