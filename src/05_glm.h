// Generalized linear model with fixed effects eta = alpha + X * beta

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

enum Family {
  UNKNOWN = 0,
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN
};

inline double predict_convergence(const vec &eps_history, double current_eps) {
  if (eps_history.n_elem < 3 || !eps_history.is_finite()) {
    return current_eps;
  }

  uvec finite_indices = find_finite(eps_history);
  if (finite_indices.n_elem < 3) {
    return current_eps;
  }

  // Linear extrapolation based on last 3 values
  vec log_eps = log(eps_history.elem(finite_indices.tail(3)));
  vec x_vals = linspace(1, 3, 3);

  // Simple regression log(eps) = a + b*x
  double x_mean = mean(x_vals);
  double y_mean = mean(log_eps);
  double slope = dot(x_vals - x_mean, log_eps - y_mean) /
                 dot(x_vals - x_mean, x_vals - x_mean);
  double intercept = y_mean - slope * x_mean;

  // Predict next value
  double hat_log_eps = intercept + slope * 4.0;
  return std::max(exp(hat_log_eps), datum::eps);
}

template <typename T>
inline T clamp(const T &value, const T &lower, const T &upper) {
  return (value < lower) ? lower : ((value > upper) ? upper : value);
}

std::string tidy_family(const std::string &family) {
  std::string fam = family;

  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  uword pos = fam.find("(");
  if (pos != std::string::npos) {
    fam.erase(pos, fam.size());
  }

  std::replace(fam.begin(), fam.end(), ' ', '_');
  std::replace(fam.begin(), fam.end(), '.', '_');

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isspace), fam.end());

  return fam;
}

Family get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", GAUSSIAN},
      {"poisson", POISSON},
      {"binomial", BINOMIAL},
      {"gamma", GAMMA},
      {"inverse_gaussian", INV_GAUSSIAN},
      {"negative_binomial", NEG_BIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

vec link_inv_gaussian(const vec &eta) { return eta; }

vec link_inv_poisson(const vec &eta) { return exp(eta); }

vec link_inv_logit(const vec &eta) { return 1.0 / (1.0 + exp(-eta)); }

vec link_inv_gamma(const vec &eta) { return 1 / eta; }

vec link_inv_invgaussian(const vec &eta) { return 1 / sqrt(eta); }

vec link_inv_negbin(const vec &eta) { return exp(eta); }

double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu));
}

double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt) {
  vec r = mu % wt;

  uvec p = find(y > 0);
  r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));

  return 2 * accu(r);
}

// Adapted from binomial_dev_resids()
// in base R it can be found in src/library/stats/src/family.c
double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
  vec r(y.n_elem, fill::zeros);
  vec s(y.n_elem, fill::zeros);

  uvec p = find(y == 1);
  uvec q = find(y == 0);
  r(p) = y(p) % log(y(p) / mu(p));
  s(q) = (1 - y(q)) % log((1 - y(q)) / (1 - mu(q)));

  return 2 * dot(wt, r + s);
}

double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  vec r = y / mu;

  uvec p = find(y == 0);
  r.elem(p).fill(1.0);
  r = wt % (log(r) - (y - mu) / mu);

  return -2 * accu(r);
}

double dev_resids_invgaussian(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

double dev_resids_negbin(const vec &y, const vec &mu, const double &theta,
                         const vec &wt) {
  vec r = y;

  uvec p = find(y < 1);
  r.elem(p).fill(1.0);
  r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2 * accu(r);
}

vec variance_gaussian(const vec &mu) { return ones<vec>(mu.n_elem); }

vec link_inv(const vec &eta, const Family family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case GAUSSIAN:
    result = link_inv_gaussian(eta);
    break;
  case POISSON:
    result = link_inv_poisson(eta);
    break;
  case BINOMIAL:
    result = link_inv_logit(eta);
    break;
  case GAMMA:
    result = link_inv_gamma(eta);
    break;
  case INV_GAUSSIAN:
    result = link_inv_invgaussian(eta);
    break;
  case NEG_BIN:
    result = link_inv_negbin(eta);
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

double dev_resids(const vec &y, const vec &mu, const double &theta,
                  const vec &wt, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return dev_resids_gaussian(y, mu, wt);
  case POISSON:
    return dev_resids_poisson(y, mu, wt);
  case BINOMIAL:
    return dev_resids_logit(y, mu, wt);
  case GAMMA:
    return dev_resids_gamma(y, mu, wt);
  case INV_GAUSSIAN:
    return dev_resids_invgaussian(y, mu, wt);
  case NEG_BIN:
    return dev_resids_negbin(y, mu, theta, wt);
  default:
    stop("Unknown family");
  }
}

bool valid_eta(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
  case POISSON:
  case BINOMIAL:
  case NEG_BIN:
    return true;
  case GAMMA:
    return eta.is_finite() && all(eta != 0.0);
  case INV_GAUSSIAN:
    return eta.is_finite() && all(eta > 0.0);
  default:
    stop("Unknown family");
  }
}

bool valid_mu(const vec &mu, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return true;
  case POISSON:
  case NEG_BIN:
    return mu.is_finite() && all(mu > 0);
  case BINOMIAL:
    return mu.is_finite() && all(mu > 0 && mu < 1);
  case GAMMA:
    return mu.is_finite() && all(mu > 0.0);
  case INV_GAUSSIAN:
    return true;
  default:
    stop("Unknown family");
  }
}

vec inverse_link_derivative(const vec &eta, const Family family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case GAUSSIAN:
    result.ones();
    break;
  case POISSON:
  case NEG_BIN:
    result = arma::exp(eta);
    break;
  case BINOMIAL: {
    vec exp_eta = arma::exp(eta);
    result = exp_eta / arma::square(1 + exp_eta);
    break;
  }
  case GAMMA:
    result = -1 / arma::square(eta);
    break;
  case INV_GAUSSIAN:
    result = -1 / (2 * arma::pow(eta, 1.5));
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

vec variance(const vec &mu, const double &theta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ones<vec>(mu.n_elem);
  case POISSON:
    return mu;
  case BINOMIAL:
    return mu % (1 - mu);
  case GAMMA:
    return square(mu);
  case INV_GAUSSIAN:
    return pow(mu, 3.0);
  case NEG_BIN:
    return mu + square(mu) / theta;
  default:
    stop("Unknown family");
  }
}

struct GlmWorkspace {
  mat XtX;
  mat XtX_cache; // Cache for reuse across iterations

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

  uword cached_n, cached_p;
  bool is_initialized;

  GlmWorkspace() : cached_n(0), cached_p(0), is_initialized(false) {}

  GlmWorkspace(uword n, uword p)
      : cached_n(n), cached_p(p), is_initialized(true) {
    uword safe_n = std::max(n, uword(1));
    uword safe_p = std::max(p, uword(1));

    XtX.set_size(safe_p, safe_p);
    XtX_cache.set_size(safe_p, safe_p);

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
  }

  void ensure_size(uword n, uword p) {
    if (!is_initialized || n > cached_n || p > cached_p) {
      uword new_n = std::max(n, cached_n);
      uword new_p = std::max(p, cached_p);

      if (XtX.n_rows < new_p || XtX.n_cols < new_p)
        XtX.set_size(new_p, new_p);
      if (XtX_cache.n_rows < new_p || XtX_cache.n_cols < new_p)
        XtX_cache.set_size(new_p, new_p);

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

      cached_n = new_n;
      cached_p = new_p;
      is_initialized = true;
    }
  }

  ~GlmWorkspace() { clear(); }

  void clear() {
    XtX.reset();
    XtX_cache.reset();
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
                       GlmWorkspace *workspace = nullptr) {
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

  check_collinearity(X, w, /*use_weights =*/true, params.collin_tol, result);

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
        
          // Adaptive X re-centering based on convergence
        uword recenter_interval = (dev_ratio > 0.01) ? 2 : 
                                  (dev_ratio > 0.001) ? 3 : 5;
        
        if (iter % recenter_interval == 0) {
          center_variables(X, w_working, fe_groups, current_hdfe_tol,
                           current_max_iter, params.iter_interrupt,
                           group_info_ptr, &centering_workspace);
          workspace->XtX_cache.reset();
        }
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

    // Compute beta directly into result struct
    get_beta(X, MNU, MNU, w_working, result, false, false, &workspace->XtX_cache);

    // Handle collinearity
    vec beta_upd_reduced;
    if (result.has_collinearity &&
        result.non_collinear_cols.n_elem > 0) {
      beta_upd_reduced =
          result.coefficients.elem(result.non_collinear_cols);
    } else {
      beta_upd_reduced = result.coefficients;
    }

    const uword full_p =
        result.has_collinearity ? result.coef_status.n_elem : p;
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

      if (result.has_collinearity &&
          result.non_collinear_cols.n_elem > 0) {
        beta = beta0;
        beta.elem(result.non_collinear_cols) = 
            beta0.elem(result.non_collinear_cols) + rho * beta_upd_reduced;
      } else {
        beta = beta0 + rho * beta_upd_reduced;
      }

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

    if (dev > dev0 && num_step_halving < max_step_halving) {
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
        if (result.has_collinearity &&
            result.non_collinear_cols.n_elem > 0) {
          x_beta = X0.cols(result.non_collinear_cols) *
                   beta.elem(result.non_collinear_cols);
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
