// Computing generalized linear models with fixed effects
// eta = X beta + alpha + offset

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

inline Family string_to_family(const std::string &fam) {
  
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", Family::GAUSSIAN},
      {"poisson", Family::POISSON},
      {"binomial", Family::BINOMIAL},
      {"gamma", Family::GAMMA},
      {"inverse_gaussian", Family::INV_GAUSSIAN},
      {"negative_binomial", Family::NEGBIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : Family::GAUSSIAN;
}

inline std::string tidy_family(const std::string &family) {
  

  std::string fam = family;

  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  size_t pos = fam.find("(");
  if (pos != std::string::npos) {
    fam.erase(pos, fam.size());
  }

  std::replace(fam.begin(), fam.end(), ' ', '_');
  std::replace(fam.begin(), fam.end(), '.', '_');

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isspace), fam.end());

  return fam;
}

inline Family get_family_type(const std::string &fam) {
  
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", Family::GAUSSIAN},
      {"poisson", Family::POISSON},
      {"binomial", Family::BINOMIAL},
      {"gamma", Family::GAMMA},
      {"inverse_gaussian", Family::INV_GAUSSIAN},
      {"negative_binomial", Family::NEGBIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : Family::GAUSSIAN;
}

inline void link_inv(const vec &eta, vec &mu, const Family family_type) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    mu = eta;
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    mu = exp(eta);
    break;
  case Family::BINOMIAL:
    mu = 1.0 / (1.0 + exp(-eta));
    break;
  case Family::GAMMA:
    mu = 1.0 / eta;
    break;
  case Family::INV_GAUSSIAN:
    mu = 1.0 / sqrt(eta);
    break;
  default:
    stop("Unknown family");
  }
}

inline void d_inv_link(const vec &eta, vec &result, const Family family_type) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    result.ones();
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    result = exp(eta);
    break;
  case Family::BINOMIAL: {
    // Use the identity: d/d_eta[1/(1+exp(-eta))] = mu*(1-mu) where mu =
    // 1/(1+exp(-eta))
    result = 1.0 / (1.0 + exp(-eta));
    result = result % (1.0 - result);
    break;
  }
  case Family::GAMMA:
    result = -1.0 / square(eta);
    break;
  case Family::INV_GAUSSIAN: {
    result = -1.0 / (2.0 * pow(abs(eta), 1.5));
    break;
  }
  default:
    stop("Unknown family");
  }
}

void initialize_family(vec &mu, vec &eta, const vec &y_orig, double mean_y,
                       const Family family_type, double binomial_mu_min,
                       double binomial_mu_max, double safe_clamp_min,
                       double safe_clamp_max) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    mu = y_orig;
    eta = y_orig;
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    mu = (y_orig + mean_y) / 2.0;
    eta = log(clamp(mu, safe_clamp_min, safe_clamp_max));
    break;
  case Family::BINOMIAL:

    mu = clamp((y_orig + mean_y) / 2.0, binomial_mu_min, binomial_mu_max);
    eta = log(mu / (1.0 - mu));
    break;
  case Family::GAMMA:
    mu = clamp(y_orig, safe_clamp_min, safe_clamp_max);
    eta = 1.0 / mu;
    break;
  case Family::INV_GAUSSIAN:
    mu = clamp(y_orig, safe_clamp_min, safe_clamp_max);
    eta = 1.0 / (mu % mu);
    break;
  default:
    stop("Unknown family");
  }
}

void initialize_family_fixed_effects(vec &mu, vec &eta, double mean_y,
                                     const Family family_type,
                                     double safe_clamp_min) {
  
  double safe_mean_y = std::max(static_cast<double>(mean_y), safe_clamp_min);

  switch (family_type) {
  case Family::GAUSSIAN:
    mu.fill(mean_y);
    eta.fill(mean_y);
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    mu.fill(safe_mean_y);
    eta.fill(log(safe_mean_y));
    break;
  case Family::BINOMIAL:
    mu.fill(0.5);
    eta.zeros();
    break;
  case Family::GAMMA:
    mu.fill(safe_mean_y);
    eta.fill(1.0 / safe_mean_y);
    break;
  case Family::INV_GAUSSIAN:
    mu.fill(safe_mean_y);
    eta.fill(1.0 / (safe_mean_y * safe_mean_y));
    break;
  default:
    stop("Unknown family");
  }
}

inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &w) {
  
  return dot(w, square(y - mu));
}

inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &w) {
  
  uvec p = find(y > 0);
  if (p.n_elem == 0) {
    return 0.0;
  }

  vec y_pos = y(p);
  vec mu_pos = mu(p);
  vec w_pos = w(p);

  return 2.0 * accu(w_pos % (y_pos % log(y_pos / mu_pos) - (y_pos - mu_pos)));
}

inline double dev_resids_binomial(const vec &y, const vec &mu, const vec &w) {
  
  vec r(y.n_elem, fill::zeros);

  uvec p = find(y == 1);
  if (p.n_elem > 0) {
    r(p) = log(y(p) / mu(p));
  }

  uvec q = find(y == 0);
  if (q.n_elem > 0) {
    r(q) = log((1.0 - y(q)) / (1.0 - mu(q)));
  }

  return 2.0 * dot(w, r);
}

inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &w,
                               double safe_clamp_min) {
  
  vec y_adj = clamp(y, safe_clamp_min, datum::inf);
  vec mu_adj = clamp(mu, safe_clamp_min, datum::inf);

  return -2.0 * dot(w, log(y_adj / mu_adj) - (y - mu) / mu_adj);
}

inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &w) {
  
  vec diff = y - mu;
  vec mu_sq = mu % mu;
  return dot(w, (diff % diff) / (y % mu_sq));
}

inline double dev_resids_negbin(const vec &y, const vec &mu,
                                const double &theta, const vec &w,
                                double safe_clamp_min) {
  
  vec y_adj = clamp(y, safe_clamp_min, datum::inf);

  return 2.0 * accu(w % (y % log(y_adj / mu) -
                         (y + theta) % log((y + theta) / (mu + theta))));
}

inline double dev_resids(const vec &y, const vec &mu, const double &theta,
                         const vec &w, const Family family_type,
                         double safe_clamp_min) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    return dev_resids_gaussian(y, mu, w);
  case Family::POISSON:
    return dev_resids_poisson(y, mu, w);
  case Family::BINOMIAL:
    return dev_resids_binomial(y, mu, w);
  case Family::GAMMA:
    return dev_resids_gamma(y, mu, w, safe_clamp_min);
  case Family::INV_GAUSSIAN:
    return dev_resids_invgaussian(y, mu, w);
  case Family::NEGBIN:
    return dev_resids_negbin(y, mu, theta, w, safe_clamp_min);
  default:
    stop("Unknown family");
  }
}

inline void variance(const vec &mu, const double &theta, vec &result,
                     const Family family_type) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    result.ones();
    break;
  case Family::POISSON:
    result = mu;
    break;
  case Family::BINOMIAL:
    result = mu % (1 - mu);
    break;
  case Family::GAMMA:
    result = mu % mu;
    break;
  case Family::INV_GAUSSIAN:
    result = pow(mu, 3.0);
    break;
  case Family::NEGBIN:
    result = mu + square(mu) / theta;
    break;
  default:
    stop("Unknown family");
  }
}

inline bool valid_eta(const vec &eta, const Family family_type) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
  case Family::POISSON:
  case Family::BINOMIAL:
  case Family::NEGBIN:
    return is_finite(eta);
  case Family::GAMMA:
    return is_finite(eta) && all(eta != 0.0);
  case Family::INV_GAUSSIAN:
    return is_finite(eta) && all(eta > 0.0);
  default:
    stop("Unknown family");
  }
}

inline bool valid_mu(const vec &mu, const Family family_type) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    return is_finite(mu);
  case Family::POISSON:
  case Family::NEGBIN:
    return is_finite(mu) && all(mu > 0);
  case Family::BINOMIAL:
    return is_finite(mu) && all(mu > 0 && mu < 1);
  case Family::GAMMA:
    return is_finite(mu) && all(mu > 0.0);
  case Family::INV_GAUSSIAN:
    return is_finite(mu) && all(mu > 0.0);
  default:
    stop("Unknown family");
  }
}

inline bool valid_response(const vec &y, const Family family_type) {
  
  switch (family_type) {
  case Family::GAUSSIAN:
    return is_finite(y);
  case Family::POISSON:
  case Family::NEGBIN:
    return is_finite(y) && all(y >= 0);
  case Family::BINOMIAL:
    return is_finite(y) && all(y >= 0) && all(y <= 1);
  case Family::GAMMA:
  case Family::INV_GAUSSIAN:
    return is_finite(y) && all(y > 0);
  default:
    stop("Unknown family");
  }
}

struct InferenceWLM {
  vec coefficients;
  vec fitted_values;
  uvec coef_status;
  bool success;

  mat hessian;
  field<vec> fixed_effects;
  uvec nb_references;
  bool is_regular;
  bool has_fe;

  InferenceWLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        coef_status(p, fill::ones), success(false), hessian(p, p, fill::zeros),
        is_regular(true), has_fe(false) {
    
  }
};

struct InferenceGLM {
  vec coefficients;
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  size_t iter;
  uvec coef_status; // 1 = estimable, 0 = collinear

  vec residuals_working;
  vec residuals_response;

  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
  bool has_fe = false;
  uvec iterations;

  mat X_dm;
  bool has_mx = false;

  InferenceGLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), eta(n, fill::zeros),
        fitted_values(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), deviance(0.0), null_deviance(0.0),
        conv(false), iter(0), coef_status(p, fill::ones),
        residuals_working(n, fill::zeros), residuals_response(n, fill::zeros),
        is_regular(true), has_fe(false), has_mx(false) {
    
  }

  cpp11::list to_list(bool keep_dmx = true) const {
    
    auto out = writable::list(
        {"coefficients"_nm = as_doubles(coefficients),
         "eta"_nm = as_doubles(eta),
         "fitted.values"_nm = as_doubles(fitted_values),
         "weights"_nm = as_doubles(weights),
         "hessian"_nm = as_doubles_matrix(hessian),
         "deviance"_nm = writable::doubles({deviance}),
         "null.deviance"_nm = writable::doubles({null_deviance}),
         "conv"_nm = writable::logicals({conv}),
         "iter"_nm = writable::integers({static_cast<int>(iter)})});

    if (has_fe && fixed_effects.n_elem > 0) {
      writable::list fe_list(fixed_effects.n_elem);
      for (size_t k = 0; k < fixed_effects.n_elem; ++k) {
        fe_list[k] = as_doubles(fixed_effects(k));
      }
      out.push_back({"fixed.effects"_nm = fe_list});
      out.push_back({"nb_references"_nm = as_integers(nb_references)});
      out.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    }

    if (keep_dmx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(X_dm)});
    }

    return out;
  }
};

inline InferenceWLM wlm_fit(const mat &X_reduced, const vec &y,
                            const vec &y_orig, const vec &w,
                            const field<uvec> &fe_indices, const uvec &nb_ids,
                            const field<uvec> &fe_id_tables,
                            const CollinearityResult &collin_result,
                            const CapybaraParameters &params,
                            bool compute_hessian = false,
                            bool compute_fixed_effects = false) {
  
  const size_t n = y.n_elem;
  const size_t p_orig = collin_result.coef_status.n_elem;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceWLM result(n, p_orig);

  bool use_weights = params.use_weights && !all(w == 1.0);

  mat X_demean;
  vec y_demean;
  DemeanResult y_demean_result(0);

  if (has_fixed_effects) {

    field<vec> y_to_demean(1);
    y_to_demean(0) = y;

    y_demean_result =
        demean_variables(y_to_demean, w, fe_indices, nb_ids, fe_id_tables,
                         compute_fixed_effects, params);
    y_demean = y_demean_result.demeaned_vars(0);

    if (X_reduced.n_cols > 0) {

      X_demean.set_size(n, X_reduced.n_cols);

      field<vec> x_columns_to_demean(X_reduced.n_cols);
      for (size_t j = 0; j < X_reduced.n_cols; ++j) {
        x_columns_to_demean(j) = X_reduced.unsafe_col(j);
      }

      DemeanResult x_demean_result =
          demean_variables(x_columns_to_demean, w, fe_indices, nb_ids,
                           fe_id_tables, false, params);

      for (size_t j = 0; j < X_reduced.n_cols; ++j) {
        X_demean.unsafe_col(j) = std::move(x_demean_result.demeaned_vars(j));
      }
    } else {
      X_demean = mat(n, 0);
    }

    result.has_fe = true;
  } else {
    y_demean = y;
    X_demean = X_reduced;
    result.has_fe = false;
  }

  InferenceBeta beta_result = get_beta(X_demean, y_demean, y, w, collin_result,
                                       use_weights, has_fixed_effects);

  result.coefficients = beta_result.coefficients;
  result.fitted_values = beta_result.fitted_values;
  result.coef_status = beta_result.coef_status;

  if (compute_hessian) {
    result.hessian = beta_result.hessian;
  }

  if (has_fixed_effects && compute_fixed_effects) {

    vec coef_reduced;
    if (collin_result.has_collinearity) {
      coef_reduced = result.coefficients(collin_result.non_collinear_cols);
    } else {
      coef_reduced = result.coefficients;
    }

    vec sum_fe = result.fitted_values - X_reduced * coef_reduced;

    
    field<field<uvec>> group_indices(fe_indices.n_elem);
    for (size_t k = 0; k < fe_indices.n_elem; ++k) {
      const uvec &fe_idx = fe_indices(k);
      const size_t n_obs = fe_idx.n_elem;
      const size_t n_groups = nb_ids(k);
      
      group_indices(k).set_size(n_groups);

      
      uvec group_sizes(n_groups, fill::zeros);
      const uword *fe_idx_ptr = fe_idx.memptr();
      uword *group_sizes_ptr = group_sizes.memptr();
      
      for (size_t obs = 0; obs < n_obs; ++obs) {
        group_sizes_ptr[fe_idx_ptr[obs]]++;
      }

      
      for (size_t g = 0; g < n_groups; ++g) {
        if (group_sizes_ptr[g] > 0) {
          group_indices(k)(g).set_size(group_sizes_ptr[g]);
        } else {
          group_indices(k)(g).reset();
        }
      }

      
      uvec group_counters(n_groups, fill::zeros);
      uword *group_counters_ptr = group_counters.memptr();
      
      for (size_t obs = 0; obs < n_obs; ++obs) {
        uword group_id = fe_idx_ptr[obs];
        group_indices(k)(group_id)(group_counters_ptr[group_id]++) = obs;
      }
    }

    InferenceAlpha alpha_result =
        get_alpha(sum_fe, group_indices, params.alpha_convergence_tol,
                  params.alpha_iter_max);
    result.fixed_effects = std::move(alpha_result.Alpha);
    result.nb_references = std::move(alpha_result.nb_references);
    result.is_regular = alpha_result.is_regular;
  }

  result.success = true;
  return result;
}

inline InferenceGLM feglm_fit(mat &X, const vec &y_orig, const vec &w,
                              const field<uvec> &fe_indices, const uvec &nb_ids,
                              const field<uvec> &fe_id_tables,
                              const std::string &family,
                              const CapybaraParameters &params,
                              const double &theta = 0.0) {
  
  const size_t n = y_orig.n_elem;
  const size_t p_orig = X.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceGLM result(n, p_orig);

  std::string fam = tidy_family(family);
  Family family_type = get_family_type(fam);

  if (!valid_response(y_orig, family_type)) {
    result.conv = false;
    return result;
  }

  bool use_weights = params.use_weights && w.n_elem > 1 && !all(w == 1.0);
  const vec &weights_vec = w;

  
  double tolerance = params.collin_tol;
  CollinearityResult collin_result =
      check_collinearity(X, weights_vec, use_weights, tolerance, true);
  
  

  double mean_y =
      use_weights ? sum(weights_vec % y_orig) / sum(weights_vec) : mean(y_orig);
  vec mu(n, fill::none);
  vec eta(n, fill::none);

  initialize_family(mu, eta, y_orig, mean_y, family_type,
                    params.binomial_mu_min, params.binomial_mu_max,
                    params.safe_clamp_min, params.safe_clamp_max);

  if (has_fixed_effects) {
    initialize_family_fixed_effects(mu, eta, mean_y, family_type,
                                    params.safe_clamp_min);
  }

  vec mu_init(n);
  vec eta_old_wls = vec(n, fill::value(params.glm_init_eta));
  link_inv(eta_old_wls, mu_init, family_type);
  double devold = dev_resids(y_orig, mu_init, theta, weights_vec, family_type,
                             params.safe_clamp_min);

  
  vec eta_old(n);
  vec z(n);
  vec working_weights(n);
  vec mu_eta_val(n);
  vec var_mu(n);
  vec mu_new(n);

  InferenceWLM wls_result(n, p_orig);
  bool converged = false;

  for (size_t iter = 0; iter < params.iter_max; iter++) {
    result.iter = iter + 1;
    eta_old = eta;

    d_inv_link(eta, mu_eta_val, family_type);
    variance(mu, theta, var_mu, family_type);

    if (!all(var_mu > 0) || !is_finite(var_mu)) {
      break;
    }

    z = eta + (y_orig - mu) / mu_eta_val;
    working_weights = weights_vec % (mu_eta_val % mu_eta_val) / var_mu;

    if (!any(working_weights > 0)) {
      break;
    }

    wls_result =
        wlm_fit(X, z, y_orig, working_weights, fe_indices, nb_ids,
                fe_id_tables, collin_result, params, false, false);

    if (!wls_result.success || !is_finite(wls_result.coefficients)) {
      if (iter == 0) {
        result.conv = false;
        return result;
      }
      break;
    }

    result.coefficients = wls_result.coefficients;
    result.coef_status = wls_result.coef_status;

    // Reuse z vector as temporary for eta_new to avoid allocation
    z = wls_result.fitted_values;
    link_inv(z, mu_new, family_type);
    double dev = dev_resids(y_orig, mu_new, theta, weights_vec, family_type,
                            params.safe_clamp_min);
    double dev_evol = dev - devold;

    bool need_step_halving = !is_finite(dev) || (dev_evol > 0) ||
                             !valid_eta(z, family_type) ||
                             !valid_mu(mu_new, family_type);

    if (need_step_halving &&
        !(std::abs(dev_evol) < params.dev_tol ||
          std::abs(dev_evol) / (params.rel_tol_denom + std::abs(dev)) <
              params.dev_tol)) {
      size_t iter_sh = 0;
      bool step_accepted = false;

      while (iter_sh < params.iter_inner_max) {
        iter_sh++;
        // eta_new = (1-factor)*eta_old + factor*new_eta
        z = (1.0 - params.step_halving_factor) * eta_old_wls +
            params.step_halving_factor * wls_result.fitted_values;
        link_inv(z, mu_new, family_type);
        dev = dev_resids(y_orig, mu_new, theta, weights_vec, family_type,
                         params.safe_clamp_min);
        dev_evol = dev - devold;

        if (is_finite(dev) && (dev_evol <= 0) && valid_eta(z, family_type) &&
            valid_mu(mu_new, family_type)) {
          step_accepted = true;
          break;
        }

        if (iter == 0 && iter_sh >= 2 && is_finite(dev) &&
            valid_eta(z, family_type) && valid_mu(mu_new, family_type)) {
          step_accepted = true;
          break;
        }
      }

      if (!step_accepted) {
        if (iter == 0) {
          result.conv = false;
          return result;
        }
        eta = eta_old;
        link_inv(eta, mu, family_type);
        result.deviance = devold;
      } else {
        eta = z;
        mu = mu_new;
        result.deviance = dev;
        dev_evol = datum::inf;
      }
    } else {
      eta = z;
      mu = mu_new;
      result.deviance = dev;
    }

    eta_old_wls = wls_result.fitted_values;

    if (std::abs(dev_evol) / (params.rel_tol_denom + std::abs(dev)) <
        params.dev_tol) {
      result.conv = true;
      converged = true;
      break;
    }

    devold = dev;
  }

  result.conv = converged;
  result.eta = eta;
  result.fitted_values = mu;
  result.weights = working_weights;
  result.residuals_response = y_orig - mu;
  result.residuals_working = z - eta;

  if (has_fixed_effects) {
    vec final_mu_eta(n);
    d_inv_link(eta, final_mu_eta, family_type);
    vec final_var_mu(n);
    variance(mu, theta, final_var_mu, family_type);
    vec final_working_weights =
        weights_vec % (final_mu_eta % final_mu_eta) / final_var_mu;
    vec final_z = eta + (y_orig - mu) / final_mu_eta;

    InferenceWLM final_wls =
        wlm_fit(X, final_z, y_orig, final_working_weights, fe_indices,
                nb_ids, fe_id_tables, collin_result, params, true, true);

    result.hessian = std::move(final_wls.hessian);
    result.fixed_effects = std::move(final_wls.fixed_effects);
    result.has_fe = true;
    result.is_regular = final_wls.is_regular;
    result.nb_references = std::move(final_wls.nb_references);
  } else {

    if (X.n_cols > 0) {

      mat X_weighted = X.each_col() % sqrt(working_weights);
      mat hess_reduced = X_weighted.t() * X_weighted;

      if (collin_result.has_collinearity) {
        result.hessian(collin_result.non_collinear_cols,
                       collin_result.non_collinear_cols) = hess_reduced;
      } else {
        result.hessian = hess_reduced;
      }
    }

    result.has_fe = false;
  }

  vec mu_null(y_orig.n_elem, fill::value(mean_y));
  result.null_deviance = dev_resids(y_orig, mu_null, theta, weights_vec,
                                    family_type, params.safe_clamp_min);

  if (params.keep_dmx) {
    result.X_dm = std::move(X);
    result.has_mx = true;
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_GLM_H
