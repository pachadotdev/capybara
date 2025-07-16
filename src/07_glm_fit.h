#ifndef CAPYBARA_GLM
#define CAPYBARA_GLM

struct SeparationResult {
  uvec separated_indices;
  bool has_separation;
  size_t n_separated;
};

struct GLMResult {
  vec coefficients;
  field<vec> fixed_effects;
  uvec nb_references;          // Number of references per dimension (fixest compatibility)
  bool is_regular;             // Whether fixed effects are regular
  bool has_fe = true;
  vec eta;
  vec fitted_values;  // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  size_t iter;
  mat mx;  // only if keep_mx = true
  bool has_mx = false;
  uvec coef_status;  // 1 = valid, 0 = collinear

  // PPML-only
  vec residuals_working;
  vec residuals_response;

  cpp11::list to_list(bool keep_mx = true) const {
    auto out = writable::list(
        {"coefficients"_nm = as_doubles(coefficients),
         "eta"_nm = as_doubles(eta),
         "fitted.values"_nm = as_doubles(fitted_values),
         "weights"_nm = as_doubles(weights),
         "hessian"_nm = as_doubles_matrix(hessian),
         "deviance"_nm = writable::doubles({deviance}),
         "null.deviance"_nm = writable::doubles({null_deviance}),
         "conv"_nm = writable::logicals({conv}),
         "iter"_nm = writable::integers({static_cast<int>(iter)})
        });
    if (has_fe && fixed_effects.n_elem > 0) {
      writable::list fe_list(fixed_effects.n_elem);
      for (size_t k = 0; k < fixed_effects.n_elem; ++k) {
        fe_list[k] = as_doubles(fixed_effects(k));
      }
      out.push_back({"fixed.effects"_nm = fe_list});
      out.push_back({"nb_references"_nm = as_integers(nb_references)});
      out.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    }
    if (keep_mx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(mx)});
    }
    return out;
  }
};

inline double ppml_deviance(const vec &y, const vec &mu) {
  // Handle log terms for non-zero y values safely
  uvec nonzero_mask = (y > 0);
  vec dev_terms = zeros<vec>(y.n_elem);

  if (any(nonzero_mask)) {
    uvec nonzero_idx = find(nonzero_mask);
    dev_terms.elem(nonzero_idx) =
        y.elem(nonzero_idx) % log(y.elem(nonzero_idx) / mu.elem(nonzero_idx));
  }

  dev_terms -= (y - mu);
  return 2.0 * sum(dev_terms);
}

// Poisson regression using PPML algorithm (ppmlhdfe approach)
GLMResult fepoisson_fit(const mat &X, const vec &y, const vec &wt,
                        const field<field<uvec>> &group_indices,
                        double center_tol, double dev_tol, bool keep_mx,
                        size_t iter_max, size_t iter_center_max,
                        size_t iter_inner_max, double collin_tol) {
  GLMResult result;
  result.conv = false;
  result.iter = 0;

  const size_t n = y.n_elem;
  const size_t p = X.n_cols;

  if (any(y < 0)) {
    return result;
  }

  double mean_y = mean(y);
  vec mu = (y + mean_y) / 2.0;
  vec eta = log(mu);

  result.deviance = ppml_deviance(y, mu);
  uvec coef_status = ones<uvec>(p);

  umat fe_matrix;
  bool has_fe = group_indices.n_elem > 0;
  if (has_fe) {
    fe_matrix.set_size(n, group_indices.n_elem);
    fe_matrix.zeros();

    for (size_t k = 0; k < group_indices.n_elem; k++) {
      for (size_t g = 0; g < group_indices(k).n_elem; g++) {
        const uvec &group_obs = group_indices(k)(g);
        if (group_obs.n_elem > 0) {
          fe_matrix.submat(group_obs, uvec{k}).fill(g);
        }
      }
    }
  }

  // Storage for IRLS algorithm  
  vec eta_old, eta_upd, nu, nu_old(n, fill::zeros), w;
  vec mu_eta_vec;
  double rho, dev_old, dev_ratio_inner;
  bool dev_crit, val_crit, imp_crit;
  
  // IRLS loop - following the old working algorithm
  for (size_t iter = 0; iter < iter_max; iter++) {
    result.iter = iter + 1;
    rho = 1.0;
    eta_old = eta;
    dev_old = result.deviance;

    // Compute weights and working dependent variable (mu_eta = exp(eta) for PPML)
    mu_eta_vec = exp(eta);  // d(mu)/d(eta) = exp(eta) for PPML
    w = mu_eta_vec % mu_eta_vec / mu;  // weights = (mu_eta)^2 / variance, variance = mu for PPML
    nu = (y - mu) / mu_eta_vec;  // working dependent variable

    // Joint centering approach following old algorithm
    vec working_y = nu;
    mat working_X = X;
    
    // Update working variables incrementally 
    working_y += (nu - nu_old);
    nu_old = nu;

    if (has_fe) {
      // Center variables using weighted demeaning (approximating center_variables_)
      mat combined = join_rows(working_y, working_X);
      WeightedDemeanResult demean_result =
          demean_variables(combined, fe_matrix, w, center_tol,
                           static_cast<int>(iter_center_max), "poisson");

      if (!demean_result.success) {
        return result;
      }

      working_y = demean_result.demeaned_data.col(0);
      working_X = demean_result.demeaned_data.cols(1, p);
    }

    // Solve for coefficient update
    beta_results ws(n, p);
    get_beta(working_X, working_y, y, mu, n, p, ws, true, collin_tol, has_fe);
    
    coef_status = ws.coef_status;
    vec beta_upd = ws.coefficients;
    
    // Compute eta update (following old algorithm)
    eta_upd = working_X * beta_upd + nu - working_y;

    // Step halving with comprehensive checks (following old algorithm)
    for (size_t iter_inner = 0; iter_inner < iter_inner_max; iter_inner++) {
      eta = eta_old + rho * eta_upd;
      mu = exp(eta);
      double deviance = ppml_deviance(y, mu);
      dev_ratio_inner = (deviance - dev_old) / (0.1 + std::abs(deviance));

      // Three checks: finite deviance, valid eta/mu, improvement
      dev_crit = std::isfinite(deviance);
      val_crit = all(eta > -700.0) && all(eta < 700.0) && all(mu > 1e-12);  // validity checks
      imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        result.deviance = deviance;
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed
    if (!dev_crit || !val_crit) {
      cpp11::stop("Inner loop failed; cannot correct step size.");
    }

    // If step halving does not improve deviance, revert
    if (!imp_crit) {
      eta = eta_old;
      mu = exp(eta);
      result.deviance = dev_old;
    }

    // Check convergence
    double dev_ratio = std::abs(result.deviance - dev_old) / (0.1 + std::abs(result.deviance));
    
    if (dev_ratio < dev_tol) {
      result.conv = true;
      break;
    }
  }

  result.coef_status = coef_status;

  // Null deviance
  double y_mean = mean(y);
  vec fitted_values_null(y.n_elem, fill::value(y_mean));
  result.null_deviance = ppml_deviance(y, fitted_values_null);

  if (keep_mx) {
    result.mx = X;
    result.has_mx = true;
  }

  if (has_fe) {
    // Use enhanced fixed effects extraction for GLMs
    GetAlphaResult alpha_result =
        extract_model_fixef(result.fitted_values, eta, X, result.coefficients,
                           group_indices, "poisson", center_tol, iter_center_max);
    result.fixed_effects = alpha_result.Alpha;
    result.nb_references = alpha_result.nb_references;
    result.is_regular = alpha_result.is_regular;
    result.has_fe = alpha_result.success;
  } else {
    result.has_fe = false;
  }

  return result;
}

// Generic IRLS - Fixed implementation based on working algorithm
template <FamilyType family_type>
GLMResult generic_irls_fit(const mat &X, const vec &y, const vec &wt,
                           const field<field<uvec>> &group_indices,
                           double center_tol, double dev_tol, bool keep_mx,
                           size_t iter_max, size_t iter_center_max,
                           size_t iter_inner_max, double collin_tol) {
  GLMResult result;
  result.conv = false;
  result.iter = 0;

  const size_t n = y.n_elem;
  const size_t p = X.n_cols;

  vec beta = zeros<vec>(p);
  vec eta = zeros<vec>(n);
  vec mu = link_inv_(eta, family_type);

  // Initialize mu and eta based on family type following R GLM approach
  double ymean = sum(wt % y) / sum(wt);
  vec ymean_vec = ymean * ones<vec>(n);
  
  if (family_type == BINOMIAL) {
    // Start with (y + mean(y))/2 clamped to avoid boundary issues
    mu = clamp((y + ymean) / 2.0, 0.001, 0.999);
    eta = log(mu / (1.0 - mu));  // logit link
  } else if (family_type == GAMMA) {
    // Start with mean for gamma
    mu.fill(ymean);
    eta = 1.0 / mu;  // inverse link
  } else {
    // For other families, start with mean
    mu.fill(ymean);
    eta = mu;  // identity link initially
  }

  // Compute initial deviance
  double deviance = dev_resids_(y, mu, 0.0, wt, family_type);
  double null_deviance = dev_resids_(y, ymean_vec, 0.0, wt, family_type);

  uvec coef_status = ones<uvec>(p);

  umat fe_matrix;
  bool has_fe = group_indices.n_elem > 0;
  if (has_fe) {
    fe_matrix.set_size(n, group_indices.n_elem);
    fe_matrix.zeros();

    for (size_t k = 0; k < group_indices.n_elem; k++) {
      for (size_t g = 0; g < group_indices(k).n_elem; g++) {
        const uvec &group_obs = group_indices(k)(g);
        if (group_obs.n_elem > 0) {
          fe_matrix.submat(group_obs, uvec{k}).fill(g);
        }
      }
    }
  }

  // Copy design matrix and dependent variable for centering
  mat MX = X;
  vec MNU = zeros<vec>(n);
  vec nu_old = zeros<vec>(n);

  // IRLS loop - following the working algorithm structure
  for (size_t iter = 0; iter < iter_max; ++iter) {
    result.iter = iter + 1;
    
    double rho = 1.0;
    vec eta_old = eta;
    vec beta_old = beta;
    double dev_old = deviance;

    // Compute IRLS weights and working dependent variable
    vec mu_eta = d_inv_link(eta, family_type);  // d mu / d eta
    vec variance_vals = variance_(mu, 0.0, family_type);
    vec w = (wt % square(mu_eta)) / variance_vals;
    vec nu = (y - mu) / mu_eta;

    // Update centered dependent variable following the working algorithm
    MNU += (nu - nu_old);
    nu_old = nu;

    // Center variables if fixed effects present
    if (has_fe) {
      // Center MNU
      mat MNU_mat = MNU;
      
      std::string family_str;
      if (family_type == BINOMIAL)
        family_str = "binomial";
      else if (family_type == GAMMA)
        family_str = "gamma";
      else
        family_str = "gaussian";

      WeightedDemeanResult demean_result_nu =
          demean_variables(MNU_mat, fe_matrix, w, center_tol,
                           static_cast<int>(iter_center_max), family_str);

      if (!demean_result_nu.success) {
        return result;
      }
      
      MNU = demean_result_nu.demeaned_data.col(0);

      // Center MX
      WeightedDemeanResult demean_result_x =
          demean_variables(MX, fe_matrix, w, center_tol,
                           static_cast<int>(iter_center_max), family_str);

      if (!demean_result_x.success) {
        return result;
      }
      
      MX = demean_result_x.demeaned_data;
    }

    // Compute update step using enhanced workspace
    beta_results ws(n, p);
    get_beta(MX, MNU, y, w, n, p, ws, true, collin_tol, has_fe);

    coef_status = ws.coef_status;
    vec beta_upd = ws.coefficients;
    
    // Compute eta update: eta_upd = MX * beta_upd + nu - MNU
    vec eta_upd = MX * beta_upd + nu - MNU;

    // Step halving with three checks following the working algorithm
    bool step_accepted = false;
    for (size_t iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + rho * eta_upd;
      beta = beta_old + rho * beta_upd;
      mu = link_inv_(eta, family_type);
      deviance = dev_resids_(y, mu, 0.0, wt, family_type);
      
      double dev_ratio_inner = (deviance - dev_old) / (0.1 + std::abs(dev_old));

      bool dev_crit = is_finite(deviance);
      bool val_crit = valid_eta_(eta, family_type) && valid_mu_(mu, family_type);
      bool imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        step_accepted = true;
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed
    if (!step_accepted) {
      // If step halving does not improve the deviance, revert
      eta = eta_old;
      beta = beta_old;
      deviance = dev_old;
      mu = link_inv_(eta, family_type);
    }

    // Check convergence
    double dev_ratio = std::abs(deviance - dev_old) / (0.1 + std::abs(dev_old));
    if (dev_ratio < dev_tol) {
      result.conv = true;
      break;
    }
  }

  // Use workspace to store final results
  beta_results ws_final(n, p);
  ws_final.coefficients = beta;
  ws_final.eta = eta;
  ws_final.mu = mu;
  
  // Compute final weights for the last iteration
  vec mu_eta_final = d_inv_link(eta, family_type);
  vec variance_vals_final = variance_(mu, 0.0, family_type);
  vec w_final = (wt % square(mu_eta_final)) / variance_vals_final;
  
  ws_final.weights = w_final;
  ws_final.deviance = deviance;
  ws_final.null_deviance = null_deviance;
  ws_final.coef_status = coef_status;
  ws_final.conv = result.conv;
  ws_final.iter = result.iter;

  // Compute Hessian: H = X^T W X where W is diagonal weight matrix
  mat H = MX.t() * (MX.each_col() % w_final);
  ws_final.hessian = H;

  // Copy workspace results to final result
  ws_final.copy_glm_results_to(result);

  if (keep_mx) {
    result.mx = MX;
    result.has_mx = true;
  }

  if (has_fe) {
    // Use enhanced fixed effects extraction for generic IRLS
    std::string family_name = "unknown";
    if (family_type == BINOMIAL) family_name = "logit";
    else if (family_type == GAMMA) family_name = "gamma";
    else if (family_type == POISSON) family_name = "poisson";
    
    GetAlphaResult alpha_result =
        extract_model_fixef(result.fitted_values, eta, X, beta,
                           group_indices, family_name, center_tol, iter_center_max);
    result.fixed_effects = alpha_result.Alpha;
    result.nb_references = alpha_result.nb_references;
    result.is_regular = alpha_result.is_regular;
    result.has_fe = alpha_result.success;
  } else {
    result.has_fe = false;
  }

  return result;
}

// Binomial/Logit
GLMResult felogit_fit(const mat &X, const vec &y, const vec &wt,
                      const field<field<uvec>> &group_indices,
                      double center_tol, double dev_tol, bool keep_mx,
                      size_t iter_max, size_t iter_center_max,
                      size_t iter_inner_max, double collin_tol) {
  return generic_irls_fit<BINOMIAL>(X, y, wt, group_indices, center_tol,
                                    dev_tol, keep_mx, iter_max, iter_center_max,
                                    iter_inner_max, collin_tol);
}

// Gaussian regression
GLMResult fegaussian_fit(const mat &X, const vec &y, const vec &wt,
                         const field<field<uvec>> &group_indices,
                         double center_tol, double dev_tol, bool keep_mx,
                         size_t iter_max, size_t iter_center_max,
                         double collin_tol) {
  LMResult lm_result =
      felm_fit(X, y, wt, group_indices, center_tol, iter_center_max, 0, 0,
               collin_tol, any(wt != 1.0));

  GLMResult result;
  result.coefficients = lm_result.coefficients;
  result.fitted_values = lm_result.fitted;
  result.eta = lm_result.fitted;
  result.weights = lm_result.weights;
  result.hessian = lm_result.hessian;
  result.coef_status = lm_result.coef_status;
  result.conv = lm_result.success;
  result.iter = 1;

  result.deviance = dev_resids_(y, result.fitted_values, 0.0, wt, GAUSSIAN);

  double ymean = sum(wt % y) / sum(wt);
  vec mu_null(y.n_elem, fill::value(ymean));
  result.null_deviance = dev_resids_(y, mu_null, 0.0, wt, GAUSSIAN);

  // Copy enhanced fixed effects metadata from LM result
  if (lm_result.has_fe) {
    result.fixed_effects = lm_result.fixed_effects;
    result.nb_references = lm_result.nb_references;
    result.is_regular = lm_result.is_regular;
    result.has_fe = true;
  } else {
    result.has_fe = false;
  }

  if (keep_mx) {
    result.mx = X;
    result.has_mx = true;
  }

  return result;
}

// Gamma
GLMResult fegamma_fit(const mat &X, const vec &y, const vec &wt,
                      const field<field<uvec>> &group_indices,
                      double center_tol, double dev_tol, bool keep_mx,
                      size_t iter_max, size_t iter_center_max,
                      size_t iter_inner_max, double collin_tol) {
  return generic_irls_fit<GAMMA>(X, y, wt, group_indices, center_tol, dev_tol,
                                 keep_mx, iter_max, iter_center_max,
                                 iter_inner_max, collin_tol);
}

inline GLMResult feglm_fit(mat &MX, vec &beta, vec &eta, const vec &y,
                           const vec &wt, const double &theta,
                           const field<field<uvec>> &group_indices,
                           double center_tol, double dev_tol, bool keep_mx,
                           size_t iter_max, size_t iter_center_max,
                           size_t iter_inner_max, size_t iter_interrupt,
                           size_t iter_ssr, const std::string &fam,
                           FamilyType family_type, double collin_tol) {
  switch (family_type) {
    case POISSON:
      return fepoisson_fit(MX, y, wt, group_indices, center_tol, dev_tol,
                           keep_mx, iter_max, iter_center_max, iter_inner_max, collin_tol);

    case BINOMIAL:  // Logistic regression
      return felogit_fit(MX, y, wt, group_indices, center_tol, dev_tol, keep_mx,
                         iter_max, iter_center_max, iter_inner_max, collin_tol);

    case GAUSSIAN:
      return fegaussian_fit(MX, y, wt, group_indices, center_tol, dev_tol,
                            keep_mx, iter_max, iter_center_max, collin_tol);

    case GAMMA:
      return fegamma_fit(MX, y, wt, group_indices, center_tol, dev_tol, keep_mx,
                         iter_max, iter_center_max, iter_inner_max, collin_tol);

    default: {
      GLMResult result;
      result.conv = false;
      return result;
    }
  }
}

#endif  // CAPYBARA_GLM
