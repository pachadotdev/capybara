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
  mat scores;

  cpp11::list to_list(bool keep_mx = true) const {
    auto out = writable::list(
        {"coefficients"_nm = as_doubles(coefficients),
         "eta"_nm = as_doubles(eta),
         "fitted.values"_nm = as_doubles(fitted_values),
         "weights"_nm = as_doubles(weights),
         "hessian"_nm = as_doubles_matrix(hessian),
         "coef.status"_nm = as_integers(coef_status),
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
                        double collin_tol) {
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

  double last_deviance = ppml_deviance(y, mu);
  double crit = 1.0;
  bool stop_iterating = false;
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

  // IRLS loop
  for (size_t iter = 0; iter < iter_max; iter++) {
    result.iter = iter + 1;

    if (stop_iterating) {
      result.conv = true;
      break;
    }

    // Step 1: Working dependent variable
    vec Z = eta + y / mu - 1.0;
    vec reg_Z = Z;

    // Step 2: Joint demeaning
    mat ZX = join_rows(reg_Z, X);

    if (has_fe) {
      WeightedDemeanResult demean_result =
          demean_variables(ZX, fe_matrix, mu, center_tol,
                           static_cast<int>(iter_center_max), "poisson");

      if (!demean_result.success) {
        return result;
      }

      reg_Z = demean_result.demeaned_data.col(0);
      mat X_resid = demean_result.demeaned_data.cols(1, p);

      // Step 3: WLS estimation using enhanced workspace
      beta_results ws(n, p);
      get_beta_glm(X_resid, reg_Z, y, mu, n, p, ws, true, collin_tol, has_fe, eta, mu);

      // Track collinearity status and extract coefficients
      coef_status = ws.coef_status;
      vec delta_new = ws.coefficients;

      vec resid = reg_Z - X_resid * delta_new;

      // Step 4: Update mu and eta
      eta = Z - resid;
      mu = exp(eta);
      
      // Store current iteration results in workspace
      ws.eta = eta;
      ws.mu = mu;
      ws.residuals_working = ws.residuals;  // WLS residuals
      ws.residuals_response = y - mu;       // Response residuals

      // Copy results to output structure
      ws.copy_glm_results_to(result);
      result.fitted_values = mu;  // Ensure fitted_values = mu for PPML

    } else {
      // No fixed effects case - use enhanced workspace
      beta_results ws(n, p);
      get_beta_glm(X, reg_Z, y, mu, n, p, ws, true, collin_tol, false, eta, mu);

      coef_status = ws.coef_status;
      vec delta_new = ws.coefficients;

      eta = X * delta_new;
      mu = exp(eta);
      
      // Update workspace with current iteration results
      ws.eta = eta;
      ws.mu = mu;
      ws.residuals_response = y - mu;

      // Copy results to output structure
      ws.copy_glm_results_to(result);
      result.fitted_values = mu;  // Ensure fitted_values = mu for PPML
    }

    // Step 5: Check convergence (fixest approach)
    double deviance = ppml_deviance(y, mu);
    crit = std::abs(deviance - last_deviance) / (0.1 + std::abs(last_deviance));
    last_deviance = deviance;

    stop_iterating = (crit < dev_tol);
    result.deviance = deviance;
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
    vec fe_residuals = eta - X * result.coefficients;
    GetAlphaResult alpha_result =
        get_alpha(fe_residuals, group_indices, center_tol, iter_center_max);
    result.fixed_effects = alpha_result.Alpha;
    result.has_fe = true;
  } else {
    result.has_fe = false;
  }

  return result;
}

// Generic IRLS
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

  double ymean = sum(wt % y) / sum(wt);
  if (family_type == BINOMIAL) {
    mu = clamp((y + ymean) / 2.0, 0.001, 0.999);
    eta = log(mu / (1.0 - mu));  // logit link
  } else if (family_type == GAMMA) {
    mu.fill(ymean);
    eta = 1.0 / mu;
  } else {
    mu.fill(ymean);
    eta = mu;
  }

  double deviance = dev_resids_(y, mu, 0.0, wt, family_type);
  double deviance_old = deviance + 1.0;

  uvec coef_status = ones<uvec>(p);

  vec W_tilde, W;

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

  // IRLS loop
  for (size_t r = 0; r < iter_max; ++r) {
    result.iter = r + 1;

    if (r > 0) {
      double crit =
          std::abs(deviance - deviance_old) / (0.1 + std::abs(deviance_old));
      if (crit < dev_tol) {
        result.conv = true;
        break;
      }
    }

    // This follows alpaca Alternating Projections and Newton-Raphson approach

    deviance_old = deviance;

    // Step 1: w_tilde(r-1) and v(r-1)
    vec detadmu = 1.0 / d_inv_link(eta, family_type);
    W = 1.0 / (square(detadmu) % variance_(mu, 0.0, family_type));

    // Step 2: Get v_tilde(r-1) and X_tilde(r-1)
    W_tilde = sqrt(W);
    mat X_tilde = X.each_col() % W_tilde;
    vec v_tilde = W_tilde % ((y - mu) % detadmu);

    // Step 3: v_dotdot(r-1) and X_dotdot(r-1) (demeaning)
    vec v_dotdot = v_tilde;
    mat X_dotdot = X_tilde;

    if (has_fe) {
      mat vX_combined = join_rows(v_tilde, X_tilde);

      std::string family_str;
      if (family_type == BINOMIAL)
        family_str = "binomial";
      else if (family_type == GAMMA)
        family_str = "gamma";
      else
        family_str = "gaussian";

      WeightedDemeanResult demean_result =
          demean_variables(vX_combined, fe_matrix, W_tilde, center_tol,
                           static_cast<int>(iter_center_max), family_str);

      if (!demean_result.success) {
        return result;
      }

      v_dotdot = demean_result.demeaned_data.col(0);
      X_dotdot = demean_result.demeaned_data.cols(1, p);
    }

    // Step 4: beta(r) - beta(r-1) and check convergence using enhanced workspace
    beta_results ws(n, p);
    get_beta_glm(X_dotdot, v_dotdot, y, W_tilde, n, p, ws, true, collin_tol,
                 has_fe, eta, mu);

    coef_status = ws.coef_status;
    vec beta_update_diff = ws.coefficients - beta;

    // Step 5: Step halving if required
    double alpha = 1.0;
    bool step_accepted = false;
    const double step_halfing_tolerance = 1e-12;

    while (alpha > step_halfing_tolerance) {
      vec beta_try = beta + alpha * beta_update_diff;
      vec eta_try = eta + alpha * (X_dotdot * beta_update_diff) / W_tilde;
      vec mu_try = link_inv_(eta_try, family_type);
      double deviance_try = dev_resids_(y, mu_try, 0.0, wt, family_type);

      bool valid =
          valid_eta_(eta_try, family_type) && valid_mu_(mu_try, family_type);

      if (valid && deviance_try < deviance_old) {
        beta = beta_try;
        eta = eta_try;
        mu = mu_try;
        deviance = deviance_try;
        step_accepted = true;
        break;
      } else {
        alpha /= 2.0;
      }
    }

    if (!step_accepted) {
      break;
    }
  }

  // Use workspace to store final results and copy to result structure
  beta_results ws_final(n, p);
  ws_final.coefficients = beta;
  ws_final.eta = eta;
  ws_final.mu = mu;
  ws_final.weights = W_tilde;
  ws_final.W = W;
  ws_final.W_tilde = W_tilde;
  ws_final.deviance = deviance;
  ws_final.coef_status = coef_status;
  ws_final.conv = result.iter < iter_max;
  ws_final.iter = result.iter;
  
  // Null deviance
  vec mu_null(n, fill::value(sum(wt % y) / sum(wt)));
  ws_final.null_deviance = dev_resids_(y, mu_null, 0.0, wt, family_type);

  // Hessian
  mat X_final = X.each_col() % sqrt(W);
  ws_final.hessian = X_final.t() * X_final;

  // Copy workspace results to final result
  ws_final.copy_glm_results_to(result);

  if (keep_mx) {
    result.mx = X;
    result.has_mx = true;
  }

  if (has_fe) {
    vec fe_residuals = eta - X * beta;
    GetAlphaResult alpha_result =
        get_alpha(fe_residuals, group_indices, center_tol, iter_center_max);
    result.fixed_effects = alpha_result.Alpha;
    result.has_fe = true;
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

// Probit
// TODO: Proper link!
GLMResult feprobit_fit(const mat &X, const vec &y, const vec &wt,
                       const field<field<uvec>> &group_indices,
                       double center_tol, double dev_tol, bool keep_mx,
                       size_t iter_max, size_t iter_center_max,
                       size_t iter_inner_max, double collin_tol) {
  return felogit_fit(X, y, wt, group_indices, center_tol, dev_tol, keep_mx,
                     iter_max, iter_center_max, iter_inner_max, collin_tol);
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
                           keep_mx, iter_max, iter_center_max, collin_tol);

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
