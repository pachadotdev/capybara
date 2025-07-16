#ifndef CAPYBARA_GLM
#define CAPYBARA_GLM

struct GLMResult {
  vec coefficients;
  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
  bool has_fe = true;
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  size_t iter;
  mat mx; // only if keep_mx = true
  bool has_mx = false;
  uvec coef_status; // 1 = valid, 0 = collinear

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

    if (keep_mx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(mx)});
    }

    return out;
  }
};

// Extract fixed effects from GLM results
inline field<vec>
extract_glm_fixed_effects(const vec &eta, const mat &X_orig,
                          const vec &coefficients,
                          const field<field<uvec>> &group_indices) {
  const size_t Q = group_indices.n_elem;
  field<vec> fixed_effects(Q);

  if (Q == 0) {
    return fixed_effects;
  }

  // Compute fixed effects component
  vec fe_component;
  if (X_orig.n_cols > 0 && coefficients.n_elem > 0) {
    vec linear_part = X_orig * coefficients;
    fe_component = eta - linear_part;
  } else {
    fe_component = eta;
  }

  // Extract fixed effects for each dimension
  for (size_t q = 0; q < Q; ++q) {
    const size_t n_groups = group_indices(q).n_elem;
    vec fe_q(n_groups, fill::zeros);

    // Compute group means
    for (size_t g = 0; g < n_groups; ++g) {
      const uvec &group_obs = group_indices(q)(g);
      if (group_obs.n_elem > 0) {
        double group_sum = 0.0;
        for (size_t i = 0; i < group_obs.n_elem; ++i) {
          group_sum += fe_component(group_obs(i));
        }
        fe_q(g) = group_sum / static_cast<double>(group_obs.n_elem);
      }
    }

    // Set reference group (first group) to zero
    // TODO: check this later to match LM logic
    if (n_groups > 0) {
      double reference_value = fe_q(0);
      fe_q -= reference_value;
    }

    fixed_effects(q) = fe_q;
  }

  return fixed_effects;
}

GLMResult feglm_fit(const mat &MX, const vec &y, const vec &wt,
                    const field<field<uvec>> &group_indices, double center_tol,
                    double dev_tol, bool keep_mx, size_t iter_max,
                    size_t iter_center_max, size_t iter_inner_max,
                    const std::string &family, double collin_tol) {
  GLMResult result;
  result.conv = false;
  result.iter = 0;

  const size_t n = y.n_elem;
  const size_t p = MX.n_cols;

  // Get family type
  FamilyType family_type = get_family_type(family);

  // Handle Gaussian case by delegating to LM
  if (family_type == GAUSSIAN) {
    LMResult lm_result =
        felm_fit(MX, y, wt, group_indices, center_tol, iter_center_max, 0, 0,
                 collin_tol, any(wt != 1.0));

    result.coefficients = lm_result.coefficients;
    result.fitted_values = lm_result.fitted;
    result.eta = lm_result.fitted;
    result.weights = lm_result.weights;
    result.hessian = lm_result.hessian;
    result.coef_status = lm_result.coef_status;
    result.conv = lm_result.success;
    result.iter = 1;

    // Gaussian deviance
    vec residuals = y - result.fitted_values;
    result.deviance = sum(wt % square(residuals));

    // Null deviance
    double ymean = sum(wt % y) / sum(wt);
    vec resid_null = y - ymean;
    result.null_deviance = sum(wt % square(resid_null));

    // Copy fixed effects from LM result
    if (lm_result.has_fe) {
      result.fixed_effects = lm_result.fixed_effects;
      result.nb_references = lm_result.nb_references;
      result.is_regular = lm_result.is_regular;
      result.has_fe = true;
    } else {
      result.has_fe = false;
    }

    if (keep_mx) {
      result.mx = MX;
      result.has_mx = true;
    }

    return result;
  }

  // For non-Gaussian families, validate response
  if (!valid_mu_(y, family_type)) {
    return result;
  }

  // Initialize
  double mean_y = sum(wt % y) / sum(wt);
  vec mu;
  vec eta;

  if (family_type == POISSON) {
    mu = (y + mean_y) / 2.0;
    eta = log(clamp(mu, 1e-12, 1e12));
  } else if (family_type == BINOMIAL) {
    mu = clamp((y + mean_y) / 2.0, 0.001, 0.999);
    eta = log(mu / (1.0 - mu)); // logit link
  } else {
    return result; // Unsupported family
  }

  result.deviance = dev_resids_(y, mu, 0.0, wt, family_type);
  uvec coef_status = ones<uvec>(p);

  // Convert group indices for demeaning
  umat fe_matrix;
  bool has_fe = group_indices.n_elem > 0;
  if (has_fe) {
    fe_matrix = convert_group_indices_to_umat(group_indices, n);
  }

  // Storage for IRLS algorithm
  vec eta_old, z, w;
  double devold = result.deviance;
  ModelResults ws(n, p);

  // IRLS loop
  for (size_t iter = 0; iter < iter_max; iter++) {
    result.iter = iter + 1;
    eta_old = eta;

    // Compute mu.eta and variance using exponential family functions
    vec mu_eta_val = d_inv_link(eta, family_type);
    vec var_mu = variance_(mu, 0.0, family_type);

    // Check variance
    if (any(var_mu <= 0) || any(var_mu != var_mu)) {
      break;
    }

    // Working response and weights
    z = eta + (y - mu) / mu_eta_val;
    w = wt % square(mu_eta_val) / var_mu;

    // Check weights
    uvec zero_w = (w <= 0);
    if (any(zero_w) && all(zero_w)) {
      break;
    }

    // Weighted least squares
    if (has_fe) {
      // Joint demeaning
      mat combined = join_rows(z, MX);
      WeightedDemeanResult demean_result = demean_variables(
          combined, fe_matrix, w, center_tol, iter_center_max, family);

      if (!demean_result.success) {
        break;
      }

      vec z_demean = demean_result.demeaned_data.col(0);
      mat X_demean = demean_result.demeaned_data.cols(1, p);
      get_beta(X_demean, z_demean, y, w, n, p, ws, true, collin_tol, has_fe);
    } else {
      mat MX_copy = MX; // get_beta needs non-const reference
      get_beta(MX_copy, z, y, w, n, p, ws, true, collin_tol, false);
    }

    if (!ws.success || any(ws.coefficients != ws.coefficients)) {
      if (iter == 0) {
        return result; // Failed at first iteration
      }
      break; // Divergence
    }

    coef_status = ws.coef_status;
    vec new_eta = ws.fitted_values;

    // Step halving
    double rho = 1.0;
    bool step_accepted = false;

    for (size_t iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + rho * (new_eta - eta_old);
      mu = link_inv_(eta, family_type);
      double deviance = dev_resids_(y, mu, 0.0, wt, family_type);

      double dev_ratio = (deviance - devold) / (0.1 + std::abs(devold));

      bool dev_finite = is_finite(deviance);
      bool mu_valid = valid_mu_(mu, family_type);
      bool eta_valid = valid_eta_(eta, family_type);
      bool dev_improved = (dev_ratio <= -dev_tol);

      if (dev_finite && mu_valid && eta_valid && dev_improved) {
        result.deviance = deviance;
        step_accepted = true;
        break;
      }

      rho *= 0.5;
    }

    if (!step_accepted) {
      eta = eta_old;
      mu = link_inv_(eta, family_type);
      result.deviance = devold;
    }

    // Check convergence
    double dev_ratio =
        std::abs(result.deviance - devold) / (0.1 + std::abs(result.deviance));
    if (dev_ratio < dev_tol) {
      result.conv = true;
      break;
    }

    devold = result.deviance;
  }

  // Final results
  result.coefficients = ws.coefficients;
  result.eta = eta;
  result.fitted_values = mu;
  result.weights = w;
  result.coef_status = coef_status;

  // Hessian
  // TODO: check this later
  if (has_fe) {
    mat combined = join_rows(z, MX);
    WeightedDemeanResult demean_result = demean_variables(
        combined, fe_matrix, w, center_tol, iter_center_max, family);
    if (demean_result.success) {
      mat X_demean = demean_result.demeaned_data.cols(1, p);
      result.hessian = X_demean.t() * (X_demean.each_col() % w);
    }
  } else {
    result.hessian = MX.t() * (MX.each_col() % w);
  }

  // Null deviance
  vec mu_null(y.n_elem, fill::value(mean_y));
  result.null_deviance = dev_resids_(y, mu_null, 0.0, wt, family_type);

  if (keep_mx) {
    result.mx = MX;
    result.has_mx = true;
  }

  // Extract fixed effects
  if (has_fe) {
    result.fixed_effects =
        extract_glm_fixed_effects(eta, MX, result.coefficients, group_indices);
    result.has_fe = true;
    result.is_regular = true;
    result.nb_references.set_size(group_indices.n_elem);
    result.nb_references.fill(1);
  } else {
    result.has_fe = false;
  }

  return result;
}

#endif // CAPYBARA_GLM
