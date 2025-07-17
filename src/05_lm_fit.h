#ifndef CAPYBARA_LM_FIT
#define CAPYBARA_LM_FIT

// Original LMResult struct for compatibility
struct LMResult {
  vec coefficients;
  vec fitted;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;  // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  uvec nb_references;  // Number of references per dimension
  bool is_regular;     // Whether fixed effects are regular
  bool has_fe = false;

  cpp11::list to_list() const {
    auto out = writable::list({"coefficients"_nm = as_doubles(coefficients),
                               "fitted.values"_nm = as_doubles(fitted),
                               "weights"_nm = as_doubles(weights),
                               "residuals"_nm = as_doubles(residuals),
                               "hessian"_nm = as_doubles_matrix(hessian)});

    if (has_fe && fixed_effects.n_elem > 0) {
      writable::list fe_list(fixed_effects.n_elem);
      for (size_t k = 0; k < fixed_effects.n_elem; ++k) {
        fe_list[k] = as_doubles(fixed_effects(k));
      }
      out.push_back({"fixed.effects"_nm = fe_list});
      out.push_back({"nb_references"_nm = as_integers(nb_references)});
      out.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    }

    return out;
  }
};

// Complete LM fitting with correct joint demeaning
struct LMFitResult {
  vec coefficients;
  vec fitted_values;  // Note: different name than LMResult.fitted
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;
  bool success;

  // Fixed effects info
  field<vec> fixed_effects;
  bool has_fe;
  uvec iterations;

  LMFitResult(size_t n, size_t p)
      : coefficients(p, fill::none),
        fitted_values(n, fill::none),
        residuals(n, fill::none),
        weights(n, fill::none),
        hessian(p, p, fill::none),
        coef_status(p, fill::none),
        success(false),
        has_fe(false) {}

  // Convert to LMResult for R interface
  LMResult to_lm_result() const {
    LMResult result;
    result.coefficients = coefficients;
    result.fitted = fitted_values;  // Map fitted_values -> fitted
    result.residuals = residuals;
    result.weights = weights;
    result.hessian = hessian;
    result.coef_status = coef_status;
    result.success = success;
    result.fixed_effects = fixed_effects;
    result.has_fe = has_fe;
    result.is_regular = true;

    if (has_fe && fixed_effects.n_elem > 0) {
      result.nb_references.set_size(fixed_effects.n_elem);
      result.nb_references.fill(1);
    }

    return result;
  }
};

// Extract fixed effects after regression - umat version
inline field<vec> extract_fixed_effects(
    const vec& y_orig, const vec& fitted_values, const mat& X_orig, 
    const vec& coefficients, const umat& fe_matrix) {
  
  // Compute sum of fixed effects per observation
  vec sum_fe = fitted_values - X_orig * coefficients;
  
  // Extract individual fixed effects using the umat-based get_alpha function
  GetAlphaResult alpha_result = get_alpha_umat(sum_fe, fe_matrix, 1e-8, 10000);
  
  return alpha_result.Alpha;
}

// Main LM fitting function that works directly with umat
inline LMResult felm_fit(const mat& X_orig, const vec& y_orig, const vec& w,
                         const umat& fe_matrix,
                         double center_tol, size_t iter_center_max,
                         size_t iter_interrupt, size_t iter_ssr,
                         double collin_tol, bool has_weights = false) {
  const size_t n = y_orig.n_elem;
  const size_t p_orig = X_orig.n_cols;
  const bool has_fixed_effects = fe_matrix.n_cols > 0;

  LMFitResult result(n, p_orig);

  mat X_demean;
  vec Y_demean;
  
  if (has_fixed_effects) {
    // Combine Y and X for joint demeaning
    mat YX_combined = join_rows(y_orig, X_orig);

    // Joint demeaning using the efficient demean_variables function
    WeightedDemeanResult demean_result = demean_variables(
        YX_combined, fe_matrix, w, center_tol, iter_center_max, "gaussian");

    if (!demean_result.success) {
      result.success = false;
      return result.to_lm_result();
    }

    // Extract demeaned Y and X
    Y_demean = demean_result.demeaned_data.col(0);
    if (p_orig > 0) {
      X_demean = demean_result.demeaned_data.cols(1, demean_result.demeaned_data.n_cols - 1);
    } else {
      X_demean = mat(n, 0);
    }
    result.has_fe = true;
  } else {
    // No fixed effects
    Y_demean = y_orig;
    X_demean = X_orig;
    result.has_fe = false;
  }

  // Regression on demeaned data
  BetaResult beta_result = get_beta(X_demean, Y_demean, y_orig, w, collin_tol,
                                    has_weights, has_fixed_effects);

  if (!beta_result.success) {
    result.success = false;
    return result.to_lm_result();
  }

  // Transfer results
  result.coefficients = std::move(beta_result.coefficients);
  result.fitted_values = std::move(beta_result.fitted_values);
  result.residuals = std::move(beta_result.residuals);
  result.weights = std::move(beta_result.weights);
  result.hessian = std::move(beta_result.hessian);
  result.coef_status = std::move(beta_result.coef_status);

  // Extract fixed effects if present
  if (has_fixed_effects) {
    result.fixed_effects = extract_fixed_effects(
        y_orig, result.fitted_values, X_orig, result.coefficients, 
        fe_matrix);
  }

  result.success = true;
  return result.to_lm_result();
}

#endif  // CAPYBARA_LM_FIT
