#ifndef CAPYBARA_LM
#define CAPYBARA_LM

struct LMResult {
  vec coefficients;
  vec fitted;
  vec residuals;
  vec weights;
  mat hessian;
  mat scores;
  uvec coef_status;  // 1 = estimable, 0 = collinear (from get_beta)
  bool success;
  
  // Fixed effects support (matching GLMResult)
  field<vec> fixed_effects;
  bool has_fe = false;

  cpp11::list to_list() const {
    auto out = writable::list(
        {"coefficients"_nm = as_doubles(coefficients),
         "fitted.values"_nm = as_doubles(fitted),
         "weights"_nm = as_doubles(weights),
         "residuals"_nm = as_doubles(residuals),
         "hessian"_nm = as_doubles_matrix(hessian),
         "coef.status"_nm = as_integers(arma::conv_to<ivec>::from(
             coef_status))
         // "scores"_nm = as_doubles_matrix(scores),
         });

    // Add fixed effects if they exist
    if (has_fe && fixed_effects.n_elem > 0) {
      writable::list fe_list(fixed_effects.n_elem);
      for (size_t k = 0; k < fixed_effects.n_elem; ++k) {
        fe_list[k] = as_doubles(fixed_effects(k));
      }
      out.push_back({"fixed.effects"_nm = fe_list});
    }
    
    return out;
  }
};

inline mat crossprod(const mat &X, const vec &w) {
  if (all(w == 1.0)) {
    return X.t() * X;
  } else {
    mat wX = X.each_col() % w;
    return X.t() * wX;
  }
}

// Core function matching Python Feols workflow exactly
inline LMResult felm_fit(const mat &X_orig, const vec &y_orig, const vec &w,
                         const field<field<uvec>> &group_indices,
                         double center_tol, size_t iter_center_max,
                         size_t iter_interrupt, size_t iter_ssr,
                         double collin_tol, bool has_weights = false) {
  LMResult res;
  res.success = false;

  const size_t n = y_orig.n_elem;
  const size_t p_orig = X_orig.n_cols;
  const bool has_fixed_effects = group_indices.n_elem > 0;

  // Step 1: Joint demeaning (matches Python demean_model)
  mat X_demean;
  vec Y_demean;

  if (has_fixed_effects) {
    // Convert field<field<uvec>> to umat format
    umat fe_matrix;
    fe_matrix.set_size(n, group_indices.n_elem);
    fe_matrix.zeros();  // CRITICAL: Initialize to zeros

    for (size_t k = 0; k < group_indices.n_elem; k++) {
      for (size_t g = 0; g < group_indices(k).n_elem; g++) {
        const uvec &group_obs = group_indices(k)(g);
        if (group_obs.n_elem > 0) {
          fe_matrix.submat(group_obs, uvec{k}).fill(g);
        }
      }
    }

    // CRITICAL: Joint demeaning as in Python (matches demean_model)
    mat YX_combined = join_rows(y_orig, X_orig);
    WeightedDemeanResult demean_result =
        demean_variables(YX_combined, fe_matrix, w, center_tol,
                         static_cast<int>(iter_center_max), "gaussian");

    if (!demean_result.success) {
      return res;  // Demeaning failed
    }

    // Extract demeaned Y and X
    Y_demean = demean_result.demeaned_data.col(0);
    X_demean = demean_result.demeaned_data.cols(1, p_orig);

  } else {
    // No fixed effects - use original data
    Y_demean = y_orig;
    X_demean = X_orig;
  }

  // Step 2: WLS transformation (matches Python wls_transform)
  mat X_final = X_demean;
  vec Y_final = Y_demean;

  if (has_weights) {
    // Apply weights: sqrt(w) * X and sqrt(w) * Y
    vec sqrt_w = sqrt(w);
    X_final = X_demean.each_col() % sqrt_w;
    Y_final = Y_demean % sqrt_w;
  }

  // Step 3: Complete regression using updated get_beta (stores ALL results in
  // workspace)
  beta_results ws(n, p_orig);
  get_beta(X_final, Y_final, y_orig, w, n, p_orig, ws, has_weights, collin_tol,
           has_fixed_effects);

  if (!ws.success) {
    return res;  // Regression failed
  }

  // Step 4: Extract all results from workspace
  res.coefficients = ws.coefficients;
  res.fitted = ws.fitted_values;
  res.residuals = ws.residuals;
  res.weights = ws.weights;
  // res.scores = ws.scores;
  res.hessian = ws.hessian;
  res.coef_status = ws.coef_status;
  res.success = ws.success;

  // Step 5: Compute and store fixed effects if they exist (matching GLM approach)
  if (has_fixed_effects) {
    // Compute fixed effects residuals: fitted_values - X * coefficients
    vec fe_residuals = res.fitted - X_orig * res.coefficients;
    
    // Use get_alpha to recover fixed effects
    GetAlphaResult alpha_result = 
        get_alpha(fe_residuals, group_indices, center_tol, iter_center_max);
    
    res.fixed_effects = alpha_result.Alpha;
    res.has_fe = true;
  } else {
    res.has_fe = false;
  }

  return res;
}

#endif  // CAPYBARA_LM
