#ifndef CAPYBARA_LM
#define CAPYBARA_LM

struct LMResult {
  vec coefficients;
  vec fitted;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
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

// Convert group_indices to umat format for compatibility with demean_variables
// TODO: find a better structure to avoid this conversion
inline umat
convert_group_indices_to_umat(const field<field<uvec>> &group_indices,
                              size_t n_obs) {
  if (group_indices.n_elem == 0) {
    return umat();
  }

  const size_t Q = group_indices.n_elem;
  umat fe_matrix(n_obs, Q);

  for (size_t k = 0; k < Q; ++k) {
    vec fe_col(n_obs, fill::zeros);
    for (size_t g = 0; g < group_indices(k).n_elem; ++g) {
      const uvec &group_obs = group_indices(k)(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_col(group_obs(i)) = static_cast<double>(g);
      }
    }
    fe_matrix.col(k) = conv_to<uvec>::from(fe_col);
  }

  return fe_matrix;
}

// Extract fixed effects from the fitted model
inline field<vec>
extract_fixed_effects(const vec &fitted_values, const mat &X_orig,
                      const vec &coefficients,
                      const field<field<uvec>> &group_indices) {
  GetAlphaResult alpha_result = extract_model_fixef(
      fitted_values, fitted_values, // For linear models, both are the same
      X_orig, coefficients, group_indices, "gaussian");

  if (alpha_result.success) {
    return alpha_result.Alpha;
  } else {
    field<vec> empty_result(group_indices.n_elem);
    for (size_t q = 0; q < group_indices.n_elem; ++q) {
      empty_result(q).zeros(group_indices(q).n_elem);
    }
    return empty_result;
  }
}

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

  // Step 1: Joint demeaning
  mat X_demean;
  vec Y_demean;

  if (has_fixed_effects) {
    // Convert group indices to umat format
    // TODO: find a better structure to avoid this conversion
    umat fe_matrix = convert_group_indices_to_umat(group_indices, n);

    mat YX_combined = join_rows(y_orig, X_orig);

    WeightedDemeanResult demean_result = demean_variables(
        YX_combined, fe_matrix, w, center_tol, iter_center_max, "gaussian");

    if (!demean_result.success) {
      return res; // Demeaning failed
    }

    // Extract demeaned Y and X
    Y_demean = demean_result.demeaned_data.col(0);
    X_demean = demean_result.demeaned_data.cols(1, p_orig);

  } else {
    // No fixed effects - use original data
    Y_demean = y_orig;
    X_demean = X_orig;
  }

  // Step 2: WLS transformation if weights are provided
  mat X_final = X_demean;
  vec Y_final = Y_demean;

  if (has_weights) {
    // Apply weights: sqrt(w) * X and sqrt(w) * Y
    vec sqrt_w = sqrt(w);
    X_final = X_demean.each_col() % sqrt_w;
    Y_final = Y_demean % sqrt_w;
  }

  // Step 3: Regression
  ModelResults ws(n, p_orig);
  get_beta(X_final, Y_final, y_orig, w, n, p_orig, ws, has_weights, collin_tol,
           has_fixed_effects);

  if (!ws.success) {
    return res; // Regression failed
  }

  // Step 4: Extract results from ModelResults workspace
  res.coefficients = ws.coefficients;
  res.fitted = ws.fitted_values;
  res.residuals = ws.residuals;
  res.weights = ws.weights;
  res.hessian = ws.hessian;
  res.coef_status = ws.coef_status;
  res.success = ws.success;

  // Step 5: Extract fixed effects
  if (has_fixed_effects) {
    res.fixed_effects = extract_fixed_effects(res.fitted, X_orig,
                                              res.coefficients, group_indices);
    res.has_fe = true;
    res.is_regular = true;

    // Set number of references (one per dimension)
    res.nb_references.set_size(group_indices.n_elem);
    res.nb_references.fill(1);
  } else {
    res.has_fe = false;
    res.is_regular = true;
  }

  return res;
}

#endif // CAPYBARA_LM
