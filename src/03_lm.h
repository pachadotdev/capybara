// Linear models with fixed effects Y = alpha + X beta + epsilon

#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {

struct InferenceLM {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
  bool has_fe = true;
  uvec iterations;

  InferenceLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), coef_status(p, fill::ones), success(false),
        is_regular(true), has_fe(false) {}
};

mat crossprod(const mat &X, const vec &w) {
  if (w.n_elem == 1) {
    return X.t() * X;
  } else {
    return X.t() * diagmat(w) * X;
  }
}

// Optimized felm_fit implementation - matching alpaca's approach for linear models
// Debug version to diagnose the fixed effects issue
InferenceLM felm_fit(mat &X, const vec &y, const vec &w, 
                           const field<field<uvec>> &fe_groups,
                           const double center_tol, 
                           const size_t iter_center_max,
                           const size_t iter_interrupt, 
                           const size_t iter_ssr) {
  // Initialize the result object
  InferenceLM result(y.n_elem, X.n_cols);
  result.weights = w;  
  
  // Keep a copy of original X and y before any modifications
  mat X_original = X;
  vec y_original = y;
  
  // Check for collinearity
  bool use_weights = !all(w == 1.0);
  double collin_tol = 1.0e-7;
  CollinearityResult collin_result = check_collinearity(X, w, use_weights, collin_tol, false);
  
  // Check if we have fixed effects
  const bool has_fixed_effects = fe_groups.n_elem > 0;
  result.has_fe = has_fixed_effects;

  // Demean variables if we have fixed effects
  vec y_demeaned = y;
  
  if (has_fixed_effects) {
    // Demean y
    center_variables(y_demeaned, w, fe_groups, center_tol, iter_center_max,
                     iter_interrupt, iter_ssr);
    
    // Demean X columns
    if (X.n_cols > 0) {
      center_variables(X, w, fe_groups, center_tol, iter_center_max, 
                       iter_interrupt, iter_ssr);
    }
  }

  // Create workspace for beta computation
  BetaWorkspace workspace(X.n_rows, X.n_cols);
  
  // Compute beta coefficients on demeaned data
  InferenceBeta beta_result = get_beta(X, y_demeaned, y_demeaned, w, collin_result, 
                                       false, false, &workspace);
  
  // Copy results from beta computation
  result.coefficients = beta_result.coefficients;
  result.coef_status = collin_result.coef_status;
  result.hessian = beta_result.hessian;
  result.success = beta_result.success;
  
  // Compute X * beta using original (non-demeaned) data
  vec x_beta;
  if (collin_result.has_collinearity && !collin_result.non_collinear_cols.is_empty()) {
    x_beta = X_original.cols(collin_result.non_collinear_cols) * 
             result.coefficients.elem(collin_result.non_collinear_cols);
  } else if (!collin_result.has_collinearity && X.n_cols > 0) {
    x_beta = X_original * result.coefficients;
  } else {
    x_beta.zeros(y.n_elem);
  }
  
  // Extract fixed effects and compute final fitted values
  if (has_fixed_effects) {
    // Compute pi = y - X*beta (using original data)
    vec pi = y_original - x_beta;
    
    // Use get_alpha to solve for individual fixed effects from pi
    InferenceAlpha alpha_result = get_alpha(pi, fe_groups);
    result.fixed_effects = alpha_result.Alpha;
    
    // Compute final fitted values = X*beta + fixed effects
    result.fitted_values = x_beta;
    for (size_t k = 0; k < fe_groups.n_elem; ++k) {
      for (size_t j = 0; j < fe_groups(k).n_elem; ++j) {
        const uvec &group_idx = fe_groups(k)(j);
        for (size_t obs_idx : group_idx) {
          result.fitted_values(obs_idx) += result.fixed_effects(k)(j);
        }
      }
    }
  } else {
    // No fixed effects - fitted values are just X*beta
    result.fitted_values = x_beta;
  }
  
  // Compute residuals
  result.residuals = y_original - result.fitted_values;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
