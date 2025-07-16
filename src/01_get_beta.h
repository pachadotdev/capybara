#ifndef CAPYBARA_BETA
#define CAPYBARA_BETA

struct ModelResults {
  // Matrix computations
  mat XtX;
  vec XtY;
  mat decomp;
  vec work;
  mat Xt;
  mat Q;
  mat XW;

  // Core results
  vec coefficients;
  uvec coef_status;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  bool success;

  // GLM-specific fields
  vec eta;                    // Linear predictor
  vec mu;                     // Mean values (response scale)
  double deviance;            // Current deviance
  double null_deviance;       // Null deviance
  size_t iter;               // Number of iterations
  bool conv;                 // Convergence status
  vec residuals_working;     // Working residuals
  vec residuals_response;    // Response residuals
  field<vec> fixed_effects;  // Recovered fixed effects
  bool has_fe;               // Whether fixed effects are present
  
  // GLM temporary workspace
  vec W;                     // IRLS weights
  vec W_tilde;              // Square root of IRLS weights
  vec Z;                    // Working dependent variable
  vec v_tilde;              // Weighted working variable
  mat X_tilde;              // Weighted design matrix
  mat X_dotdot;             // Demeaned weighted design matrix
  vec v_dotdot;             // Demeaned weighted variable

  ModelResults(size_t n, size_t p)
      : XtX(p, p, fill::none),
        XtY(p, fill::none),
        decomp(p, p, fill::none),
        work(p, fill::none),
        Xt(p, n, fill::none),
        Q(p, 0, fill::none),
        XW(n, 0, fill::none),
        coefficients(p, fill::zeros),
        coef_status(p, fill::ones),
        fitted_values(n, fill::none),
        residuals(n, fill::none),
        weights(n, fill::none),
        hessian(p, p, fill::none),
        success(false),
        // GLM-specific initialization
        eta(n, fill::none),
        mu(n, fill::none),
        deviance(0.0),
        null_deviance(0.0),
        iter(0),
        conv(false),
        residuals_working(n, fill::none),
        residuals_response(n, fill::none),
        has_fe(false),
        // GLM workspace initialization
        W(n, fill::none),
        W_tilde(n, fill::none),
        Z(n, fill::none),
        v_tilde(n, fill::none),
        X_tilde(n, p, fill::none),
        X_dotdot(n, p, fill::none),
        v_dotdot(n, fill::none) {}

  // Template method to copy GLM-specific results to any result structure
  template<typename ResultType>
  void copy_glm_results_to(ResultType &result) const {
    result.coefficients = coefficients;
    result.eta = eta;
    result.fitted_values = mu.n_elem > 0 ? mu : fitted_values;
    result.weights = W_tilde.n_elem > 0 ? W_tilde : weights;
    result.hessian = hessian;
    result.coef_status = coef_status;
    result.conv = conv;
    result.iter = iter;
    result.deviance = deviance;
    result.null_deviance = null_deviance;
    
    // PPML-specific fields (only copy if they exist in both structures)
    if (residuals_working.n_elem > 0) {
      result.residuals_working = residuals_working;
    }
    if (residuals_response.n_elem > 0) {
      result.residuals_response = residuals_response;
    }
    if (has_fe && fixed_effects.n_elem > 0) {
      result.fixed_effects = fixed_effects;
      result.has_fe = true;
    }
  }
};

// Typedef for consistency with usage in other files
using beta_results = ModelResults;

// Rank-revealing Cholesky decomposition with implicit pivoting
// Based on fixest's cpp_cholesky implementation
inline void get_beta(mat &MX, const vec &MNU, const vec &y_orig, const vec &w,
                     const uword n, const uword p, ModelResults &ws,
                     bool use_weights, double collin_tol,
                     bool has_fixed_effects = false) {
  ws.coefficients.set_size(p);
  ws.coefficients.fill(datum::nan);
  ws.coef_status.zeros(p);
  ws.success = false;

  if (p == 0) {
    ws.success = true;
    return;
  }

  // Compute X'X and X'Y
  if (use_weights) {
    const vec sqrt_w = sqrt(w);
    ws.XW = MX.each_col() % sqrt_w;
    ws.XtX = ws.XW.t() * ws.XW;
    ws.XtY = MX.t() * (w % MNU);
  } else {
    ws.XtX = MX.t() * MX;
    ws.XtY = MX.t() * MNU;
  }

  // Rank-revealing Cholesky decomposition with implicit pivoting
  mat R(p, p, fill::zeros);
  uvec id_excl(p, fill::zeros);
  uword n_excl = 0;
  double min_norm = ws.XtX(0, 0);
  
  // Rank-revealing Cholesky with implicit pivoting
  for (uword j = 0; j < p; ++j) {
    // Compute diagonal element
    double R_jj = ws.XtX(j, j);
    
    for (uword k = 0; k < j; ++k) {
      if (id_excl(k)) continue;
      R_jj -= R(k, j) * R(k, j);
    }
    
    // Check for collinearity
    if (R_jj < collin_tol) {
      n_excl++;
      id_excl(j) = 1;
      
      // Corner case: all variables excluded
      if (n_excl == p) {
        ws.coefficients.fill(datum::nan);
        ws.coef_status.zeros();
        ws.success = false;
        return;
      }
      continue;
    }
    
    if (min_norm > R_jj) min_norm = R_jj;
    
    R_jj = sqrt(R_jj);
    R(j, j) = R_jj;
    
    // Compute off-diagonal elements
    for (uword i = j + 1; i < p; ++i) {
      double value = ws.XtX(i, j);
      for (uword k = 0; k < j; ++k) {
        if (id_excl(k)) continue;
        value -= R(k, i) * R(k, j);
      }
      R(j, i) = value / R_jj;
    }
  }
  
  // Reconstruct R matrix if variables were excluded
  uword p_reduced = p - n_excl;
  if (n_excl > 0) {
    // Find first excluded variable
    uword j_start = 0;
    while (j_start < p && !id_excl(j_start)) ++j_start;
    
    // Compact the matrix
    uword n_j_excl = 0;
    for (uword j = j_start; j < p; ++j) {
      if (id_excl(j)) {
        ++n_j_excl;
        continue;
      }
      
      uword n_i_excl = 0;
      for (uword i = 0; i <= j; ++i) {
        if (id_excl(i)) {
          ++n_i_excl;
          continue;
        }
        R(i - n_i_excl, j - n_j_excl) = R(i, j);
      }
    }
  }
  
  // Store decomposition
  ws.decomp = R.submat(0, 0, p_reduced - 1, p_reduced - 1);
  
  // Set coefficient status
  for (uword j = 0; j < p; ++j) {
    ws.coef_status(j) = id_excl(j) ? 0 : 1;
  }
  
  // Solve for coefficients using forward/backward substitution
  if (p_reduced > 0) {
    // Extract non-excluded elements from XtY
    vec XtY_reduced(p_reduced);
    uword idx = 0;
    for (uword j = 0; j < p; ++j) {
      if (!id_excl(j)) {
        XtY_reduced(idx++) = ws.XtY(j);
      }
    }
    
    // Forward substitution: solve L * z = XtY_reduced
    vec z = solve(trimatl(ws.decomp), XtY_reduced, solve_opts::fast);
    
    // Backward substitution: solve L^T * coef_reduced = z  
    vec coef_reduced = solve(trimatu(ws.decomp.t()), z, solve_opts::fast);
    
    // Place coefficients back in original positions
    idx = 0;
    for (uword j = 0; j < p; ++j) {
      if (!id_excl(j)) {
        ws.coefficients(j) = coef_reduced(idx++);
      }
    }
  }

  // 1. Fitted values
  if (has_fixed_effects) {
    vec prediction_demeaned = MX * ws.coefficients;
    ws.fitted_values = y_orig - (MNU - prediction_demeaned);
  } else {
    ws.fitted_values = MX * ws.coefficients;
  }

  // 2. Residuals
  ws.residuals = y_orig - ws.fitted_values;

  if (use_weights) {
    ws.residuals = ws.residuals / sqrt(w);
  }

  // 3. Weights
  ws.weights = w;

  // 4. Hessian (use X'X or weighted version)
  ws.hessian = ws.XtX;

  ws.success = true;
}

#endif  // CAPYBARA_BETA
