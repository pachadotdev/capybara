// Linear models with fixed effects Y = alpha + X beta + epsilon

#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

// Forward declaration
struct CapybaraParameters;

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
  bool has_fe = true;
  uvec iterations;

  mat TX; // Centered design matrix
  bool has_tx = false;

  InferenceLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), coef_status(p, fill::ones), success(false),
        has_fe(false), has_tx(false) {}
};

// Workspace structure to eliminate repeated allocations in felm_fit
struct FelmWorkspace {
  vec y_demeaned;           // Demeaned response vector
  vec x_beta;               // X*beta terms  
  vec pi;                   // Residual for fixed effects computation
  mat X_original;           // Original X matrix (view or copy as needed)
  vec y_original;           // Original y vector (view or copy as needed)
  
  // Cache for reducing allocations
  size_t cached_N, cached_P;
  bool is_initialized;
  
  // Default constructor
  FelmWorkspace() : cached_N(0), cached_P(0), is_initialized(false) {}
  
  FelmWorkspace(size_t N, size_t P) : cached_N(N), cached_P(P), is_initialized(true) {
    size_t safe_N = std::max(N, size_t(1));
    size_t safe_P = std::max(P, size_t(1));
    
    y_demeaned.set_size(safe_N);
    x_beta.set_size(safe_N);
    pi.set_size(safe_N);
    X_original.set_size(safe_N, safe_P);
    y_original.set_size(safe_N);
  }
  
  // Efficient resize that avoids reallocation when possible
  void ensure_size(size_t N, size_t P) {
    if (!is_initialized || N > cached_N || P > cached_P) {
      size_t new_N = std::max(N, cached_N);
      size_t new_P = std::max(P, cached_P);
      
      if (y_demeaned.n_elem < new_N) y_demeaned.set_size(new_N);
      if (x_beta.n_elem < new_N) x_beta.set_size(new_N);
      if (pi.n_elem < new_N) pi.set_size(new_N);
      if (X_original.n_rows < new_N || X_original.n_cols < new_P) X_original.set_size(new_N, new_P);
      if (y_original.n_elem < new_N) y_original.set_size(new_N);
      
      cached_N = new_N;
      cached_P = new_P;
      is_initialized = true;
    }
  }
  
  // Destructor to ensure proper cleanup
  ~FelmWorkspace() {
    clear();
  }
  
  // Method to clear and release memory
  void clear() {
    y_demeaned.reset();
    x_beta.reset();
    pi.reset();
    X_original.reset();
    y_original.reset();
    cached_N = 0;
    cached_P = 0;
    is_initialized = false;
  }
};

// Optimized crossprod function with better branch prediction
mat crossprod(const mat &X, const vec &w) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  
  if (w.n_elem == 1) {
    // Unweighted case - use optimized BLAS operations
    return X.t() * X;
  } else {
    // Weighted case - optimize for cache efficiency
    mat result(p, p);
    const double* w_ptr = w.memptr();
    
    for (size_t i = 0; i < p; ++i) {
      const double* Xi_ptr = X.colptr(i);
      for (size_t j = i; j < p; ++j) {
        const double* Xj_ptr = X.colptr(j);
        
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
          sum += Xi_ptr[k] * Xj_ptr[k] * w_ptr[k];
        }
        
        result(i, j) = sum;
        if (i != j) result(j, i) = sum; // Symmetry
      }
    }
    return result;
  }
}

// Optimized vectorized accumulation of fixed effects
inline void accumulate_fixed_effects(vec &fitted_values,
                                               const field<vec> &fixed_effects,
                                               const field<field<uvec>> &fe_groups) {
  const size_t K = fe_groups.n_elem;
  
  for (size_t k = 0; k < K; ++k) {
    const size_t J = fe_groups(k).n_elem;
    const vec &fe_k = fixed_effects(k);
    
    for (size_t j = 0; j < J; ++j) {
      const uvec &group_idx = fe_groups(k)(j);
      const double fe_value = fe_k(j);
      
      // Vectorized addition using pointer arithmetic for cache efficiency
      const size_t group_size = group_idx.n_elem;
      const uword* idx_ptr = group_idx.memptr();
      double* fitted_ptr = fitted_values.memptr();
      
      for (size_t t = 0; t < group_size; ++t) {
        fitted_ptr[idx_ptr[t]] += fe_value;
      }
    }
  }
}

// Optimized fitted values computation using workspace and vectorization
inline void fitted_values(FelmWorkspace *ws, InferenceLM &result,
                                           const CollinearityResult &collin_result,
                                           const field<field<uvec>> &fe_groups,
                                           const CapybaraParameters &params) {
  const size_t N = ws->y_original.n_elem;
  
  // Compute X*beta using workspace to avoid temporary allocations
  if (collin_result.has_collinearity && !collin_result.non_collinear_cols.is_empty()) {
    // Use subview to avoid creating temporary matrix
    auto X_sub = ws->X_original.cols(collin_result.non_collinear_cols);
    auto coef_sub = result.coefficients.elem(collin_result.non_collinear_cols);
    ws->x_beta = X_sub * coef_sub;
  } else if (!collin_result.has_collinearity && ws->X_original.n_cols > 0) {
    ws->x_beta = ws->X_original * result.coefficients;
  } else {
    ws->x_beta.zeros(N);
  }

  const bool has_fixed_effects = fe_groups.n_elem > 0;
  
  if (has_fixed_effects) {
    // pi = y_original - x_beta (using original data)
    ws->pi = ws->y_original - ws->x_beta;
    
    // Create workspace for alpha computation
    AlphaWorkspace alpha_workspace;
    
    // Use get_alpha to solve for individual fixed effects from pi
    result.fixed_effects = get_alpha(ws->pi, fe_groups, params.alpha_tol, 
                                   params.iter_alpha_max, &alpha_workspace);
    result.has_fe = true;

    // Initialize fitted values with X*beta
    result.fitted_values = ws->x_beta;
    
    // Vectorized accumulation of fixed effects
    accumulate_fixed_effects(result.fitted_values, result.fixed_effects, fe_groups);
  } else {
    // No fixed effects - fitted values are just X*beta
    result.fitted_values = ws->x_beta;
  }
}

InferenceLM felm_fit(mat &X, const vec &y, const vec &w,
                     const field<field<uvec>> &fe_groups,
                     const CapybaraParameters &params,
                     FelmWorkspace *workspace = nullptr) {
  const size_t N = y.n_elem;
  const size_t P = X.n_cols;
  
  // Initialize the result object
  InferenceLM result(N, P);
  result.weights = w;

  // Create or use provided workspace
  FelmWorkspace local_workspace;
  FelmWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, P);

  // Use workspace to avoid copies - store original data as views when possible
  ws->X_original = X;  // This will be a copy, but we need original for fitted values
  ws->y_original = y;  // This will be a copy, but we need original for residuals

  // Check for collinearity using parameters from control
  bool use_weights = !all(w == 1.0);
  CollinearityResult collin_result =
      check_collinearity(X, w, use_weights, params.collin_tol);

  // Check if we have fixed effects
  const bool has_fixed_effects = fe_groups.n_elem > 0;
  result.has_fe = has_fixed_effects;

  // Demean variables if we have fixed effects
  if (has_fixed_effects) {
    // Use workspace vector for demeaned y
    ws->y_demeaned = y;  // Copy once to workspace
    
    // Demean y
    center_variables(ws->y_demeaned, w, fe_groups, params.center_tol,
                     params.iter_center_max, params.iter_interrupt,
                     params.iter_ssr, params.accel_start, params.use_cg);

    // Demean X columns (X is modified in place)
    if (X.n_cols > 0) {
      center_variables(X, w, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       params.iter_ssr, params.accel_start, params.use_cg);
    }
  } else {
    // No fixed effects - use original y
    ws->y_demeaned = y;
  }

  // Store centered design matrix if requested
  if (params.keep_tx && X.n_cols > 0) {
    result.TX = X;
    result.has_tx = true;
  }

  // Create workspace for beta computation
  BetaWorkspace beta_workspace(X.n_rows, X.n_cols);

  // Compute beta coefficients on demeaned data
  InferenceBeta beta_result = get_beta(X, ws->y_demeaned, ws->y_demeaned, w,
                                       collin_result, false, false, &beta_workspace);

  // Copy results from beta computation
  result.coefficients = beta_result.coefficients;
  result.coef_status = collin_result.coef_status;
  result.hessian = beta_result.hessian;
  result.success = beta_result.success;

  // Compute fitted values using workspace to avoid temporary allocations
  fitted_values(ws, result, collin_result, fe_groups, params);

  // Compute residuals using workspace
  result.residuals = ws->y_original - result.fitted_values;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
