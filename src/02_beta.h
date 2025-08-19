// Computing beta in a model with fixed effects Y = alpha + X beta

#ifndef CAPYBARA_BETA_H
#define CAPYBARA_BETA_H

namespace capybara {

// Define InferenceBeta structure
struct InferenceBeta {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear
  double scale;
  uvec pivot;
  double rank;
  bool success;

  // Default constructor
  InferenceBeta() : scale(0.0), rank(0.0), success(false) {}

  // Constructor with size parameters
  InferenceBeta(size_t n, size_t p)
      : coefficients(p, fill::zeros),
        fitted_values(n, fill::zeros), residuals(n, fill::zeros),
        weights(n, fill::ones), hessian(p, p, fill::zeros),
        coef_status(p, fill::ones), scale(0.0), rank(0.0), success(false) {}
};

struct CollinearityResult {
  uvec coef_status;        // 1 = estimable, 0 = collinear
  uvec collinear_cols;     // Vector of collinear columns (if any)
  uvec non_collinear_cols; // Vector of non-collinear columns (if any)
  bool has_collinearity;   // Collinearity detected?
  size_t n_valid;          // Number of valid (non-collinear) columns
  mat R;                   // R matrix from QR decomposition

  // Default constructor
  CollinearityResult() : has_collinearity(false), n_valid(0) {}

  // Constructor with size
  CollinearityResult(size_t p)
      : coef_status(p, fill::ones), collinear_cols(p), non_collinear_cols(p),
        has_collinearity(false), n_valid(0) {}
};

inline bool rank_revealing_cholesky(uvec &excluded, const mat &XtX,
                                    double tol) {
  const size_t p = XtX.n_cols;
  excluded.zeros(p);

  if (p == 0)
    return true;

  mat R(p, p, fill::zeros);

  double *R_ptr = R.memptr();
  uword *excluded_ptr = excluded.memptr();
  const double *XtX_ptr = XtX.memptr();

  // Track number of excluded variables to avoid expensive checks
  size_t n_excluded = 0;

  for (size_t j = 0; j < p; ++j) {

    double R_jj = XtX_ptr[j + j * p]; // Direct pointer access

    if (j > 0) {
      // Vectorized sum of squares computation
      double sum_squares = 0.0;
      const double *R_j_ptr = R_ptr + j * p;
      
      for (size_t k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          double R_kj = R_j_ptr[k];
          sum_squares += R_kj * R_kj;
        }
      }
      R_jj -= sum_squares;
    }

    if (R_jj < tol) {
      excluded_ptr[j] = 1;
      n_excluded++;

      // Early exit when all variables excluded
      if (n_excluded >= p)
        return false;
      continue;
    }

    R_jj = std::sqrt(R_jj);
    R_ptr[j + j * p] = R_jj;
    const double inv_R_jj = 1.0 / R_jj;

    // Vectorized computation of remaining columns
    for (size_t col = j + 1; col < p; ++col) {
      double R_j_col = XtX_ptr[j + col * p];

      // Optimize inner loop with pointer arithmetic
      const double *R_col_ptr = R_ptr + col * p;
      const double *R_j_ptr = R_ptr + j * p;
      
      for (size_t k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          R_j_col -= R_j_ptr[k] * R_col_ptr[k];
        }
      }

      R_ptr[j + col * p] = R_j_col * inv_R_jj;
    }
  }

  return n_excluded < p;
}

inline CollinearityResult
check_collinearity(mat &X, const vec &w, bool has_weights, double tolerance) {

  const size_t p = X.n_cols;
  const size_t n = X.n_rows;

  CollinearityResult result(p);

  if (p == 0) {
    result.coef_status = uvec();
    return result;
  }

  if (p == 1) {
    const double *col_ptr = X.colptr(0);
    const double *w_ptr = has_weights ? w.memptr() : nullptr;

    double mean_val = 0.0, sum_sq = 0.0, sum_w = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double val = col_ptr[i];
      double weight = has_weights ? w_ptr[i] : 1.0;

      if (has_weights) {
        val *= std::sqrt(weight);
        sum_w += weight;
      } else {
        sum_w += 1.0;
      }
      mean_val += val;
      sum_sq += val * val;
    }

    mean_val /= sum_w;
    double variance = (sum_sq / sum_w) - (mean_val * mean_val);

    if (variance < tolerance * tolerance) {
      result.coef_status.zeros();
      result.has_collinearity = true;
      result.n_valid = 0;
      result.non_collinear_cols = uvec();
      X.reset();
    }
    return result;
  }

  mat XtX(p, p, fill::none);
  if (has_weights) {
    const double *w_ptr = w.memptr();
    
    // Blocked computation for better cache locality
    const size_t block_size = 4; // Optimize for cache lines
    
    // Initialize upper triangle
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = i; j < p; ++j) {
        XtX(i, j) = 0.0;
      }
    }
    
    // Process in blocks to improve cache utilization
    for (size_t block_start = 0; block_start < n; block_start += block_size) {
      size_t block_end = std::min(block_start + block_size, n);
      
      for (size_t i = 0; i < p; ++i) {
        const double *Xi_ptr = X.colptr(i);
        for (size_t j = i; j < p; ++j) {
          const double *Xj_ptr = X.colptr(j);
          double block_sum = 0.0;
          
          // Vectorized inner loop over block
          for (size_t obs = block_start; obs < block_end; ++obs) {
            block_sum += Xi_ptr[obs] * Xj_ptr[obs] * w_ptr[obs];
          }
          XtX(i, j) += block_sum;
        }
      }
    }
    
    // Fill lower triangle
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = 0; j < i; ++j) {
        XtX(i, j) = XtX(j, i);
      }
    }
  } else {
    XtX = X.t() * X;
  }

  uvec excluded(p);
  bool success = rank_revealing_cholesky(excluded, XtX, tolerance);

  if (!success) {
    result.coef_status.zeros();
    result.has_collinearity = true;
    result.n_valid = 0;
    result.non_collinear_cols = uvec();
    X.reset();
    return result;
  }

  const uvec indep = find(excluded == 0);

  result.coef_status.zeros();
  if (!indep.is_empty()) {
    result.coef_status.elem(indep).ones();
  }
  result.has_collinearity = (indep.n_elem < p);
  result.n_valid = indep.n_elem;
  result.non_collinear_cols = indep;

  if (result.has_collinearity && !indep.is_empty()) {
    X = X.cols(indep);
  } else if (result.has_collinearity && indep.is_empty()) {
    X.reset();
  }

  return result;
}

struct BetaWorkspace {
  mat XtX;
  vec XtY;
  mat L;
  vec beta_work;
  mat X_weighted;
  vec y_weighted;
  vec sqrt_w;
  vec z_solve;
  
  // Cache for reducing allocations
  size_t cached_n, cached_p;
  bool is_initialized;

  // Default constructor
  BetaWorkspace() : cached_n(0), cached_p(0), is_initialized(false) {}

  BetaWorkspace(size_t n, size_t p) : cached_n(n), cached_p(p), is_initialized(true) {
    size_t safe_n = std::max(n, size_t(1));
    size_t safe_p = std::max(p, size_t(1));

    XtX.set_size(safe_p, safe_p);
    XtY.set_size(safe_p);
    L.set_size(safe_p, safe_p);
    beta_work.set_size(safe_p);
    X_weighted.set_size(safe_n, safe_p);
    y_weighted.set_size(safe_n);
    sqrt_w.set_size(safe_n);
    z_solve.set_size(safe_p);
  }
  
  // Efficient resize that avoids reallocation when possible
  void ensure_size(size_t n, size_t p) {
    if (!is_initialized || n > cached_n || p > cached_p) {
      size_t new_n = std::max(n, cached_n);
      size_t new_p = std::max(p, cached_p);
      
      if (XtX.n_rows < new_p || XtX.n_cols < new_p) XtX.set_size(new_p, new_p);
      if (XtY.n_elem < new_p) XtY.set_size(new_p);
      if (L.n_rows < new_p || L.n_cols < new_p) L.set_size(new_p, new_p);
      if (beta_work.n_elem < new_p) beta_work.set_size(new_p);
      if (X_weighted.n_rows < new_n || X_weighted.n_cols < new_p) X_weighted.set_size(new_n, new_p);
      if (y_weighted.n_elem < new_n) y_weighted.set_size(new_n);
      if (sqrt_w.n_elem < new_n) sqrt_w.set_size(new_n);
      if (z_solve.n_elem < new_p) z_solve.set_size(new_p);
      
      cached_n = new_n;
      cached_p = new_p;
      is_initialized = true;
    }
  }
  
  // Explicit cleanup method
  void clear() {
    XtX.reset();
    XtY.reset();
    L.reset();
    beta_work.reset();
    X_weighted.reset();
    y_weighted.reset();
    sqrt_w.reset();
    z_solve.reset();
    cached_n = 0;
    cached_p = 0;
    is_initialized = false;
  }
  
  // Destructor to ensure proper cleanup
  ~BetaWorkspace() {
    clear();
  }
};

// Optimized get_beta function
inline InferenceBeta get_beta(const mat &X, const vec &y, const vec &y_orig,
                              const vec &w,
                              const CollinearityResult &collin_result,
                              bool weighted, bool scale_X,
                              BetaWorkspace *workspace) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  const size_t p_orig =
      collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;
  const bool has_weights = !all(w == 1.0);

  // Initialize result with appropriate dimensions
  InferenceBeta result(n, p_orig);

  // If p = 0, there's nothing to estimate
  if (p == 0) {
    result.success = true;
    result.coefficients.zeros();
    result.coef_status = collin_result.coef_status;
    result.fitted_values.zeros();
    result.residuals = y_orig;
    result.weights = w;
    result.hessian.zeros();
    return result;
  }

  // If there's no workspace provided or y has zero size, return with error
  if (!workspace || y.n_elem == 0) {
    result.success = false;
    return result;
  }

  // Ensure workspace has sufficient capacity
  workspace->ensure_size(n, p);

  // Use workspace references to avoid pointer overhead
  mat &XtX = workspace->XtX;
  vec &Xty = workspace->XtY;
  mat &L = workspace->L;
  vec &beta_work = workspace->beta_work;
  vec &z_solve = workspace->z_solve;
  
  // Now that workspace is properly sized, use subviews to avoid resizing
  // Only use the portion we need without reallocating
  mat XtX_view = XtX.submat(0, 0, p-1, p-1);
  vec Xty_view = Xty.subvec(0, p-1);
  mat L_view = L.submat(0, 0, p-1, p-1);
  vec beta_work_view = beta_work.subvec(0, p-1);
  vec z_solve_view = z_solve.subvec(0, p-1);

  // Compute XtX and Xty with optimized memory access
  if (has_weights) {
    // Use workspace vectors to eliminate temporary allocations
    vec &sqrt_w = workspace->sqrt_w;
    mat &X_weighted = workspace->X_weighted;
    vec &y_weighted = workspace->y_weighted;
    
    // Use subviews to avoid resizing
    mat X_weighted_view = X_weighted.submat(0, 0, n-1, p-1);
    vec y_weighted_view = y_weighted.subvec(0, n-1);
    vec sqrt_w_view = sqrt_w.subvec(0, n-1);
    
    // Compute sqrt(w) once and reuse
    const double* w_ptr = w.memptr();
    double* sqrt_w_ptr = sqrt_w_view.memptr();
    for (size_t i = 0; i < n; ++i) {
      sqrt_w_ptr[i] = std::sqrt(w_ptr[i]);
    }
    
    // Compute weighted X and y in-place without copies
    const double* sqrt_w_ptr_const = sqrt_w_view.memptr();
    
    // Weighted X: reuse workspace matrix
    for (size_t j = 0; j < p; ++j) {
      const double* X_col = X.colptr(j);
      double* X_weighted_col = X_weighted_view.colptr(j);
      for (size_t i = 0; i < n; ++i) {
        X_weighted_col[i] = X_col[i] * sqrt_w_ptr_const[i];
      }
    }
    
    // Weighted y: reuse workspace vector
    const double* y_ptr = y.memptr();
    double* y_weighted_ptr = y_weighted_view.memptr();
    for (size_t i = 0; i < n; ++i) {
      y_weighted_ptr[i] = y_ptr[i] * sqrt_w_ptr_const[i];
    }

    // Use optimized BLAS operations
    XtX_view = X_weighted_view.t() * X_weighted_view;
    Xty_view = X_weighted_view.t() * y_weighted_view;
  } else {
    // Unweighted case - direct BLAS operations
    XtX_view = X.t() * X;
    Xty_view = X.t() * y;
  }

  // Solve the system using workspace to avoid allocations
  bool chol_success = chol(L_view, XtX_view, "lower");

  if (chol_success) {
    // Use workspace vectors for intermediate results
    z_solve_view = solve(trimatl(L_view), Xty_view, solve_opts::fast);
    beta_work_view = solve(trimatu(L_view.t()), z_solve_view, solve_opts::fast);
  } else {
    // Fallback to standard solve if Cholesky fails
    beta_work_view = solve(XtX_view, Xty_view, solve_opts::likely_sympd);
  }

  // Set the coefficient vector in the result (minimize copies)
  result.coefficients.fill(datum::nan); // Initialize with NaN for collinear variables
  if (collin_result.has_collinearity) {
    // Only assign the coefficients for non-collinear columns
    if (collin_result.non_collinear_cols.n_elem > 0 && 
        collin_result.non_collinear_cols.n_elem <= p_orig &&
        beta_work_view.n_elem >= collin_result.non_collinear_cols.n_elem) {
      // Extract only the coefficients we computed
      vec beta_subset = beta_work_view.subvec(0, collin_result.non_collinear_cols.n_elem - 1);
      result.coefficients.elem(collin_result.non_collinear_cols) = beta_subset;
    }
  } else {
    if (beta_work_view.n_elem >= p_orig) {
      result.coefficients = beta_work_view.subvec(0, p_orig - 1);
    } else {
      result.coefficients = beta_work_view;
    }
  }

  result.coef_status = collin_result.coef_status;

  // Calculate fitted values - handle collinearity case properly
  if (collin_result.has_collinearity && collin_result.non_collinear_cols.n_elem > 0) {
    // Extract only the coefficients we computed for matrix multiplication
    vec beta_subset = beta_work_view.subvec(0, collin_result.non_collinear_cols.n_elem - 1);
    result.fitted_values = X * beta_subset;
  } else {
    // No collinearity or edge case
    if (beta_work_view.n_elem >= p) {
      result.fitted_values = X * beta_work_view.subvec(0, p - 1);
    } else {
      result.fitted_values = X * beta_work_view;
    }
  }

  // Calculate residuals in-place to avoid temporary
  result.residuals = y_orig - result.fitted_values;

  result.weights = w;

  // Store hessian for standard errors (minimize copies)
  result.hessian.set_size(p_orig, p_orig);
  result.hessian.zeros();

  if (collin_result.has_collinearity) {
    if (collin_result.non_collinear_cols.n_elem > 0) {
      // Only assign the portion of XtX that corresponds to non-collinear variables
      mat XtX_subset = XtX_view.submat(0, 0, 
                                  collin_result.non_collinear_cols.n_elem - 1,
                                  collin_result.non_collinear_cols.n_elem - 1);
      result.hessian(collin_result.non_collinear_cols,
                     collin_result.non_collinear_cols) = XtX_subset;
    }
  } else {
    result.hessian = XtX_view;
  }

  result.success = true;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_BETA_H