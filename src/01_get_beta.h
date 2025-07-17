#ifndef CAPYBARA_BETA
#define CAPYBARA_BETA

using namespace arma;

// High-performance rank-revealing Cholesky with vectorization
struct BetaResult {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;
  bool success;

  BetaResult(size_t n, size_t p)
      : coefficients(p, fill::none),
        fitted_values(n, fill::none),
        residuals(n, fill::none),
        weights(n, fill::none),
        hessian(p, p, fill::none),
        coef_status(p, fill::none),
        success(false) {}
};

// Rank-revealing Cholesky - vectorized operations
inline bool cholesky_decomp(mat& R, uvec& excluded, const mat& XtX,
                            double tol) {
  const size_t p = XtX.n_cols;
  R.zeros(p, p);
  excluded.zeros(p);

  size_t n_excluded = 0;

  for (size_t j = 0; j < p; ++j) {
    // Compute diagonal element efficiently
    double R_jj = XtX(j, j);

    // Vectorized computation of sum of squares
    if (j > 0) {
      const rowvec R_row = R.submat(0, j, j - 1, j);
      R_jj -= dot(R_row, R_row);
    }

    // Check for collinearity
    if (R_jj < tol) {
      excluded(j) = 1;
      ++n_excluded;
      if (n_excluded == p) {
        return false;  // All variables excluded
      }
      continue;
    }

    R_jj = std::sqrt(R_jj);
    R(j, j) = R_jj;

    // Vectorized computation of off-diagonal elements
    if (j < p - 1) {
      vec values = XtX.submat(j + 1, j, p - 1, j);

      if (j > 0) {
        const mat R_prev = R.submat(0, j + 1, j - 1, p - 1);
        const vec R_j_prev = R.submat(0, j, j - 1, j);
        values -= R_prev.t() * R_j_prev;
      }

      values /= R_jj;
      R.submat(j, j + 1, j, p - 1) = values.t();
    }
  }

  return true;
}

// Forward/backward substitution
inline vec solve_triangular(const mat& R, const vec& b, bool upper = true) {
  const size_t n = R.n_cols;
  vec x(n, fill::none);

  if (upper) {
    // Backward substitution
    for (size_t i = n; i-- > 0;) {
      double sum = b(i);
      for (size_t j = i + 1; j < n; ++j) {
        sum -= R(i, j) * x(j);
      }
      x(i) = sum / R(i, i);
    }
  } else {
    // Forward substitution
    for (size_t i = 0; i < n; ++i) {
      double sum = b(i);
      for (size_t j = 0; j < i; ++j) {
        sum -= R(i, j) * x(j);
      }
      x(i) = sum / R(i, i);
    }
  }

  return x;
}

// Beta computation with pre-allocated workspace
inline BetaResult get_beta(const mat& X, const vec& y, const vec& y_orig,
                           const vec& w, double collin_tol,
                           bool has_weights = false,
                           bool has_fixed_effects = false) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;

  BetaResult result(n, p);

  if (p == 0) {
    result.success = true;
    return result;
  }

  // Pre-allocate matrices with appropriate size
  mat XtX(p, p, fill::none);
  vec XtY(p, fill::none);

  // Compute X'X and X'Y efficiently
  if (has_weights) {
    // Weighted case - vectorized computation
    const vec sqrt_w = sqrt(w);
    const mat X_weighted = X.each_col() % sqrt_w;

    XtX = X_weighted.t() * X_weighted;
    XtY = X.t() * (w % y);
  } else {
    // Unweighted case - use BLAS
    XtX = X.t() * X;
    XtY = X.t() * y;
  }

  // Rank-revealing Cholesky decomposition
  mat R(p, p, fill::none);
  uvec excluded(p, fill::none);

  if (!cholesky_decomp(R, excluded, XtX, collin_tol)) {
    result.coefficients.fill(datum::nan);
    result.coef_status.zeros();
    result.success = false;
    return result;
  }

  // Count non-excluded variables
  const size_t p_reduced = p - sum(excluded);

  if (p_reduced == 0) {
    result.coefficients.fill(datum::nan);
    result.coef_status.zeros();
    result.success = false;
    return result;
  }

  // Extract non-excluded elements efficiently
  uvec included_idx = find(excluded == 0);
  mat R_reduced = R.submat(included_idx, included_idx);
  vec XtY_reduced = XtY.elem(included_idx);

  // Solve triangular system efficiently
  vec z = solve_triangular(R_reduced, XtY_reduced, false);      // Forward
  vec coef_reduced = solve_triangular(R_reduced.t(), z, true);  // Backward

  // Place coefficients back in original positions
  result.coefficients.fill(datum::nan);
  result.coef_status = 1 - excluded;  // 1 for included, 0 for excluded

  for (size_t i = 0; i < p_reduced; ++i) {
    result.coefficients(included_idx(i)) = coef_reduced(i);
  }

  // Compute fitted values and residuals efficiently
  if (has_fixed_effects) {
    // For fixed effects models: fitted = y_orig - (y_demeaned - X_demeaned *
    // beta)
    const vec pred_demeaned = X * result.coefficients;
    result.fitted_values = y_orig - (y - pred_demeaned);
  } else {
    // Standard case: fitted = X * beta
    result.fitted_values = X * result.coefficients;
  }

  result.residuals = y_orig - result.fitted_values;

  if (has_weights) {
    result.residuals = result.residuals / sqrt(w);
  }

  result.weights = w;
  result.hessian = XtX;
  result.success = true;

  return result;
}

#endif  // CAPYBARA_BETA
