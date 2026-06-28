// Computing beta in a model with fixed effects Y = alpha + X beta
#ifndef CAPYBARA_BETA_H
#define CAPYBARA_BETA_H

namespace capybara {

struct InferenceBeta {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;
  double scale;
  double rank;
  bool success;

  InferenceBeta() : scale(0.0), rank(0.0), success(false) {}

  InferenceBeta(uword n, uword p)
      : coefficients(p, fill::none), fitted_values(n, fill::none),
        residuals(n, fill::none), weights(n, fill::ones),
        hessian(p, p, fill::zeros), coef_status(p, fill::ones), scale(0.0),
        rank(0.0), success(false) {}
};

struct CollinearityResult {
  uvec coef_status;
  uvec collinear_cols;
  uvec non_collinear_cols;
  bool has_collinearity;
  uword n_valid;
  mat R;

  CollinearityResult() : has_collinearity(false), n_valid(0) {}

  CollinearityResult(uword p)
      : coef_status(p, fill::ones), collinear_cols(p), non_collinear_cols(p),
        has_collinearity(false), n_valid(0) {}
};

// X'X with optional weights (X'WX where W = diag(w))
// Uses BLAS-optimized approach for all matrix sizes
inline mat crossprod(const mat &X, const vec &w = vec()) {
  if (X.is_empty()) {
    return mat();
  }

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  if (w.is_empty()) {
    return symmatu(X.t() * X);
  }

  // Compute sqrt(w) * X, then use BLAS for (sqrt(w)*X)' * (sqrt(w)*X)
  // This avoids creating the full N*N diagmat(w) and lets BLAS handle
  // memory access patterns and vectorization
  mat Xw(n, p);
  const double *w_ptr = w.memptr();
  for (uword j = 0; j < p; ++j) {
    double *xw_col = Xw.colptr(j);
    const double *x_col = X.colptr(j);
    for (uword i = 0; i < n; ++i) {
      xw_col[i] = std::sqrt(w_ptr[i]) * x_col[i];
    }
  }
  return symmatu(Xw.t() * Xw);
}

// X'Wy (or X'y if w empty)
inline vec crossprod_Xy(const mat &X, const vec &w, const vec &y) {
  if (w.is_empty()) {
    return X.t() * y;
  }
  return X.t() * (w % y);
}

// Flag regressors that are (numerically) absorbed by the fixed effects.
//
// After FE-centering, a regressor whose retained weighted variation is a
// negligible fraction of its original variation is collinear with the fixed
// effects (e.g. a dyadic/pair-constant variable together with a pair fixed
// effect). The normal equations for such a column are singular: solving them
// returns a meaningless (exploding or denormalized) coefficient instead of NA.
// Detecting this requires a *relative* test, since an absolute pivot tolerance
// on the centered cross-product fails when several absorbed columns remain
// mutually independent at the centering-noise level.
//
//   orig_ss(j)     = sum_i w_i * X(i,j)^2     (uncentered column)
//   centered_ss(j) = sum_i w_i * Xc(i,j)^2    (FE-centered column)
//   absorbed(j)    = orig_ss(j) <= 0 || centered_ss(j) / orig_ss(j) < tol
//
// X and Xc must have matching columns; `w` may be empty for the unweighted
// case. Returns a 0/1 vector (1 = absorbed) of length X.n_cols.
inline uvec flag_fe_absorbed(const mat &X, const mat &Xc, const vec &w,
                             double tol) {
  const uword p = X.n_cols;
  uvec absorbed(p, fill::zeros);
  const bool weighted = !w.is_empty();
  for (uword j = 0; j < p; ++j) {
    const double orig_ss =
        weighted ? dot(w % X.col(j), X.col(j)) : dot(X.col(j), X.col(j));
    if (orig_ss <= 0.0) {
      absorbed(j) = 1; // zero/empty regressor is not estimable
      continue;
    }
    const double cent_ss =
        weighted ? dot(w % Xc.col(j), Xc.col(j)) : dot(Xc.col(j), Xc.col(j));
    if (cent_ss / orig_ss < tol) {
      absorbed(j) = 1;
    }
  }
  return absorbed;
}

inline InferenceBeta get_beta(const mat &X, const vec &y, const vec &y_orig,
                              const vec &w,
                              const CollinearityResult &collin_result,
                              bool weighted, bool scale_X,
                              mat *cached_XtX = nullptr) {
  const uword n = X.n_rows;
  const uword p = X.n_cols;
  const uword p_orig =
      collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;

  InferenceBeta result(n, p_orig);

  // Early exit: no predictors
  if (p == 0) {
    result.success = true;
    result.coef_status = collin_result.coef_status;
    result.residuals = y_orig;
    result.weights = w;
    return result;
  }

  // Early exit: no observations
  if (y.is_empty()) {
    return result;
  }

  // Build or use cached X'WX
  mat XtX;
  if (cached_XtX && cached_XtX->n_rows == p) {
    XtX = *cached_XtX;
  } else {
    XtX = crossprod(X, w);
    if (cached_XtX) {
      *cached_XtX = XtX;
    }
  }

  // Regularization for numerical stability
  XtX.diag() += 1e-10;

  // X'Wy
  const vec Xty = crossprod_Xy(X, w, y);

  // Rank-revealing Cholesky
  mat R_rank;
  Col<uword> excluded_cols;
  uword rank_out = 0;

  if (!chol_rank(R_rank, excluded_cols, rank_out, XtX, "upper", 1e-10)) {
    cpp4r::stop("Failed to solve normal equations for beta estimation.");
  }

  // Independent columns (excluded_cols == 0 means kept)
  const uvec indep = find(excluded_cols == 0);
  const uword rnk = indep.n_elem;

  vec beta_reduced(p, fill::value(datum::nan));

  if (rnk == 0) {
    result.coef_status.zeros();
    result.hessian.zeros();
  } else {
    // Extract submatrices for independent columns
    const mat R_sub = R_rank.submat(indep, indep);
    const vec Xty_sub = Xty.elem(indep);

    // Solve R'R * beta = Xty via two triangular solves
    vec beta_sub;
    if (!solve(beta_sub, trimatl(R_sub.t()), Xty_sub, solve_opts::fast) ||
        !solve(beta_sub, trimatu(R_sub), beta_sub, solve_opts::fast)) {
      cpp4r::stop("Failed to solve triangular system in beta estimation.");
    }

    // Scatter into full beta vector
    beta_reduced.elem(indep) = beta_sub;

    // Coefficient status
    result.coef_status.zeros();
    result.coef_status.elem(indep).ones();

    // Hessian: place XtX into correct positions
    if (collin_result.has_collinearity) {
      const uvec &valid_cols = collin_result.non_collinear_cols;
      result.hessian.submat(valid_cols, valid_cols) = XtX;
    } else {
      result.hessian = XtX;
    }

    result.rank = static_cast<double>(rank_out);
  }

  // Expand coefficients if collinearity mapping exists
  if (collin_result.has_collinearity) {
    result.coefficients.fill(datum::nan);
    result.coefficients.elem(collin_result.non_collinear_cols) =
        beta_reduced.head(collin_result.non_collinear_cols.n_elem);
    result.coef_status = collin_result.coef_status;
  } else {
    result.coefficients = beta_reduced;
  }

  result.fitted_values = X * beta_reduced;
  result.residuals = y_orig - result.fitted_values;
  result.weights = w;
  result.success = true;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_BETA_H
