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
  uvec pivot;
  double rank;
  bool success;

  InferenceBeta() : scale(0.0), rank(0.0), success(false) {}

  InferenceBeta(uword n, uword p)
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
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

inline mat crossprod(const mat &X) {
  if (X.is_empty()) {
    return mat();
  }
  return symmatu(X.t() * X);
}

inline mat crossprod(const mat &X, const vec &w) {
  if (X.is_empty()) {
    return mat();
  }

  if (w.is_empty() || !any(w != 1.0)) {
    return symmatu(X.t() * X);
  }

  const mat Xw = X.each_col() % sqrt(w);
  return symmatu(Xw.t() * Xw);
}

inline vec crossprod_Xy(const mat &X, const vec &w, const vec &y) {
  return X.t() * (w % y);
}

inline CollinearityResult
check_collinearity(mat &X, const vec &w, bool has_weights, double tolerance) {

  const uword p = X.n_cols;

  CollinearityResult result(p);

  if (p == 0) {
    result.coef_status.reset();
    return result;
  }

  // For single column, check if variance is near zero
  if (p == 1) {
    double variance;

    if (has_weights) {
      const double sum_w = accu(w);
      const vec &x = X.col(0);
      const double mean_val = dot(x, w) / sum_w;
      variance = dot(w, square(x)) / sum_w - mean_val * mean_val;
    } else {
      variance = var(X.col(0), 1);
    }

    if (variance < tolerance * tolerance) {
      result.coef_status.zeros();
      result.has_collinearity = true;
      result.n_valid = 0;
      result.non_collinear_cols.reset();
      X.reset();
    }
    return result;
  }

  const mat XtX = has_weights ? crossprod(X, w) : crossprod(X);

  mat Q, R;
  qr_econ(Q, R, XtX);

  // Vectorized collinearity detection using abs() on diagonal
  const vec diag_R = abs(diagvec(R));
  const uvec excluded = conv_to<uvec>::from(diag_R < tolerance);
  const uvec indep = find(excluded == 0);

  result.coef_status.zeros();
  if (indep.n_elem > 0) {
    result.coef_status.elem(indep).ones();
  }

  result.has_collinearity = (indep.n_elem < p);
  result.n_valid = indep.n_elem;
  result.non_collinear_cols = indep;

  if (result.has_collinearity) {
    X = indep.n_elem > 0 ? X.cols(indep) : mat();
  }

  return result;
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

  const bool has_weights = w.n_elem > 0 && any(w != 1.0);

  InferenceBeta result(n, p_orig);

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

  if (y.n_elem == 0) {
    result.success = false;
    return result;
  }

  // Solve normal equations
  mat XtX;

  if (cached_XtX && cached_XtX->n_rows == p && cached_XtX->n_cols == p) {
    XtX = *cached_XtX;
  } else {
    XtX = has_weights ? crossprod(X, w) : crossprod(X);
    if (cached_XtX) {
      *cached_XtX = XtX;
    }
  }

  XtX.diag() += 1e-10; // Regularization for numerical stability

  const vec Xty = has_weights ? crossprod_Xy(X, w, y) : X.t() * y;

  vec beta_reduced;
  const bool solve_success = solve(beta_reduced, XtX, Xty, solve_opts::fast);

  if (!solve_success) {
    cpp4r::stop("Failed to solve normal equations for beta estimation.");
  }

  // Assign coefficients
  // NaN for collinear columns => replaced with NA in the R wrappers
  result.coefficients.fill(datum::nan);

  if (collin_result.has_collinearity) {
    result.coefficients.elem(collin_result.non_collinear_cols) = beta_reduced;
  } else {
    result.coefficients = beta_reduced;
  }

  result.coef_status = collin_result.coef_status;

  result.fitted_values = X * beta_reduced;
  result.residuals = y_orig - result.fitted_values;
  result.weights = w;

  // Build full Hessian
  result.hessian.zeros(p_orig, p_orig);

  if (collin_result.has_collinearity) {
    const uvec &valid_cols = collin_result.non_collinear_cols;
    result.hessian.submat(valid_cols, valid_cols) = XtX;
  } else {
    result.hessian = XtX;
  }

  result.success = true;
  return result;
}

} // namespace capybara

#endif // CAPYBARA_BETA_H
