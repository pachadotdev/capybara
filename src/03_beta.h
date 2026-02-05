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

  // Use rank-revealing Cholesky to get triangular factor and excluded cols
  mat R_rank;
  Col<uword> excluded_cols;
  uword rank_out = 0;

  const double chol_tol = 1e-10;

  const bool chol_ok = chol_rank(R_rank, excluded_cols, rank_out, XtX, "upper", chol_tol);

  vec beta_reduced;

  if (!chol_ok) {
    cpp4r::stop("Failed to solve normal equations for beta estimation.");
  } else {
    // Determine independent columns from excluded_cols (0 = independent)
    const uvec indep = find(excluded_cols == 0);
    const uword rnk = indep.n_elem;

    if (rnk == 0) {
      beta_reduced.set_size(p);
      beta_reduced.fill(datum::nan);
      result.coef_status = uvec(p, fill::zeros);
      result.hessian.zeros(p_orig, p_orig);
    } else {
      // Extract reduced R and corresponding XtY
      vec XtY_sub = Xty(indep);
      mat R_sub = R_rank.submat(indep, indep);

      // Solve triangular systems R_sub^T * y_sub = XtY_sub, then R_sub * beta_sub = y_sub
      vec y_sub;
      if (!solve(y_sub, R_sub.t(), XtY_sub, solve_opts::fast)) {
        cpp4r::stop("Failed to solve triangular system (R^T) in beta estimation.");
      }

      vec beta_sub;
      if (!solve(beta_sub, R_sub, y_sub, solve_opts::fast)) {
        cpp4r::stop("Failed to solve triangular system (R) in beta estimation.");
      }

      // Scatter reduced solution into full-length reduced vector (size p)
      beta_reduced.set_size(p);
      beta_reduced.fill(datum::nan);
      for (uword k = 0; k < rnk; ++k) {
        beta_reduced(indep(k)) = beta_sub(k);
      }

      // Coefficient status within current X
      result.coef_status.zeros();
      if (indep.n_elem > 0) {
        result.coef_status.elem(indep).ones();
      }

      // Build Hessian for estimable columns (place into full p_orig if needed)
      result.hessian.zeros(p_orig, p_orig);
      if (rnk > 0) {
        // XtX corresponds to current X (possibly filtered)
        if (collin_result.has_collinearity) {
          const uvec &valid_cols = collin_result.non_collinear_cols;
          // valid_cols maps current X columns to original indices; place XtX into those positions
          result.hessian.submat(valid_cols, valid_cols) = XtX;
        } else {
          result.hessian = XtX;
        }
      }

      result.rank = static_cast<double>(rank_out);
    }
  }

  // Assign coefficients: if original collinearity mapping exists, expand to full length
  if (collin_result.has_collinearity) {
    result.coefficients.fill(datum::nan);
    const uvec &orig_idx = collin_result.non_collinear_cols;
    for (uword j = 0; j < orig_idx.n_elem && j < beta_reduced.n_elem; ++j) {
      result.coefficients(orig_idx(j)) = beta_reduced(j);
    }
    // Ensure coef_status corresponds to original full length
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
