// Computing beta a in a model with fixed effects Y = alpha + X beta

#ifndef CAPYBARA_BETA_H
#define CAPYBARA_BETA_H

namespace capybara {

struct InferenceGLM {
  vec coefficients;
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  bool success;      // collinearity
  uword iter;
  uvec coef_status; // 1 = estimable, 0 = collinear

  // Collinearity detection results
  uvec collinear_cols;
  uvec non_collinear_cols;
  bool has_collinearity;
  uword n_valid;

  field<vec> fixed_effects;
  bool has_fe = false;
  uvec iterations;

  mat TX;
  bool has_tx = false;

  vec means;

  InferenceGLM(uword n, uword p)
      : coefficients(p, fill::zeros), eta(n, fill::zeros),
        fitted_values(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), deviance(0.0), null_deviance(0.0),
        conv(false), success(false), iter(0), coef_status(p, fill::ones),
        collinear_cols(p), non_collinear_cols(p), has_collinearity(false), n_valid(0),
        has_fe(false), has_tx(false) {}
};

inline mat crossprod(const mat &X, const vec &w = vec()) {
  if (X.is_empty()) {
    return mat();
  }

  if (w.is_empty() || w.n_elem == 1) {
    return arma::symmatu(X.t() * X);
  }

  // Use BLAS-backed operations: scale rows by sqrt(w) and compute Xs' * Xs
  // This creates a temporary scaled matrix but leverages highly optimized BLAS/SYRK
  vec sqrtw = arma::sqrt(w);
  mat Xs = X;
  // scale each column by the row-wise sqrt weights
  Xs.each_col() %= sqrtw;
  return arma::symmatu(Xs.t() * Xs);
}

inline vec crossprod_Xy(const mat &X, const vec &y, bool has_weights, const vec &w = vec()) {
  const uword p = X.n_cols;
  vec Xty(p, fill::zeros);
  
  if (has_weights && !w.is_empty()) {
    vec wy = w % y;
    Xty = X.t() * wy;
  } else {
    Xty = X.t() * y;
  }
  return Xty;
}

inline bool rank_revealing_cholesky(uvec &excluded, const mat &XtX,
                                    double tol) {
  const uword p = XtX.n_cols;
  excluded.zeros(p);

  if (p == 0)
    return true;

  mat R(p, p, fill::zeros);

  double *R_ptr = R.memptr();
  uword *excluded_ptr = excluded.memptr();
  const double *XtX_ptr = XtX.memptr();

  uword n_excluded = 0;

  for (uword j = 0; j < p; ++j) {

    double R_jj = XtX_ptr[j + j * p];

    if (j > 0) {
      const double *R_j_ptr = R_ptr + j * p;
      for (uword k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          double R_jk = R_j_ptr[k];
          R_jj -= R_jk * R_jk;
        }
      }
    }

    if (R_jj < tol) {
      excluded_ptr[j] = 1;
      n_excluded++;
      continue;
    }

    R_jj = std::sqrt(R_jj);
    R_ptr[j + j * p] = R_jj;
    const double inv_R_jj = 1.0 / R_jj;

    for (uword col = j + 1; col < p; ++col) {
      double R_j_col = XtX_ptr[j + col * p];

      const double *R_col_ptr = R_ptr + col * p;
      const double *R_j_ptr = R_ptr + j * p;

      for (uword k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          R_j_col -= R_j_ptr[k] * R_col_ptr[k];
        }
      }

      R_ptr[j + col * p] = R_j_col * inv_R_jj;
    }
  }

  return n_excluded < p;
}

inline void
check_collinearity(mat &X, const vec &w, bool has_weights, double tolerance, InferenceGLM &result) {

  const uword p = X.n_cols;
  const uword n = X.n_rows;

  result.coef_status.set_size(p);
  result.coef_status.ones();
  result.collinear_cols.set_size(p);
  result.non_collinear_cols.set_size(p);
  result.has_collinearity = false;
  result.n_valid = 0;

  if (p == 0) {
    result.coef_status = uvec();
    return;
  }

  if (p == 1) {
    const double *col_ptr = X.colptr(0);
    const double *w_ptr = has_weights ? w.memptr() : nullptr;

    double mean_val = 0.0, sum_sq = 0.0, sum_w = 0.0;

    for (uword i = 0; i < n; ++i) {
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
    return;
  }

  mat XtX = has_weights ? crossprod(X, w) : crossprod(X);

  uvec excluded(p);
  bool success = rank_revealing_cholesky(excluded, XtX, tolerance);

  if (!success) {
    result.coef_status.zeros();
    result.has_collinearity = true;
    result.n_valid = 0;
    result.non_collinear_cols = uvec();
    X.reset();
    return;
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
}

inline void get_beta(const mat &X, const vec &y, const vec &y_orig,
                     const vec &w, InferenceGLM &result,
                     bool weighted, bool scale_X,
                     mat *cached_XtX = nullptr) {
  const uword p = X.n_cols;
  const uword p_orig =
      result.has_collinearity ? result.coef_status.n_elem : p;
  const bool has_weights = !all(w == 1.0);

  if (p == 0) {
    result.success = true;
    result.coefficients.zeros();
    result.fitted_values.zeros();
    result.weights = w;
    result.hessian.zeros();
    return;
  }

  if (y.n_elem == 0) {
    result.success = false;
    return;
  }

  mat XtX(p, p);
  vec Xty(p);

  // Reuse cached XtX if provided, otherwise compute
  if (cached_XtX && cached_XtX->n_rows == p && cached_XtX->n_cols == p) {
    XtX = *cached_XtX;
  } else {
    XtX = has_weights ? crossprod(X, w) : crossprod(X);
    // Update cache if pointer provided
    if (cached_XtX) {
      *cached_XtX = XtX;
    }
  }

  Xty = crossprod_Xy(X, y, has_weights, w);

  mat L(p, p);
  bool chol_success = chol(L, XtX, "lower");

  vec beta_reduced(p);
  if (chol_success) {
    vec z = solve(trimatl(L), Xty);
    beta_reduced = solve(trimatu(L.t()), z);
  } else {
    beta_reduced = solve(XtX, Xty);
  }

  result.coefficients.fill(datum::nan);
  if (result.has_collinearity) {
    result.coefficients.elem(result.non_collinear_cols) = beta_reduced;
  } else {
    result.coefficients = beta_reduced;
  }

  if (result.has_collinearity &&
      result.non_collinear_cols.n_elem > 0) {
    result.fitted_values = X * beta_reduced;
  } else {
    result.fitted_values = X * result.coefficients;
  }

  result.weights = w;

  result.hessian.set_size(p_orig, p_orig);
  result.hessian.zeros();

  if (result.has_collinearity) {
    const uvec &valid_cols = result.non_collinear_cols;
    for (uword i = 0; i < valid_cols.n_elem; ++i) {
      for (uword j = 0; j < valid_cols.n_elem; ++j) {
        result.hessian(valid_cols(i), valid_cols(j)) = XtX(i, j);
      }
    }
  } else {
    result.hessian = XtX;
  }

  result.success = true;
}

} // namespace capybara

#endif // CAPYBARA_BETA_H
