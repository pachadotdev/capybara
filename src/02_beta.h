// Computing beta a in a model with fixed effects Y = alpha + X beta

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

  InferenceBeta(size_t n, size_t p)
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
  size_t n_valid;
  mat R;

  CollinearityResult() : has_collinearity(false), n_valid(0) {}

  CollinearityResult(size_t p)
      : coef_status(p, fill::ones), collinear_cols(p), non_collinear_cols(p),
        has_collinearity(false), n_valid(0) {}
};

inline mat crossprod(const mat &X, const vec &w = vec()) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  mat result(p, p, fill::zeros);

  const size_t block_size = get_block_size(n, p);

  if (w.is_empty() || w.n_elem == 1) {
    for (size_t block_start = 0; block_start < n; block_start += block_size) {
      const size_t block_end = std::min(block_start + block_size, n);

      for (size_t i = 0; i < p; ++i) {
        const double *Xi_ptr = X.colptr(i) + block_start;
        for (size_t j = i; j < p; ++j) {
          const double *Xj_ptr = X.colptr(j) + block_start;

          double sum = 0.0;
          for (size_t k = 0; k < (block_end - block_start); ++k) {
            sum += Xi_ptr[k] * Xj_ptr[k];
          }

          result(i, j) += sum;
          if (i != j) {
            result(j, i) += sum;
          }
        }
      }
    }
  } else {
    const double *w_ptr = w.memptr();

    for (size_t block_start = 0; block_start < n; block_start += block_size) {
      const size_t block_end = std::min(block_start + block_size, n);

      for (size_t i = 0; i < p; ++i) {
        const double *Xi_ptr = X.colptr(i) + block_start;
        for (size_t j = i; j < p; ++j) {
          const double *Xj_ptr = X.colptr(j) + block_start;

          double sum = 0.0;
          for (size_t k = 0; k < (block_end - block_start); ++k) {
            sum += Xi_ptr[k] * Xj_ptr[k] * w_ptr[block_start + k];
          }

          result(i, j) += sum;
          if (i != j) {
            result(j, i) += sum;
          }
        }
      }
    }
  }
  return result;
}

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

  size_t n_excluded = 0;

  for (size_t j = 0; j < p; ++j) {

    double R_jj = XtX_ptr[j + j * p];

    if (j > 0) {
      const double *R_j_ptr = R_ptr + j * p;
      for (size_t k = 0; k < j; ++k) {
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

    for (size_t col = j + 1; col < p; ++col) {
      double R_j_col = XtX_ptr[j + col * p];

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
    XtX = crossprod(X, w);
  } else {
    XtX = crossprod(X);
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

inline InferenceBeta get_beta(const mat &X, const vec &y, const vec &y_orig,
                              const vec &w,
                              const CollinearityResult &collin_result,
                              bool weighted, bool scale_X) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  const size_t p_orig =
      collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;
  const bool has_weights = !all(w == 1.0);

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

  mat XtX(p, p);
  vec Xty(p);

  if (has_weights) {
    XtX = crossprod(X, w);
    Xty = X.t() * (w % y);
  } else {
    XtX = crossprod(X);
    Xty = X.t() * y;
  }

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
  if (collin_result.has_collinearity) {
    result.coefficients.elem(collin_result.non_collinear_cols) = beta_reduced;
  } else {
    result.coefficients = beta_reduced;
  }

  result.coef_status = collin_result.coef_status;

  if (collin_result.has_collinearity &&
      collin_result.non_collinear_cols.n_elem > 0) {
    result.fitted_values = X * beta_reduced;
  } else {
    result.fitted_values = X * result.coefficients;
  }

  result.residuals = y_orig - result.fitted_values;
  result.weights = w;

  result.hessian.set_size(p_orig, p_orig);
  result.hessian.zeros();

  if (collin_result.has_collinearity) {
    const uvec &valid_cols = collin_result.non_collinear_cols;
    for (size_t i = 0; i < valid_cols.n_elem; ++i) {
      for (size_t j = 0; j < valid_cols.n_elem; ++j) {
        result.hessian(valid_cols(i), valid_cols(j)) = XtX(i, j);
      }
    }
  } else {
    result.hessian = XtX;
  }

  result.success = true;
  return result;
}

} // namespace capybara

#endif // CAPYBARA_BETA_H
