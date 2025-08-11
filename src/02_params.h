// Computing beta and then alpha in a model with fixed effects
// Y = alpha + X beta

#ifndef CAPYBARA_PARAMETERS_H
#define CAPYBARA_PARAMETERS_H

namespace capybara {

// Define InferenceBeta structure
struct InferenceBeta {
  vec beta;         // Main coefficient vector (same as coefficients)
  vec coefficients; // Alias for beta
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
      : beta(p, fill::zeros), coefficients(p, fill::zeros),
        fitted_values(n, fill::zeros), residuals(n, fill::zeros),
        weights(n, fill::ones), hessian(p, p, fill::zeros),
        coef_status(p, fill::ones), scale(0.0), rank(0.0), success(false) {}
};

struct InferenceAlpha {
  field<vec> Alpha;
  uvec nb_references;
  bool is_regular;
  bool success;
  field<std::string> fe_names;         // Names for fixed effects categories
  field<field<std::string>> fe_levels; // Names for levels within each category

  InferenceAlpha() : is_regular(true), success(false) {}
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

  for (size_t j = 0; j < p; ++j) {

    double R_jj = XtX(j, j);

    if (j > 0) {

      double sum_squares = 0.0;
      for (size_t k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          double R_kj = R_ptr[k + j * p];
          sum_squares += R_kj * R_kj;
        }
      }
      R_jj -= sum_squares;
    }

    if (R_jj < tol) {
      excluded_ptr[j] = 1;

      bool all_excluded = true;
      for (size_t k = 0; k < p; ++k) {
        if (excluded_ptr[k] == 0) {
          all_excluded = false;
          break;
        }
      }
      if (all_excluded)
        return false;
      continue;
    }

    R_jj = std::sqrt(R_jj);
    R_ptr[j + j * p] = R_jj;

    for (size_t col = j + 1; col < p; ++col) {
      double R_j_col = XtX(j, col);

      for (size_t k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          R_j_col -= R_ptr[k + j * p] * R_ptr[k + col * p];
        }
      }

      R_ptr[j + col * p] = R_j_col / R_jj;
    }
  }

  size_t n_excluded = 0;
  for (size_t j = 0; j < p; ++j) {
    if (excluded_ptr[j] == 1)
      n_excluded++;
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
    for (size_t i = 0; i < p; ++i) {
      const double *Xi_ptr = X.colptr(i);
      for (size_t j = i; j < p; ++j) {
        const double *Xj_ptr = X.colptr(j);
        double sum = 0.0;
        for (size_t obs = 0; obs < n; ++obs) {
          sum += Xi_ptr[obs] * Xj_ptr[obs] * w_ptr[obs];
        }
        XtX(i, j) = sum;
        if (i != j) {
          XtX(j, i) = sum;
        }
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

  // Add default constructor
  BetaWorkspace() {}

  BetaWorkspace(size_t n, size_t p) {
    size_t safe_n = std::max(n, size_t(1));
    size_t safe_p = std::max(p, size_t(1));

    XtX.set_size(safe_p, safe_p);
    XtY.set_size(safe_p);
    L.set_size(safe_p, safe_p);
    beta_work.set_size(safe_p);
    X_weighted.set_size(safe_n, safe_p);
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

  // Use workspace for computation
  mat &XtX = workspace->XtX;
  vec &Xty = workspace->XtY;

  // Resize workspace matrices if needed
  if (XtX.n_rows < p || XtX.n_cols < p) {
    XtX.set_size(p, p);
  }
  if (Xty.n_elem < p) {
    Xty.set_size(p);
  }

  // Compute XtX and Xty with vectorization
  if (has_weights) {
    vec sqrt_w = sqrt(w);
    mat X_weighted = X.each_col() % sqrt_w;
    vec y_weighted = y % sqrt_w;

    XtX = X_weighted.t() * X_weighted;
    Xty = X_weighted.t() * y_weighted;
  } else {
    XtX = X.t() * X;
    Xty = X.t() * y;
  }

  // Solve the system using Cholesky decomposition
  vec beta_reduced(p, fill::none);

  mat L;
  bool chol_success = chol(L, XtX, "lower");

  if (chol_success) {
    // Solve L * z = Xty
    vec z = solve(trimatl(L), Xty, solve_opts::fast);

    // Solve Lt * beta = z
    beta_reduced = solve(trimatu(L.t()), z, solve_opts::fast);
  } else {
    // Fallback to standard solve if Cholesky fails
    beta_reduced = solve(XtX, Xty, solve_opts::likely_sympd);
  }

  // Set the coefficient vector in the result
  result.coefficients.zeros();
  if (collin_result.has_collinearity) {
    result.coefficients.elem(collin_result.non_collinear_cols) = beta_reduced;
  } else {
    result.coefficients = beta_reduced;
  }

  // Copy to beta field as well
  result.beta = result.coefficients;
  result.coef_status = collin_result.coef_status;

  // Calculate fitted values - vectorized
  result.fitted_values = X * beta_reduced;

  // Calculate residuals
  result.residuals = y_orig - result.fitted_values;

  result.weights = w;

  // Store hessian for standard errors
  result.hessian.set_size(p_orig, p_orig);
  result.hessian.zeros();

  if (collin_result.has_collinearity) {
    if (collin_result.non_collinear_cols.n_elem > 0) {
      result.hessian(collin_result.non_collinear_cols,
                     collin_result.non_collinear_cols) = XtX;
    }
  } else {
    result.hessian = XtX;
  }

  result.success = true;

  return result;
}

// Simplified, optimized get_alpha function that returns only field<vec>
inline field<vec> get_alpha(const vec &pi,
                            const field<field<uvec>> &group_indices,
                            double tol = 1e-8, size_t iter_max = 10000) {
  const size_t K = group_indices.n_elem;
  const size_t N = pi.n_elem;
  field<vec> Alpha(K);
  if (K == 0 || N == 0) {
    return Alpha;
  }
  // Pre-allocate Alpha vectors
  for (size_t k = 0; k < K; ++k) {
    const size_t J = group_indices(k).n_elem;
    Alpha(k).set_size(J);
    Alpha(k).zeros();
  }
  // Alternating algorithm
  vec y(N);
  field<vec> Alpha0(K);
  double crit = 1.0;
  size_t iter = 0;
  while (crit > tol && iter < iter_max) {
    // Store previous iteration
    Alpha0 = Alpha;
    // Solve normal equations for each category k
    for (size_t k = 0; k < K; ++k) {
      // Compute adjusted dependent variable y = pi - sum of other fixed effects
      y = pi;
      for (size_t kk = 0; kk < K; ++kk) {
        if (kk != k) {
          const size_t J = group_indices(kk).n_elem;
          for (size_t j = 0; j < J; ++j) {
            const uvec &indexes = group_indices(kk)(j);
            const size_t T = indexes.n_elem;
            for (size_t t = 0; t < T; ++t) {
              y(indexes[t]) -= Alpha(kk)(j);
            }
          }
        }
      }
      // Compute group means for category k
      const size_t J = group_indices(k).n_elem;
      for (size_t j = 0; j < J; ++j) {
        const uvec &indexes = group_indices(k)(j);
        const size_t T = indexes.n_elem;
        if (T > 0) {
          // Compute group sum
          double sum = 0.0;
          for (size_t t = 0; t < T; ++t) {
            sum += y(indexes[t]);
          }
          // Store group mean
          Alpha(k)(j) = sum / T;
        } else {
          Alpha(k)(j) = 0.0;
        }
      }
    }
    // Compute convergence criterion
    double num = 0.0, denom = 0.0;
    for (size_t k = 0; k < K; ++k) {
      num += arma::accu(arma::square(Alpha(k) - Alpha0(k)));
      denom += arma::accu(arma::square(Alpha0(k)));
    }
    // Handle case where denom is zero (all Alpha0 values are zero)
    if (denom > 0.0) {
      crit = std::sqrt(num / denom);
    } else if (num > 0.0) {
      crit = std::sqrt(num);
    } else {
      crit = 0.0;
    }
    ++iter;
  }
  return Alpha;
}

} // namespace capybara

#endif // CAPYBARA_PARAMETERS_H
