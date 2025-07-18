// Computing beta and then alpha in a model with fixed effects
// Y = alpha + X beta

#ifndef CAPYBARA_PARAMETERS_H
#define CAPYBARA_PARAMETERS_H

namespace capybara {
namespace parameters {

//////////////////////////////////////////////////////////////////////////////
// RESULT STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// Beta computation result
struct InferenceBeta {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;
  bool success;

  InferenceBeta(size_t n, size_t p)
      : coefficients(p, fill::none),
        fitted_values(n, fill::none),
        residuals(n, fill::none),
        weights(n, fill::none),
        hessian(p, p, fill::none),
        coef_status(p, fill::none),
        success(false) {}
};

// Fixed effects extraction result
struct InferenceAlpha {
  field<vec> Alpha;
  uvec nb_references;  // Number of references per dimension (fixest compatibility)
  bool is_regular;     // Whether fixed effects are regular
  bool success;        // Whether extraction succeeded

  InferenceAlpha() : is_regular(true), success(false) {}

  cpp11::list to_list() const {
    writable::list Alpha_r(Alpha.n_elem);
    for (size_t k = 0; k < Alpha.n_elem; ++k) {
      Alpha_r[k] = as_doubles_matrix(Alpha(k).eval());
    }

    // Add fixest-style metadata
    writable::list result;
    result.push_back({"fixed_effects"_nm = Alpha_r});
    result.push_back({"nb_references"_nm = as_integers(nb_references)});
    result.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    result.push_back({"success"_nm = writable::logicals({success})});

    return result;
  }
};

//////////////////////////////////////////////////////////////////////////////
// STRUCTURAL PARAMETERS ESTIMATION
//////////////////////////////////////////////////////////////////////////////

// QR-based beta computation (matching original implementation)
inline void get_beta_qr(mat &X, const vec &y, const vec &w, InferenceBeta &result,
                        bool has_weights, double qr_collin_tol_multiplier = 1.0) {
  const size_t p = X.n_cols;
  
  mat Q, R;
  
  if (has_weights) {
    mat X_weighted = X.each_col() % sqrt(w);
    qr_econ(Q, R, X_weighted);
  } else {
    qr_econ(Q, R, X);
  }

  vec QTy = Q.t() * y;

  const vec diag_abs = abs(R.diag());
  const double max_diag = diag_abs.max();
  // Use configurable tolerance for collinearity detection
  const double tol = qr_collin_tol_multiplier * 1e-7 * max_diag;
  const uvec indep = find(diag_abs > tol);

  result.coefficients.fill(datum::nan);
  result.coef_status.zeros();
  result.coef_status(indep).ones();

  if (indep.n_elem == p) {
    result.coefficients = solve(trimatu(R), QTy, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = R.submat(indep, indep);
    const vec Yr = QTy.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    result.coefficients(indep) = br;
    // Keep NaN for invalid coefficients
  }
}

inline InferenceBeta get_beta(const mat &X, const vec &y, const vec &y_orig,
                           const vec &w, double collin_tol,
                           bool has_weights = false,
                           bool has_fixed_effects = false,
                           double direct_qr_threshold = 0.9,
                           double qr_collin_tol_multiplier = 1.0,
                           double chol_stability_threshold = 1e-12) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;

  InferenceBeta result(n, p);

  if (p == 0) {
    result.success = true;
    return result;
  }
  
  // For very wide matrices, use QR directly
  const bool direct_qr = (p > direct_qr_threshold * n);

  if (direct_qr) {
    mat X_copy = X;  // QR modifies the matrix
    get_beta_qr(X_copy, y, w, result, has_weights, qr_collin_tol_multiplier);
  } else {
    // Try Cholesky first (faster), fall back to QR if needed
    mat XtX, XtY;
    
    if (has_weights) {
      const vec sqrt_w = sqrt(w);
      const mat X_weighted = X.each_col() % sqrt_w;
      XtX = X_weighted.t() * X_weighted;
      XtY = X.t() * (w % y);
    } else {
      XtX = X.t() * X;
      XtY = X.t() * y;
    }

    mat L;
    const bool chol_ok = chol(L, XtX, "lower");

    if (chol_ok) {
      const vec d = abs(L.diag());
      const double mind = d.min();
      const double avgd = mean(d);
      
      if (mind > chol_stability_threshold * avgd) {
        vec work = solve(trimatl(L), XtY, solve_opts::fast);
        result.coefficients = solve(trimatu(L.t()), work, solve_opts::fast);
        result.coef_status.ones();
      } else {
        // Cholesky failed due to conditioning, fall back to QR
        mat X_copy = X;
        get_beta_qr(X_copy, y, w, result, has_weights, qr_collin_tol_multiplier);
      }
    } else {
      // Cholesky failed, fall back to QR
      mat X_copy = X;
      get_beta_qr(X_copy, y, w, result, has_weights, qr_collin_tol_multiplier);
    }
  }

  // Compute fitted values and residuals
  if (has_fixed_effects) {
    // For fixed effects models: fitted = y_orig - (y_demeaned - X_demeaned * beta)
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

  if (has_weights) {
    const vec sqrt_w = sqrt(w);
    const mat X_weighted = X.each_col() % sqrt_w;
    result.hessian = X_weighted.t() * X_weighted;
  } else {
    result.hessian = X.t() * X;
  }
  result.success = true;

  return result;
}

//////////////////////////////////////////////////////////////////////////////
// FIXED EFFECTS ESTIMATION
//////////////////////////////////////////////////////////////////////////////

inline InferenceAlpha get_alpha(const vec &p,
                                const field<field<uvec>> &group_indices,
                                double tol = 1e-8, size_t iter_max = 10000) {
  const size_t K = group_indices.n_elem;
  InferenceAlpha result;

  if (K == 0) {
    // No fixed effects => return intercept
    result.Alpha.set_size(1);
    result.Alpha(0) = vec(1);
    result.Alpha(0)(0) = mean(p);
    result.nb_references.set_size(1);
    result.nb_references(0) = 0;
    result.is_regular = true;
    result.success = true;
    return result;
  }

  if (K == 1) {
    // Single FE case - inline implementation
    const field<uvec> &groups = group_indices(0);
    uvec fe_id(p.n_elem);

    for (size_t g = 0; g < groups.n_elem; ++g) {
      const uvec &group_obs = groups(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_id(group_obs(i)) = g; // 0-based indexing consistently
      }
    }

    // Extract single fixed effect directly
    uvec myOrder = sort_index(fe_id);
    uvec sorted_id = fe_id(myOrder);

    // Find positions where ID changes (first occurrence of each unique ID)
    uvec select;
    select.resize(0);

    if (sorted_id.n_elem > 0) {
      select.resize(1);
      select(0) = myOrder(0);  // First element

      for (size_t i = 1; i < sorted_id.n_elem; ++i) {
        if (sorted_id(i) != sorted_id(i - 1)) {
          select.resize(select.n_elem + 1);
          select(select.n_elem - 1) = myOrder(i);
        }
      }
    }

    // Extract fixed effects at selected positions
    result.Alpha.set_size(1);
    result.Alpha(0) = p(select);

    // For single FE, no references needed
    result.nb_references.set_size(1);
    result.nb_references(0) = 0;
    result.is_regular = true;
    result.success = true;
    return result;
  }

  // Multi-FE case

  // Initialize fixed effects storage
  field<vec> Alpha(K);
  for (size_t k = 0; k < K; ++k) {
    const size_t n_groups = group_indices(k).n_elem;
    Alpha(k).zeros(n_groups);
  }

  const size_t N = p.n_elem;
  umat dumMat(N, K);

  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &groups_k = group_indices(k);
    for (size_t g = 0; g < groups_k.n_elem; ++g) {
      const uvec &group_obs = groups_k(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        dumMat(group_obs(i), k) = g;
      }
    }
  }

  // Alpaca-like alternating projections
  field<vec> Alpha_old(K);
  for (size_t k = 0; k < K; ++k) {
    Alpha_old(k).zeros(Alpha(k).n_elem);
  }

  double ratio = 0.0;
  size_t iter = 0;

  for (; iter < iter_max; ++iter) {
    Alpha_old = Alpha;

    // Update each FE dimension
    for (size_t k = 0; k < K; ++k) {
      const size_t n_groups_k = Alpha(k).n_elem;

      // Compute residual: p - sum of other FEs
      vec resid = p;
      for (size_t l = 0; l < K; ++l) {
        if (l == k)
          continue;

        for (size_t obs = 0; obs < N; ++obs) {
          resid(obs) -= Alpha(l)(dumMat(obs, l));
        }
      }

      // Update FE k
      Alpha(k).zeros();
      uvec group_counts(n_groups_k, fill::zeros);

      for (size_t obs = 0; obs < N; ++obs) {
        size_t group_id = dumMat(obs, k);
        Alpha(k)(group_id) += resid(obs);
        group_counts(group_id)++;
      }

      // Convert sums to means
      for (size_t g = 0; g < n_groups_k; ++g) {
        if (group_counts(g) > 0) {
          Alpha(k)(g) /= group_counts(g);
        }
      }
    }

    // Check convergence
    double num = 0.0, denom = 0.0;
    for (size_t k = 0; k < K; ++k) {
      const vec &diff = Alpha(k) - Alpha_old(k);
      num += dot(diff, diff);
      denom += dot(Alpha_old(k), Alpha_old(k));
    }
    ratio = sqrt(num / (denom + 1e-16));
    if (ratio < tol)
      break;
  }

  // By construction, the elements of the first fixed-effect dimension
  // are never set as references
  result.nb_references.set_size(K);
  result.nb_references.zeros();

  if (K >= 2) {
    // Set references for all FE dimensions except the first
    // In the presence of regular fixed-effects, there should be Q-1 references
    for (size_t k = 1; k < K; ++k) {
      result.nb_references(k) = 1;

      if (Alpha(k).n_elem > 0) {
        double reference_value = Alpha(k)(Alpha(k).n_elem - 1);
        Alpha(k) -= reference_value;
      }
    }
  }

  result.Alpha = Alpha;
  result.success = (iter < iter_max);
  result.is_regular = (K <= 2);  // Simplified regularity check

  return result;
}

}  // namespace parameters
}  // namespace capybara

#endif // CAPYBARA_PARAMETERS_H
