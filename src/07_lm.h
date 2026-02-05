// Linear model with fixed effects Y = alpha + X beta
#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {

struct InferenceLM {
  mat coef_table; // Coefficient table: [estimate, std.error, z, p-value]
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  mat vcov;         // Covariance matrix (inverse Hessian or sandwich)
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  bool has_fe = true;
  uvec iterations;

  mat TX; // Centered design matrix T(X)
  double r_squared;
  double adj_r_squared;
  bool has_tx = false;

  InferenceLM(uword n, uword p)
      : coef_table(p, 4, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), vcov(p, p, fill::zeros),
        coef_status(p, fill::ones), success(false), has_fe(false),
        r_squared(0.0), adj_r_squared(0.0), has_tx(false) {}
};

struct FelmWorkspace {
  vec y_demeaned;
  vec x_beta;
  vec pi;
  mat X_original;
  vec y_original;
  vec scratch;

  uword cached_N, cached_P;
  bool is_initialized;

  FelmWorkspace() : cached_N(0), cached_P(0), is_initialized(false) {}

  FelmWorkspace(uword N, uword P)
      : cached_N(N), cached_P(P), is_initialized(true) {
    const uword safe_N = std::max(N, uword(1));
    const uword safe_P = std::max(P, uword(1));

    y_demeaned.set_size(safe_N);
    x_beta.set_size(safe_N);
    pi.set_size(safe_N);
    X_original.set_size(safe_N, safe_P);
    y_original.set_size(safe_N);
    scratch.set_size(safe_N);
  }

  void ensure_size(uword N, uword P) {
    if (!is_initialized || N > cached_N || P > cached_P) {
      const uword new_N = std::max(N, cached_N);
      const uword new_P = std::max(P, cached_P);

      // Resize only if necessary
      if (y_demeaned.n_elem < new_N)
        y_demeaned.set_size(new_N);
      if (x_beta.n_elem < new_N)
        x_beta.set_size(new_N);
      if (pi.n_elem < new_N)
        pi.set_size(new_N);
      if (scratch.n_elem < new_N)
        scratch.set_size(new_N);
      if (X_original.n_rows < new_N || X_original.n_cols < new_P) {
        X_original.set_size(new_N, new_P);
      }
      if (y_original.n_elem < new_N)
        y_original.set_size(new_N);

      cached_N = new_N;
      cached_P = new_P;
      is_initialized = true;
    }
  }

  ~FelmWorkspace() { clear(); }

  void clear() {
    y_demeaned.reset();
    x_beta.reset();
    pi.reset();
    X_original.reset();
    y_original.reset();
    scratch.reset();
    cached_N = 0;
    cached_P = 0;
    is_initialized = false;
  }
};

// Precompute observation-to-group mapping
struct FEAccumInfo {
  uvec obs_to_group;
  uword n_groups;
};

inline field<FEAccumInfo>
precompute_fe_accum_info(const field<field<uvec>> &fe_groups, uword N) {
  const uword K = fe_groups.n_elem;
  field<FEAccumInfo> info(K);

  for (uword k = 0; k < K; ++k) {
    const uword J = fe_groups(k).n_elem;
    info(k).n_groups = J;
    info(k).obs_to_group.set_size(N);

    for (uword j = 0; j < J; ++j) {
      info(k).obs_to_group.elem(fe_groups(k)(j)).fill(j);
    }
  }
  return info;
}

// Fixed effects accumulation
inline void
accumulate_fixed_effects_vectorized(vec &fitted_values,
                                    const field<vec> &fixed_effects,
                                    const field<FEAccumInfo> &fe_info) {
  const uword K = fe_info.n_elem;

  for (uword k = 0; k < K; ++k) {
    fitted_values += fixed_effects(k).elem(fe_info(k).obs_to_group);
  }
}

// Fallback for when precomputed info isn't available
inline void accumulate_fixed_effects(vec &fitted_values,
                                     const field<vec> &fixed_effects,
                                     const field<field<uvec>> &fe_groups) {
  const uword K = fe_groups.n_elem;
  const uword N = fitted_values.n_elem;

  for (uword k = 0; k < K; ++k) {
    const uword J = fe_groups(k).n_elem;
    const vec &fe_k = fixed_effects(k);

    uvec obs_to_group(N);
    for (uword j = 0; j < J; ++j) {
      obs_to_group.elem(fe_groups(k)(j)).fill(j);
    }

    fitted_values += fe_k.elem(obs_to_group);
  }
}

inline void compute_fitted_values(FelmWorkspace *ws, InferenceLM &result,
                                  const CollinearityResult &collin_result,
                                  const field<field<uvec>> &fe_groups,
                                  const CapybaraParameters &params) {
  const uword N = ws->y_original.n_elem;
  const vec &coef = result.coef_table.col(0);

  if (collin_result.has_collinearity &&
      !collin_result.non_collinear_cols.is_empty()) {
    const uvec &cols = collin_result.non_collinear_cols;
    ws->x_beta = ws->X_original.cols(cols) * coef.elem(cols);
  } else if (!collin_result.has_collinearity && ws->X_original.n_cols > 0) {
    ws->x_beta = ws->X_original * coef;
  } else {
    ws->x_beta.zeros(N);
  }

  const bool has_fixed_effects = fe_groups.n_elem > 0;

  if (has_fixed_effects) {
    // Vectorized residual pi = y - X*beta
    ws->pi = ws->y_original - ws->x_beta;

    const vec *coef_weights = nullptr;
    // Only pass weights if they are not all 1.
    // However, result.weights should already be set to input weights w.
    // If w was all 1s, result.weights is all 1s.
    // But efficiently, we might want to check.
    // For feglm, weights are definitely not all 1s.
    coef_weights = &result.weights;

    result.fixed_effects =
        get_alpha(ws->pi, fe_groups, params.alpha_tol, params.iter_alpha_max,
                  nullptr, nullptr, coef_weights);
    result.has_fe = true;

    result.fitted_values = ws->x_beta;

    accumulate_fixed_effects(result.fitted_values, result.fixed_effects,
                             fe_groups);
  } else {
    result.fitted_values = ws->x_beta;
  }
}

inline void compute_r_squared(InferenceLM &result, const vec &y,
                              const vec &residuals, const vec &w, uword n_coef,
                              const field<field<uvec>> &fe_groups) {
  const uword N = y.n_elem;
  const double y_mean = dot(w, y) / accu(w);

  const vec y_centered = y - y_mean;

  const double tss = dot(w % y_centered, y_centered);
  const double rss = dot(w % residuals, residuals);

  result.r_squared = (tss > 1e-12) ? (1.0 - rss / tss) : 0.0;

  uword k = n_coef;
  const uword K = fe_groups.n_elem;
  for (uword j = 0; j < K; ++j) {
    k += fe_groups(j).n_elem - 1;
  }

  const double denom = std::max(1.0, static_cast<double>(N - k));
  result.adj_r_squared =
      1.0 - (1.0 - result.r_squared) * (static_cast<double>(N - 1)) / denom;
}

// Expand reduced vcov matrix to full size accounting for collinearity
inline void expand_vcov(mat &vcov_full, const mat &vcov_reduced,
                        const uvec &non_collinear_cols, uword n_coef) {
  vcov_full.set_size(n_coef, n_coef);
  vcov_full.fill(datum::nan);

  const uword n_idx = non_collinear_cols.n_elem;
  for (uword j = 0; j < n_idx; ++j) {
    const uword col_j = non_collinear_cols(j);
    for (uword i = 0; i < n_idx; ++i) {
      vcov_full(non_collinear_cols(i), col_j) = vcov_reduced(i, j);
    }
  }
}

InferenceLM felm_fit(mat &X, const vec &y, const vec &w,
                     const field<field<uvec>> &fe_groups,
                     const CapybaraParameters &params,
                     FelmWorkspace *workspace = nullptr,
                     const field<uvec> *cluster_groups = nullptr,
                     bool run_from_glm = false) {
  const uword N = y.n_elem;
  const uword P = X.n_cols;

  InferenceLM result(N, P);
  result.weights = w;

  FelmWorkspace local_workspace;
  FelmWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, P);
  ws->y_original = y;

  const bool has_fixed_effects = fe_groups.n_elem > 0;
  result.has_fe = has_fixed_effects;

  // Add intercept column if no fixed effects
  if (!has_fixed_effects && !run_from_glm) {
    X = join_horiz(ones<vec>(N), X);
  }

  ws->X_original = X;

  const bool use_weights = any(w != 1.0);

  CollinearityResult collin_result =
      check_collinearity(X, w, use_weights, params.collin_tol);

  if (has_fixed_effects) {
    CenteringWorkspace centering_workspace;
    ObsToGroupMapping group_info = precompute_group_info(fe_groups, w);
    const ObsToGroupMapping *group_info_ptr = &group_info;

    ws->y_demeaned = y;

    center_variables(ws->y_demeaned, w, fe_groups, params.center_tol,
                     params.iter_center_max, params.iter_interrupt,
                     group_info_ptr, &centering_workspace);

    if (X.n_cols > 0) {
      center_variables(X, w, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       group_info_ptr, &centering_workspace);
    }
  } else {
    ws->y_demeaned = y;
  }

  if (params.keep_tx && X.n_cols > 0) {
    result.TX = X;
    result.has_tx = true;
  }

  InferenceBeta beta_result = get_beta(X, ws->y_demeaned, ws->y_demeaned, w,
                                       collin_result, false, false, nullptr);

  const uword n_coef = beta_result.coefficients.n_elem;
  if (result.coef_table.n_rows != n_coef) {
    result.coef_table.set_size(n_coef, 4);
  }

  result.coef_table.col(0) = beta_result.coefficients;
  result.coef_status = std::move(collin_result.coef_status);
  result.hessian = std::move(beta_result.hessian);
  result.success = beta_result.success;

  compute_fitted_values(ws, result, collin_result, fe_groups, params);

  if (run_from_glm) {
    // Only return coefficients
    return result;
  }

  result.residuals = ws->y_original - result.fitted_values;

  compute_r_squared(result, ws->y_original, result.residuals, w, n_coef,
                    fe_groups);

  mat vcov_reduced;
  const double rss = dot(w % result.residuals, result.residuals);

  if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
    vcov_reduced =
        compute_sandwich_vcov(X, ws->y_original, result.fitted_values,
                              result.hessian, *cluster_groups);
  } else {
    mat H_inv;
    bool success = inv(H_inv, result.hessian);
    if (!success) {
      H_inv.set_size(result.hessian.n_rows, result.hessian.n_cols);
      H_inv.fill(datum::inf);
    }

    const double sigma2 = rss / std::max(1.0, static_cast<double>(N - P));
    vcov_reduced = sigma2 * H_inv;
  }

  // Expand vcov if there's collinearity
  if (collin_result.has_collinearity &&
      collin_result.non_collinear_cols.n_elem > 0) {
    expand_vcov(result.vcov, vcov_reduced, collin_result.non_collinear_cols,
                n_coef);
  } else {
    result.vcov = std::move(vcov_reduced);
  }

  // Coefficients table
  const vec se = sqrt(diagvec(result.vcov));
  const vec &coefficients = result.coef_table.col(0);
  const vec z_values = coefficients / se;

  result.coef_table.col(1) = se;
  result.coef_table.col(2) = z_values;
  result.coef_table.col(3) = 2.0 * normcdf(-abs(z_values));

  // Mark collinear coefficients as NaN
  if (collin_result.has_collinearity) {
    const uvec collinear_idx = find(result.coef_status == 0);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      result.coef_table.row(collinear_idx(i)).fill(datum::nan);
    }
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
