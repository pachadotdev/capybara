// Linear model with fixed effects Y = alpha + X beta

#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {

struct InferenceLM {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  bool has_fe = true;
  uvec iterations;

  mat TX; // Centered design matrix
  bool has_tx = false;

  InferenceLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), coef_status(p, fill::ones), success(false),
        has_fe(false), has_tx(false) {}
};

struct FelmWorkspace {
  vec y_demeaned;
  vec x_beta;
  vec pi;
  mat X_original;
  vec y_original;

  mat XtX;
  vec XtY;
  mat L;
  vec beta_work;

  vec y;
  vec alpha0;
  vec group_sums;

  size_t cached_N, cached_P;
  bool is_initialized;

  FelmWorkspace() : cached_N(0), cached_P(0), is_initialized(false) {}

  FelmWorkspace(size_t N, size_t P)
      : cached_N(N), cached_P(P), is_initialized(true) {
    size_t safe_N = std::max(N, size_t(1));
    size_t safe_P = std::max(P, size_t(1));

    y_demeaned.set_size(safe_N);
    x_beta.set_size(safe_N);
    pi.set_size(safe_N);
    X_original.set_size(safe_N, safe_P);
    y_original.set_size(safe_N);

    XtX.set_size(safe_P, safe_P);
    XtY.set_size(safe_P);
    L.set_size(safe_P, safe_P);
    beta_work.set_size(safe_P);

    y.set_size(safe_N);
    alpha0.set_size(safe_N);
    group_sums.set_size(safe_N);
  }

  void ensure_size(size_t N, size_t P) {
    if (!is_initialized || N > cached_N || P > cached_P) {
      size_t new_N = std::max(N, cached_N);
      size_t new_P = std::max(P, cached_P);

      if (y_demeaned.n_elem < new_N)
        y_demeaned.set_size(new_N);
      if (x_beta.n_elem < new_N)
        x_beta.set_size(new_N);
      if (pi.n_elem < new_N)
        pi.set_size(new_N);
      if (X_original.n_rows < new_N || X_original.n_cols < new_P)
        X_original.set_size(new_N, new_P);
      if (y_original.n_elem < new_N)
        y_original.set_size(new_N);

      if (XtX.n_rows < new_P || XtX.n_cols < new_P)
        XtX.set_size(new_P, new_P);
      if (XtY.n_elem < new_P)
        XtY.set_size(new_P);
      if (L.n_rows < new_P || L.n_cols < new_P)
        L.set_size(new_P, new_P);
      if (beta_work.n_elem < new_P)
        beta_work.set_size(new_P);

      if (y.n_elem < new_N)
        y.set_size(new_N);
      if (alpha0.n_elem < new_N)
        alpha0.set_size(new_N);
      if (group_sums.n_elem < new_N)
        group_sums.set_size(new_N);

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
    XtX.reset();
    XtY.reset();
    L.reset();
    beta_work.reset();
    y.reset();
    alpha0.reset();
    group_sums.reset();
    cached_N = 0;
    cached_P = 0;
    is_initialized = false;
  }
};

inline void accumulate_fixed_effects(vec &fitted_values,
                                     const field<vec> &fixed_effects,
                                     const field<field<uvec>> &fe_groups) {
  const size_t K = fe_groups.n_elem;
  const size_t N = fitted_values.n_elem;

  const size_t obs_block_size = get_block_size(N, K);

  for (size_t k = 0; k < K; ++k) {
    const size_t J = fe_groups(k).n_elem;
    const vec &fe_k = fixed_effects(k);

    for (size_t j = 0; j < J; ++j) {
      const uvec &group_idx = fe_groups(k)(j);
      const double fe_value = fe_k(j);

      const size_t group_size = group_idx.n_elem;
      const uword *idx_ptr = group_idx.memptr();
      double *fitted_ptr = fitted_values.memptr();

      for (size_t block_start = 0; block_start < group_size;
           block_start += obs_block_size) {
        const size_t block_end =
            std::min(block_start + obs_block_size, group_size);

        for (size_t t = block_start; t < block_end; ++t) {
          fitted_ptr[idx_ptr[t]] += fe_value;
        }
      }
    }
  }
}

inline void fitted_values(FelmWorkspace *ws, InferenceLM &result,
                          const CollinearityResult &collin_result,
                          const field<field<uvec>> &fe_groups,
                          const CapybaraParameters &params) {
  const size_t N = ws->y_original.n_elem;

  if (collin_result.has_collinearity &&
      !collin_result.non_collinear_cols.is_empty()) {
    auto X_sub = ws->X_original.cols(collin_result.non_collinear_cols);
    auto coef_sub = result.coefficients.elem(collin_result.non_collinear_cols);
    ws->x_beta = X_sub * coef_sub;
  } else if (!collin_result.has_collinearity && ws->X_original.n_cols > 0) {
    ws->x_beta = ws->X_original * result.coefficients;
  } else {
    ws->x_beta.zeros(N);
  }

  const bool has_fixed_effects = fe_groups.n_elem > 0;

  if (has_fixed_effects) {
    ws->pi = ws->y_original - ws->x_beta;

    result.fixed_effects =
        get_alpha(ws->pi, fe_groups, params.alpha_tol, params.iter_alpha_max);
    result.has_fe = true;

    result.fitted_values = ws->x_beta;

    accumulate_fixed_effects(result.fitted_values, result.fixed_effects,
                             fe_groups);
  } else {
    result.fitted_values = ws->x_beta;
  }
}

InferenceLM felm_fit(mat &X, const vec &y, const vec &w,
                     const field<field<uvec>> &fe_groups,
                     const CapybaraParameters &params,
                     FelmWorkspace *workspace = nullptr) {
  const size_t N = y.n_elem;
  const size_t P = X.n_cols;

  InferenceLM result(N, P);
  result.weights = w;

  FelmWorkspace local_workspace;
  FelmWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, P);

  ws->X_original = X;
  ws->y_original = y;

  bool use_weights = !all(w == 1.0);
  CollinearityResult collin_result =
      check_collinearity(X, w, use_weights, params.collin_tol);

  const bool has_fixed_effects = fe_groups.n_elem > 0;
  result.has_fe = has_fixed_effects;

  if (has_fixed_effects) {
    ws->y_demeaned = y;

    center_variables(ws->y_demeaned, w, fe_groups, params.center_tol,
                     params.iter_center_max, params.iter_interrupt,
                     params.iter_ssr, params.accel_start, params.project_tol_factor,
                     params.grand_accel_tol, params.project_group_tol, params.irons_tuck_tol,
                     params.grand_accel_interval, params.irons_tuck_interval, 
                     params.ssr_check_interval, params.convergence_factor, params.tol_multiplier);

    if (X.n_cols > 0) {
      center_variables(X, w, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       params.iter_ssr, params.accel_start, params.project_tol_factor,
                       params.grand_accel_tol, params.project_group_tol, params.irons_tuck_tol,
                       params.grand_accel_interval, params.irons_tuck_interval, 
                       params.ssr_check_interval, params.convergence_factor, params.tol_multiplier);
    }
  } else {
    ws->y_demeaned = y;
  }

  if (params.keep_tx && X.n_cols > 0) {
    result.TX = X;
    result.has_tx = true;
  }

  InferenceBeta beta_result = get_beta(X, ws->y_demeaned, ws->y_demeaned, w,
                                       collin_result, false, false);

  result.coefficients = std::move(beta_result.coefficients);
  result.coef_status = std::move(collin_result.coef_status);
  result.hessian = std::move(beta_result.hessian);
  result.success = std::move(beta_result.success);

  fitted_values(ws, result, collin_result, fe_groups, params);

  result.residuals = ws->y_original - result.fitted_values;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
