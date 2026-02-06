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
  mat X_original;  // Uncentered X for fitted values computation
  mat X_centered;  // Working copy that gets centered in place
  vec y_original;

  uword cached_N, cached_P;

  FelmWorkspace() : cached_N(0), cached_P(0) {}

  void ensure_size(uword N, uword P) {
    if (N > cached_N) {
      y_demeaned.set_size(N);
      x_beta.set_size(N);
      pi.set_size(N);
      y_original.set_size(N);
      cached_N = N;
    }
    if (P > cached_P || N > X_original.n_rows) {
      X_original.set_size(N, P);
      X_centered.set_size(N, P);
      cached_P = P;
    }
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
    const field<uvec> &groups_k = fe_groups(k);
    const uword J = groups_k.n_elem;
    info(k).n_groups = J;
    info(k).obs_to_group.set_size(N);

    // Vectorized scatter: fill group index for all obs in each group
    for (uword j = 0; j < J; ++j) {
      info(k).obs_to_group.elem(groups_k(j)).fill(j);
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

  for (uword k = 0; k < K; ++k) {
    const field<uvec> &groups_k = fe_groups(k);
    const vec &fe_k = fixed_effects(k);
    const uword J = groups_k.n_elem;

    // Scatter fixed effects to observations (vectorized per group)
    for (uword j = 0; j < J; ++j) {
      fitted_values.elem(groups_k(j)) += fe_k(j);
    }
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
                  nullptr, coef_weights);
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
  const double w_sum = accu(w);
  const double y_mean = dot(w, y) / w_sum;

  const vec y_centered = y - y_mean;

  const double tss = dot(w % y_centered, y_centered);
  const double rss = dot(w % residuals, residuals);

  result.r_squared = (tss > 1e-12) ? (1.0 - rss / tss) : 0.0;

  // Count degrees of freedom for FE
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
  if (n_idx > 0 && n_idx == vcov_reduced.n_rows &&
      n_idx == vcov_reduced.n_cols) {
    vcov_full.submat(non_collinear_cols, non_collinear_cols) = vcov_reduced;
  }
}

InferenceLM felm_fit(const mat &X, const vec &y, const vec &w,
                     const field<field<uvec>> &fe_groups,
                     const CapybaraParameters &params,
                     FelmWorkspace *workspace = nullptr,
                     const field<uvec> *cluster_groups = nullptr,
                     bool run_from_glm = false) {
  const uword N = y.n_elem;
  const uword P_input = X.n_cols;
  const bool has_fixed_effects = fe_groups.n_elem > 0;

  // Determine final P (with or without intercept)
  const uword P = (!has_fixed_effects && !run_from_glm) ? P_input + 1 : P_input;

  InferenceLM result(N, P);
  result.weights = w;

  FelmWorkspace local_workspace;
  FelmWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, P);
  ws->y_original = y;

  result.has_fe = has_fixed_effects;

  // Copy X to workspace buffers (only copy needed per iteration)
  if (!has_fixed_effects && !run_from_glm) {
    // Add intercept column
    ws->X_original.col(0).ones();
    if (P_input > 0) {
      ws->X_original.cols(1, P - 1) = X;
    }
  } else {
    ws->X_original = X;
  }
  ws->X_centered = ws->X_original;  // Copy for centering

  const bool use_weights = any(w != 1.0);

  #ifdef CAPYBARA_DEBUG
  auto tcenter0 = std::chrono::high_resolution_clock::now();
  #endif

  if (has_fixed_effects) {
    FlatFEMap fe_map = build_fe_map(fe_groups, w);

    ws->y_demeaned = y;
    center_variables(ws->y_demeaned, w, fe_map, params.center_tol,
                     params.iter_center_max);

    if (P > 0) {
      center_variables(ws->X_centered, w, fe_map, params.center_tol,
                       params.iter_center_max);
    }
  } else {
    ws->y_demeaned = y;
  }

  #ifdef CAPYBARA_DEBUG
  auto tcenter1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> center_duration = tcenter1 - tcenter0;
  std::ostringstream center_msg;
  center_msg << "Centering time: " << center_duration.count() << " seconds.\n";
  cpp4r::message(center_msg.str());
  #endif

  if (params.keep_tx && P > 0) {
    result.TX = ws->X_centered;
    result.has_tx = true;
  }

  const uword P_final = P;

  // Solve normal equations using rank-revealing Cholesky
  #ifdef CAPYBARA_DEBUG
  auto tsolve0 = std::chrono::high_resolution_clock::now();
  #endif

  mat XtX;
  vec XtY_vec;

  if (use_weights) {
    XtX = crossprod(ws->X_centered, w);
    XtY_vec = ws->X_centered.t() * (w % ws->y_demeaned);
  } else {
    XtX = crossprod(ws->X_centered);
    XtY_vec = ws->X_centered.t() * ws->y_demeaned;
  }

  mat R_rank;
  uvec excl;
  uword rank;

  if (!chol_rank(R_rank, excl, rank, XtX, "upper", params.collin_tol)) {
    throw std::runtime_error("chol_rank failed in felm_fit");
  }

  // Populate collinearity result from excl vector
  CollinearityResult collin_result(P_final);
  collin_result.has_collinearity = any(excl);
  collin_result.non_collinear_cols = find(excl == 0);
  collin_result.collinear_cols = find(excl != 0);
  collin_result.coef_status = 1 - excl;

  // Solve reduced system for non-excluded columns
  const uword n_coef = P_final;
  vec beta_solved(n_coef, fill::value(datum::nan));
  mat hessian_reduced;

  if (rank > 0) {
    const uvec keep_idx = collin_result.non_collinear_cols;
    const mat R_sub = R_rank.submat(keep_idx, keep_idx);
    const vec XtY_sub = XtY_vec.elem(keep_idx);

    // Solve triangular systems: R'y = XtY, then Rb = y
    const vec y_sub = solve(trimatl(R_sub.t()), XtY_sub);
    const vec beta_sub = solve(trimatu(R_sub), y_sub);

    // Scatter into full beta (excluded entries remain NaN)
    beta_solved.elem(keep_idx) = beta_sub;

    // Hessian from Cholesky factor: X'X = R'R
    hessian_reduced = R_sub.t() * R_sub;
  }

  if (result.coef_table.n_rows != n_coef) {
    result.coef_table.set_size(n_coef, 4);
  }
  result.coef_status = std::move(collin_result.coef_status);

  result.coef_table.col(0) = beta_solved;
  if (collin_result.has_collinearity) {
    expand_vcov(result.hessian, hessian_reduced,
                collin_result.non_collinear_cols, n_coef);
  } else {
    result.hessian = hessian_reduced;
  }
  result.success = true;

  compute_fitted_values(ws, result, collin_result, fe_groups, params);

  if (run_from_glm) {
    // Only return coefficients
    return result;
  }

  #ifdef CAPYBARA_DEBUG
  auto tsolve1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_duration = tsolve1 - tsolve0;
  std::ostringstream solve_msg;
  solve_msg << "Solving time: " << solve_duration.count() << " seconds.\n";
  cpp4r::message(solve_msg.str());
  #endif

  result.residuals = ws->y_original - result.fitted_values;

  compute_r_squared(result, ws->y_original, result.residuals, w, n_coef,
                    fe_groups);

  mat vcov_reduced;
  const double rss = dot(w % result.residuals, result.residuals);

  if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
    vcov_reduced =
        compute_sandwich_vcov(ws->X_centered, ws->y_original, result.fitted_values,
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
  // if (collin_result.has_collinearity) {
  //   const uvec collinear_idx = find(result.coef_status == 0);
  //   for (uword i = 0; i < collinear_idx.n_elem; ++i) {
  //     result.coef_table.row(collinear_idx(i)).fill(datum::nan);
  //   }
  // }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
