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

  const uword P_final = X.n_cols;

  // New solver
  CollinearityResult collin_result(P_final);
  mat XtX;
  vec XtY_vec;

  if (use_weights) {
    XtX = crossprod(X, w);
    XtY_vec = X.t() * (w % ws->y_demeaned);
  } else {
    XtX = crossprod(X);
    XtY_vec = X.t() * ws->y_demeaned;
  }

  mat R_rank;
  uvec excl;
  uword rank;

  // Adaptive tolerance strategy
  double current_tol = params.collin_tol;

  // Ensure tolerance is not too small relative to matrix scale
  if (XtX.n_rows > 0) {
    double max_diag = max(XtX.diag());
    double rel_tol = max_diag * 1e-12; // 1e-12 relative tolerance
    if (current_tol < rel_tol) {
      current_tol = rel_tol;
    }
  }

  // First pass
  if (!chol_rank(R_rank, excl, rank, XtX, "upper", current_tol)) {
    throw std::runtime_error("chol_rank failed in felm_fit");
  }

  // Robustness check: Ensure diagonal elements of R are above tolerance
  // If we find small pivots that were not excluded, we must re-run chol_rank
  const double pivot_thresh = std::sqrt(current_tol) * 10.0;
  const double max_diag_val = (XtX.n_rows > 0) ? max(XtX.diag()) : 1.0;

  // Vectorized check: find non-excluded indices with bad pivots
  const vec R_diag = R_rank.diag();
  const uvec kept = find(excl == 0);
  const vec kept_diag = R_diag.elem(kept);

  // Check for NaN or small pivots in kept columns
  const uvec bad_pivots =
      find((kept_diag != kept_diag) || // NaN check (x != x for NaN)
           (abs(kept_diag) < pivot_thresh));

  if (bad_pivots.n_elem > 0) {
    // Find maximum tolerance needed
    const vec bad_vals = kept_diag.elem(bad_pivots);
    const uvec finite_idx = find_finite(bad_vals);
    const double max_bad =
        finite_idx.n_elem > 0 ? max(abs(bad_vals.elem(finite_idx))) : 0.0;

    // Propose tolerance based on worst case
    const double proposed_tol = std::max(
        {max_diag_val * 1e-8, current_tol * 100.0, max_bad * max_bad * 1.1});

    // Second pass with stricter tolerance
    if (!chol_rank(R_rank, excl, rank, XtX, "upper", proposed_tol)) {
      const double last_resort_tol = max_diag_val * 1e-6;
      if (!chol_rank(R_rank, excl, rank, XtX, "upper", last_resort_tol)) {
        throw std::runtime_error("chol_rank failed in felm_fit rerun");
      }
    }
  }

  collin_result.has_collinearity = any(excl);

  if (collin_result.has_collinearity) {
    collin_result.non_collinear_cols = find(excl == 0);
    collin_result.collinear_cols = find(excl != 0);
    collin_result.coef_status = 1 - excl;
  } else {
    collin_result.non_collinear_cols = regspace<uvec>(0, P_final - 1);
    collin_result.coef_status.fill(1);
  }

  vec beta_solved(P_final, fill::value(datum::nan));

  // Extract submatrix and subvector using Armadillo indexing
  const uvec keep_idx = find(excl == 0);

  // Use .submat() for proper submatrix extraction with index vectors
  const mat R_sub = R_rank.submat(keep_idx, keep_idx);
  const vec XtY_sub = XtY_vec.elem(keep_idx);

  // Solve triangular systems
  const vec y_sub = solve(trimatl(R_sub.t()), XtY_sub);
  const vec beta_sub = solve(trimatu(R_sub), y_sub);

  // Place results back (vectorized)
  beta_solved.elem(keep_idx) = beta_sub;

  const uword n_coef = P_final;
  if (result.coef_table.n_rows != n_coef) {
    result.coef_table.set_size(n_coef, 4);
  }
  result.coef_status = std::move(collin_result.coef_status);

  result.coef_table.col(0) = beta_solved;
  mat hessian_reduced = R_sub.t() * R_sub;
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
