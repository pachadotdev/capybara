// Separation detection for Poisson models
// Based on Correia, Guimarães, Zylkin (2019)
// "Verifying the existence of maximum likelihood estimates for generalized
// linear models"

#ifndef CAPYBARA_SEPARATION_H
#define CAPYBARA_SEPARATION_H

namespace capybara {

// Result structure for separation detection
struct SeparationResult {
  uvec separated_obs; // Indices of separated observations (0-based)
  vec support;        // z vector proving separation (supporting hyperplane)
  uword num_separated;
  bool converged;
  uword iterations;

  SeparationResult() : num_separated(0), converged(false), iterations(0) {}
};

// Parameters for separation detection
struct SeparationParameters {
  double tol;      // Convergence tolerance
  double zero_tol; // Tolerance for treating values as zero
  uword max_iter;
  uword simplex_max_iter;
  bool use_relu;
  bool use_simplex;

  SeparationParameters()
      : tol(1e-8), zero_tol(1e-12), max_iter(1000), simplex_max_iter(10000),
        use_relu(true), use_simplex(true) {}
};

// ============================================================================
// ReLU Separation Detection
// Algorithm: Iterative least squares with ReLU activation
// Reference: Section 3.2 of Correia, Guimarães, Zylkin (2019)
// ============================================================================

// Solve weighted least squares: minimize ||sqrt(W)(y - X*beta)||^2
// Uses normal equations with rank-revealing Cholesky for robustness
inline vec solve_wls(const mat &X, const vec &y, const vec &w, vec &residuals) {
  if (X.n_cols == 0) {
    residuals = y;
    return vec();
  }

  const uword k = X.n_cols;

  // Form normal equations: X'WX * beta = X'Wy
  const mat Xw = X.each_col() % w;
  const mat XtWX = X.t() * Xw;
  const vec XtWy = Xw.t() * y;

  // Use rank-revealing Cholesky for robustness
  mat R;
  uvec excluded;
  uword rank;
  chol_rank(R, excluded, rank, XtWX, "upper");

  vec beta(k, arma::fill::zeros);

  if (rank == k) {
    // Full rank: solve R'R * beta = XtWy via back-substitution
    vec z;
    arma::solve(z, arma::trimatl(R.t()), XtWy, arma::solve_opts::fast);
    arma::solve(beta, arma::trimatu(R), z, arma::solve_opts::fast);
  } else if (rank > 0) {
    // Rank-deficient: solve on non-excluded columns
    const uvec included = arma::find(excluded == 0);
    if (included.n_elem > 0) {
      const mat R_sub = R.submat(included, included);
      const vec XtWy_sub = XtWy.elem(included);
      vec z;
      arma::solve(z, arma::trimatl(R_sub.t()), XtWy_sub,
                  arma::solve_opts::fast);
      vec beta_sub;
      arma::solve(beta_sub, arma::trimatu(R_sub), z, arma::solve_opts::fast);
      beta.elem(included) = beta_sub;
    }
  }

  residuals = y - X * beta;
  return beta;
}

inline vec solve_lse_weighted(const mat &X, const vec &y,
                              const uvec &constrained_sample, double M_weight,
                              vec &residuals) {
  if (X.n_cols == 0) {
    residuals = y;
    return vec();
  }

  vec weights(y.n_elem, arma::fill::ones);
  if (constrained_sample.n_elem > 0) {
    weights.elem(constrained_sample).fill(M_weight);
  }

  return solve_wls(X, y, weights, residuals);
}

// ReLU separation detection with fixed effects support
inline SeparationResult
detect_separation_relu_fe(const vec &y, const mat &X, const vec &w,
                          const field<field<uvec>> &fe_groups,
                          const SeparationParameters &params,
                          CenteringWorkspace *centering_workspace,
                          const CapybaraParameters &cap_params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;
  const bool has_fe = fe_groups.n_elem > 0;

  const uvec boundary_sample = arma::find(y == 0);
  const uvec interior_sample = arma::find(y > 0);
  const uword num_boundary = boundary_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  vec u = arma::conv_to<vec>::from(y == 0);
  const double M = 1.0 / std::sqrt(arma::datum::eps);

  vec xbd(n, arma::fill::zeros);
  vec weights(n);
  double uu_old = arma::dot(u, u);

  for (uword iter = 0; iter < params.max_iter; ++iter) {
    if (iter % 100 == 0)
      check_user_interrupt();

    // Weight construction: M for interior, 1 for boundary
    weights.ones();
    if (interior_sample.n_elem > 0) {
      weights.elem(interior_sample).fill(M);
    }

    vec u_centered = u;
    mat X_centered = X;

    if (has_fe) {
      const ObsToGroupMapping group_info =
          precompute_group_info(fe_groups, weights);
      center_variables(u_centered, weights, fe_groups, cap_params.center_tol,
                       cap_params.iter_center_max, cap_params.iter_interrupt,
                       &group_info, centering_workspace);
      center_variables(X_centered, weights, fe_groups, cap_params.center_tol,
                       cap_params.iter_center_max, cap_params.iter_interrupt,
                       &group_info, centering_workspace);
    }

    vec resid;
    solve_wls(X_centered, u_centered, weights, resid);
    xbd = u - resid;

    const double epsilon = arma::dot(resid, resid) + params.tol;
    const double delta = epsilon + params.tol;

    if (interior_sample.n_elem > 0) {
      xbd.elem(interior_sample).zeros();
    }

    // Zero out boundary values within tolerance
    vec boundary_vals = xbd.elem(boundary_sample);
    const uvec near_zero =
        arma::find((boundary_vals > -0.1 * delta) % (boundary_vals < delta));
    if (near_zero.n_elem > 0) {
      xbd.elem(boundary_sample.elem(near_zero)).zeros();
      boundary_vals = xbd.elem(boundary_sample);
    }

    // Check separation
    if (arma::all(boundary_vals >= 0)) {
      const uvec sep_ind_local = arma::find(boundary_vals > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    resid.clean(params.zero_tol);
    const vec boundary_resid = resid.elem(boundary_sample);

    if (boundary_resid.min() >= 0) {
      const uvec pos_resid_idx = arma::find(boundary_resid > delta);
      if (pos_resid_idx.n_elem > 0) {
        xbd.elem(boundary_sample.elem(pos_resid_idx)).zeros();
      }
      boundary_vals = xbd.elem(boundary_sample);
      const uvec sep_ind_local = arma::find(boundary_vals > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // ReLU: u = max(xbd, 0)
    u.zeros();
    u.elem(boundary_sample) = arma::clamp(boundary_vals, 0.0, arma::datum::inf);

    const double uu = arma::dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.tol * 0.01) {
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  if (!result.converged) {
    result.iterations = params.max_iter;
    const vec boundary_vals = xbd.elem(boundary_sample);
    const uvec sep_ind_local = arma::find(boundary_vals > params.tol);
    if (sep_ind_local.n_elem > 0) {
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
    }
  }

  return result;
}

// Main ReLU separation detection algorithm (without FE)
inline SeparationResult
detect_separation_relu(const vec &y, const mat &X, const vec &w,
                       const SeparationParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;

  const uvec boundary_sample = arma::find(y == 0);
  const uvec interior_sample = arma::find(y > 0);
  const uword num_boundary = boundary_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  vec u = arma::conv_to<vec>::from(y == 0);
  const double M = 1.0 / std::sqrt(arma::datum::eps);

  vec xbd(n, arma::fill::zeros);
  double uu_old = arma::dot(u, u);

  // Early termination tracking
  double ee_prev = uu_old;
  uword stall_count = 0;

  for (uword iter = 0; iter < params.max_iter; ++iter) {
    if (iter % 100 == 0)
      check_user_interrupt();

    vec resid;
    solve_lse_weighted(X, u, interior_sample, M, resid);
    xbd = u - resid;

    const double ee = arma::dot(resid, resid);
    const double epsilon = ee + params.tol;
    const double delta = epsilon + params.tol;

    // Early termination: detect stalled convergence
    if (iter > 3 && std::abs(ee - ee_prev) < params.tol * ee_prev) {
      ++stall_count;
      if (stall_count > 3) {
        result.iterations = iter + 1;
        break;
      }
    } else {
      stall_count = 0;
    }
    ee_prev = ee;

    // Enforce constraints on interior
    if (interior_sample.n_elem > 0) {
      xbd.elem(interior_sample).zeros();
    }

    // Zero out near-zero boundary values
    vec boundary_xbd = xbd.elem(boundary_sample);
    const uvec near_zero =
        arma::find((boundary_xbd > -0.1 * delta) % (boundary_xbd < delta));
    if (near_zero.n_elem > 0) {
      xbd.elem(boundary_sample.elem(near_zero)).zeros();
      boundary_xbd = xbd.elem(boundary_sample);
    }

    // Check separation
    if (arma::all(boundary_xbd >= 0)) {
      const uvec sep_ind_local = arma::find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    resid.clean(params.zero_tol);
    const vec boundary_resid = resid.elem(boundary_sample);

    if (boundary_resid.min() >= 0) {
      const uvec pos_resid_idx = arma::find(boundary_resid > delta);
      if (pos_resid_idx.n_elem > 0) {
        xbd.elem(boundary_sample.elem(pos_resid_idx)).zeros();
      }
      boundary_xbd = xbd.elem(boundary_sample);
      const uvec sep_ind_local = arma::find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // ReLU update
    u.zeros();
    u.elem(boundary_sample) = arma::clamp(boundary_xbd, 0.0, arma::datum::inf);

    const double uu = arma::dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.tol * 0.01) {
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  if (!result.converged) {
    result.iterations = params.max_iter;
    const vec boundary_xbd = xbd.elem(boundary_sample);
    const uvec sep_ind_local = arma::find(boundary_xbd > params.tol);
    if (sep_ind_local.n_elem > 0) {
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
    }
  }

  return result;
}

// ============================================================================
// Simplex Separation Detection
// ============================================================================

inline void simplex_presolve(mat &X, uvec &basic_vars, uvec &nonbasic_vars,
                             uvec &keep_mask, uword &k, uword &n) {
  mat A(n, k, arma::fill::zeros);
  keep_mask.ones(n);
  uvec is_dropped(k, arma::fill::zeros);

  std::vector<uword> nonbasic_list;
  nonbasic_list.reserve(k);

  const double pivot_tol = arma::datum::eps * 100;

  for (uword j = 0; j < k; ++j) {
    const uvec candidates = arma::find(arma::abs(X.col(j)) > pivot_tol, 1);

    if (candidates.n_elem == 0) {
      is_dropped(j) = 1;
      continue;
    }

    const uword pivot_row = candidates(0);
    nonbasic_list.push_back(pivot_row);
    keep_mask(pivot_row) = 0;

    const double pivot_inv = -1.0 / X(pivot_row, j);
    const vec pivot_col = X.col(j);

    // Rank-1 updates
    A(pivot_row, j) = 1.0;
    A += pivot_col * (pivot_inv * A.row(pivot_row));
    X += pivot_col * (pivot_inv * X.row(pivot_row));

    X.clean(1e-14);
    A.clean(1e-14);
  }

  const uvec kept_rows = arma::find(keep_mask);
  X = kept_rows.n_elem > 0 ? A.rows(kept_rows) : mat();

  const uvec not_dropped = arma::find(is_dropped == 0);
  if (not_dropped.n_elem > 0 && not_dropped.n_elem < k) {
    X = X.cols(not_dropped);
    k = not_dropped.n_elem;
  }

  nonbasic_vars = arma::conv_to<uvec>::from(nonbasic_list);
  basic_vars = arma::find(keep_mask);
  n = basic_vars.n_elem;
}

// Main simplex algorithm
inline SeparationResult
detect_separation_simplex(const mat &residuals,
                          const SeparationParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  if (residuals.n_elem == 0) {
    result.converged = true;
    return result;
  }

  mat X = residuals;
  uword n = X.n_rows;
  uword k = X.n_cols;

  // Test 1: Check column bounds
  rowvec col_min = arma::min(X, 0);
  rowvec col_max = arma::max(X, 0);
  col_min.clean(params.tol);
  col_max.clean(params.tol);

  // Find special columns
  const uvec all_zero = arma::find((col_min == 0) && (col_max == 0));
  const uvec all_positive = arma::find(col_min >= 0);
  const uvec all_negative = arma::find(col_max <= 0);

  uvec dropped_obs(n, arma::fill::zeros);
  uvec dropped_vars(k, arma::fill::zeros);

  if (all_zero.n_elem > 0)
    dropped_vars.elem(all_zero).ones();
  if (all_positive.n_elem > 0)
    dropped_vars.elem(all_positive).ones();
  if (all_negative.n_elem > 0)
    dropped_vars.elem(all_negative).ones();

  // Find separated observations from positive/negative columns
  if (all_positive.n_elem > 0) {
    dropped_obs
        .elem(arma::find(arma::max(X.cols(all_positive), 1) > params.tol))
        .ones();
  }
  if (all_negative.n_elem > 0) {
    dropped_obs
        .elem(arma::find(arma::min(X.cols(all_negative), 1) < -params.tol))
        .ones();
  }

  // Test 2: Apply simplex on remaining
  const uvec kept_obs = arma::find(dropped_obs == 0);
  const uvec kept_vars = arma::find(dropped_vars == 0);

  if (kept_vars.n_elem > 1 && kept_obs.n_elem > 0) {
    mat X_simplex = X.submat(kept_obs, kept_vars);
    uword n_simp = X_simplex.n_rows;
    uword k_simp = X_simplex.n_cols;

    if (n_simp > k_simp) {
      uvec basic_vars, nonbasic_vars, keep_mask;
      simplex_presolve(X_simplex, basic_vars, nonbasic_vars, keep_mask, k_simp,
                       n_simp);

      if (X_simplex.n_elem > 0) {
        vec c_basic(basic_vars.n_elem, arma::fill::ones);
        vec c_nonbasic(nonbasic_vars.n_elem, arma::fill::ones);

        for (uword iter = 0; iter < params.simplex_max_iter; ++iter) {
          if (iter % 100 == 0)
            check_user_interrupt();

          vec r = c_nonbasic - X_simplex.t() * c_basic;
          r.clean(1e-14);

          if (r.max() <= 0) {
            result.converged = true;
            break;
          }

          const uword pivot_col = r.index_max();
          const vec pivot_column = X_simplex.col(pivot_col);

          if (pivot_column.max() <= 0) {
            c_nonbasic(pivot_col) = 0;
            c_basic.elem(arma::find(pivot_column < 0)).zeros();
            continue;
          }

          const uword pivot_row = pivot_column.index_max();
          const double pivot = X_simplex(pivot_row, pivot_col);

          if (std::abs(pivot) < 1e-14)
            continue;

          // Pivot operations
          X_simplex.row(pivot_row) /= pivot;

          vec col_mult = pivot_column;
          col_mult(pivot_row) = 0.0;
          X_simplex -= col_mult * X_simplex.row(pivot_row);

          std::swap(basic_vars(pivot_row), nonbasic_vars(pivot_col));
          std::swap(c_basic(pivot_row), c_nonbasic(pivot_col));

          X_simplex.clean(1e-10);
        }

        // Identify separated observations
        const uvec all_vars = arma::join_vert(basic_vars, nonbasic_vars);
        const vec all_costs = arma::join_vert(c_basic, c_nonbasic);

        const uvec zero_cost_idx = arma::find(all_costs == 0);
        if (zero_cost_idx.n_elem > 0) {
          dropped_obs.elem(kept_obs.elem(all_vars.elem(zero_cost_idx))).ones();
        }
      }
    }
  }

  result.separated_obs = arma::find(dropped_obs);
  result.num_separated = result.separated_obs.n_elem;
  result.converged = true;

  return result;
}

// ============================================================================
// Combined separation detection
// ============================================================================

inline SeparationResult check_separation(const vec &y, const mat &X,
                                         const vec &w,
                                         const SeparationParameters &params) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  const uvec boundary_sample = arma::find(y == 0);
  if (boundary_sample.n_elem == 0) {
    return result;
  }

  // Partial out: center X on y > 0
  mat X_centered = X;
  if (X.n_cols > 0) {
    vec w_interior = w;
    w_interior.elem(boundary_sample).zeros();
    const double sum_w = arma::accu(w_interior);

    if (sum_w > 0) {
      X_centered.each_row() -= (X.t() * w_interior).t() / sum_w;
    }
  }

  // Simplex
  if (params.use_simplex && X_centered.n_cols > 0) {
    SeparationResult simplex_result =
        detect_separation_simplex(X_centered.rows(boundary_sample), params);

    if (simplex_result.num_separated > 0) {
      result.separated_obs = boundary_sample.elem(simplex_result.separated_obs);
      result.num_separated = result.separated_obs.n_elem;
    }
  }

  // ReLU
  if (params.use_relu) {
    SeparationResult relu_result =
        detect_separation_relu(y, X_centered, w, params);

    if (relu_result.num_separated > 0) {
      if (result.num_separated > 0) {
        result.separated_obs = arma::unique(
            arma::join_vert(result.separated_obs, relu_result.separated_obs));
        result.num_separated = result.separated_obs.n_elem;
      } else {
        result.separated_obs = relu_result.separated_obs;
        result.num_separated = relu_result.num_separated;
      }
      result.support = relu_result.support;
      result.iterations = relu_result.iterations;
    }
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_H
