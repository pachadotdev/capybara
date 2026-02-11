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

  vec beta(k, fill::zeros);

  if (rank == k) {
    // Full rank: solve R'R * beta = XtWy via back-substitution
    vec z;
    solve(z, trimatl(R.t()), XtWy, solve_opts::fast);
    solve(beta, trimatu(R), z, solve_opts::fast);
  } else if (rank > 0) {
    // Rank-deficient: solve on non-excluded columns
    const uvec included = find(excluded == 0);
    if (included.n_elem > 0) {
      const mat R_sub = R.submat(included, included);
      const vec XtWy_sub = XtWy.elem(included);
      vec z;
      solve(z, trimatl(R_sub.t()), XtWy_sub, solve_opts::fast);
      vec beta_sub;
      solve(beta_sub, trimatu(R_sub), z, solve_opts::fast);
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

  vec weights(y.n_elem, fill::ones);
  if (constrained_sample.n_elem > 0) {
    weights.elem(constrained_sample).fill(M_weight);
  }

  return solve_wls(X, y, weights, residuals);
}

// ReLU separation detection with fixed effects support
inline SeparationResult
detect_separation_relu_fe(const vec &y, const mat &X, const vec &w,
                          const FlatFEMap &fe_map,
                          const CapybaraParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;
  const bool has_fe = fe_map.K > 0;

  const uvec boundary_sample = find(y == 0);
  const uvec interior_sample = find(y > 0);
  const uword num_boundary = boundary_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  vec u = conv_to<vec>::from(y == 0);
  const double M = 1.0 / std::sqrt(datum::eps);

  vec xbd(n, fill::zeros);
  vec weights(n);
  double uu_old = dot(u, u);

  // Pre-build FE map once (weights are constant M for interior, 1 for boundary)
  weights.ones();
  if (interior_sample.n_elem > 0) {
    weights.elem(interior_sample).fill(M);
  }

  // Use the shared FE map; build a mutable copy for weight updates
  FlatFEMap local_fe_map;
  if (has_fe) {
    local_fe_map = fe_map; // copy structure
    local_fe_map.update_weights(weights);
  }

  // Reusable buffers
  vec u_centered(n), resid(n);
  mat X_centered;

  for (uword iter = 0; iter < params.sep_max_iter; ++iter) {
    if (iter % 100 == 0)
      check_user_interrupt();

    u_centered = u;
    X_centered = X;

    if (has_fe) {
      center_variables(u_centered, weights, local_fe_map, params.center_tol,
                       params.iter_center_max, params.grand_acc_period);
      center_variables(X_centered, weights, local_fe_map, params.center_tol,
                       params.iter_center_max, params.grand_acc_period);
    }

    solve_wls(X_centered, u_centered, weights, resid);
    xbd = u - resid;

    const double epsilon = dot(resid, resid) + params.sep_tol;
    const double delta = epsilon + params.sep_tol;

    if (interior_sample.n_elem > 0) {
      xbd.elem(interior_sample).zeros();
    }

    // Zero out boundary values within tolerance
    vec boundary_vals = xbd.elem(boundary_sample);
    const uvec near_zero =
        find((boundary_vals > -0.1 * delta) % (boundary_vals < delta));
    if (near_zero.n_elem > 0) {
      xbd.elem(boundary_sample.elem(near_zero)).zeros();
      boundary_vals = xbd.elem(boundary_sample);
    }

    // Check separation
    if (all(boundary_vals >= 0)) {
      const uvec sep_ind_local = find(boundary_vals > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    resid.clean(params.sep_zero_tol);
    const vec boundary_resid = resid.elem(boundary_sample);

    if (boundary_resid.min() >= 0) {
      const uvec pos_resid_idx = find(boundary_resid > delta);
      if (pos_resid_idx.n_elem > 0) {
        xbd.elem(boundary_sample.elem(pos_resid_idx)).zeros();
      }
      boundary_vals = xbd.elem(boundary_sample);
      const uvec sep_ind_local = find(boundary_vals > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // ReLU: u = max(xbd, 0)
    u.zeros();
    u.elem(boundary_sample) = clamp(boundary_vals, 0.0, datum::inf);

    const double uu = dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.sep_tol * 0.01) {
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  if (!result.converged) {
    result.iterations = params.sep_max_iter;
    const vec boundary_vals = xbd.elem(boundary_sample);
    const uvec sep_ind_local = find(boundary_vals > params.sep_tol);
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
                       const CapybaraParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;

  const uvec boundary_sample = find(y == 0);
  const uvec interior_sample = find(y > 0);
  const uword num_boundary = boundary_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  vec u = conv_to<vec>::from(y == 0);
  const double M = 1.0 / std::sqrt(datum::eps);

  vec xbd(n, fill::zeros);
  vec xbd_prev1(n, fill::zeros);
  vec xbd_prev2(n, fill::zeros);
  vec resid(n);
  double uu_old = dot(u, u);

  // Progress tracking for acceleration (from ppmlhdfe)
  double ee_cumulative = 0.0;
  const double ee_boundary = uu_old;
  double progress_ratio_prev1 = 0.0;
  double progress_ratio_prev2 = 0.0;
  uword num_candidates_prev1 = 0;
  uword num_candidates_prev2 = 0;
  bool convergence_is_stuck = false;
  double acceleration_value = 1.0;

  for (uword iter = 0; iter < params.sep_max_iter; ++iter) {
    if (iter % 100 == 0)
      check_user_interrupt();

    // Shift xbd history for acceleration detection
    std::swap(xbd_prev2, xbd_prev1);
    std::swap(xbd_prev1, xbd);

    // Build weights with potential acceleration
    vec weights(n, fill::ones);
    if (interior_sample.n_elem > 0) {
      weights.elem(interior_sample).fill(M);
    }

    // Apply acceleration to stuck negative boundary observations
    if (convergence_is_stuck && iter > 3) {
      const vec xbd_b = xbd_prev1.elem(boundary_sample);
      const vec xbd_b_p1 = xbd_prev2.elem(boundary_sample);
      // Find obs stuck at negative values
      for (uword i = 0; i < num_boundary; ++i) {
        if (xbd_b(i) < -0.1 * params.sep_tol && xbd_b_p1(i) < 1.01 * xbd_b(i)) {
          weights(boundary_sample(i)) = acceleration_value;
        }
      }
    }

    solve_wls(X, u, weights, resid);
    xbd = u - resid;

    const double ee = dot(resid, resid);
    const double epsilon = ee + params.sep_tol;
    const double delta = epsilon + params.sep_tol;

    // Track cumulative progress (from ppmlhdfe)
    ee_cumulative += ee;
    const double progress_ratio =
        ee_boundary > 0 ? 100.0 * ee_cumulative / ee_boundary : 100.0;

    // Count candidates for separation
    uword num_candidates = 0;
    {
      const vec boundary_xbd_tmp = xbd.elem(boundary_sample);
      for (uword i = 0; i < num_boundary; ++i) {
        if (boundary_xbd_tmp(i) > delta)
          num_candidates++;
      }
    }

    // Detect stuck convergence and enable acceleration (from ppmlhdfe)
    if (!convergence_is_stuck && iter > 3) {
      if ((progress_ratio - progress_ratio_prev2 < 1.0) &&
          (num_candidates == num_candidates_prev2)) {
        convergence_is_stuck = true;
        acceleration_value = 4.0;
      }
    } else if (convergence_is_stuck) {
      acceleration_value = std::min(256.0, 4.0 * acceleration_value);
    }

    // Update history
    progress_ratio_prev2 = progress_ratio_prev1;
    progress_ratio_prev1 = progress_ratio;
    num_candidates_prev2 = num_candidates_prev1;
    num_candidates_prev1 = num_candidates;

    // Enforce constraints on interior
    if (interior_sample.n_elem > 0) {
      xbd.elem(interior_sample).zeros();
    }

    // Zero out near-zero boundary values
    vec boundary_xbd = xbd.elem(boundary_sample);
    const uvec near_zero =
        find((boundary_xbd > -0.1 * delta) % (boundary_xbd < delta));
    if (near_zero.n_elem > 0) {
      xbd.elem(boundary_sample.elem(near_zero)).zeros();
      boundary_xbd = xbd.elem(boundary_sample);
    }

    // Check separation - all non-negative means we found it
    if (all(boundary_xbd >= 0)) {
      const uvec sep_ind_local = find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    resid.clean(params.sep_zero_tol);
    const vec boundary_resid = resid.elem(boundary_sample);

    if (boundary_resid.min() >= 0) {
      const uvec pos_resid_idx = find(boundary_resid > delta);
      if (pos_resid_idx.n_elem > 0) {
        xbd.elem(boundary_sample.elem(pos_resid_idx)).zeros();
      }
      boundary_xbd = xbd.elem(boundary_sample);
      const uvec sep_ind_local = find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // ReLU update: u = max(xbd, 0) on boundary
    u.zeros();
    u.elem(boundary_sample) = clamp(boundary_xbd, 0.0, datum::inf);

    const double uu = dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.sep_tol * 0.01) {
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  if (!result.converged) {
    result.iterations = params.sep_max_iter;
    const vec boundary_xbd = xbd.elem(boundary_sample);
    const uvec sep_ind_local = find(boundary_xbd > params.sep_tol);
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
  // First pass: identify and drop empty columns (from ppmlhdfe)
  uvec is_dropped(k, fill::zeros);
  for (uword j = 0; j < k; ++j) {
    if (accu(abs(X.col(j))) == 0) {
      is_dropped(j) = 1;
    }
  }

  // Remove empty columns early
  const uvec non_empty = find(is_dropped == 0);
  if (non_empty.n_elem < k) {
    if (non_empty.n_elem == 0) {
      // All columns empty - trivial case
      keep_mask.ones(n);
      basic_vars = regspace<uvec>(0, n - 1);
      nonbasic_vars = uvec();
      k = 0;
      return;
    }
    X = X.cols(non_empty);
    k = non_empty.n_elem;
    is_dropped.zeros(k); // Reset for remaining columns
  }

  mat A(n, k, fill::zeros);
  keep_mask.ones(n);

  std::vector<uword> nonbasic_list;
  nonbasic_list.reserve(k);

  const double pivot_tol = datum::eps * 100;

  for (uword j = 0; j < k; ++j) {
    const uvec candidates = find(abs(X.col(j)) > pivot_tol, 1);

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

  const uvec kept_rows = find(keep_mask);
  X = kept_rows.n_elem > 0 ? A.rows(kept_rows) : mat();

  const uvec not_dropped = find(is_dropped == 0);
  if (not_dropped.n_elem > 0 && not_dropped.n_elem < k) {
    X = X.cols(not_dropped);
    k = not_dropped.n_elem;
  }

  nonbasic_vars = conv_to<uvec>::from(nonbasic_list);
  basic_vars = find(keep_mask);
  n = basic_vars.n_elem;
}

// Main simplex algorithm - optimized with early exit
inline SeparationResult
detect_separation_simplex(const mat &residuals,
                          const CapybaraParameters &params) {
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

  // Test 1: Check column bounds (vectorized)
  rowvec col_min = min(X, 0);
  rowvec col_max = max(X, 0);
  col_min.clean(params.sep_tol);
  col_max.clean(params.sep_tol);

  uvec dropped_obs(n, fill::zeros);
  uvec dropped_vars(k, fill::zeros);

  // Find and mark special columns
  for (uword j = 0; j < k; ++j) {
    if (col_min(j) == 0 && col_max(j) == 0) {
      dropped_vars(j) = 1; // all zero
    } else if (col_min(j) >= 0) {
      dropped_vars(j) = 1; // all positive
      // Mark obs with positive values as separated
      for (uword i = 0; i < n; ++i) {
        if (X(i, j) > params.sep_tol)
          dropped_obs(i) = 1;
      }
    } else if (col_max(j) <= 0) {
      dropped_vars(j) = 1; // all negative
      // Mark obs with negative values as separated
      for (uword i = 0; i < n; ++i) {
        if (X(i, j) < -params.sep_tol)
          dropped_obs(i) = 1;
      }
    }
  }

  // Test 2: Apply simplex on remaining (only if needed)
  const uvec kept_obs = find(dropped_obs == 0);
  const uvec kept_vars = find(dropped_vars == 0);

  if (kept_vars.n_elem > 1 && kept_obs.n_elem > 0) {
    mat X_simplex = X.submat(kept_obs, kept_vars);
    uword n_simp = X_simplex.n_rows;
    uword k_simp = X_simplex.n_cols;

    if (n_simp > k_simp) {
      uvec basic_vars, nonbasic_vars, keep_mask;
      simplex_presolve(X_simplex, basic_vars, nonbasic_vars, keep_mask, k_simp,
                       n_simp);

      if (X_simplex.n_elem > 0 && basic_vars.n_elem > 0) {
        vec c_basic(basic_vars.n_elem, fill::ones);
        vec c_nonbasic(nonbasic_vars.n_elem, fill::ones);

        // Reduced iteration limit for speed
        const uword effective_max_iter =
            std::min(params.sep_simplex_max_iter, (size_t)(100 * k_simp));

        for (uword iter = 0; iter < effective_max_iter; ++iter) {
          if (iter % 100 == 0)
            check_user_interrupt();

          vec r = c_nonbasic - X_simplex.t() * c_basic;
          r.clean(1e-14);

          double r_max = r.max();
          if (r_max <= 0) {
            result.converged = true;
            break;
          }

          const uword pivot_col = r.index_max();
          const vec pivot_column = X_simplex.col(pivot_col);
          double pivot_max = pivot_column.max();

          if (pivot_max <= 0) {
            c_nonbasic(pivot_col) = 0;
            c_basic.elem(find(pivot_column < 0)).zeros();
            continue;
          }

          const uword pivot_row = pivot_column.index_max();
          const double pivot = X_simplex(pivot_row, pivot_col);

          if (std::abs(pivot) < 1e-14)
            continue;

          // Pivot operations (in-place)
          X_simplex.row(pivot_row) /= pivot;

          for (uword i = 0; i < X_simplex.n_rows; ++i) {
            if (i != pivot_row && std::abs(pivot_column(i)) > 1e-14) {
              X_simplex.row(i) -= pivot_column(i) * X_simplex.row(pivot_row);
            }
          }

          std::swap(basic_vars(pivot_row), nonbasic_vars(pivot_col));
          std::swap(c_basic(pivot_row), c_nonbasic(pivot_col));
        }

        // Identify separated observations
        const uvec all_vars = join_vert(basic_vars, nonbasic_vars);
        const vec all_costs = join_vert(c_basic, c_nonbasic);

        const uvec zero_cost_idx = find(all_costs == 0);
        if (zero_cost_idx.n_elem > 0) {
          dropped_obs.elem(kept_obs.elem(all_vars.elem(zero_cost_idx))).ones();
        }
      }
    }
  }

  result.separated_obs = find(dropped_obs);
  result.num_separated = result.separated_obs.n_elem;
  result.converged = true;

  return result;
}

// ============================================================================
// Combined separation detection
// ============================================================================

inline SeparationResult check_separation(const vec &y, const mat &X,
                                         const vec &w,
                                         const CapybaraParameters &params) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  const uvec boundary_sample = find(y == 0);
  if (boundary_sample.n_elem == 0) {
    return result;
  }

  // Partial out: center X on y > 0
  mat X_centered = X;
  if (X.n_cols > 0) {
    vec w_interior = w;
    w_interior.elem(boundary_sample).zeros();
    const double sum_w = accu(w_interior);

    if (sum_w > 0) {
      X_centered.each_row() -= (X.t() * w_interior).t() / sum_w;
    }
  }

  // Simplex
  if (params.sep_use_simplex && X_centered.n_cols > 0) {
    SeparationResult simplex_result =
        detect_separation_simplex(X_centered.rows(boundary_sample), params);

    if (simplex_result.num_separated > 0) {
      result.separated_obs = boundary_sample.elem(simplex_result.separated_obs);
      result.num_separated = result.separated_obs.n_elem;
    }
  }

  // ReLU
  if (params.sep_use_relu) {
    SeparationResult relu_result =
        detect_separation_relu(y, X_centered, w, params);

    if (relu_result.num_separated > 0) {
      if (result.num_separated > 0) {
        result.separated_obs =
            unique(join_vert(result.separated_obs, relu_result.separated_obs));
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
