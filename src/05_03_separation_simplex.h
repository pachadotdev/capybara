// Simplex Separation Detection
// Ported from ppmlhdfe (Correia, Guimaraes, Zylkin 2019)
// Key insight: Test 1 checks RESIDUALS of collinear variables, not raw values

#ifndef CAPYBARA_SEPARATION_SIMPLEX_H
#define CAPYBARA_SEPARATION_SIMPLEX_H

namespace capybara {

// Find indices of columns that are NOT collinear (full rank subset)
// Uses rank-revealing Cholesky on X'X
// Returns indices of non-collinear columns
inline uvec find_noncollinear_cols(const mat &X, const vec &w,
                                   double tol = 1e-10) {
  if (X.n_cols == 0)
    return uvec();

  const mat Xw = X.each_col() % sqrt(w);
  const mat XtX = Xw.t() * Xw;

  mat R;
  uvec excluded;
  uword rank;
  chol_rank(R, excluded, rank, XtX, "upper", tol);

  return find(excluded == 0);
}

inline void simplex_presolve(mat &X, uvec &basic_vars, uvec &nonbasic_vars,
                             Col<unsigned char> &keep_mask, uword &k,
                             uword &n) {
  // First pass: identify and drop empty columns (byte mask for memory
  // efficiency)
  Col<unsigned char> is_dropped(k, fill::zeros);
  uword num_dropped = 0;
  for (uword j = 0; j < k; ++j) {
    if (accu(abs(X.col(j))) == 0) {
      is_dropped(j) = 1;
      ++num_dropped;
    }
  }

  // Remove empty columns early
  if (num_dropped > 0) {
    if (num_dropped == k) {
      // All columns empty - trivial case
      keep_mask.ones(n);
      basic_vars = regspace<uvec>(0, n - 1);
      nonbasic_vars = uvec();
      k = 0;
      return;
    }
    // Build index of non-empty columns
    uvec non_empty(k - num_dropped);
    uword idx = 0;
    for (uword j = 0; j < k; ++j) {
      if (!is_dropped(j))
        non_empty(idx++) = j;
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

  // Count and extract non-dropped columns (from byte mask)
  uword num_kept = 0;
  for (uword j = 0; j < k; ++j) {
    if (!is_dropped(j))
      ++num_kept;
  }
  if (num_kept > 0 && num_kept < k) {
    uvec not_dropped(num_kept);
    uword idx = 0;
    for (uword j = 0; j < k; ++j) {
      if (!is_dropped(j))
        not_dropped(idx++) = j;
    }
    X = X.cols(not_dropped);
    k = not_dropped.n_elem;
  }

  nonbasic_vars = conv_to<uvec>::from(nonbasic_list);
  basic_vars = find(keep_mask);
  n = basic_vars.n_elem;
}

// Main simplex algorithm following ppmlhdfe logic:
// 1. Check collinearity on interior (y>0) sample using zero weights on boundary
// 2. Compute residuals of collinear vars by regressing on non-collinear vars
// 3. Test 1: Check if residuals have uniform sign
// 4. Test 2: Full simplex on residuals
inline SeparationResult
detect_separation_simplex(const mat &X_centered, const uvec &boundary_sample,
                          const uvec &interior_sample, const vec &w,
                          const CapybaraParameters &params) {
  (void)interior_sample; // Unused but kept for API compatibility

  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  if (boundary_sample.n_elem == 0 || X_centered.n_cols == 0) {
    result.converged = true;
    return result;
  }

  const uword n_boundary = boundary_sample.n_elem;
  const uword k = X_centered.n_cols;

  // Use byte masks for memory efficiency
  Col<unsigned char> dropped_obs(n_boundary, fill::zeros);

  // Create interior weights: zero out boundary observations instead of copying
  vec w_interior = w;
  w_interior.elem(boundary_sample).zeros();

  // Step 1: Find collinear columns on interior sample (using zero-weight
  // strategy)
  const uvec ok_cols =
      find_noncollinear_cols(X_centered, w_interior, params.collin_tol);
  const uword n_ok = ok_cols.n_elem;

  if (n_ok == k) {
    // All variables are non-collinear on interior - no separation via this test
    result.converged = true;
    return result;
  }

  // Build mask of collinear (flagged) variables
  Col<unsigned char> flagged_var_mask(k, fill::ones);
  flagged_var_mask.elem(ok_cols).zeros();
  const uvec flagged_vars = find(flagged_var_mask);
  const uword n_flagged = flagged_vars.n_elem;

  if (n_flagged == 0) {
    result.converged = true;
    return result;
  }

  // Step 2: Compute residuals of flagged (collinear) variables on boundary
  // By regressing them on non-flagged vars using interior weights (boundary=0),
  // then computing residuals for boundary observations only
  mat residuals(n_boundary, n_flagged);

  if (n_ok == 0) {
    // All variables are collinear - extract boundary rows directly
    // Note: Armadillo subview chaining requires explicit submat
    for (uword j = 0; j < n_flagged; ++j) {
      residuals.col(j) = X_centered(boundary_sample, uvec{flagged_vars(j)});
    }
  } else {
    // Regress flagged vars on ok vars using interior weights (zero-weight
    // strategy) Compute: b = (X_ok' W X_ok)^-1 X_ok' W X_flag where W has zeros
    // for boundary observations

    // Use BLAS-optimized matrix operations
    // Extract submatrices first, then apply sqrt weights
    const vec sqrt_w_int = sqrt(w_interior);
    const mat X_ok = X_centered.cols(ok_cols);
    const mat X_flag = X_centered.cols(flagged_vars);

    // Apply sqrt weights via broadcasting (BLAS)
    const mat X_ok_w = X_ok.each_col() % sqrt_w_int;
    const mat X_flag_w = X_flag.each_col() % sqrt_w_int;

    // Form normal equations using BLAS gemm: O(n * p^2) ops
    const mat XtX_ok = X_ok_w.t() * X_ok_w;
    const mat XtX_ok_flag = X_ok_w.t() * X_flag_w;

    mat b;
    if (!solve(b, XtX_ok, XtX_ok_flag, solve_opts::likely_sympd)) {
      b = pinv(XtX_ok) * XtX_ok_flag;
    }

    // Compute residuals only for boundary observations using matrix ops
    // Extract boundary rows of submatrices
    const mat X_flag_bnd = X_flag.rows(boundary_sample);
    const mat X_ok_bnd = X_ok.rows(boundary_sample);
    residuals = X_flag_bnd - X_ok_bnd * b;
  }

  // Clean up tiny values
  residuals.clean(params.sep_tol);

  // Step 3: Test 1 - Check for uniform signs in each residual column
  // Following ppmlhdfe: if residual column is all >= 0 (or all <= 0),
  // then observations with non-zero values are separated
  Col<unsigned char> dropped_vars(n_flagged, fill::zeros);

  // Use vectorized min/max for all columns at once (single pass)
  rowvec col_mins = min(residuals, 0);
  rowvec col_maxs = max(residuals, 0);

  for (uword j = 0; j < n_flagged; ++j) {
    const double col_min = col_mins(j);
    const double col_max = col_maxs(j);

    if (col_min >= 0 && col_max > params.sep_tol) {
      // Uniformly non-negative with some positive values
      // Mark observations with positive values as separated
      const double *col_ptr = residuals.colptr(j);
      for (uword i = 0; i < n_boundary; ++i) {
        if (col_ptr[i] > params.sep_tol) {
          dropped_obs(i) = 1;
        }
      }
      dropped_vars(j) = 1;
    } else if (col_max <= 0 && col_min < -params.sep_tol) {
      // Uniformly non-positive with some negative values
      // Mark observations with negative values as separated
      const double *col_ptr = residuals.colptr(j);
      for (uword i = 0; i < n_boundary; ++i) {
        if (col_ptr[i] < -params.sep_tol) {
          dropped_obs(i) = 1;
        }
      }
      dropped_vars(j) = 1;
    } else if (col_min >= -params.sep_tol && col_max <= params.sep_tol) {
      // All zeros - drop variable but not observations
      dropped_vars(j) = 1;
    }
  }

  // Step 4: Test 2 - Full simplex on remaining observations and variables
  // (only if there are remaining flagged variables with mixed signs)
  // Count first, then allocate once (avoid O(n^2) resizing)
  uword n_remaining_vars = 0;
  for (uword j = 0; j < n_flagged; ++j) {
    if (!dropped_vars(j))
      ++n_remaining_vars;
  }
  uvec remaining_vars_idx(n_remaining_vars);
  uword var_idx = 0;
  for (uword j = 0; j < n_flagged; ++j) {
    if (!dropped_vars(j))
      remaining_vars_idx(var_idx++) = j;
  }

  uword n_remaining_obs = 0;
  for (uword i = 0; i < n_boundary; ++i) {
    if (!dropped_obs(i))
      ++n_remaining_obs;
  }
  uvec remaining_obs_idx(n_remaining_obs);
  uword obs_idx = 0;
  for (uword i = 0; i < n_boundary; ++i) {
    if (!dropped_obs(i))
      remaining_obs_idx(obs_idx++) = i;
  }

  if (remaining_vars_idx.n_elem > 1 && remaining_obs_idx.n_elem > 0) {
    mat X_simplex = residuals.submat(remaining_obs_idx, remaining_vars_idx);
    uword n_simp = X_simplex.n_rows;
    uword k_simp = X_simplex.n_cols;

    if (n_simp > k_simp) {
      uvec basic_vars, nonbasic_vars;
      Col<unsigned char> keep_mask;
      simplex_presolve(X_simplex, basic_vars, nonbasic_vars, keep_mask, k_simp,
                       n_simp);

      if (X_simplex.n_elem > 0 && basic_vars.n_elem > 0) {
        vec c_basic(basic_vars.n_elem, fill::ones);
        vec c_nonbasic(nonbasic_vars.n_elem, fill::ones);

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

          // Pivot operations
          X_simplex.row(pivot_row) /= pivot;

          for (uword i = 0; i < X_simplex.n_rows; ++i) {
            if (i != pivot_row && std::abs(pivot_column(i)) > 1e-14) {
              X_simplex.row(i) -= pivot_column(i) * X_simplex.row(pivot_row);
            }
          }

          std::swap(basic_vars(pivot_row), nonbasic_vars(pivot_col));
          std::swap(c_basic(pivot_row), c_nonbasic(pivot_col));
        }

        // Identify separated observations from simplex
        const uvec all_vars = join_vert(basic_vars, nonbasic_vars);
        const vec all_costs = join_vert(c_basic, c_nonbasic);

        const uvec zero_cost_idx = find(all_costs == 0);
        if (zero_cost_idx.n_elem > 0) {
          for (uword i = 0; i < zero_cost_idx.n_elem; ++i) {
            uword var_idx = all_vars(zero_cost_idx(i));
            if (var_idx < remaining_obs_idx.n_elem) {
              dropped_obs(remaining_obs_idx(var_idx)) = 1;
            }
          }
        }
      }
    }
  }

  // Convert dropped_obs mask to separated_obs indices (in boundary_sample)
  result.separated_obs = find(dropped_obs);
  result.num_separated = result.separated_obs.n_elem;

  // Track separated coefficients: flagged_vars that were dropped due to
  // uniform sign (these perfectly predict y=0, so their coef -> -Inf)
  const uvec dropped_var_idx = find(dropped_vars);
  if (dropped_var_idx.n_elem > 0) {
    result.separated_coefs = flagged_vars.elem(dropped_var_idx);
  }

  result.converged = true;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_SIMPLEX_H
