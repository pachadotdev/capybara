// Simplex Separation Detection

#ifndef CAPYBARA_SEPARATION_SIMPLEX_H
#define CAPYBARA_SEPARATION_SIMPLEX_H

namespace capybara {

inline void simplex_presolve(mat &X, uvec &basic_vars, uvec &nonbasic_vars,
                             Col<unsigned char> &keep_mask, uword &k, uword &n) {
  // First pass: identify and drop empty columns (byte mask for memory efficiency)
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
      if (!is_dropped(j)) non_empty(idx++) = j;
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
    if (!is_dropped(j)) ++num_kept;
  }
  if (num_kept > 0 && num_kept < k) {
    uvec not_dropped(num_kept);
    uword idx = 0;
    for (uword j = 0; j < k; ++j) {
      if (!is_dropped(j)) not_dropped(idx++) = j;
    }
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

  // Use byte masks instead of uvec (8x memory reduction)
  Col<unsigned char> dropped_obs(n, fill::zeros);
  Col<unsigned char> dropped_vars(k, fill::zeros);

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
  // Build indices from byte masks
  uword num_kept_obs = 0, num_kept_vars = 0;
  for (uword i = 0; i < n; ++i) if (!dropped_obs(i)) ++num_kept_obs;
  for (uword j = 0; j < k; ++j) if (!dropped_vars(j)) ++num_kept_vars;

  uvec kept_obs(num_kept_obs);
  uvec kept_vars(num_kept_vars);
  {
    uword oi = 0, vi = 0;
    for (uword i = 0; i < n; ++i) if (!dropped_obs(i)) kept_obs(oi++) = i;
    for (uword j = 0; j < k; ++j) if (!dropped_vars(j)) kept_vars(vi++) = j;
  }

  if (kept_vars.n_elem > 1 && kept_obs.n_elem > 0) {
    mat X_simplex = X.submat(kept_obs, kept_vars);
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

} // namespace capybara

#endif // CAPYBARA_SEPARATION_SIMPLEX_H
