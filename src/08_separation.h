// Separation detection for Poisson models
// Based on Correia, Guimarães, Zylkin (2019)
// "Verifying the existence of maximum likelihood estimates for generalized
// linear models"

#ifndef CAPYBARA_SEPARATION_H
#define CAPYBARA_SEPARATION_H

namespace capybara {

// Result structure for separation detection
struct SeparationResult {
  uvec separated_obs;  // Indices of separated observations (0-based)
  vec certificate;     // z vector proving separation (certificate)
  uword num_separated; // Count of separated observations
  bool converged;      // Whether the algorithm converged
  uword iterations;    // Number of iterations used

  SeparationResult() : num_separated(0), converged(false), iterations(0) {}
};

// Parameters for separation detection
struct SeparationParameters {
  double tol;             // Convergence tolerance
  double zero_tol;        // Tolerance for treating values as zero
  uword max_iter;         // Maximum iterations for ReLU method
  uword simplex_max_iter; // Maximum iterations for simplex method
  bool use_relu;          // Whether to use ReLU method
  bool use_simplex;       // Whether to use simplex method
  bool verbose;           // Verbose output

  SeparationParameters()
      : tol(1e-8), zero_tol(1e-12), max_iter(1000), simplex_max_iter(10000),
        use_relu(true), use_simplex(true), verbose(false) {}
};

// ============================================================================
// Helper functions
// ============================================================================

// Create a mask vector: 1 for kept observations, 0 for removed
inline uvec create_mask(uword n, uword default_val, const uvec &indices,
                        uword index_val) {
  uvec mask(n, arma::fill::value(default_val));
  for (uword i = 0; i < indices.n_elem; ++i) {
    if (indices(i) < n) {
      mask(indices(i)) = index_val;
    }
  }
  return mask;
}

// Update mask at specified indices
inline void update_mask(uvec &mask, const uvec &indices, uword val) {
  for (uword i = 0; i < indices.n_elem; ++i) {
    if (indices(i) < mask.n_elem) {
      mask(indices(i)) = val;
    }
  }
}

// Update vector at specified indices
inline void update_vec(vec &v, const uvec &indices, double val) {
  for (uword i = 0; i < indices.n_elem; ++i) {
    if (indices(i) < v.n_elem) {
      v(indices(i)) = val;
    }
  }
}

// Edit values to zero if below tolerance
inline void edit_to_zero_tol(vec &v, double tol) {
  for (uword i = 0; i < v.n_elem; ++i) {
    if (std::abs(v(i)) < tol) {
      v(i) = 0.0;
    }
  }
}

// Check if value is in range
inline bool in_range(double x, double low, double high) {
  return x >= low && x <= high;
}

// ============================================================================
// ReLU Separation Detection
// Algorithm: Iterative least squares with ReLU activation
// Reference: Section 3.2 of Correia, Guimarães, Zylkin (2019)
// ============================================================================

// Solve weighted least squares: minimize ||sqrt(W)(y - X*beta)||^2
// Returns coefficients and residuals
inline vec solve_wls(const mat &X, const vec &y, const vec &w, vec &residuals) {
  if (X.n_cols == 0) {
    residuals = y;
    return vec();
  }

  // Weighted design matrix and response
  vec sqrt_w = sqrt(w);
  mat Xw = X.each_col() % sqrt_w;
  vec yw = y % sqrt_w;

  // Solve via QR decomposition for numerical stability
  vec beta;
  bool success = solve(beta, Xw, yw, arma::solve_opts::fast);

  if (!success) {
    // Fall back to normal equations with regularization
    mat XtWX = X.t() * (X.each_col() % w);
    XtWX.diag() += 1e-10; // Ridge regularization
    vec XtWy = X.t() * (y % w);
    beta = solve(XtWX, XtWy);
  }

  residuals = y - X * beta;
  return beta;
}

// Solve LSE (Least Squares with Equality constraints) via weighting method
// Based on Van Loan (1985) and Stewart (1997)
// Equality constraints are: X[constrained_sample, :] * beta =
// y[constrained_sample]
inline vec solve_lse_weighted(const mat &X, const vec &y,
                              const uvec &unconstrained_sample,
                              const uvec &constrained_sample, double M_weight,
                              vec &residuals) {
  const uword n = y.n_elem;

  if (X.n_cols == 0) {
    residuals = y;
    return vec();
  }

  // Create weight vector: high weight for constrained (y > 0) observations
  vec weights(n, arma::fill::ones);
  for (uword i = 0; i < constrained_sample.n_elem; ++i) {
    weights(constrained_sample(i)) = M_weight;
  }

  return solve_wls(X, y, weights, residuals);
}

// Main ReLU separation detection algorithm
inline SeparationResult
detect_separation_relu(const vec &y, const mat &X, const vec &w,
                       const SeparationParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;

  // Identify boundary (y=0) and interior (y>0) observations
  uvec boundary_sample = find(y == 0);
  uvec interior_sample = find(y > 0);

  uword num_boundary = boundary_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  // Initialize working variable u = (y == 0)
  vec u = conv_to<vec>::from(y == 0);

  // Method of weighting parameter (from Van Loan 1985)
  // M should be large enough to enforce constraints
  double M = 1.0 / std::sqrt(arma::datum::eps);
  double reg_tol = params.tol * params.tol;
  if (reg_tol < 1e-13)
    reg_tol = 1e-13;

  // Convergence tracking
  vec xbd(n, arma::fill::zeros);
  vec xbd_prev1(n), xbd_prev2(n);
  double uu_old = dot(u, u);

  // For acceleration detection
  double progress_ratio_prev1 = 0.0, progress_ratio_prev2 = 0.0;
  uword num_candidates_prev1 = 0, num_candidates_prev2 = 0;
  double ee_cumulative = 0.0;
  double ee_boundary = uu_old;
  bool convergence_is_stuck = false;

  // Main iteration loop
  for (uword iter = 0; iter < params.max_iter; ++iter) {
    // Check for user interrupt periodically
    if (iter % 100 == 0) {
      check_user_interrupt();
    }

    // Save previous predictions for acceleration detection
    xbd_prev2 = xbd_prev1;
    xbd_prev1 = xbd;

    // Solve weighted LSE problem
    vec resid;
    vec beta =
        solve_lse_weighted(X, u, boundary_sample, interior_sample, M, resid);
    xbd = u - resid;

    // Compute statistics
    double ee = dot(resid, resid);
    ee_cumulative += ee;
    double epsilon = ee + params.tol;
    double delta = epsilon + params.tol;

    // Progress tracking
    double progress_ratio = 100.0 * ee_cumulative / (ee_boundary + 1e-10);
    uword num_candidates = 0;
    for (uword i = 0; i < boundary_sample.n_elem; ++i) {
      if (xbd(boundary_sample(i)) > delta) {
        num_candidates++;
      }
    }

    // Check for stuck convergence (acceleration trigger)
    if (!convergence_is_stuck && iter > 3) {
      convergence_is_stuck = (progress_ratio - progress_ratio_prev2 < 1.0) &&
                             (num_candidates == num_candidates_prev2);
    }

    // Update tracking variables
    progress_ratio_prev2 = progress_ratio_prev1;
    progress_ratio_prev1 = progress_ratio;
    num_candidates_prev2 = num_candidates_prev1;
    num_candidates_prev1 = num_candidates;

    // Update xbd -> 0 for interior sample (y > 0)
    update_vec(xbd, interior_sample, 0.0);

    // Update xbd -> 0 for boundary observations within tolerance of zero
    uvec within_tol;
    {
      std::vector<uword> indices;
      for (uword i = 0; i < boundary_sample.n_elem; ++i) {
        double val = xbd(boundary_sample(i));
        if (in_range(val, -0.1 * delta, delta)) {
          indices.push_back(boundary_sample(i));
        }
      }
      if (!indices.empty()) {
        within_tol = uvec(indices);
        update_vec(xbd, within_tol, 0.0);
      }
    }

    // Check convergence: separation found if all boundary xbd >= 0
    bool all_nonnegative = true;
    for (uword i = 0; i < boundary_sample.n_elem; ++i) {
      if (xbd(boundary_sample(i)) < 0) {
        all_nonnegative = false;
        break;
      }
    }

    if (all_nonnegative) {
      // Count separated observations
      std::vector<uword> sep_indices;
      for (uword i = 0; i < boundary_sample.n_elem; ++i) {
        if (xbd(boundary_sample(i)) > 0) {
          sep_indices.push_back(boundary_sample(i));
        }
      }
      result.separated_obs = uvec(sep_indices);
      result.num_separated = sep_indices.size();
      result.certificate = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // Check for no negative residuals (implies no more separation to find)
    edit_to_zero_tol(resid, params.zero_tol);
    vec boundary_resid(boundary_sample.n_elem);
    for (uword i = 0; i < boundary_sample.n_elem; ++i) {
      boundary_resid(i) = resid(boundary_sample(i));
    }

    if (min(boundary_resid) >= 0) {
      // Update xbd for observations with positive residuals
      std::vector<uword> pos_resid_indices;
      for (uword i = 0; i < boundary_sample.n_elem; ++i) {
        if (boundary_resid(i) > delta) {
          pos_resid_indices.push_back(boundary_sample(i));
        }
      }
      if (!pos_resid_indices.empty()) {
        uvec pos_idx(pos_resid_indices);
        update_vec(xbd, pos_idx, 0.0);
      }

      // Count final separated observations
      std::vector<uword> sep_indices;
      for (uword i = 0; i < boundary_sample.n_elem; ++i) {
        if (xbd(boundary_sample(i)) > 0) {
          sep_indices.push_back(boundary_sample(i));
        }
      }
      result.separated_obs = uvec(sep_indices);
      result.num_separated = sep_indices.size();
      result.certificate = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // Apply ReLU: u = max(xbd, 0) for boundary observations
    for (uword i = 0; i < boundary_sample.n_elem; ++i) {
      uword idx = boundary_sample(i);
      u(idx) = std::max(xbd(idx), 0.0);
    }

    // Check for overall convergence
    double uu = dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.tol * 0.01) {
      // Essentially converged
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  // If we exit the loop without explicit convergence, collect what we found
  if (!result.converged) {
    result.iterations = params.max_iter;
    // Still mark any positive xbd values as potentially separated
    std::vector<uword> sep_indices;
    for (uword i = 0; i < boundary_sample.n_elem; ++i) {
      if (xbd(boundary_sample(i)) > params.tol) {
        sep_indices.push_back(boundary_sample(i));
      }
    }
    if (!sep_indices.empty()) {
      result.separated_obs = uvec(sep_indices);
      result.num_separated = sep_indices.size();
      result.certificate = xbd;
    }
  }

  return result;
}

// ============================================================================
// Simplex Separation Detection
// Algorithm: Linear programming via simplex to find separating hyperplane
// Reference: Section 3.3 of Correia, Guimarães, Zylkin (2019)
// ============================================================================

// Simplex presolve: remove free variables
// Based on Gaussian elimination to identify basic variables
inline void simplex_presolve(mat &X, uvec &basic_vars, uvec &nonbasic_vars,
                             uvec &keep_mask, uword &k, uword &n) {
  mat A(n, k, arma::fill::zeros);
  std::vector<uword> nonbasic_list;
  keep_mask.ones(n);
  uvec is_dropped(k, arma::fill::zeros);

  for (uword j = 0; j < k; ++j) {
    // Find first non-zero in column j
    uword pivot_row = n; // Invalid
    for (uword i = 0; i < n; ++i) {
      if (std::abs(X(i, j)) > arma::datum::eps * 100) {
        pivot_row = i;
        break;
      }
    }

    if (pivot_row >= n) {
      // Column is empty, mark as dropped
      is_dropped(j) = 1;
      continue;
    }

    nonbasic_list.push_back(pivot_row);
    keep_mask(pivot_row) = 0;

    // Gaussian elimination
    double pivot = -1.0 / X(pivot_row, j);
    vec pivot_row_vals = X.row(pivot_row).t() * pivot;
    vec pivot_col = X.col(j);

    A(pivot_row, j) = 1.0;
    A += pivot_col * (pivot * A.row(pivot_row));
    X += pivot_col * pivot_row_vals.t();

    // Clean numerical noise
    X.elem(find(abs(X) < 1e-14)).zeros();
    A.elem(find(abs(A) < 1e-14)).zeros();
  }

  // Update outputs
  uvec kept_rows = find(keep_mask);
  if (kept_rows.n_elem > 0) {
    X = A.rows(kept_rows);
  } else {
    X.reset();
  }

  // Remove dropped variables
  uvec not_dropped = find(is_dropped == 0);
  if (not_dropped.n_elem > 0 && not_dropped.n_elem < k) {
    X = X.cols(not_dropped);
    k = not_dropped.n_elem;
  }

  // Build nonbasic_vars
  nonbasic_vars = uvec(nonbasic_list);

  // Build basic_vars (remaining observations)
  basic_vars = find(keep_mask);

  n = basic_vars.n_elem;
}

// Main simplex algorithm for separation detection
inline SeparationResult
detect_separation_simplex(const mat &residuals,
                          const SeparationParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  if (residuals.n_rows == 0 || residuals.n_cols == 0) {
    result.converged = true;
    return result;
  }

  uword n = residuals.n_rows;
  uword k = residuals.n_cols;

  // =========================================================================
  // Test 1: Check if any column is all positive or all negative
  // =========================================================================
  vec col_min = min(residuals, 0).t();
  vec col_max = max(residuals, 0).t();

  // Apply tolerance
  for (uword j = 0; j < k; ++j) {
    if (std::abs(col_min(j)) < params.tol)
      col_min(j) = 0.0;
    if (std::abs(col_max(j)) < params.tol)
      col_max(j) = 0.0;
  }

  uvec all_zero = find((col_min == 0) && (col_max == 0));
  uvec all_positive = find(col_min >= 0);
  uvec all_negative = find(col_max <= 0);

  uvec dropped_obs(n, arma::fill::zeros);
  uvec dropped_vars(k, arma::fill::zeros);

  // Variables that are all positive have coefficient of -infinity
  // Drop observations where these residuals are positive
  if (all_positive.n_elem > 0) {
    for (uword i = 0; i < all_positive.n_elem; ++i) {
      uword j = all_positive(i);
      dropped_vars(j) = 1;
      for (uword row = 0; row < n; ++row) {
        if (residuals(row, j) > params.tol) {
          dropped_obs(row) = 1;
        }
      }
    }
  }

  // Variables that are all negative have coefficient of +infinity
  if (all_negative.n_elem > 0) {
    for (uword i = 0; i < all_negative.n_elem; ++i) {
      uword j = all_negative(i);
      dropped_vars(j) = 1;
      for (uword row = 0; row < n; ++row) {
        if (residuals(row, j) < -params.tol) {
          dropped_obs(row) = 1;
        }
      }
    }
  }

  // Mark all-zero variables
  for (uword i = 0; i < all_zero.n_elem; ++i) {
    dropped_vars(all_zero(i)) = 1;
  }

  // =========================================================================
  // Test 2: Apply simplex if needed
  // =========================================================================
  uvec kept_obs = find(dropped_obs == 0);
  uvec kept_vars = find(dropped_vars == 0);

  if (kept_vars.n_elem > 1 && kept_obs.n_elem > 0) {
    mat X_simplex = residuals.submat(kept_obs, kept_vars);

    uword n_simp = X_simplex.n_rows;
    uword k_simp = X_simplex.n_cols;

    if (n_simp > k_simp) { // Only proceed if system is overdetermined
      // Presolve
      uvec basic_vars, nonbasic_vars, keep_mask;
      simplex_presolve(X_simplex, basic_vars, nonbasic_vars, keep_mask, k_simp,
                       n_simp);

      if (X_simplex.n_rows > 0 && X_simplex.n_cols > 0) {
        // Initialize costs (all 1 = minimize sum of artificial variables)
        vec c_basic(basic_vars.n_elem, arma::fill::ones);
        vec c_nonbasic(nonbasic_vars.n_elem, arma::fill::ones);

        // Simplex iterations
        for (uword iter = 0; iter < params.simplex_max_iter; ++iter) {
          if (iter % 100 == 0) {
            check_user_interrupt();
          }

          // Compute reduced costs: r = c_nonbasic - c_basic' * X
          vec r = c_nonbasic - X_simplex.t() * c_basic;
          edit_to_zero_tol(r, 1e-14);

          // Check optimality: all reduced costs <= 0
          if (max(r) <= 0) {
            result.converged = true;
            break;
          }

          // Find entering variable (maximum reduced cost)
          uword pivot_col = index_max(r);
          vec pivot_column = X_simplex.col(pivot_col);

          // Check unboundedness: all entries in pivot column <= 0
          if (max(pivot_column) <= 0) {
            // Mark entering variable cost to 0
            c_nonbasic(pivot_col) = 0;

            // Also mark basic vars with negative pivot column entries
            for (uword i = 0; i < pivot_column.n_elem; ++i) {
              if (pivot_column(i) < 0) {
                c_basic(i) = 0;
              }
            }
            continue;
          }

          // Find leaving variable (minimum ratio test, but we use max since our
          // RHS is 0)
          uword pivot_row = index_max(pivot_column);
          double pivot = X_simplex(pivot_row, pivot_col);

          if (std::abs(pivot) < 1e-14) {
            continue; // Skip degenerate pivot
          }

          // Pivot operation
          X_simplex.row(pivot_row) /= pivot;
          for (uword i = 0; i < X_simplex.n_rows; ++i) {
            if (i != pivot_row) {
              X_simplex.row(i) -=
                  X_simplex(i, pivot_col) * X_simplex.row(pivot_row);
            }
          }

          // Swap variables
          uword temp_var = basic_vars(pivot_row);
          basic_vars(pivot_row) = nonbasic_vars(pivot_col);
          nonbasic_vars(pivot_col) = temp_var;

          double temp_cost = c_basic(pivot_row);
          c_basic(pivot_row) = c_nonbasic(pivot_col);
          c_nonbasic(pivot_col) = temp_cost;

          // Clean numerical noise
          X_simplex.elem(find(abs(X_simplex) < 1e-10)).zeros();
        }

        // Determine separated observations from final costs
        uvec all_vars = join_vert(basic_vars, nonbasic_vars);
        vec all_costs = join_vert(c_basic, c_nonbasic);

        // Sort by variable index
        uvec sort_idx = sort_index(all_vars);
        all_vars = all_vars.elem(sort_idx);
        all_costs = all_costs.elem(sort_idx);

        // Mark observations with zero cost as separated
        for (uword i = 0; i < all_vars.n_elem; ++i) {
          if (all_costs(i) == 0) {
            dropped_obs(kept_obs(all_vars(i))) = 1;
          }
        }
      }
    }
  }

  // Collect results
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
                                         const SeparationParameters &params) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  // Step 1: Identify suspect observations (y = 0)
  uvec boundary_sample = find(y == 0);
  if (boundary_sample.n_elem == 0) {
    return result;
  }

  uvec interior_sample = find(y > 0);

  // Step 2: Center X on y > 0 sample (partial out)
  mat X_centered = X;
  if (interior_sample.n_elem > 0 && X.n_cols > 0) {
    // Compute weighted means on interior sample
    vec w_interior = w;
    for (uword i = 0; i < boundary_sample.n_elem; ++i) {
      w_interior(boundary_sample(i)) = 0.0;
    }
    double sum_w = sum(w_interior);

    if (sum_w > 0) {
      for (uword j = 0; j < X.n_cols; ++j) {
        double wmean = dot(X.col(j), w_interior) / sum_w;
        X_centered.col(j) -= wmean;
      }
    }
  }

  // Step 3: Apply simplex method first (faster pre-check)
  if (params.use_simplex && X_centered.n_cols > 0) {
    mat X_boundary = X_centered.rows(boundary_sample);
    SeparationResult simplex_result =
        detect_separation_simplex(X_boundary, params);

    if (simplex_result.num_separated > 0) {
      // Convert local indices back to global
      result.separated_obs = boundary_sample.elem(simplex_result.separated_obs);
      result.num_separated = result.separated_obs.n_elem;
    }
  }

  // Step 4: Apply ReLU method (more thorough)
  if (params.use_relu) {
    SeparationResult relu_result =
        detect_separation_relu(y, X_centered, w, params);

    if (relu_result.num_separated > 0) {
      // Merge results (union of separated observations)
      if (result.num_separated > 0) {
        uvec combined =
            join_vert(result.separated_obs, relu_result.separated_obs);
        result.separated_obs = unique(combined);
        result.num_separated = result.separated_obs.n_elem;
      } else {
        result.separated_obs = relu_result.separated_obs;
        result.num_separated = relu_result.num_separated;
      }
      result.certificate = relu_result.certificate;
      result.iterations = relu_result.iterations;
    }
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_H
