#ifndef CAPYBARA_CENTER_VARIABLES_H
#define CAPYBARA_CENTER_VARIABLES_H

// This implements the fastest MAP from reghdfe:
// symmetric Kaczmarz + conjugate gradient

// Projection for a single fixed effect using unified Z matrix
inline void project_fe(mat &Z, size_t k, const indices_info &indices,
                       const vec &w, bool use_weights) {
  const size_t J = indices.fe_sizes(k);

  for (size_t j = 0; j < J; ++j) {
    const uvec &idx = indices.get_group(k, j);
    if (idx.is_empty())
      continue;

    const size_t n_idx = idx.n_elem;
    double inv_sumw;

    if (use_weights) {
      double sumw = accu(w(idx));
      inv_sumw = (sumw > 0) ? (1.0 / sumw) : 0.0;
    } else {
      inv_sumw = 1.0 / n_idx;
    }

    // Project all columns (X cols + y col) in unified Z matrix
    if (use_weights) {
      const vec w_subset = w(idx);
      // Computation across all columns
      mat Z_subset = Z.rows(idx);
      rowvec zbars = sum(Z_subset.each_col() % w_subset, 0) * inv_sumw;
      Z.rows(idx) -= ones<vec>(idx.n_elem) * zbars;
    } else {
      // Computation across all columns
      mat Z_subset = Z.rows(idx);
      rowvec zbars = mean(Z_subset, 0);
      Z.rows(idx) -= ones<vec>(idx.n_elem) * zbars;
    }
  }
}

// Symmetric Kaczmarz transformation (forward + backward pass)
inline void transform_symmetric_kaczmarz(mat &Z, const vec &w,
                                         const indices_info &indices,
                                         bool use_weights) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  // Forward pass
  for (size_t k = 0; k < K; ++k) {
    project_fe(Z, k, indices, w, use_weights);
  }

  // Backward pass
  for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
    project_fe(Z, static_cast<size_t>(k), indices, w, use_weights);
  }
}

// Weighted sum of squares for convergence check using unified Z matrix
inline double weighted_ssr(const mat &Z, const vec &w) {
  if (w.n_elem > 1) {
    // Weighted computation - column-wise
    double ssr = 0.0;
    for (size_t p = 0; p < Z.n_cols; ++p) {
      ssr += dot(w, Z.col(p) % Z.col(p));
    }
    return ssr;
  } else {
    // Unweighted computation
    return accu(Z % Z);
  }
}

// Helper to compute weighted quadratic sum for vectors
inline double weighted_quadsum_vec(const vec &x, const vec &y, const vec &w) {
  if (w.n_elem > 1) {
    return dot(x % y, w);
  } else {
    return dot(x, y);
  }
}

// Helper to compute weighted quadratic sum for matrices (column-wise)
inline rowvec weighted_quadsum_mat(const mat &x, const mat &y, const vec &w) {
  if (w.n_elem > 1) {
    const size_t P = x.n_cols;
    if (P < 14) {
      // For thin matrices, use more efficient computation
      const mat xy = x % y;
      return sum(xy.each_col() % w, 0);
    } else {
      // For wide matrices, use column-wise computation
      rowvec result(P);
      for (size_t p = 0; p < P; ++p) {
        result(p) = accu((x.col(p) % y.col(p)) % w);
      }
      return result;
    }
  } else {
    // Unweighted case
    const size_t P = x.n_cols;
    if (P < 25) {
      return diagvec(trans(x) * y).t();
    } else {
      return sum(x % y, 0);
    }
  }
}

// Conjugate gradient acceleration using unified Z matrix
inline void center_variables_cg(mat &Z, const vec &w,
                                const indices_info &indices, double tol,
                                size_t max_iter, size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  const bool use_weights = (w.n_elem > 1);
  size_t iint = iter_interrupt;

  // Workspace for CG
  const size_t N = Z.n_rows;
  const size_t P_cols = Z.n_cols;
  mat r_Z, u_Z, v_Z;
  rowvec ssr_Z, ssr_old_Z, alpha_Z, beta_Z, improvement_potential_Z;

  const size_t d = 1; // Number of recent SSR values for convergence
  mat recent_ssr_Z;

  r_Z.set_size(N, P_cols);
  u_Z.set_size(N, P_cols);
  v_Z.set_size(N, P_cols);
  ssr_Z.set_size(P_cols);
  ssr_old_Z.set_size(P_cols);
  alpha_Z.set_size(P_cols);
  beta_Z.set_size(P_cols);
  improvement_potential_Z.set_size(P_cols);
  recent_ssr_Z.set_size(d, P_cols);

  // Initialize improvement potential
  improvement_potential_Z = weighted_quadsum_mat(Z, Z, w);

  // Initialize: r = T(Z) (first transformation)
  r_Z = Z;
  transform_symmetric_kaczmarz(r_Z, w, indices, use_weights);
  r_Z = Z - r_Z; // Get residual
  ssr_Z = weighted_quadsum_mat(r_Z, r_Z, w);
  u_Z = r_Z;

  // Main CG loop
  for (size_t iter = 1; iter <= max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Apply transformation to u: v = T(u) - reuse existing matrices
    v_Z = u_Z;
    transform_symmetric_kaczmarz(v_Z, w, indices, use_weights);
    v_Z = u_Z - v_Z; // Get residual in-place

    // Compute alpha = ssr / (u'v)
    rowvec uv = weighted_quadsum_mat(u_Z, v_Z, w);
    for (size_t p = 0; p < alpha_Z.n_elem; ++p) {
      alpha_Z(p) = (uv(p) > 0) ? ssr_Z(p) / uv(p) : 0.0;
    }

    // Update: Z = Z - alpha * u, r = r - alpha * v
    Z -= u_Z.each_row() % alpha_Z;
    r_Z -= v_Z.each_row() % alpha_Z;

    // SSR for convergence check
    rowvec alpha_ssr = alpha_Z % ssr_Z;
    recent_ssr_Z.row(iter % d) = alpha_ssr;
    improvement_potential_Z -= alpha_ssr;

    // Update SSR and compute beta
    ssr_old_Z = ssr_Z;
    ssr_Z = weighted_quadsum_mat(r_Z, r_Z, w);

    // Compute beta = ssr / ssr_old (Fletcher-Reeves)
    // Add small epsilon to avoid division by zero
    beta_Z = ssr_Z / (ssr_old_Z + 1e-16);

    // Update u = r + beta * u
    u_Z = r_Z + u_Z.each_row() % beta_Z;

    // Check convergence (Hestenes-Stiefel)
    bool converged = false;
    rowvec recent_sum = sum(recent_ssr_Z.rows(0, std::min(iter, d) - 1), 0);
    for (size_t p = 0; p < recent_sum.n_elem; ++p) {
      double eps_threshold = 1e-15;
      double ratio =
          recent_sum(p) / (improvement_potential_Z(p) + eps_threshold);
      if (sqrt(ratio) <= tol) {
        converged = true;
        break;
      }
    }

    if (converged)
      break;
  }
}

// Hybrid acceleration: start with simple Kaczmarz, then switch to CG
// This follows reghdfe's accelerate_hybrid approach
inline void center_variables_hybrid(mat &Z, const vec &w,
                                    const indices_info &indices, double tol,
                                    size_t max_iter, size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;
  
  const bool use_weights = (w.n_elem > 1);
  
  const size_t accel_start = 6; // Start CG after 6 iterations, like reghdfe
  size_t iint = iter_interrupt;
  
  // First phase: simple symmetric Kaczmarz
  double prev_ssr = weighted_ssr(Z, w);
  
  for (size_t iter = 1; iter <= std::min(accel_start, max_iter); ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }
    
    // Apply symmetric Kaczmarz transformation
    transform_symmetric_kaczmarz(Z, w, indices, use_weights);
    
    // Check convergence using SSR (avoid matrix copy)
    double curr_ssr = weighted_ssr(Z, w);
    double rel_change = std::abs(curr_ssr - prev_ssr) / (1.0 + prev_ssr);
    if (rel_change <= tol * tol) return;  // Square tolerance since we're comparing SSR
    prev_ssr = curr_ssr;
  }
  
  // Second phase: switch to CG if not converged
  if (accel_start < max_iter) {
    // Adjust remaining iterations
    size_t remaining_iter = max_iter - accel_start;
    center_variables_cg(Z, w, indices, tol, remaining_iter, iter_interrupt);
  }
}

// Main centering function for matrix (X only, no y)
inline void center_variables(mat &V, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter,
                             size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;
    
  // Handle constant-only case (no fixed effects)
  if (K == 1) {
    // Check if this is a trivial case with only one group
    const size_t J = indices.fe_sizes(0);
    if (J == 1) {
      // Single group
      const bool use_weights = (w.n_elem > 1);
      if (use_weights) {
        double weight_sum = accu(w);
        rowvec means(V.n_cols);
        for (size_t p = 0; p < V.n_cols; ++p) {
          means(p) = dot(V.col(p), w) / weight_sum;
        }
        V.each_row() -= means;
      } else {
        rowvec means = mean(V, 0);
        V.each_row() -= means;
      }
      return;
    }
  }
  
  // Use unified Z matrix approach - V is already our Z matrix
  center_variables_hybrid(V, w, indices, tol, max_iter, iter_interrupt);
}

// Main centering function for vector (y only, no X)
inline void center_variables(vec &y, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter,
                             size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;
    
  // Handle constant-only case (no fixed effects)
  if (K == 1) {
    const size_t J = indices.fe_sizes(0);
    if (J == 1) {
      // Single group
      const bool use_weights = (w.n_elem > 1);
      if (use_weights) {
        double mean_y = dot(y, w) / accu(w);
        y -= mean_y;
      } else {
        double mean_y = mean(y);
        y -= mean_y;
      }
      return;
    }
  }
  
  // Convert y to Z matrix (P=0 case, single column)
  mat Z(y.n_elem, 1);
  Z.col(0) = y;
  center_variables_hybrid(Z, w, indices, tol, max_iter, iter_interrupt);
  y = Z.col(0);
}

// Main centering function for both X and y using unified Z matrix
inline void center_variables(mat &X_work, vec &y, const vec &w,
                             const mat &X_orig, const indices_info &indices,
                             double tol, size_t max_iter,
                             size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;
    
  if (&X_work != &X_orig) {
    X_work = X_orig;
  }
  
  // Handle constant-only case (no fixed effects)
  if (K == 1) {
    const size_t J = indices.fe_sizes(0);
    if (J == 1) {
      // Single group
      const bool use_weights = (w.n_elem > 1);
      if (use_weights) {
        double weight_sum = accu(w);
        rowvec means_X(X_work.n_cols);
        for (size_t p = 0; p < X_work.n_cols; ++p) {
          means_X(p) = dot(X_work.col(p), w) / weight_sum;
        }
        double mean_y = dot(y, w) / weight_sum;
        X_work.each_row() -= means_X;
        y -= mean_y;
      } else {
        rowvec means_X = mean(X_work, 0);
        double mean_y = mean(y);
        X_work.each_row() -= means_X;
        y -= mean_y;
      }
      return;
    }
  }
  
  // Use in-place operations to avoid copying - resize X_work to include y
  const size_t P = X_work.n_cols;
  const size_t N = X_work.n_rows;
  
  // Resize X_work to P+1 columns and copy y into the last column
  X_work.resize(N, P + 1);
  X_work.col(P) = y;
  
  // Apply centering to unified matrix in-place
  center_variables_hybrid(X_work, w, indices, tol, max_iter, iter_interrupt);
  
  // Extract y back and resize X_work
  y = X_work.col(P);
  X_work.resize(N, P);
}

#endif // CAPYBARA_CENTER_VARIABLES_H
