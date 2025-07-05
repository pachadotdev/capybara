#ifndef CAPYBARA_CENTER_VARIABLES_H
#define CAPYBARA_CENTER_VARIABLES_H

// This implements the fastest MAP from reghdfe:
// symmetric Kaczmarz + conjugate gradient

// Projection for a single fixed effect
inline void project_fe(mat *X, vec *y, size_t k, const indices_info &indices,
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

    // Project matrix columns if present
    if (X != nullptr) {
      const size_t P = X->n_cols;
      
      if (use_weights) {
        const vec w_subset = w(idx);
        for (size_t p = 0; p < P; ++p) {
          vec x_subset = X->submat(idx, uvec{p});
          double xbar = dot(x_subset, w_subset) * inv_sumw;
          X->submat(idx, uvec{p}) -= xbar;
        }
      } else {
        for (size_t p = 0; p < P; ++p) {
          vec x_subset = X->submat(idx, uvec{p});
          double xbar = mean(x_subset);
          X->submat(idx, uvec{p}) -= xbar;
        }
      }
    }

    // Project vector if present
    if (y != nullptr) {
      if (use_weights) {
        const vec w_subset = w(idx);
        const vec y_subset = y->elem(idx);
        double ybar = dot(y_subset, w_subset) * inv_sumw;
        y->elem(idx) -= ybar;
      } else {
        vec y_subset = y->elem(idx);
        double ybar = mean(y_subset);
        y->elem(idx) -= ybar;
      }
    }
  }
}

// Symmetric Kaczmarz transformation (forward + backward pass)
inline void transform_symmetric_kaczmarz(mat *X, vec *y, const vec &w,
                                         const indices_info &indices,
                                         bool use_weights) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  // Forward pass
  for (size_t k = 0; k < K; ++k) {
    project_fe(X, y, k, indices, w, use_weights);
  }

  // Backward pass
  for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
    project_fe(X, y, static_cast<size_t>(k), indices, w, use_weights);
  }
}

// Weighted sum of squares for convergence check
inline double weighted_ssr(const mat *X, const vec *y, const vec &w) {
  double ssr = 0.0;

  if (X != nullptr) {
    const size_t N = X->n_rows;
    const size_t P = X->n_cols;
    for (size_t p = 0; p < P; ++p) {
      for (size_t i = 0; i < N; ++i) {
        double val = (*X)(i, p);
        ssr += w.n_elem > 1 ? w(i) * val * val : val * val;
      }
    }
  }

  if (y != nullptr) {
    const size_t N = y->n_elem;
    for (size_t i = 0; i < N; ++i) {
      double val = (*y)(i);
      ssr += w.n_elem > 1 ? w(i) * val * val : val * val;
    }
  }

  return ssr;
}

// Helper to compute weighted quadratic sum for vectors
inline double weighted_quadsum_vec(const vec &x, const vec &y, const vec &w) {
  if (w.n_elem > 1) {
    return dot(x % y, w);
  } else {
    return dot(x, y);
  }
}

// Helper to compute weighted quadratic sum for matrices (column-wise) - optimized
inline rowvec weighted_quadsum_mat(const mat &x, const mat &y, const vec &w) {
  if (w.n_elem > 1) {
    // Use the same approach as reghdfe's weighted_quadcolsum
    const size_t P = x.n_cols;
    if (P < 14) {
      // For thin matrices, use quadcross approach
      return diagvec(trans(x % y) * diagmat(w)).t();
    } else {
      // For wide matrices, use column-wise computation
      rowvec result(P);
      for (size_t p = 0; p < P; ++p) {
        result(p) = dot(x.col(p) % y.col(p), w);
      }
      return result;
    }
  } else {
    // Unweighted case - use different thresholds like reghdfe
    const size_t P = x.n_cols;
    if (P < 25) {
      return diagvec(trans(x) * y).t();
    } else {
      return sum(x % y, 0);
    }
  }
}

// Conjugate gradient acceleration
inline void center_variables_cg(mat *X, vec *y, const vec &w,
                                const indices_info &indices, double tol,
                                size_t max_iter, size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  const bool has_X = (X != nullptr);
  const bool has_y = (y != nullptr);
  const bool use_weights = (w.n_elem > 1);
  size_t iint = iter_interrupt;

  // Workspace for CG
  mat r_X, u_X, v_X;
  vec r_y, u_y, v_y;
  rowvec ssr_X, ssr_old_X, alpha_X, beta_X, improvement_potential_X;
  double ssr_y, ssr_old_y, alpha_y, beta_y, improvement_potential_y;

  const size_t d = 1; // Number of recent SSR values for convergence
  mat recent_ssr_X;
  vec recent_ssr_y;

  if (has_X) {
    const size_t N = X->n_rows;
    const size_t P = X->n_cols;
    r_X.set_size(N, P);
    u_X.set_size(N, P);
    v_X.set_size(N, P);
    ssr_X.set_size(P);
    ssr_old_X.set_size(P);
    alpha_X.set_size(P);
    beta_X.set_size(P);
    improvement_potential_X.set_size(P);
    recent_ssr_X.set_size(d, P);

    // Initialize improvement potential
    improvement_potential_X = weighted_quadsum_mat(*X, *X, w);
  }

  if (has_y) {
    const size_t N = y->n_elem;
    r_y.set_size(N);
    u_y.set_size(N);
    v_y.set_size(N);
    recent_ssr_y.set_size(d);

    // Initialize improvement potential
    improvement_potential_y = weighted_quadsum_vec(*y, *y, w);
  }

  // Initialize: r = T(X,y) (first transformation)
  if (has_X) {
    r_X = *X;
    transform_symmetric_kaczmarz(&r_X, nullptr, w, indices, use_weights);
    r_X = *X - r_X; // Get residual
    ssr_X = weighted_quadsum_mat(r_X, r_X, w);
    u_X = r_X;
  }

  if (has_y) {
    r_y = *y;
    transform_symmetric_kaczmarz(nullptr, &r_y, w, indices, use_weights);
    r_y = *y - r_y; // Get residual
    ssr_y = weighted_quadsum_vec(r_y, r_y, w);
    u_y = r_y;
  }

  // Main CG loop
  for (size_t iter = 1; iter <= max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Apply transformation to u: v = T(u) - reuse existing matrices
    if (has_X) {
      v_X = u_X;
      transform_symmetric_kaczmarz(&v_X, nullptr, w, indices, use_weights);
      v_X = u_X - v_X; // Get residual in-place

      // Compute alpha = ssr / (u'v)
      rowvec uv = weighted_quadsum_mat(u_X, v_X, w);
      for (size_t p = 0; p < alpha_X.n_elem; ++p) {
        alpha_X(p) = (uv(p) > 0) ? ssr_X(p) / uv(p) : 0.0;
      }

      // Update: x = x - alpha * u, r = r - alpha * v
      *X -= u_X.each_row() % alpha_X;
      r_X -= v_X.each_row() % alpha_X;

      // SSR for convergence check
      rowvec alpha_ssr = alpha_X % ssr_X;
      recent_ssr_X.row(iter % d) = alpha_ssr;
      improvement_potential_X -= alpha_ssr;
    }

    if (has_y) {
      v_y = u_y;
      transform_symmetric_kaczmarz(nullptr, &v_y, w, indices, use_weights);
      v_y = u_y - v_y; // Get residual

      // alpha = ssr / (u'v)
      double uv = weighted_quadsum_vec(u_y, v_y, w);
      alpha_y = (uv > 0) ? ssr_y / uv : 0.0;

      // Update: y = y - alpha * u, r = r - alpha * v
      *y -= alpha_y * u_y;
      r_y -= alpha_y * v_y;

      // Store recent SSR for convergence check
      recent_ssr_y(iter % d) = alpha_y * ssr_y;
      improvement_potential_y -= alpha_y * ssr_y;
    }

    // Update SSR and compute beta
    if (has_X) {
      ssr_old_X = ssr_X;
      ssr_X = weighted_quadsum_mat(r_X, r_X, w);

      // Compute beta = ssr / ssr_old (Fletcher-Reeves)
      // Add small epsilon to avoid division by zero
      beta_X = ssr_X / (ssr_old_X + 1e-16);

      // Update u = r + beta * u
      u_X = r_X + u_X.each_row() % beta_X;
    }

    if (has_y) {
      ssr_old_y = ssr_y;
      ssr_y = weighted_quadsum_vec(r_y, r_y, w);

      // Compute beta = ssr / ssr_old (Fletcher-Reeves)
      beta_y = (ssr_old_y > 0) ? ssr_y / ssr_old_y : 0.0;

      // Update u = r + beta * u
      u_y = r_y + beta_y * u_y;
    }

    // Check convergence (Hestenes-Stiefel)
    bool converged = false;
    if (has_X) {
      rowvec recent_sum = sum(recent_ssr_X.rows(0, std::min(iter, d) - 1), 0);
      for (size_t p = 0; p < recent_sum.n_elem; ++p) {
        double eps_threshold = 1e-15;
        double ratio =
            recent_sum(p) / (improvement_potential_X(p) + eps_threshold);
        if (sqrt(ratio) <= tol) {
          converged = true;
          break;
        }
      }
    }

    if (has_y && !converged) {
      double recent_sum = sum(recent_ssr_y.rows(0, std::min(iter, d) - 1));
      double eps_threshold = 1e-15;
      double ratio = recent_sum / (improvement_potential_y + eps_threshold);
      if (sqrt(ratio) <= tol) {
        converged = true;
      }
    }

    if (converged)
      break;
  }
}

// Hybrid acceleration: start with simple Kaczmarz, then switch to CG
// This follows reghdfe's accelerate_hybrid approach
inline void center_variables_hybrid(mat *X, vec *y, const vec &w,
                                    const indices_info &indices, double tol,
                                    size_t max_iter, size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;
  
  const bool has_X = (X != nullptr);
  const bool has_y = (y != nullptr);
  const bool use_weights = (w.n_elem > 1);
  
  const size_t accel_start = 6; // Start CG after 6 iterations, like reghdfe
  size_t iint = iter_interrupt;
  
  // First phase: simple symmetric Kaczmarz
  for (size_t iter = 1; iter <= std::min(accel_start, max_iter); ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }
    
    // Store previous state for convergence check
    mat X_old;
    vec y_old;
    if (has_X) X_old = *X;
    if (has_y) y_old = *y;
    
    // Apply symmetric Kaczmarz transformation
    transform_symmetric_kaczmarz(X, y, w, indices, use_weights);
    
    // Check convergence
    bool converged = true;
    if (has_X) {
      double rel_change = norm(*X - X_old, "fro") / (1.0 + norm(X_old, "fro"));
      if (rel_change > tol) converged = false;
    }
    if (has_y) {
      double rel_change = norm(*y - y_old) / (1.0 + norm(y_old));
      if (rel_change > tol) converged = false;
    }
    
    if (converged) return;
  }
  
  // Second phase: switch to CG if not converged
  if (accel_start < max_iter) {
    // Adjust remaining iterations
    size_t remaining_iter = max_iter - accel_start;
    center_variables_cg(X, y, w, indices, tol, remaining_iter, iter_interrupt);
  }
}

// Main centering function following reghdfe's MAP solver
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
  
  // Follow reghdfe's poolsize logic
  const size_t P = V.n_cols;
  const size_t poolsize = P; // all columns at once
  
  if (poolsize >= P) {
    // Process all columns together - hybrid method
    center_variables_hybrid(&V, nullptr, w, indices, tol, max_iter, iter_interrupt);
  } else {
    // Process columns individually
    for (size_t p = 0; p < P; ++p) {
      vec col_p = V.col(p);
      center_variables_hybrid(nullptr, &col_p, w, indices, tol, max_iter, iter_interrupt);
      V.col(p) = col_p;
    }
  }
}

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
  
  // Use hybrid method for vector
  center_variables_hybrid(nullptr, &y, w, indices, tol, max_iter, iter_interrupt);
}

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
  
  // Follow reghdfe's poolsize logic for joint processing
  const size_t P = X_work.n_cols;
  const size_t poolsize = P + 1; // X columns + y vector
  
  if (poolsize >= P + 1) {
    // Process X and y together
    center_variables_hybrid(&X_work, &y, w, indices, tol, max_iter, iter_interrupt);
  } else {
    // Process separately
    center_variables_hybrid(&X_work, nullptr, w, indices, tol, max_iter, iter_interrupt);
    center_variables_hybrid(nullptr, &y, w, indices, tol, max_iter, iter_interrupt);
  }
}

#endif // CAPYBARA_CENTER_VARIABLES_H
