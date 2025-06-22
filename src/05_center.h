#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

// MAP implementation - keeping for future optimization but not using it now

// Here we implement the Method of Alternating Projections (MAP) for
// centering/demeaning variables

struct map_workspace {
  // Working vectors for transformation
  vec residual;     // Current residual (after transform)
  vec residual_old; // Previous residual for convergence check
  vec projection;   // Current projection

  // CG acceleration vectors
  vec u; // Search direction
  vec v; // Transformed search direction
  vec r; // Residual in CG context

  // Group means data structures
  field<vec> group_means;   // Means by group and fixed effect
  field<vec> group_weights; // Weights by group and fixed effect

  // Sorting structures for cache efficiency
  field<uvec> sorted_indices; // Sorted indices for each fixed effect
  field<uvec> group_id;       // Group ID for each observation

  // Convergence tracking
  double tolerance;
  size_t max_iter;
  vec recent_ssr;            // Recent sum of squared residuals
  vec improvement_potential; // Total potential improvement

  map_workspace(size_t N, size_t k) {
    residual.set_size(N);
    residual_old.set_size(N);
    projection.set_size(N);
    u.set_size(N);
    v.set_size(N);
    r.set_size(N);

    group_means = field<vec>(k);
    group_weights = field<vec>(k);
    sorted_indices = field<uvec>(k);
    group_id = field<uvec>(k);

    tolerance = 1e-8;
    max_iter = 100;
  }
};

// Transformation Methods

inline void project_one_fe(const indices_info &indices, uword fe_idx, vec &x,
                           const vec &weights, map_workspace &ws) {
  const uword n_groups = indices.fe_sizes(fe_idx);

  // Resize group means if needed
  if (ws.group_means(fe_idx).n_elem != n_groups) {
    ws.group_means(fe_idx).set_size(n_groups);
    ws.group_weights(fe_idx).set_size(n_groups);
  }

  // Zero out group means and weights
  ws.group_means(fe_idx).zeros();
  ws.group_weights(fe_idx).zeros();

  // Compute group means and weights
  for (uword j = 0; j < n_groups; ++j) {
    const uvec &group = indices.get_group(fe_idx, j);
    if (group.is_empty())
      continue;

    // Use cache-efficient sorted indices if available
    if (indices.cache_optimized) {
      const uvec &sorted_group = indices.get_sorted_group(fe_idx, j);

      double sum = 0.0;
      double weight_sum = 0.0;

      // Process group with better cache locality
      for (uword i = 0; i < sorted_group.n_elem; ++i) {
        uword idx = sorted_group(i);
        double w = weights.n_elem > 1 ? weights(idx) : 1.0;
        sum += x(idx) * w;
        weight_sum += w;
      }

      if (weight_sum > 0) {
        double mean = sum / weight_sum;
        ws.group_means(fe_idx)(j) = mean;
        ws.group_weights(fe_idx)(j) = weight_sum;

        // Subtract mean from each observation in the group
        for (uword i = 0; i < sorted_group.n_elem; ++i) {
          x(sorted_group(i)) -= mean;
        }
      }
    } else {
      // Fallback to standard group processing
      double sum = 0.0;
      double weight_sum = 0.0;

      for (uword i = 0; i < group.n_elem; ++i) {
        uword idx = group(i);
        double w = weights.n_elem > 1 ? weights(idx) : 1.0;
        sum += x(idx) * w;
        weight_sum += w;
      }

      if (weight_sum > 0) {
        double mean = sum / weight_sum;
        ws.group_means(fe_idx)(j) = mean;
        ws.group_weights(fe_idx)(j) = weight_sum;

        for (uword i = 0; i < group.n_elem; ++i) {
          x(group(i)) -= mean;
        }
      }
    }
  }
}

// inline void transform_kaczmarz(const indices_info &indices, const vec &y,
//                                vec &ans, const vec &weights,
//                                map_workspace &ws) {
//   const uword k = indices.fe_sizes.n_elem;
//   ans = y;

//   // For each fixed effect, compute means and subtract
//   for (uword j = 0; j < k; ++j) {
//     project_one_fe(indices, j, ans, weights, ws);
//   }
// }

// Symmetric Kaczmarz (forward and backward pass)
inline void transform_sym_kaczmarz(const indices_info &indices, const vec &y,
                                   vec &ans, const vec &weights,
                                   map_workspace &ws) {
  const uword k = indices.fe_sizes.n_elem;
  ans = y;

  // Forward pass
  for (uword j = 0; j < k; ++j) {
    project_one_fe(indices, j, ans, weights, ws);
  }

  // Backward pass
  for (int j = k - 1; j >= 0; --j) {
    project_one_fe(indices, j, ans, weights, ws);
  }
}

// Acceleration Methods

inline void accelerate_none(const indices_info &indices, vec &y, vec &result,
                            const vec &weights, map_workspace &ws,
                            void (*transform)(const indices_info &, const vec &,
                                              vec &, const vec &,
                                              map_workspace &)) {
  vec residual(y.n_elem);

  for (size_t iter = 0; iter < ws.max_iter; ++iter) {
    ws.residual_old = y;

    transform(indices, y, residual, weights, ws);

    // Check convergence
    double rel_diff =
        norm(residual - ws.residual_old, 2) / (1.0 + norm(ws.residual_old, 2));
    if (rel_diff < ws.tolerance) {
      break;
    }

    // Update for next iteration
    y = residual;
  }

  result = residual;
}

// Conjugate Gradient
inline void accelerate_cg(const indices_info &indices, vec &y, vec &result,
                          const vec &weights, map_workspace &ws,
                          void (*transform)(const indices_info &, const vec &,
                                            vec &, const vec &,
                                            map_workspace &)) {
  const double eps = 1e-14;
  const size_t memory_length = 3; // Number of SSR values to track
  ws.recent_ssr.set_size(memory_length);
  ws.recent_ssr.zeros();

  // Initial projection
  transform(indices, y, ws.r, weights, ws);

  // Initial search direction
  ws.u = ws.r;

  // Initial SSR
  double ssr = dot(ws.r % weights, ws.r);
  ws.improvement_potential = dot(y % weights, y);

  for (size_t iter = 0; iter < ws.max_iter; ++iter) {
    transform(indices, ws.u, ws.v, weights, ws);

    // Compute step size
    double u_dot_v = dot(ws.u % weights, ws.v);
    double alpha = (std::abs(u_dot_v) > eps) ? (ssr / u_dot_v) : 0.0;

    // Update solution and residual
    y = y - alpha * ws.u;
    ws.r = ws.r - alpha * ws.v;

    // Track SSR history
    ws.recent_ssr(iter % memory_length) = alpha * ssr;
    ws.improvement_potential -= alpha * ssr;

    // Compute new SSR
    double ssr_old = ssr;
    ssr = dot(ws.r % weights, ws.r);

    // Update search direction using Fletcher-Reeves formula
    double beta = (std::abs(ssr_old) > eps) ? (ssr / ssr_old) : 0.0;
    ws.u = ws.r + beta * ws.u;

    // Check convergence using Hestenes stopping criteria
    double sum_recent_ssr_scalar = sum(ws.recent_ssr);
    double potential_scalar =
        as_scalar(ws.tolerance * ws.tolerance * ws.improvement_potential);
    if (sum_recent_ssr_scalar < potential_scalar) {
      break;
    }
  }

  result = y;
}

inline void center_variables(mat &X, vec &y, const vec &weights,
                             const indices_info &indices, double tolerance,
                             size_t max_iter) {
  const uword N = X.n_rows;
  const uword K = indices.fe_sizes.n_elem;

  // Early exit for empty case
  if (K == 0)
    return;

  // For K >= 2, use MAP acceleration by default
  if (K >= 2) {
    try {
      // Optimize indices for cache efficiency if not already done
      indices_info cached_indices = indices;
      if (!cached_indices.cache_optimized) {
        cached_indices.optimize_cache_access();
      }

      // Create and configure workspace
      map_workspace ws(N, K);
      ws.tolerance = tolerance;
      ws.max_iter = max_iter;

      // Function pointer for symmetric Kaczmarz transformation
      void (*transform)(const indices_info &, const vec &, vec &, const vec &,
                        map_workspace &) = &transform_sym_kaczmarz;

      // Center y first
      vec y_centered = y;
      accelerate_cg(cached_indices, y_centered, y, weights, ws, transform);

      // Center each column of X
      for (uword j = 0; j < X.n_cols; ++j) {
        vec x(X.colptr(j), N, false, false); // Non-owning view
        vec result;
        accelerate_cg(cached_indices, x, result, weights, ws, transform);
        X.col(j) = result;
      }

      // Validate results aren't NaN or Inf
      if (!y.is_finite() || !X.is_finite()) {
        throw std::runtime_error("MAP solver produced non-finite values");
      }

      return;
    } catch (const std::exception &e) {
      cpp11::warning(
          "MAP acceleration failed, falling back to standard algorithm: %s",
          e.what());
      // Continue to fallback implementation
    }
  }

  // Fallback to traditional implementation for K=1 or if acceleration failed
  bool use_weights = weights.n_elem > 1;
  const uword P = X.n_cols;
  const double sw = use_weights ? accu(weights) : N;

  // Pre-allocate working vectors
  vec x_old(N, fill::none);
  vec weights_vec = use_weights ? weights : vec(N, fill::ones);
  vec abs_diff(N, fill::none);
  vec abs_old(N, fill::none);

  for (uword p = 0; p < P; ++p) {
    vec x_col(X.colptr(p), N, false, false); // Non-owning view

    for (uword iter = 0; iter < max_iter; ++iter) {
      if (iter % 100 == 0 && iter > 0)
        check_user_interrupt();

      x_old = x_col;

      // Branch once at the category level
      if (use_weights) {
        for (uword k = 0; k < K; ++k) {
          const uword J = indices.fe_sizes(k);
          for (uword j = 0; j < J; ++j) {
            uvec grp = indices.get_group(k, j);
            if (grp.is_empty())
              continue;

            vec w_grp = weights.elem(grp);
            vec x_grp = x_col.elem(grp);
            double num = dot(w_grp, x_grp);
            double denom = accu(w_grp);

            if (denom > 0) {
              double meanj = num / denom;
              x_col.elem(grp) -= meanj;
            }
          }
        }
      } else {
        for (uword k = 0; k < K; ++k) {
          const uword J = indices.fe_sizes(k);
          for (uword j = 0; j < J; ++j) {
            uvec grp = indices.get_group(k, j);
            if (grp.is_empty())
              continue;

            double num = accu(x_col.elem(grp));
            double denom = grp.n_elem;

            if (denom > 0) {
              double meanj = num / denom;
              x_col.elem(grp) -= meanj;
            }
          }
        }
      }

      // Convergence check
      abs_diff = abs(x_col - x_old);
      abs_old = 1.0 + abs(x_old);
      double delta = accu((abs_diff / abs_old) % weights_vec) / sw;
      if (delta < tolerance)
        break;
    }
  }
}

inline void center_variables_batch(mat &X_work, vec &y, const vec &w,
                                   const mat &X_orig,
                                   const indices_info &indices, double tol,
                                   size_t max_iter, size_t iter_interrupt,
                                   bool use_weights,
                                   bool use_acceleration) {
  const uword N = X_orig.n_rows;
  const uword K = indices.fe_sizes.n_elem;

  if (K == 0) return;

  // Copy from original to avoid external matrix copying
  if (&X_work != &X_orig) {
    X_work = X_orig;
  }

  // For K >= 2, try to use MAP acceleration only if enabled
  if (N >= 5000 && K >= 2 && use_acceleration) {
    try {
      center_variables(X_work, y, w, indices, tol, max_iter);
      return;
    } catch (const std::exception &e) {
      cpp11::warning(
          "MAP batch acceleration failed, falling back to standard "
          "algorithm: %s",
          e.what());
      // Continue to fallback implementation
    }
  }

  // Traditional implementation (rest of the function unchanged)
  const uword P = X_work.n_cols;
  const double sw = use_weights ? accu(w) : N;

  vec y_old(N, fill::none);
  mat X_old(N, P, fill::none);

  bool y_converged = false;
  uvec x_converged(P, fill::zeros);
  vec weights_vec = use_weights ? w : vec(N, fill::ones);

  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter % iter_interrupt == 0 && iter > 0) check_user_interrupt();

    if (!y_converged) y_old = y;
    for (uword p = 0; p < P; ++p) {
      if (!x_converged(p)) X_old.col(p) = X_work.col(p);
    }

    // Alternate between categories
    for (uword k = 0; k < K; ++k) {
      const uword J = indices.fe_sizes(k);
      for (uword j = 0; j < J; ++j) {
        uvec grp = indices.get_group(k, j);
        if (grp.is_empty()) continue;

        // Center y
        if (!y_converged) {
          if (use_weights) {
            vec w_grp = w.elem(grp);
            vec y_grp = y.elem(grp);
            double num = dot(w_grp, y_grp);
            double denom = accu(w_grp);
            if (denom > 0) {
              double meanj = num / denom;
              y.elem(grp) -= meanj;
            }
          } else {
            double num = accu(y.elem(grp));
            double denom = grp.n_elem;
            if (denom > 0) {
              double meanj = num / denom;
              y.elem(grp) -= meanj;
            }
          }
        }

        // Center each X column
        for (uword p = 0; p < P; ++p) {
          if (x_converged(p)) continue;

          vec x_col(X_work.colptr(p), N, false, false);
          if (use_weights) {
            vec w_grp = w.elem(grp);
            vec x_grp = x_col.elem(grp);
            double num = dot(w_grp, x_grp);
            double denom = accu(w_grp);
            if (denom > 0) {
              double meanj = num / denom;
              x_col.elem(grp) -= meanj;
            }
          } else {
            double num = accu(x_col.elem(grp));
            double denom = grp.n_elem;
            if (denom > 0) {
              double meanj = num / denom;
              x_col.elem(grp) -= meanj;
            }
          }
        }
      }
    }

    // Check convergence
    if (!y_converged) {
      double delta =
          accu(abs(y - y_old) / (1.0 + abs(y_old)) % weights_vec) / sw;
      if (delta < tol) y_converged = true;
    }

    bool all_x_converged = true;
    for (uword p = 0; p < P; ++p) {
      if (!x_converged(p)) {
        double delta = accu(abs(X_work.col(p) - X_old.col(p)) /
                            (1.0 + abs(X_old.col(p))) % weights_vec) /
                       sw;
        if (delta < tol) {
          x_converged(p) = 1;
        } else {
          all_x_converged = false;
        }
      }
    }

    if (y_converged && all_x_converged) break;
  }
}

#endif // CAPYBARA_CENTER_H
