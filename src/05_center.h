#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

inline indices_info indices_for_cache(const indices_info &indices) {
  indices_info cached;
  cached.all_indices = indices.all_indices;
  cached.group_offsets = indices.group_offsets;
  cached.group_sizes = indices.group_sizes;
  cached.fe_offsets = indices.fe_offsets;
  cached.fe_sizes = indices.fe_sizes;
  cached.nonempty_groups = indices.nonempty_groups;
  cached.precomputed_groups = indices.precomputed_groups;
  cached.optimize_cache_access();
  return cached;
}

namespace accelerated_centering {

inline bool irons_tuck_update(uword nb_coef_no_Q, vec &X, const vec &GX,
                              const vec &GGX, vec &delta_GX, vec &delta2_X) {
  delta_GX = GGX - GX;
  delta2_X = delta_GX - GX + X;

  double ssq = dot(delta2_X, delta2_X);
  if (ssq == 0)
    return true;

  double coef = dot(delta_GX, delta2_X) / ssq;
  X = GGX - coef * delta_GX;
  return false;
}

// Convergence criterion
inline bool continue_crit(double x_new, double x_old, double tol) {
  return std::abs(x_new - x_old) / (1.0 + std::abs(x_old)) > tol;
}

// SSR stopping criterion
inline bool stopping_crit(double ssr_old, double ssr_new, double tol) {
  return std::abs(ssr_new - ssr_old) / (0.1 + std::abs(ssr_new)) < tol;
}

// Pre-compute group info structure for efficient access
struct GroupInfo {
  vec group_weights;
  uvec group_start; // Starting index for each group in sorted data
  uvec group_end;   // Ending index for each group in sorted data
  uvec sort_index;  // Maps original indices to sorted positions
  uvec inv_sort;    // Maps sorted positions back to original indices
};

// Pre-compute group information for efficient processing
inline field<GroupInfo> precompute_group_info(const indices_info &indices,
                                              const vec &w, uword n) {
  uword K = indices.fe_sizes.n_elem;
  field<GroupInfo> group_info(K);
  bool use_weights = (w.n_elem > 1);

  for (uword k = 0; k < K; ++k) {
    GroupInfo &info = group_info(k);
    uword n_groups = indices.fe_sizes(k);

    info.group_weights.set_size(n_groups);
    info.group_start.set_size(n_groups);
    info.group_end.set_size(n_groups);

    // Count group sizes and compute weights
    uvec group_counts(n_groups, fill::zeros);
    for (uword j = 0; j < n_groups; ++j) {
      const uvec &grp = indices.get_sorted_group(k, j);
      group_counts(j) = grp.n_elem;

      double weight_sum = 0.0;
      if (use_weights) {
        for (uword idx = 0; idx < grp.n_elem; ++idx) {
          if (grp(idx) < n)
            weight_sum += w(grp(idx));
        }
      } else {
        weight_sum = grp.n_elem;
      }
      info.group_weights(j) = weight_sum > 0 ? weight_sum : 1.0;
    }

    // Compute cumulative positions
    info.group_start(0) = 0;
    for (uword j = 1; j < n_groups; ++j) {
      info.group_start(j) = info.group_start(j - 1) + group_counts(j - 1);
    }
    for (uword j = 0; j < n_groups; ++j) {
      info.group_end(j) = info.group_start(j) + group_counts(j);
    }

    // Build sort indices
    info.sort_index.set_size(n);
    info.inv_sort.set_size(n);
    uword pos = 0;
    for (uword j = 0; j < n_groups; ++j) {
      const uvec &grp = indices.get_sorted_group(k, j);
      for (uword idx = 0; idx < grp.n_elem; ++idx) {
        if (grp(idx) < n) {
          info.sort_index(grp(idx)) = pos;
          info.inv_sort(pos) = grp(idx);
          pos++;
        }
      }
    }
  }

  return group_info;
}

// K=1 (closed form solution)
inline void demean_single_k1(vec &x_col, const vec &w,
                             const indices_info &indices) {
  const uword n = x_col.n_elem;
  const uword n_groups = indices.fe_sizes(0);
  bool use_weights = (w.n_elem > 1);

  vec group_sums(n_groups, fill::zeros);
  vec group_weights(n_groups, fill::zeros);

  // Single pass to compute all statistics
  for (uword j = 0; j < n_groups; ++j) {
    const uvec &grp = indices.get_sorted_group(0, j);
    if (grp.is_empty())
      continue;

    uvec valid_idx = grp(find(grp < n));
    if (valid_idx.is_empty())
      continue;

    if (use_weights) {
      vec grp_vals = x_col(valid_idx);
      vec grp_weights = w(valid_idx);
      group_sums(j) = dot(grp_vals, grp_weights);
      group_weights(j) = sum(grp_weights);
    } else {
      group_sums(j) = sum(x_col(valid_idx));
      group_weights(j) = valid_idx.n_elem;
    }
  }

  vec group_means = group_sums / group_weights;

  for (uword j = 0; j < n_groups; ++j) {
    const uvec &grp = indices.get_sorted_group(0, j);
    uvec valid_idx = grp(find(grp < n));
    if (!valid_idx.is_empty()) {
      x_col(valid_idx) -= group_means(j);
    }
  }
}

// K=2
inline bool demean_k2_accelerated(vec &x_col, const vec &w,
                                  const indices_info &indices, double tol,
                                  int max_iter) {
  const uword n = x_col.n_elem;
  const uword n1 = indices.fe_sizes(0);
  const uword n2 = indices.fe_sizes(1);
  bool use_weights = (w.n_elem > 1);

  // Pre-sort data for better cache access
  field<GroupInfo> group_info = precompute_group_info(indices, w, n);

  // Working vectors in sorted order
  vec x_sorted1 = x_col(group_info(0).inv_sort);
  vec x_sorted2 = x_col(group_info(1).inv_sort);
  vec w_sorted1 = use_weights ? w(group_info(0).inv_sort) : vec();
  vec w_sorted2 = use_weights ? w(group_info(1).inv_sort) : vec();

  // FE coefficients
  vec alpha(n1, fill::zeros);
  vec beta(n2, fill::zeros);

  // Acceleration variables
  vec X = alpha;
  vec GX(n1), GGX(n1);
  vec delta_GX(n1), delta2_X(n1);

  // Update using sorted data
  auto update_beta = [&]() {
    beta.zeros();
    for (uword j = 0; j < n2; ++j) {
      uword start = group_info(1).group_start(j);
      uword end = group_info(1).group_end(j);
      if (start >= end)
        continue;

      vec residuals(end - start);
      for (uword i = start; i < end; ++i) {
        uword orig_idx = group_info(1).inv_sort(i);
        uword group1 = group_info(0).sort_index(orig_idx);
        // Find which group in FE1 this belongs to
        uword g1 = 0;
        for (uword k = 1; k < n1; ++k) {
          if (group1 >= group_info(0).group_start(k))
            g1 = k;
          else
            break;
        }
        residuals(i - start) = x_sorted2(i) - alpha(g1);
      }

      if (use_weights) {
        vec grp_weights = w_sorted2.subvec(start, end - 1);
        beta(j) = dot(residuals, grp_weights) / group_info(1).group_weights(j);
      } else {
        beta(j) = mean(residuals);
      }
    }
  };

  auto update_alpha = [&]() {
    alpha.zeros();
    for (uword j = 0; j < n1; ++j) {
      uword start = group_info(0).group_start(j);
      uword end = group_info(0).group_end(j);
      if (start >= end)
        continue;

      vec residuals(end - start);
      for (uword i = start; i < end; ++i) {
        uword orig_idx = group_info(0).inv_sort(i);
        uword group2 = group_info(1).sort_index(orig_idx);
        // Find which group in FE2 this belongs to
        uword g2 = 0;
        for (uword k = 1; k < n2; ++k) {
          if (group2 >= group_info(1).group_start(k))
            g2 = k;
          else
            break;
        }
        residuals(i - start) = x_sorted1(i) - beta(g2);
      }

      if (use_weights) {
        vec grp_weights = w_sorted1.subvec(start, end - 1);
        alpha(j) = dot(residuals, grp_weights) / group_info(0).group_weights(j);
      } else {
        alpha(j) = mean(residuals);
      }
    }
  };

  // First iteration
  update_beta();
  update_alpha();
  GX = alpha;

  // Main iteration loop
  for (int iter = 0; iter < max_iter; ++iter) {
    // Check convergence
    if (norm(X - GX, "inf") / (1.0 + norm(GX, "inf")) < tol)
      break;

    X = GX;
    update_beta();
    update_alpha();
    GGX = alpha;

    // Irons-Tuck acceleration
    if (irons_tuck_update(n1, X, GX, GGX, delta_GX, delta2_X))
      break;

    alpha = X;
    update_beta();
    update_alpha();
    GX = alpha;
  }

  // Apply final centering
  for (uword i = 0; i < n; ++i) {
    // Find groups
    uword g1 = 0, g2 = 0;
    uword pos1 = group_info(0).sort_index(i);
    uword pos2 = group_info(1).sort_index(i);

    for (uword k = 1; k < n1; ++k) {
      if (pos1 >= group_info(0).group_start(k))
        g1 = k;
      else
        break;
    }
    for (uword k = 1; k < n2; ++k) {
      if (pos2 >= group_info(1).group_start(k))
        g2 = k;
      else
        break;
    }

    x_col(i) -= (alpha(g1) + beta(g2));
  }

  return true;
}

// K>=3
inline bool demean_k_general_accelerated(vec &x_col, const vec &w,
                                         const indices_info &indices,
                                         double tol, int max_iter) {
  const uword K = indices.fe_sizes.n_elem;
  const uword n = x_col.n_elem;
  bool use_weights = (w.n_elem > 1);

  // Pre-compute group information
  field<GroupInfo> group_info = precompute_group_info(indices, w, n);

  // FE coefficients
  field<vec> fe_coefs(K);
  for (uword k = 0; k < K; ++k) {
    fe_coefs(k).set_size(indices.fe_sizes(k));
    fe_coefs(k).zeros();
  }

  // Create observation-to-group mapping for fast lookup
  field<uvec> obs_to_group(K);
  for (uword k = 0; k < K; ++k) {
    obs_to_group(k).set_size(n);
    for (uword j = 0; j < indices.fe_sizes(k); ++j) {
      const uvec &grp = indices.get_sorted_group(k, j);
      for (uword idx = 0; idx < grp.n_elem; ++idx) {
        if (grp(idx) < n)
          obs_to_group(k)(grp(idx)) = j;
      }
    }
  }

  // Gauss-Seidel iterations with vectorized operations
  vec residual(n);
  for (int iter = 0; iter < max_iter; ++iter) {
    field<vec> fe_coefs_old = fe_coefs;

    // Update each FE in reverse order
    for (int k = K - 1; k >= 0; --k) {
      fe_coefs(k).zeros();

      // Compute residuals efficiently
      residual = x_col;
      for (uword h = 0; h < K; ++h) {
        if (h != static_cast<uword>(k)) {
          // Vectorized subtraction
          for (uword i = 0; i < n; ++i) {
            residual(i) -= fe_coefs(h)(obs_to_group(h)(i));
          }
        }
      }

      // Accumulate by groups using sorted data
      vec x_sorted = residual(group_info(k).inv_sort);
      vec w_sorted = use_weights ? w(group_info(k).inv_sort) : vec();

      for (uword j = 0; j < indices.fe_sizes(k); ++j) {
        uword start = group_info(k).group_start(j);
        uword end = group_info(k).group_end(j);
        if (start >= end)
          continue;

        if (use_weights) {
          vec grp_vals = x_sorted.subvec(start, end - 1);
          vec grp_weights = w_sorted.subvec(start, end - 1);
          fe_coefs(k)(j) =
              dot(grp_vals, grp_weights) / group_info(k).group_weights(j);
        } else {
          fe_coefs(k)(j) = mean(x_sorted.subvec(start, end - 1));
        }
      }
    }

    // Check convergence every 5 iterations
    if (iter % 5 == 0) {
      double max_change = 0;
      for (uword k = 0; k < K; ++k) {
        double change = norm(fe_coefs(k) - fe_coefs_old(k), "inf") /
                        (1.0 + norm(fe_coefs_old(k), "inf"));
        max_change = std::max(max_change, change);
      }
      if (max_change < tol)
        break;
    }
  }

  // Apply final centering
  for (uword i = 0; i < n; ++i) {
    double sum = 0.0;
    for (uword k = 0; k < K; ++k) {
      sum += fe_coefs(k)(obs_to_group(k)(i));
    }
    x_col(i) -= sum;
  }

  return true;
}

} // namespace accelerated_centering

inline void center_variables(mat &X, const vec &w,
                             const indices_info &indices_in, double tol,
                             size_t max_iter, size_t iter_interrupt,
                             bool use_weights) {
  indices_info indices = indices_in;
  indices.optimize_cache_access();

  const uword n = X.n_rows;
  const uword K = indices.fe_sizes.n_elem;
  const uword P = X.n_cols;

  if (K == 0)
    return;

  // Process columns in chunks for better cache usage
  const uword chunk_size = 4; // Process 4 columns at a time

  for (uword p_start = 0; p_start < P; p_start += chunk_size) {
    uword p_end = std::min(p_start + chunk_size, P);

    // Process chunk
    for (uword p = p_start; p < p_end; ++p) {
      vec x_col(X.colptr(p), n, false, false);

      if (K == 1) {
        accelerated_centering::demean_single_k1(x_col, w, indices);
      } else if (K == 2) {
        accelerated_centering::demean_k2_accelerated(x_col, w, indices, tol,
                                                     max_iter);
      } else {
        accelerated_centering::demean_k_general_accelerated(x_col, w, indices,
                                                            tol, max_iter);
      }
    }

    // Check interrupt less frequently
    if ((p_start / chunk_size) % 10 == 0 && p_start > 0) {
      check_user_interrupt();
    }
  }
}

inline void center_variables_batch(mat &X_work, mat &y, const vec &w,
                                   const mat &X_orig,
                                   const indices_info &indices_in, double tol,
                                   size_t max_iter, size_t iter_interrupt,
                                   bool use_weights) {
  indices_info indices = indices_in;
  indices.optimize_cache_access();

  const uword K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  // Copy from the original to avoid external matrix copying
  // Skip copy if working matrix and original are the same (in-place
  // modification)
  if (&X_work != &X_orig) {
    X_work = X_orig;
  }

  const uword n = X_work.n_rows;
  const uword P = X_work.n_cols;

  // For K=1, batch process everything
  if (K == 1) {
    // Process y
    vec y_col(y.colptr(0), n, false, false);
    accelerated_centering::demean_single_k1(y_col, w, indices);

    // Process X in chunks
    const uword chunk_size = 16;
    for (uword p = 0; p < P; p += chunk_size) {
      uword end = std::min(p + chunk_size, P);
      for (uword j = p; j < end; ++j) {
        vec x_col(X_work.colptr(j), n, false, false);
        accelerated_centering::demean_single_k1(x_col, w, indices);
      }
    }
    return;
  }

  // For K>=2, use appropriate algorithm
  vec y_col(y.colptr(0), n, false, false);

  if (K == 2) {
    accelerated_centering::demean_k2_accelerated(y_col, w, indices, tol,
                                                 max_iter);
    for (uword p = 0; p < P; ++p) {
      vec x_col(X_work.colptr(p), n, false, false);
      accelerated_centering::demean_k2_accelerated(x_col, w, indices, tol,
                                                   max_iter);
      if (p % iter_interrupt == 0 && p > 0) {
        check_user_interrupt();
      }
    }
  } else {
    accelerated_centering::demean_k_general_accelerated(y_col, w, indices, tol,
                                                        max_iter);
    for (uword p = 0; p < P; ++p) {
      vec x_col(X_work.colptr(p), n, false, false);
      accelerated_centering::demean_k_general_accelerated(x_col, w, indices,
                                                          tol, max_iter);
      if (p % iter_interrupt == 0 && p > 0) {
        check_user_interrupt();
      }
    }
  }
}

#endif // CAPYBARA_CENTER_H
