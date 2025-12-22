// Symmetric Kaczmarz centering with CG acceleration (reghdfe-style)

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Store group information
struct GroupInfo {
  const uvec *coords;
  double inv_weight;
  uword n_elem;
  bool is_singleton;
};

// Workspace for centering to avoid repeated allocations
struct CenteringWorkspace {
  vec x;  // Current solution (y in reghdfe)
  vec r;  // Residual
  vec u;  // CG direction
  vec v;  // Projected direction
  uword cached_n;
  bool is_initialized;

  CenteringWorkspace() : cached_n(0), is_initialized(false) {}

  void ensure_size(uword n) {
    if (!is_initialized || n > cached_n) {
      x.set_size(n);
      r.set_size(n);
      u.set_size(n);
      v.set_size(n);
      cached_n = n;
      is_initialized = true;
    }
  }

  void clear() {
    x.reset();
    r.reset();
    u.reset();
    v.reset();
    cached_n = 0;
    is_initialized = false;
  }
};

// Precompute group information for all variables
inline field<field<GroupInfo>>
precompute_group_info(const field<field<uvec>> &group_indices, const vec &w) {
  const uword K = group_indices.n_elem;
  const double *w_ptr = w.memptr();

  field<field<GroupInfo>> group_info(K);

  for (uword k = 0; k < K; ++k) {
    const field<uvec> &fe_groups = group_indices(k);
    group_info(k).set_size(fe_groups.n_elem);

    for (uword l = 0; l < fe_groups.n_elem; ++l) {
      const uvec &coords = fe_groups(l);
      GroupInfo info;
      info.coords = &coords;
      info.n_elem = coords.n_elem;
      info.is_singleton = (coords.n_elem <= 1);

      if (!info.is_singleton) {
        double sum_w = 0.0;
        const uword *coord_ptr = coords.memptr();
        for (uword i = 0; i < coords.n_elem; ++i) {
          sum_w += w_ptr[coord_ptr[i]];
        }
        info.inv_weight = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
      } else {
        info.inv_weight = 0.0;
      }

      group_info(k)(l) = info;
    }
  }

  return group_info;
}

// Weighted sum of squared values: sum(w * x^2)
// Uses Armadillo's vectorized operations (SIMD-optimized)
inline double weighted_quadsum(const vec &x, const vec &w) {
  return dot(w, square(x));
}

// Weighted cross product: sum(w * x * y)
inline double weighted_quadcross(const vec &x, const vec &y, const vec &w) {
  return dot(w, x % y);
}

// Safe division avoiding division by zero
inline double safe_divide(double num, double denom) {
  return (std::abs(denom) > 1e-14) ? (num / denom) : 0.0;
}

// Project onto one FE group: subtract weighted mean
// Returns the mean that was subtracted (the "projection component")
inline double project_one_group(double *v, const double *w, 
                                 const GroupInfo &info) {
  if (info.is_singleton) return 0.0;

  double weighted_sum = 0.0;
  const uword *coord_ptr = info.coords->memptr();
  const uword n = info.n_elem;

  uword i = 0;
  for (; i + 4 <= n; i += 4) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
    weighted_sum += w[coord_ptr[i + 1]] * v[coord_ptr[i + 1]];
    weighted_sum += w[coord_ptr[i + 2]] * v[coord_ptr[i + 2]];
    weighted_sum += w[coord_ptr[i + 3]] * v[coord_ptr[i + 3]];
  }
  for (; i < n; ++i) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
  }

  double mean = weighted_sum * info.inv_weight;

  // Subtract mean from all elements in group
  i = 0;
  for (; i + 4 <= n; i += 4) {
    v[coord_ptr[i]] -= mean;
    v[coord_ptr[i + 1]] -= mean;
    v[coord_ptr[i + 2]] -= mean;
    v[coord_ptr[i + 3]] -= mean;
  }
  for (; i < n; ++i) {
    v[coord_ptr[i]] -= mean;
  }

  return mean;
}

// Symmetric Kaczmarz transform: computes y - P(y) where P is the projection
// If get_proj=true, returns P(y) instead (what was projected out)
// This mimics reghdfe's transform_sym_kaczmarz()
inline void transform_sym_kaczmarz(double *y, double *ans, const double *w,
                                    const field<field<GroupInfo>> &all_group_info,
                                    uword N, bool get_proj = false) {
  const uword K = all_group_info.n_elem;
  
  // Copy y to ans, then work in-place on ans
  std::memcpy(ans, y, N * sizeof(double));
  
  // Forward pass: FE1, FE2, ..., FEK
  for (uword k = 0; k < K; ++k) {
    for (uword l = 0; l < all_group_info(k).n_elem; ++l) {
      project_one_group(ans, w, all_group_info(k)(l));
    }
  }
  
  // Backward pass: FE(K-1), ..., FE1
  for (uword k = K; k-- > 1;) {
    for (uword l = all_group_info(k-1).n_elem; l-- > 0;) {
      project_one_group(ans, w, all_group_info(k-1)(l));
    }
  }
  
  // ans now contains y - P(y) (the residual after projection)
  // If get_proj, we want P(y) = y - ans
  if (get_proj) {
    for (uword i = 0; i < N; ++i) {
      ans[i] = y[i] - ans[i];
    }
  }
}

// Single FE transform (simpler case)
inline void transform_single_fe(double *y, double *ans, const double *w,
                                 const field<GroupInfo> &fe_info,
                                 uword N, bool get_proj = false) {
  std::memcpy(ans, y, N * sizeof(double));
  
  // Forward pass
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(ans, w, fe_info(l));
  }
  
  // Backward pass
  for (uword l = fe_info.n_elem; l-- > 0;) {
    project_one_group(ans, w, fe_info(l));
  }
  
  if (get_proj) {
    for (uword i = 0; i < N; ++i) {
      ans[i] = y[i] - ans[i];
    }
  }
}

// CG acceleration for centering (following reghdfe's accelerate_cg)
// Returns the centered vector (y with FE projected out)
inline void center_one_column_cg(
    vec &y, const vec &w, const double *w_ptr,
    const field<field<GroupInfo>> &all_group_info,
    double tol, uword max_iter, uword iter_interrupt,
    CenteringWorkspace &ws) {
  
  const uword K = all_group_info.n_elem;
  const uword N = y.n_elem;
  double *y_ptr = y.memptr();
  double *r_ptr = ws.r.memptr();
  double *u_ptr = ws.u.memptr();
  double *v_ptr = ws.v.memptr();
  
  // improvement_potential = ||y||^2_w
  double improvement_potential = weighted_quadsum(y, w);
  
  // Initial transform: r = P(y) where P is the projection onto FE space
  if (K == 1) {
    transform_single_fe(y_ptr, r_ptr, w_ptr, all_group_info[0], N, true);
  } else {
    transform_sym_kaczmarz(y_ptr, r_ptr, w_ptr, all_group_info, N, true);
  }
  
  // ssr = ||r||^2_w
  double ssr = weighted_quadsum(ws.r, w);
  
  // Initialize u = r (BLAS copy)
  ws.u = ws.r;
  
  double recent_ssr = ssr;
  uword iint = iter_interrupt;
  
  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }
    
    // v = P(u) (projection of u)
    if (K == 1) {
      transform_single_fe(u_ptr, v_ptr, w_ptr, all_group_info[0], N, true);
    } else {
      transform_sym_kaczmarz(u_ptr, v_ptr, w_ptr, all_group_info, N, true);
    }
    
    // alpha = ssr / <u, v>_w
    double uv = weighted_quadcross(ws.u, ws.v, w);
    double alpha = safe_divide(ssr, uv);
    
    // Track convergence
    recent_ssr = alpha * ssr;
    improvement_potential -= recent_ssr;
    
    // y = y - alpha * u (BLAS axpy)
    y -= alpha * ws.u;
    
    // r = r - alpha * v (BLAS axpy)
    ws.r -= alpha * ws.v;
    
    double ssr_old = ssr;
    ssr = weighted_quadsum(ws.r, w);
    
    // Convergence check (Hestenes-Stiefel criterion from reghdfe)
    if (improvement_potential > 1e-14) {
      double ratio = recent_ssr / improvement_potential;
      if (ratio < tol * tol) {
        break;
      }
    }
    
    // Also check if ssr is essentially zero
    if (ssr < 1e-30) {
      break;
    }
    
    // beta = ssr_new / ssr_old (Fletcher-Reeves)
    double beta = safe_divide(ssr, ssr_old);
    
    // u = r + beta * u (BLAS operations)
    ws.u = ws.r + beta * ws.u;
  }
}

// Simple iteration without acceleration (for single FE or fallback)
inline void center_one_column_simple(
    vec &y, const vec &w, const double *w_ptr,
    const field<field<GroupInfo>> &all_group_info,
    double tol, uword max_iter, uword iter_interrupt,
    CenteringWorkspace &ws) {
  
  const uword K = all_group_info.n_elem;
  const uword N = y.n_elem;
  double *y_ptr = y.memptr();
  double *r_ptr = ws.r.memptr();
  
  double ssr_old = std::numeric_limits<double>::infinity();
  uword iint = iter_interrupt;
  
  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }
    
    // Apply symmetric Kaczmarz transform: r = y - P(y)
    if (K == 1) {
      transform_single_fe(y_ptr, r_ptr, w_ptr, all_group_info[0], N, false);
    } else {
      transform_sym_kaczmarz(y_ptr, r_ptr, w_ptr, all_group_info, N, false);
    }
    
    // Check convergence
    double ssr = weighted_quadsum(ws.r, w);
    
    // Copy result back to y (BLAS copy)
    y = ws.r;
    
    // Convergence based on relative change
    if (ssr < tol * tol) break;
    if (ssr_old < std::numeric_limits<double>::infinity()) {
      double rel_change = std::abs(ssr - ssr_old) / (1.0 + ssr_old);
      if (rel_change < tol) break;
    }
    ssr_old = ssr;
  }
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const uword &max_iter, const uword &iter_interrupt,
    const field<field<GroupInfo>> *precomputed_group_info = nullptr,
    CenteringWorkspace *workspace = nullptr) {
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem)
    return;

  const uword K = group_indices.n_elem;
  if (K == 0)
    return;

  const uword N = V.n_rows, P = V.n_cols;
  const double *w_ptr = w.memptr();

  // Reuse precomputed group metadata when available
  const field<field<GroupInfo>> *group_info_ptr = precomputed_group_info;
  field<field<GroupInfo>> local_group_info;
  if (!group_info_ptr) {
    local_group_info = precompute_group_info(group_indices, w);
    group_info_ptr = &local_group_info;
  }
  const field<field<GroupInfo>> &all_group_info = *group_info_ptr;

  CenteringWorkspace local_workspace;
  CenteringWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N);

  // Use CG for K >= 2, simple iteration for K == 1
  const bool use_cg = (K >= 2);

  for (uword col = 0; col < P; ++col) {
    // Use subview to avoid copy - work directly on column
    ws->x = V.col(col);
    
    if (use_cg) {
      center_one_column_cg(ws->x, w, w_ptr, all_group_info,
                           tol, max_iter, iter_interrupt, *ws);
    } else {
      center_one_column_simple(ws->x, w, w_ptr, all_group_info,
                               tol, max_iter, iter_interrupt, *ws);
    }
    
    // Copy result back
    V.col(col) = ws->x;
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
