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
  vec x;        // Current solution
  vec x0;       // Previous iteration
  vec Gx;       // After one projection
  vec GGx;      // After two projections
  vec Y;        // Grand accel history 1
  vec GY;       // Grand accel history 2
  vec GGY;      // Grand accel history 3
  vec delta_G;  // Workspace for acceleration
  vec delta2;   // Workspace for acceleration
  uword cached_n;
  bool is_initialized;

  CenteringWorkspace() : cached_n(0), is_initialized(false) {}

  void ensure_size(uword n) {
    if (!is_initialized || n > cached_n) {
      x.set_size(n);
      x0.set_size(n);
      Gx.set_size(n);
      GGx.set_size(n);
      Y.set_size(n);
      GY.set_size(n);
      GGY.set_size(n);
      delta_G.set_size(n);
      delta2.set_size(n);
      cached_n = n;
      is_initialized = true;
    }
  }

  void clear() {
    x.reset();
    x0.reset();
    Gx.reset();
    GGx.reset();
    Y.reset();
    GY.reset();
    GGY.reset();
    delta_G.reset();
    delta2.reset();
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
// Direct loop to avoid temporary from square()
inline double weighted_quadsum(const vec &x, const vec &w) {
  const uword n = x.n_elem;
  const double *x_ptr = x.memptr();
  const double *w_ptr = w.memptr();
  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    sum += w_ptr[i] * x_ptr[i] * x_ptr[i];
  }
  return sum;
}

// Safe division avoiding division by zero
inline double safe_divide(double num, double denom) {
  return (std::abs(denom) > 1e-14) ? (num / denom) : 0.0;
}

// Project onto one FE group: subtract weighted mean
// Returns the mean that was subtracted (the "projection component")
inline double project_one_group(vec &v, const vec &w,
                                const GroupInfo &info) {
  if (info.is_singleton)
    return 0.0;

  const uvec &coords = *info.coords;
  const uword n = info.n_elem;
  const uword *idx = coords.memptr();
  double *v_ptr = v.memptr();
  const double *w_ptr = w.memptr();
  
  // Compute weighted sum with direct pointer access (no temporaries)
  double weighted_sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    const uword j = idx[i];
    weighted_sum += w_ptr[j] * v_ptr[j];
  }
  
  double mean = weighted_sum * info.inv_weight;

  // Subtract mean from all elements in group
  for (uword i = 0; i < n; ++i) {
    v_ptr[idx[i]] -= mean;
  }

  return mean;
}

// Symmetric Kaczmarz projection: applies P(y) in-place
inline void apply_projection(vec &y, const vec &w,
                             const field<field<GroupInfo>> &all_group_info) {
  const uword K = all_group_info.n_elem;

  // Forward pass: FE1, FE2, ..., FEK
  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
    for (uword l = 0; l < fe_info.n_elem; ++l) {
      project_one_group(y, w, fe_info(l));
    }
  }

  // Backward pass: FE(K-1), ..., FE1 (symmetric Kaczmarz)
  for (uword k = K; k-- > 1;) {
    const field<GroupInfo> &fe_info = all_group_info(k - 1);
    for (uword l = 0; l < fe_info.n_elem; ++l) {
      project_one_group(y, w, fe_info(l));
    }
  }
}

// Single FE projection (simpler case)
inline void apply_projection_single_fe(vec &y, const vec &w,
                                       const field<GroupInfo> &fe_info) {
  // Forward pass
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(y, w, fe_info(l));
  }

  // Backward pass (symmetric Kaczmarz)
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(y, w, fe_info(l));
  }
}

// Irons-Tuck acceleration: extrapolate using two successive projections
// Returns true if converged
// Uses workspace vectors to avoid allocations
inline bool irons_tuck_acceleration(vec &x, const vec &Gx, const vec &GGx,
                                    vec &delta_G_ws, vec &delta2_ws,
                                    double tol) {
  const uword n = x.n_elem;
  double *dG = delta_G_ws.memptr();
  double *d2 = delta2_ws.memptr();
  const double *x_ptr = x.memptr();
  const double *Gx_ptr = Gx.memptr();
  const double *GGx_ptr = GGx.memptr();
  
  // Compute delta_G = GGx - Gx and delta2 = delta_G - (Gx - x) in one pass
  double ssq = 0.0;
  double vprod = 0.0;
  for (uword i = 0; i < n; ++i) {
    dG[i] = GGx_ptr[i] - Gx_ptr[i];
    d2[i] = dG[i] - (Gx_ptr[i] - x_ptr[i]);
    ssq += d2[i] * d2[i];
    vprod += dG[i] * d2[i];
  }
  
  if (ssq < tol * tol) {
    return true; // Converged
  }

  double coef = safe_divide(vprod, ssq);

  // Apply acceleration if coefficient is in valid range
  double *x_out = x.memptr();
  if (coef > 0.0 && coef < 2.0) {
    for (uword i = 0; i < n; ++i) {
      x_out[i] = GGx_ptr[i] - coef * dG[i];
    }
  } else {
    std::memcpy(x_out, GGx_ptr, n * sizeof(double));
  }

  return false;
}

// Grand acceleration (fixest-style): uses 3-iteration history
// Returns true if converged
// Uses workspace vectors to avoid allocations
inline bool grand_acceleration(vec &x, vec &Y, vec &GY, vec &GGY,
                               vec &delta_G_ws, vec &delta2_ws,
                               int &grand_acc_state, double tol) {
  const uword n = x.n_elem;
  
  if (grand_acc_state == 0) {
    std::memcpy(Y.memptr(), x.memptr(), n * sizeof(double));
  } else if (grand_acc_state == 1) {
    std::memcpy(GY.memptr(), x.memptr(), n * sizeof(double));
  } else {
    std::memcpy(GGY.memptr(), x.memptr(), n * sizeof(double));

    double *dG = delta_G_ws.memptr();
    double *d2 = delta2_ws.memptr();
    const double *Y_ptr = Y.memptr();
    const double *GY_ptr = GY.memptr();
    const double *GGY_ptr = GGY.memptr();
    
    // Compute delta_G = GGY - GY and delta2 = delta_G - (GY - Y) in one pass
    double ssq = 0.0;
    double vprod = 0.0;
    for (uword i = 0; i < n; ++i) {
      dG[i] = GGY_ptr[i] - GY_ptr[i];
      d2[i] = dG[i] - (GY_ptr[i] - Y_ptr[i]);
      ssq += d2[i] * d2[i];
      vprod += dG[i] * d2[i];
    }

    if (ssq < tol * tol) {
      grand_acc_state = -1; // Reset and signal convergence
      return true;
    }

    double coef = safe_divide(vprod, ssq);

    double *x_out = x.memptr();
    if (coef > 0.0 && coef < 2.0) {
      for (uword i = 0; i < n; ++i) {
        x_out[i] = GGY_ptr[i] - coef * dG[i];
      }
    } else {
      std::memcpy(x_out, GGY_ptr, n * sizeof(double));
    }

    grand_acc_state = -1; // Reset for next cycle
  }

  ++grand_acc_state;
  return false;
}

// Center one column using Irons-Tuck + Grand acceleration
inline void
center_one_column_accel(vec &y, const vec &w,
                        const field<field<GroupInfo>> &all_group_info,
                        double tol, uword max_iter, uword iter_interrupt,
                        CenteringWorkspace &ws) {

  const uword K = all_group_info.n_elem;
  const uword N = y.n_elem;

  // Use workspace vectors directly - avoid Armadillo copy
  std::memcpy(ws.x.memptr(), y.memptr(), N * sizeof(double));

  // Acceleration parameters
  const uword irons_tuck_interval = 2;
  const uword grand_accel_interval = 3;
  const uword accel_start = 1;

  int grand_acc_state = 0;
  uword iint = iter_interrupt;
  double ssr_old = std::numeric_limits<double>::infinity();
  uword stall_count = 0;

  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Save previous iteration
    std::memcpy(ws.x0.memptr(), ws.x.memptr(), N * sizeof(double));

    // Apply projection
    if (K == 1) {
      apply_projection_single_fe(ws.x, w, all_group_info[0]);
    } else {
      apply_projection(ws.x, w, all_group_info);
    }

    // Grand acceleration (every 3 iterations after iteration 1)
    if (iter >= accel_start && iter % grand_accel_interval == 0) {
      if (grand_acceleration(ws.x, ws.Y, ws.GY, ws.GGY, ws.delta_G, ws.delta2,
                             grand_acc_state, tol)) {
        break; // Converged
      }
      // Re-project after acceleration
      if (K == 1) {
        apply_projection_single_fe(ws.x, w, all_group_info[0]);
      } else {
        apply_projection(ws.x, w, all_group_info);
      }
    }

    // Irons-Tuck acceleration (every 2 iterations, not overlapping with grand)
    if (iter >= accel_start && iter % irons_tuck_interval == 0 &&
        iter % grand_accel_interval != 0) {
      // Gx = P(x)
      std::memcpy(ws.Gx.memptr(), ws.x.memptr(), N * sizeof(double));
      if (K == 1) {
        apply_projection_single_fe(ws.Gx, w, all_group_info[0]);
      } else {
        apply_projection(ws.Gx, w, all_group_info);
      }

      // GGx = P(Gx)
      std::memcpy(ws.GGx.memptr(), ws.Gx.memptr(), N * sizeof(double));
      if (K == 1) {
        apply_projection_single_fe(ws.GGx, w, all_group_info[0]);
      } else {
        apply_projection(ws.GGx, w, all_group_info);
      }

      if (irons_tuck_acceleration(ws.x, ws.Gx, ws.GGx, ws.delta_G, ws.delta2,
                                  tol)) {
        break; // Converged
      }
    }

    // Check convergence based on weighted SSR
    double ssr = weighted_quadsum(ws.x, w);

    if (ssr < tol * tol)
      break;

    if (ssr_old < std::numeric_limits<double>::infinity()) {
      double rel_change = std::abs(ssr - ssr_old) / (1.0 + ssr_old);
      if (rel_change < tol)
        break;

      if (rel_change < tol * 10.0) {
        stall_count++;
        if (stall_count > 3)
          break;
      } else {
        stall_count = 0;
      }
    }
    ssr_old = ssr;
  }

  // Copy result back - avoid Armadillo copy
  std::memcpy(y.memptr(), ws.x.memptr(), N * sizeof(double));
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const uword &max_iter, const uword &iter_interrupt,
    const field<field<GroupInfo>> *precomputed_group_info = nullptr,
    CenteringWorkspace *workspace = nullptr) {
#ifdef _OPENMP
  set_omp_threads_from_config();
#endif
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem)
    return;

  const uword K = group_indices.n_elem;
  if (K == 0)
    return;

  const uword N = V.n_rows, P = V.n_cols;

  // Reuse precomputed group metadata when available
  const field<field<GroupInfo>> *group_info_ptr = precomputed_group_info;
  field<field<GroupInfo>> local_group_info;
  if (!group_info_ptr) {
    local_group_info = precompute_group_info(group_indices, w);
    group_info_ptr = &local_group_info;
  }
  const field<field<GroupInfo>> &all_group_info = *group_info_ptr;

  // Parallelize at column level - each column is independent
  // Each thread gets its own workspace to avoid contention
#if defined(_OPENMP)
#pragma omp parallel
  {
    CenteringWorkspace thread_ws;
    thread_ws.ensure_size(N);

#pragma omp for schedule(static)
    for (uword col = 0; col < P; ++col) {
      vec y_col = V.col(col);
      center_one_column_accel(y_col, w, all_group_info, tol, max_iter,
                              iter_interrupt, thread_ws);
      V.col(col) = y_col;
    }
  }
#else
  CenteringWorkspace local_workspace;
  CenteringWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N);

  for (uword col = 0; col < P; ++col) {
    vec y_col = V.col(col);
    center_one_column_accel(y_col, w, all_group_info, tol, max_iter,
                            iter_interrupt, *ws);
    V.col(col) = y_col;
  }
#endif
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
