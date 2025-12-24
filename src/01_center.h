// Symmetric Kaczmarz centering with CG acceleration (reghdfe-style)

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {
// Configure OpenMP threads from configure-time macro
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
#ifndef CAPYBARA_DEFAULT_OMP_THREADS
#define CAPYBARA_DEFAULT_OMP_THREADS -1
#endif
inline void set_omp_threads_from_config() {
  static bool done = false;
  if (!done) {
#if defined(_OPENMP) && (CAPYBARA_DEFAULT_OMP_THREADS > 0)
    omp_set_num_threads(CAPYBARA_DEFAULT_OMP_THREADS);
#endif
    done = true;
  }
}
#endif

// Store group information
struct GroupInfo {
  const uvec *coords;
  double inv_weight;
  uword n_elem;
  bool is_singleton;
};

// Workspace for centering to avoid repeated allocations
struct CenteringWorkspace {
  vec x;   // Current solution
  vec x0;  // Previous iteration
  vec Gx;  // After one projection
  vec GGx; // After two projections
  vec Y;   // Grand accel history 1
  vec GY;  // Grand accel history 2
  vec GGY; // Grand accel history 3
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

// Safe division avoiding division by zero
inline double safe_divide(double num, double denom) {
  return (std::abs(denom) > 1e-14) ? (num / denom) : 0.0;
}

// Project onto one FE group: subtract weighted mean
// Returns the mean that was subtracted (the "projection component")
inline double project_one_group(double *v, const double *w,
                                const GroupInfo &info) {
  if (info.is_singleton)
    return 0.0;

  double weighted_sum = 0.0;
  const uword *coord_ptr = info.coords->memptr();
  const uword n = info.n_elem;

// Vectorized accumulation of w * v using SIMD-friendly canonical loop
#if defined(_OPENMP)
#pragma omp simd reduction(+ : weighted_sum)
#endif
  for (uword i = 0; i < n; ++i) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
  }

  double mean = weighted_sum * info.inv_weight;

// Subtract mean from all elements in group (SIMD-friendly canonical loop)
#if defined(_OPENMP)
#pragma omp simd
#endif
  for (uword i = 0; i < n; ++i) {
    v[coord_ptr[i]] -= mean;
  }

  return mean;
}

// Symmetric Kaczmarz projection: applies P(y) in-place
inline void apply_projection(double *y, const double *w,
                             const field<field<GroupInfo>> &all_group_info,
                             uword N) {
  const uword K = all_group_info.n_elem;

  // Forward pass: FE1, FE2, ..., FEK
  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uword l = 0; l < fe_info.n_elem; ++l) {
      project_one_group(y, w, fe_info(l));
    }
  }

  // Backward pass: FE(K-1), ..., FE1 (symmetric Kaczmarz)
  for (uword k = K; k-- > 1;) {
    const field<GroupInfo> &fe_info = all_group_info(k - 1);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uword l = 0; l < fe_info.n_elem; ++l) {
      project_one_group(y, w, fe_info(l));
    }
  }
}

// Single FE projection (simpler case)
inline void apply_projection_single_fe(double *y, const double *w,
                                       const field<GroupInfo> &fe_info,
                                       uword N) {
// Forward pass
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(y, w, fe_info(l));
  }

// Backward pass (symmetric Kaczmarz)
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(y, w, fe_info(l));
  }
}

// Irons-Tuck acceleration: extrapolate using two successive projections
// Returns true if converged
inline bool irons_tuck_acceleration(double *x, const double *Gx,
                                    const double *GGx, uword N, double tol) {
  double vprod = 0.0, ssq = 0.0;

  // Compute delta_G = GGx - Gx and delta2 = delta_G - (Gx - x)
  for (uword i = 0; i < N; ++i) {
    double delta_G = GGx[i] - Gx[i];
    double delta2 = delta_G - (Gx[i] - x[i]);
    vprod += delta_G * delta2;
    ssq += delta2 * delta2;
  }

  if (ssq < tol * tol) {
    return true; // Converged
  }

  double coef = safe_divide(vprod, ssq);

  // Apply acceleration if coefficient is in valid range
  if (coef > 0.0 && coef < 2.0) {
    for (uword i = 0; i < N; ++i) {
      x[i] = GGx[i] - coef * (GGx[i] - Gx[i]);
    }
  } else {
    // Fall back to GGx
    std::memcpy(x, GGx, N * sizeof(double));
  }

  return false;
}

// Grand acceleration (fixest-style): uses 3-iteration history
// Returns true if converged
inline bool grand_acceleration(double *x, double *Y, double *GY, double *GGY,
                               uword N, int &grand_acc_state, double tol) {
  if (grand_acc_state == 0) {
    std::memcpy(Y, x, N * sizeof(double));
  } else if (grand_acc_state == 1) {
    std::memcpy(GY, x, N * sizeof(double));
  } else {
    std::memcpy(GGY, x, N * sizeof(double));

    // Apply Irons-Tuck-style extrapolation with 3-iteration history
    double vprod = 0.0, ssq = 0.0;

    for (uword i = 0; i < N; ++i) {
      double delta_G = GGY[i] - GY[i];
      double delta2 = delta_G - (GY[i] - Y[i]);
      vprod += delta_G * delta2;
      ssq += delta2 * delta2;
    }

    if (ssq < tol * tol) {
      grand_acc_state = -1; // Reset and signal convergence
      return true;
    }

    double coef = safe_divide(vprod, ssq);

    if (coef > 0.0 && coef < 2.0) {
      for (uword i = 0; i < N; ++i) {
        x[i] = GGY[i] - coef * (GGY[i] - GY[i]);
      }
    } else {
      std::memcpy(x, GGY, N * sizeof(double));
    }

    grand_acc_state = -1; // Reset for next cycle
  }

  ++grand_acc_state;
  return false;
}

// Center one column using Irons-Tuck + Grand acceleration
inline void
center_one_column_accel(vec &y, const vec &w, const double *w_ptr,
                        const field<field<GroupInfo>> &all_group_info,
                        double tol, uword max_iter, uword iter_interrupt,
                        CenteringWorkspace &ws) {

  const uword K = all_group_info.n_elem;
  const uword N = y.n_elem;

  double *x_ptr = ws.x.memptr();
  double *x0_ptr = ws.x0.memptr();
  double *Gx_ptr = ws.Gx.memptr();
  double *GGx_ptr = ws.GGx.memptr();
  double *Y_ptr = ws.Y.memptr();
  double *GY_ptr = ws.GY.memptr();
  double *GGY_ptr = ws.GGY.memptr();

  // Copy input to workspace
  std::memcpy(x_ptr, y.memptr(), N * sizeof(double));

  // Acceleration parameters (more aggressive than fixest for faster
  // convergence)
  const uword irons_tuck_interval = 2;  // Every 2 iterations (was 3)
  const uword grand_accel_interval = 3; // Every 3 iterations (was 5)
  const uword accel_start = 1;          // Start immediately (was 2)

  int grand_acc_state = 0;
  uword iint = iter_interrupt;
  double ssr_old = std::numeric_limits<double>::infinity();
  uword stall_count = 0; // Track iterations without progress

  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Save previous iteration
    std::memcpy(x0_ptr, x_ptr, N * sizeof(double));

    // Apply projection
    if (K == 1) {
      apply_projection_single_fe(x_ptr, w_ptr, all_group_info[0], N);
    } else {
      apply_projection(x_ptr, w_ptr, all_group_info, N);
    }

    // Grand acceleration (every 3 iterations after iteration 1)
    if (iter >= accel_start && iter % grand_accel_interval == 0) {
      if (grand_acceleration(x_ptr, Y_ptr, GY_ptr, GGY_ptr, N, grand_acc_state,
                             tol)) {
        break; // Converged
      }
      // Re-project after acceleration
      if (K == 1) {
        apply_projection_single_fe(x_ptr, w_ptr, all_group_info[0], N);
      } else {
        apply_projection(x_ptr, w_ptr, all_group_info, N);
      }
    }

    // Irons-Tuck acceleration (every 2 iterations after iteration 1, not
    // overlapping with grand)
    if (iter >= accel_start && iter % irons_tuck_interval == 0 &&
        iter % grand_accel_interval != 0) {
      // Gx = P(x)
      std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));
      if (K == 1) {
        apply_projection_single_fe(Gx_ptr, w_ptr, all_group_info[0], N);
      } else {
        apply_projection(Gx_ptr, w_ptr, all_group_info, N);
      }

      // GGx = P(Gx)
      std::memcpy(GGx_ptr, Gx_ptr, N * sizeof(double));
      if (K == 1) {
        apply_projection_single_fe(GGx_ptr, w_ptr, all_group_info[0], N);
      } else {
        apply_projection(GGx_ptr, w_ptr, all_group_info, N);
      }

      if (irons_tuck_acceleration(x_ptr, Gx_ptr, GGx_ptr, N, tol)) {
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

      // Early exit if progress stalls
      if (rel_change < tol * 10.0) {
        stall_count++;
        if (stall_count > 3)
          break; // Stop if stuck for 3 iterations
      } else {
        stall_count = 0; // Reset if making progress
      }
    }
    ssr_old = ssr;
  }

  // Copy result back
  y = ws.x;
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const uword &max_iter, const uword &iter_interrupt,
    const field<field<GroupInfo>> *precomputed_group_info = nullptr,
    CenteringWorkspace *workspace = nullptr) {
// Ensure OpenMP uses the configure-detected thread count
#ifdef _OPENMP
  set_omp_threads_from_config();
#endif
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

  for (uword col = 0; col < P; ++col) {
    vec y_col = V.col(col);
    center_one_column_accel(y_col, w, w_ptr, all_group_info, tol, max_iter,
                            iter_interrupt, *ws);
    V.col(col) = y_col;
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
