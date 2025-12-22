// Symmetric Kaczmarz with Irons-Tuck acceleration

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

#ifdef _OPENMP
#include <omp.h>
#endif

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
  vec x, x0, Gx, GGx;
  vec Y, GY, GGY;
  vec delta_GX, delta2_X;
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
      delta_GX.set_size(n);
      delta2_X.set_size(n);
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
    delta_GX.reset();
    delta2_X.reset();
    cached_n = 0;
    is_initialized = false;
  }
};

// Cache-friendly block size calculation
inline uword get_block_size(uword n, uword p) {
  const uword L1_CACHE_SIZE = 32 * 1024;
  const uword ELEMENT_SIZE = sizeof(double);
  uword max_block_elements = L1_CACHE_SIZE / ELEMENT_SIZE;
  uword max_block_size = max_block_elements / (p + 1);
  const uword MIN_BLOCK_SIZE = 64;
  const uword MAX_BLOCK_SIZE = 1024;
  uword block_size =
      std::max(MIN_BLOCK_SIZE, std::min(max_block_size, MAX_BLOCK_SIZE));
  return std::min(block_size, n);
}

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

// Convergence checks
inline bool stopping_criterion(double ssr_old, double ssr_new, double tol) {
  if (ssr_old == 0.0)
    return false;
  return std::abs(ssr_new - ssr_old) / (1.0 + std::abs(ssr_old)) < tol;
}

inline bool continue_criterion(double x_old, double x_new, double tol) {
  return std::abs(x_new - x_old) / (1.0 + std::abs(x_old)) > tol;
}

// Grand acceleration (based on fixest)
inline bool grand_acceleration(double *x, double *Y, double *GY, double *GGY,
                               vec &delta_GX, vec &delta2_X, uword n,
                               size_t &grand_acc, double tol) {
  bool converged = false;

  if (grand_acc == 0) {
    std::memcpy(Y, x, n * sizeof(double));
  } else if (grand_acc == 1) {
    std::memcpy(GY, x, n * sizeof(double));
  } else {
    std::memcpy(GGY, x, n * sizeof(double));

    // Apply Irons-Tuck with the saved vectors
    double vprod = 0.0, ssq = 0.0;
    double *delta_GX_ptr = delta_GX.memptr();
    double *delta2_X_ptr = delta2_X.memptr();

    // Unrolled loop for better vectorization
    uword i = 0;
    for (; i + 4 <= n; i += 4) {
      double dG0 = GGY[i] - GY[i];
      double dG1 = GGY[i + 1] - GY[i + 1];
      double dG2 = GGY[i + 2] - GY[i + 2];
      double dG3 = GGY[i + 3] - GY[i + 3];

      double d2_0 = dG0 - GY[i] + Y[i];
      double d2_1 = dG1 - GY[i + 1] + Y[i + 1];
      double d2_2 = dG2 - GY[i + 2] + Y[i + 2];
      double d2_3 = dG3 - GY[i + 3] + Y[i + 3];

      delta_GX_ptr[i] = dG0;
      delta_GX_ptr[i + 1] = dG1;
      delta_GX_ptr[i + 2] = dG2;
      delta_GX_ptr[i + 3] = dG3;

      delta2_X_ptr[i] = d2_0;
      delta2_X_ptr[i + 1] = d2_1;
      delta2_X_ptr[i + 2] = d2_2;
      delta2_X_ptr[i + 3] = d2_3;

      vprod += dG0 * d2_0 + dG1 * d2_1 + dG2 * d2_2 + dG3 * d2_3;
      ssq += d2_0 * d2_0 + d2_1 * d2_1 + d2_2 * d2_2 + d2_3 * d2_3;
    }
    for (; i < n; ++i) {
      double delta_G = GGY[i] - GY[i];
      double delta2 = delta_G - GY[i] + Y[i];
      delta_GX_ptr[i] = delta_G;
      delta2_X_ptr[i] = delta2;
      vprod += delta_G * delta2;
      ssq += delta2 * delta2;
    }

    if (ssq < tol) {
      converged = true;
    } else {
      double coef = vprod / ssq;
      if (coef > 0.0 && coef < 2.0) {
        i = 0;
        for (; i + 4 <= n; i += 4) {
          x[i] = GGY[i] - coef * delta_GX_ptr[i];
          x[i + 1] = GGY[i + 1] - coef * delta_GX_ptr[i + 1];
          x[i + 2] = GGY[i + 2] - coef * delta_GX_ptr[i + 2];
          x[i + 3] = GGY[i + 3] - coef * delta_GX_ptr[i + 3];
        }
        for (; i < n; ++i) {
          x[i] = GGY[i] - coef * delta_GX_ptr[i];
        }
      } else {
        std::memcpy(x, GGY, n * sizeof(double));
      }
    }

    grand_acc = -1; // Reset for next cycle
  }

  ++grand_acc;
  return converged;
}

// Adaptive SSR checking
inline bool adaptive_ssr_check(double *x, const double *w, uword n,
                               double &ssr_old, double inv_sw, double tol) {
  double ssr = 0.0;

  // Unrolled loop for better vectorization
  uword i = 0;
  for (; i + 4 <= n; i += 4) {
    ssr += w[i] * x[i] * x[i];
    ssr += w[i + 1] * x[i + 1] * x[i + 1];
    ssr += w[i + 2] * x[i + 2] * x[i + 2];
    ssr += w[i + 3] * x[i + 3] * x[i + 3];
  }
  for (; i < n; ++i) {
    ssr += w[i] * x[i] * x[i];
  }
  ssr *= inv_sw;

  bool converged = stopping_criterion(ssr_old, ssr, tol);
  ssr_old = ssr;
  return converged;
}

inline void project_group(double *v, const double *w, const GroupInfo &info,
                          bool &any_change, double tol) {
  if (info.is_singleton)
    return;

  double weighted_sum = 0.0;
  const uword *coord_ptr = info.coords->memptr();
  const uword n = info.n_elem;

  // Unrolled loop for better vectorization (4x unroll)
  uword i = 0;
  for (; i + 4 <= n; i += 4) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
    weighted_sum += w[coord_ptr[i + 1]] * v[coord_ptr[i + 1]];
    weighted_sum += w[coord_ptr[i + 2]] * v[coord_ptr[i + 2]];
    weighted_sum += w[coord_ptr[i + 3]] * v[coord_ptr[i + 3]];
  }
  // Handle remainder
  for (; i < n; ++i) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
  }

  double mean = weighted_sum * info.inv_weight;

  // Check if change is significant for early convergence detection
  if (std::abs(mean) > tol) {
    any_change = true;
    // Unrolled subtraction for better vectorization
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
  }
}

inline bool irons_tuck_acceleration(double *x, const double *Gx,
                                    const double *GGx, double *x0, uword n,
                                    double tol) {
  double vprod = 0.0, ssq = 0.0;

  // Unrolled loop for better vectorization
  uword i = 0;
  for (; i + 4 <= n; i += 4) {
    double delta_G0 = GGx[i] - Gx[i];
    double delta_G1 = GGx[i + 1] - Gx[i + 1];
    double delta_G2 = GGx[i + 2] - Gx[i + 2];
    double delta_G3 = GGx[i + 3] - Gx[i + 3];

    double delta2_0 = delta_G0 - Gx[i] + x[i];
    double delta2_1 = delta_G1 - Gx[i + 1] + x[i + 1];
    double delta2_2 = delta_G2 - Gx[i + 2] + x[i + 2];
    double delta2_3 = delta_G3 - Gx[i + 3] + x[i + 3];

    vprod += delta_G0 * delta2_0 + delta_G1 * delta2_1 + delta_G2 * delta2_2 +
             delta_G3 * delta2_3;
    ssq += delta2_0 * delta2_0 + delta2_1 * delta2_1 + delta2_2 * delta2_2 +
           delta2_3 * delta2_3;
  }
  for (; i < n; ++i) {
    double delta_G = GGx[i] - Gx[i];
    double delta2 = delta_G - Gx[i] + x[i];
    vprod += delta_G * delta2;
    ssq += delta2 * delta2;
  }

  if (ssq < tol) {
    return true; // Converged
  }

  double coef = vprod / ssq;

  // Apply acceleration if coefficient is reasonable
  if (coef > 0.0 && coef < 2.0) {
    i = 0;
    for (; i + 4 <= n; i += 4) {
      x[i] = GGx[i] - coef * (GGx[i] - Gx[i]);
      x[i + 1] = GGx[i + 1] - coef * (GGx[i + 1] - Gx[i + 1]);
      x[i + 2] = GGx[i + 2] - coef * (GGx[i + 2] - Gx[i + 2]);
      x[i + 3] = GGx[i + 3] - coef * (GGx[i + 3] - Gx[i + 3]);
    }
    for (; i < n; ++i) {
      x[i] = GGx[i] - coef * (GGx[i] - Gx[i]);
    }
  } else {
    // Fall back to GGx
    std::memcpy(x, GGx, n * sizeof(double));
  }

  return false;
}

inline bool project_2fe(double *x, const double *w,
                        const field<GroupInfo> &fe1_info,
                        const field<GroupInfo> &fe2_info, double tol) {
  bool any_change = false;

  // Forward pass: FE1 then FE2
  for (uword i = 0; i < fe1_info.n_elem; ++i) {
    project_group(x, w, fe1_info(i), any_change, tol);
  }
  for (uword i = 0; i < fe2_info.n_elem; ++i) {
    project_group(x, w, fe2_info(i), any_change, tol);
  }

  // Backward pass: FE2 then FE1 (symmetric Kaczmarz)
  for (uword i = fe2_info.n_elem; i-- > 0;) {
    project_group(x, w, fe2_info(i), any_change, tol);
  }
  for (uword i = fe1_info.n_elem; i-- > 0;) {
    project_group(x, w, fe1_info(i), any_change, tol);
  }

  return !any_change; // Converged if no changes
}

inline bool
project_kfe(double *x, const double *w,
            const field<field<GroupInfo>> &all_group_info, double tol) {
  const uword K = all_group_info.n_elem;
  bool any_change = false;

  // Forward pass
  for (uword k = 0; k < K; ++k) {
    const uword n_groups = all_group_info(k).n_elem;
#ifdef _OPENMP
    // Parallel processing for large group counts
    if (n_groups > 100) {
      bool local_change = false;
#pragma omp parallel for reduction(|| : local_change) schedule(dynamic, 32)
      for (uword l = 0; l < n_groups; ++l) {
        bool group_change = false;
        project_group(x, w, all_group_info(k)(l), group_change, tol);
        local_change = local_change || group_change;
      }
      any_change = any_change || local_change;
    } else {
#endif
      for (uword l = 0; l < n_groups; ++l) {
        project_group(x, w, all_group_info(k)(l), any_change, tol);
      }
#ifdef _OPENMP
    }
#endif
  }

  // Backward pass (symmetric Kaczmarz)
  for (uword k = K; k-- > 0;) {
    const uword n_groups = all_group_info(k).n_elem;
#ifdef _OPENMP
    if (n_groups > 100) {
      bool local_change = false;
#pragma omp parallel for reduction(|| : local_change) schedule(dynamic, 32)
      for (uword l = 0; l < n_groups; ++l) {
        uword rev_l = n_groups - 1 - l;
        bool group_change = false;
        project_group(x, w, all_group_info(k)(rev_l), group_change, tol);
        local_change = local_change || group_change;
      }
      any_change = any_change || local_change;
    } else {
#endif
      for (uword l = n_groups; l-- > 0;) {
        project_group(x, w, all_group_info(k)(l), any_change, tol);
      }
#ifdef _OPENMP
    }
#endif
  }

  return !any_change; // Converged if no changes
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const uword &max_iter, const uword &iter_interrupt,
    const uword &iter_ssr, const uword &accel_start,
    const double &project_tol_factor, const double &grand_accel_tol,
    const double &project_group_tol, const double &irons_tuck_tol,
    const uword &grand_accel_interval, const uword &irons_tuck_interval,
  const uword &ssr_check_interval, const double &convergence_factor,
  const double &tol_multiplier,
  const field<field<GroupInfo>> *precomputed_group_info = nullptr,
  CenteringWorkspace *workspace = nullptr) {
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem)
    return;

  const uword K = group_indices.n_elem;
  if (K == 0)
    return;

  const uword N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);
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

  const uword col_block_size = get_block_size(N, P);

  for (uword col_block = 0; col_block < P; col_block += col_block_size) {
    const uword col_end = std::min(col_block + col_block_size, P);

    for (uword col = col_block; col < col_end; ++col) {
      double *col_ptr = V.colptr(col);

        double *x_ptr = ws->x.memptr();
        double *x0_ptr = ws->x0.memptr();
        double *Gx_ptr = ws->Gx.memptr();
        double *GGx_ptr = ws->GGx.memptr();
        double *Y_ptr = ws->Y.memptr();
        double *GY_ptr = ws->GY.memptr();
        double *GGY_ptr = ws->GGY.memptr();

      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      uword iint = iter_interrupt;
      uword isr = iter_ssr;
      size_t grand_acc = 0;

      std::memcpy(x_ptr, col_ptr, N * sizeof(double));

      // Projection with convergence checking
      auto project_with_check = [&]() -> bool {
        if (K == 1) {
          bool any_change = false;
          for (const auto &info : all_group_info[0]) {
            project_group(x_ptr, w_ptr, info, any_change,
                          tol * project_tol_factor);
          }
          return !any_change;
        } else if (K == 2) {
          return project_2fe(x_ptr, w_ptr, all_group_info[0], all_group_info[1],
                             tol * project_tol_factor);
        }
        return project_kfe(x_ptr, w_ptr, all_group_info,
                           tol * project_tol_factor);
      };

      for (uword iter = 0; iter < max_iter; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iter_interrupt;
        }

        std::memcpy(x0_ptr, x_ptr, N * sizeof(double));

        if (project_with_check()) {
          break; // Converged at projection level
        }

        if (iter >= accel_start && iter % grand_accel_interval == 0) {
          if (grand_acceleration(x_ptr, Y_ptr, GY_ptr, GGY_ptr, ws->delta_GX,
                                 ws->delta2_X, N, grand_acc, grand_accel_tol)) {
            break; // Converged via grand acceleration
          }
          // Apply projection after acceleration
          project_with_check();
        }

        if (iter >= accel_start && iter % irons_tuck_interval == 0 &&
            iter % grand_accel_interval != 0) {
          std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));
          project_with_check();
          std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));

          std::memcpy(GGx_ptr, Gx_ptr, N * sizeof(double));
          project_with_check();
          std::memcpy(GGx_ptr, x_ptr, N * sizeof(double));

          if (irons_tuck_acceleration(x_ptr, Gx_ptr, GGx_ptr, x0_ptr, N,
                                      irons_tuck_tol)) {
            break; // Converged via Irons-Tuck
          }
        }

        // Check convergence based on relative change
        double weighted_diff = 0.0;
        for (uword i = 0; i < N; ++i) {
          double rel_diff =
              std::abs(x_ptr[i] - x0_ptr[i]) / (1.0 + std::abs(x0_ptr[i]));
          weighted_diff += w_ptr[i] * rel_diff;
        }
        double ratio = weighted_diff * inv_sw;

        if (ratio < tol)
          break;

        // Adaptive SSR-based convergence check (like fixest every
        // ssr_check_interval iterations)
        if (iter > 0 && iter % ssr_check_interval == 0) {
          if (adaptive_ssr_check(x_ptr, w_ptr, N, ssr0, inv_sw, tol)) {
            break; // Converged via SSR check
          }
        }

        // Standard SSR check at specified intervals
        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += iter_ssr;
          double ssr = 0.0;
          for (uword i = 0; i < N; ++i) {
            ssr += w_ptr[i] * x_ptr[i] * x_ptr[i];
          }
          ssr *= inv_sw;
          if (stopping_criterion(ssr0, ssr, tol))
            break;
          ssr0 = ssr;
        }

        // Early termination if slow convergence
        if (iter > 3 && (ratio0 / ratio) < convergence_factor &&
            ratio < tol * tol_multiplier)
          break;
        ratio0 = ratio;
      }

      std::memcpy(col_ptr, x_ptr, N * sizeof(double));
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
