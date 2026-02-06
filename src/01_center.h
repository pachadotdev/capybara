// Centering using flat observation-to-group mapping
// Optimized for speed: raw pointer access, blocked processing
#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// In-place row scaling for all columns: A(i,j) *= scale[i] for all j
inline void scale_rows_inplace(mat &A, const vec &scale) {
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  const double *s_ptr = scale.memptr();

  for (uword j = 0; j < n_cols; ++j) {
    double *col = A.colptr(j);
    for (uword i = 0; i < n_rows; ++i) {
      col[i] *= s_ptr[i];
    }
  }
}

// Flat FE structure using std::vector for guaranteed contiguous memory
struct FlatFEMap {
  std::vector<std::vector<uword>>
      fe_map;                   // K x n_obs: fe_map[k][i] = group of obs i
  std::vector<vec> inv_weights; // K: precomputed 1/sum(w) per group
  std::vector<uword> n_groups;  // K: number of groups per FE
  uword n_obs;
  uword K;

  void build(const field<field<uvec>> &group_indices) {
    K = group_indices.n_elem;
    if (K == 0)
      return;

    n_groups.resize(K);
    n_obs = 0;

    // First pass: find n_obs and n_groups
    for (uword k = 0; k < K; ++k) {
      n_groups[k] = group_indices(k).n_elem;
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        if (idx.n_elem > 0) {
          n_obs = std::max(n_obs, idx.max() + 1);
        }
      }
    }

    // Allocate fe_map
    fe_map.resize(K);
    for (uword k = 0; k < K; ++k) {
      fe_map[k].assign(n_obs, 0);
    }

    // Fill fe_map
    for (uword k = 0; k < K; ++k) {
      uword *map_k = fe_map[k].data();
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        const uword *idx_ptr = idx.memptr();
        const uword cnt = idx.n_elem;
        for (uword j = 0; j < cnt; ++j) {
          map_k[idx_ptr[j]] = g;
        }
      }
    }
  }

  void update_weights(const vec &w) {
    if (K == 0)
      return;

    inv_weights.resize(K);
    const bool use_w = (w.n_elem == n_obs);
    const double *w_ptr = w.memptr();

    for (uword k = 0; k < K; ++k) {
      inv_weights[k].zeros(n_groups[k]);
      double *inv_w_ptr = inv_weights[k].memptr();
      const uword *map_k = fe_map[k].data();

      // Accumulate weights per group
      if (use_w) {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += w_ptr[i];
        }
      } else {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += 1.0;
        }
      }

      // Invert (clamp to avoid division by zero)
      for (uword g = 0; g < n_groups[k]; ++g) {
        inv_w_ptr[g] = (inv_w_ptr[g] > 1e-12) ? (1.0 / inv_w_ptr[g]) : 0.0;
      }
    }
  }
};

// Blocked iteration for 2FE case - update alpha1
// Block size 4 with manual unrolling for portability (works with GCC, Clang,
// MSVC)
inline void center_2fe_block_alpha1(const uword n_obs, const uword p_start,
                                    const uword b_sz, const double *w_ptr,
                                    const uword *g1, const uword *g2, mat &V,
                                    mat &alpha1, const mat &alpha2) {
  // Get column pointers for this block
  const double *v0 = V.colptr(p_start);
  const double *v1 = (b_sz > 1) ? V.colptr(p_start + 1) : nullptr;
  const double *v2 = (b_sz > 2) ? V.colptr(p_start + 2) : nullptr;
  const double *v3 = (b_sz > 3) ? V.colptr(p_start + 3) : nullptr;

  double *a1_0 = alpha1.colptr(p_start);
  double *a1_1 = (b_sz > 1) ? alpha1.colptr(p_start + 1) : nullptr;
  double *a1_2 = (b_sz > 2) ? alpha1.colptr(p_start + 2) : nullptr;
  double *a1_3 = (b_sz > 3) ? alpha1.colptr(p_start + 3) : nullptr;

  const double *a2_0 = alpha2.colptr(p_start);
  const double *a2_1 = (b_sz > 1) ? alpha2.colptr(p_start + 1) : nullptr;
  const double *a2_2 = (b_sz > 2) ? alpha2.colptr(p_start + 2) : nullptr;
  const double *a2_3 = (b_sz > 3) ? alpha2.colptr(p_start + 3) : nullptr;

  for (uword i = 0; i < n_obs; ++i) {
    const double wi = w_ptr[i];
    if (wi > 1e-14) {
      const uword ug1 = g1[i];
      const uword ug2 = g2[i];

      // Manual unroll - compiler will optimize away null pointer branches
      a1_0[ug1] += wi * (v0[i] - a2_0[ug2]);
      if (b_sz > 1)
        a1_1[ug1] += wi * (v1[i] - a2_1[ug2]);
      if (b_sz > 2)
        a1_2[ug1] += wi * (v2[i] - a2_2[ug2]);
      if (b_sz > 3)
        a1_3[ug1] += wi * (v3[i] - a2_3[ug2]);
    }
  }
}

// Blocked iteration for 2FE case - update alpha2
inline void center_2fe_block_alpha2(const uword n_obs, const uword p_start,
                                    const uword b_sz, const double *w_ptr,
                                    const uword *g1, const uword *g2, mat &V,
                                    const mat &alpha1, mat &alpha2) {
  const double *v0 = V.colptr(p_start);
  const double *v1 = (b_sz > 1) ? V.colptr(p_start + 1) : nullptr;
  const double *v2 = (b_sz > 2) ? V.colptr(p_start + 2) : nullptr;
  const double *v3 = (b_sz > 3) ? V.colptr(p_start + 3) : nullptr;

  const double *a1_0 = alpha1.colptr(p_start);
  const double *a1_1 = (b_sz > 1) ? alpha1.colptr(p_start + 1) : nullptr;
  const double *a1_2 = (b_sz > 2) ? alpha1.colptr(p_start + 2) : nullptr;
  const double *a1_3 = (b_sz > 3) ? alpha1.colptr(p_start + 3) : nullptr;

  double *a2_0 = alpha2.colptr(p_start);
  double *a2_1 = (b_sz > 1) ? alpha2.colptr(p_start + 1) : nullptr;
  double *a2_2 = (b_sz > 2) ? alpha2.colptr(p_start + 2) : nullptr;
  double *a2_3 = (b_sz > 3) ? alpha2.colptr(p_start + 3) : nullptr;

  for (uword i = 0; i < n_obs; ++i) {
    const double wi = w_ptr[i];
    if (wi > 1e-14) {
      const uword ug1 = g1[i];
      const uword ug2 = g2[i];

      a2_0[ug2] += wi * (v0[i] - a1_0[ug1]);
      if (b_sz > 1)
        a2_1[ug2] += wi * (v1[i] - a1_1[ug1]);
      if (b_sz > 2)
        a2_2[ug2] += wi * (v2[i] - a1_2[ug1]);
      if (b_sz > 3)
        a2_3[ug2] += wi * (v3[i] - a1_3[ug1]);
    }
  }
}

// 2FE centering with Irons-Tuck acceleration + grand acceleration
inline void center_2fe(mat &V, const vec &w, const FlatFEMap &map, double tol,
                       uword max_iter, uword grand_acc_period) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = map.n_groups[0];
  const uword n2 = map.n_groups[1];

  mat alpha1(n1, P, fill::zeros);
  mat alpha2(n2, P, fill::zeros);

  const uword *g1 = map.fe_map[0].data();
  const uword *g2 = map.fe_map[1].data();
  const double *w_ptr = w.memptr();

  constexpr uword block_size = 4;

  // Ring buffer for inner IT acceleration history (3 matrices)
  mat hist0(n1, P, fill::zeros);
  mat hist1(n1, P, fill::zeros);
  mat hist2(n1, P, fill::zeros);
  mat *hist[3] = {&hist0, &hist1, &hist2};
  uword hist_idx = 0;

  // Grand acceleration: track trajectory every grand_acc_period iterations
  // Y -> GY -> GGY cycle for second-level IT
  mat grand_Y(n1, P, fill::zeros);
  mat grand_GY(n1, P, fill::zeros);
  mat grand_GGY(n1, P, fill::zeros);
  uword grand_stage = 0; // 0, 1, 2 for Y, GY, GGY

  const uword total_elem = n1 * P;

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Save current alpha1 to history ring buffer
    mat *alpha1_old = hist[(hist_idx + 2) % 3];
    std::memcpy(alpha1_old->memptr(), alpha1.memptr(),
                total_elem * sizeof(double));

    // Update alpha1
    alpha1.zeros();
    for (uword p = 0; p < P; p += block_size) {
      uword b_sz = std::min(block_size, P - p);
      center_2fe_block_alpha1(n_obs, p, b_sz, w_ptr, g1, g2, V, alpha1, alpha2);
    }
    scale_rows_inplace(alpha1, map.inv_weights[0]);

    // Update alpha2
    alpha2.zeros();
    for (uword p = 0; p < P; p += block_size) {
      uword b_sz = std::min(block_size, P - p);
      center_2fe_block_alpha2(n_obs, p, b_sz, w_ptr, g1, g2, V, alpha1, alpha2);
    }
    scale_rows_inplace(alpha2, map.inv_weights[1]);

    // Inner Irons-Tuck acceleration (after warmup)
    if (iter >= 3) {
      const mat *alpha1_prev = hist[(hist_idx + 1) % 3];
      const mat *alpha1_prev2 = hist[hist_idx];

      const double *curr = alpha1.memptr();
      const double *prev = alpha1_prev->memptr();
      const double *prev2 = alpha1_prev2->memptr();

      // Compute Irons-Tuck coefficient inline
      double ssq = 0.0, vprod = 0.0;
      for (uword i = 0; i < total_elem; ++i) {
        double d1 = prev[i] - prev2[i];
        double d2 = curr[i] - prev[i];
        double dd = d2 - d1;
        ssq += dd * dd;
        vprod += d2 * dd;
      }

      if (ssq > 1e-14) {
        double coef = vprod / ssq;
        if (coef > 0.0 && coef < 2.0) {
          // Extrapolate alpha1 in place
          double *a1_ptr = alpha1.memptr();
          for (uword i = 0; i < total_elem; ++i) {
            a1_ptr[i] += coef * (a1_ptr[i] - prev[i]);
          }
          // Recompute alpha2 after acceleration
          alpha2.zeros();
          for (uword p = 0; p < P; p += block_size) {
            uword b_sz = std::min(block_size, P - p);
            center_2fe_block_alpha2(n_obs, p, b_sz, w_ptr, g1, g2, V, alpha1,
                                    alpha2);
          }
          scale_rows_inplace(alpha2, map.inv_weights[1]);
        }
      }
    }

    // Grand acceleration: second-level IT on overall trajectory
    if (iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        // Save Y
        std::memcpy(grand_Y.memptr(), alpha1.memptr(),
                    total_elem * sizeof(double));
        grand_stage = 1;
      } else if (grand_stage == 1) {
        // Save GY
        std::memcpy(grand_GY.memptr(), alpha1.memptr(),
                    total_elem * sizeof(double));
        grand_stage = 2;
      } else {
        // Save GGY and apply grand IT
        std::memcpy(grand_GGY.memptr(), alpha1.memptr(),
                    total_elem * sizeof(double));

        // Apply Irons-Tuck on (Y, GY, GGY)
        const double *Y_ptr = grand_Y.memptr();
        const double *GY_ptr = grand_GY.memptr();
        const double *GGY_ptr = grand_GGY.memptr();

        double ssq = 0.0, vprod = 0.0;
        for (uword i = 0; i < total_elem; ++i) {
          double delta_GX = GGY_ptr[i] - GY_ptr[i];
          double delta2_X = delta_GX - GY_ptr[i] + Y_ptr[i];
          ssq += delta2_X * delta2_X;
          vprod += delta_GX * delta2_X;
        }

        if (ssq > 1e-14) {
          double coef = vprod / ssq;
          if (coef > 0.0 && coef < 2.0) {
            // Update alpha1 with grand-accelerated value
            double *a1_ptr = alpha1.memptr();
            for (uword i = 0; i < total_elem; ++i) {
              double delta_GX = GGY_ptr[i] - GY_ptr[i];
              a1_ptr[i] = GGY_ptr[i] - coef * delta_GX;
            }
            // Recompute alpha2 after grand acceleration
            alpha2.zeros();
            for (uword p = 0; p < P; p += block_size) {
              uword b_sz = std::min(block_size, P - p);
              center_2fe_block_alpha2(n_obs, p, b_sz, w_ptr, g1, g2, V, alpha1,
                                      alpha2);
            }
            scale_rows_inplace(alpha2, map.inv_weights[1]);
          }
        }
        grand_stage = 0; // Reset cycle
      }
    }

    // Convergence check
    double diff_sq = 0.0;
    const double *curr = alpha1.memptr();
    const double *old_ptr = alpha1_old->memptr();
    for (uword i = 0; i < total_elem; ++i) {
      double d = curr[i] - old_ptr[i];
      diff_sq += d * d;
    }
    if (std::sqrt(diff_sq) < tol)
      break;

    hist_idx = (hist_idx + 1) % 3;
  }

  // Final subtraction (blocked)
  for (uword p = 0; p < P; p += block_size) {
    uword b_sz = std::min(block_size, P - p);
    double *v_ptrs[4];
    const double *a1_ptrs[4];
    const double *a2_ptrs[4];

    for (uword j = 0; j < b_sz; ++j) {
      v_ptrs[j] = V.colptr(p + j);
      a1_ptrs[j] = alpha1.colptr(p + j);
      a2_ptrs[j] = alpha2.colptr(p + j);
    }

    for (uword i = 0; i < n_obs; ++i) {
      uword u_g1 = g1[i];
      uword u_g2 = g2[i];
      for (uword j = 0; j < b_sz; ++j) {
        v_ptrs[j][i] -= a1_ptrs[j][u_g1] + a2_ptrs[j][u_g2];
      }
    }
  }
}

// General K-FE centering with blocked processing
// Includes grand acceleration (second-level IT) on overall trajectory
inline void center_kfe(mat &V, const vec &w, const FlatFEMap &map, double tol,
                       uword max_iter, uword grand_acc_period) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  // Allocate alpha for each FE
  std::vector<mat> alpha(K);
  for (uword k = 0; k < K; ++k) {
    alpha[k].zeros(map.n_groups[k], P);
  }

  const double *w_ptr = w.memptr();
  constexpr uword block_size = 4;

  // Prepare map pointers
  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

  // Ring buffer for inner IT acceleration history on alpha[0]
  const uword n0 = map.n_groups[0];
  const uword total_elem0 = n0 * P;
  mat hist0(n0, P, fill::zeros);
  mat hist1(n0, P, fill::zeros);
  mat hist2(n0, P, fill::zeros);
  mat *hist[3] = {&hist0, &hist1, &hist2};
  uword hist_idx = 0;

  // Grand acceleration: track trajectory every grand_acc_period iterations
  mat grand_Y(n0, P, fill::zeros);
  mat grand_GY(n0, P, fill::zeros);
  mat grand_GGY(n0, P, fill::zeros);
  uword grand_stage = 0;

  // Pre-allocate other_ptrs outside loop
  std::vector<std::vector<const double *>> other_ptrs(K);
  for (uword o = 0; o < K; ++o) {
    other_ptrs[o].resize(block_size);
  }

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Save current alpha[0] to history ring buffer
    mat *alpha0_old = hist[(hist_idx + 2) % 3];
    std::memcpy(alpha0_old->memptr(), alpha[0].memptr(),
                total_elem0 * sizeof(double));

    // Gauss-Seidel sweep over all K fixed effects
    for (uword k = 0; k < K; ++k) {
      mat &alpha_k = alpha[k];
      const uword *gk = map_ptrs[k];

      alpha_k.zeros();

      // Blocked accumulation
      for (uword p = 0; p < P; p += block_size) {
        uword b_sz = std::min(block_size, P - p);

        double *ak_ptrs[4];
        const double *v_ptrs[4];
        for (uword j = 0; j < b_sz; ++j) {
          ak_ptrs[j] = alpha_k.colptr(p + j);
          v_ptrs[j] = V.colptr(p + j);
        }

        // Prepare other alpha pointers (reuse pre-allocated vectors)
        for (uword o = 0; o < K; ++o) {
          if (o != k) {
            for (uword j = 0; j < b_sz; ++j) {
              other_ptrs[o][j] = alpha[o].colptr(p + j);
            }
          }
        }

        for (uword i = 0; i < n_obs; ++i) {
          double wi = w_ptr[i];
          if (wi > 1e-14) {
            uword g_k = gk[i];

            for (uword j = 0; j < b_sz; ++j) {
              double sum_others = 0.0;
              for (uword o = 0; o < K; ++o) {
                if (o != k) {
                  sum_others += other_ptrs[o][j][map_ptrs[o][i]];
                }
              }
              ak_ptrs[j][g_k] += wi * (v_ptrs[j][i] - sum_others);
            }
          }
        }
      }

      // Apply inverse weights
      scale_rows_inplace(alpha_k, map.inv_weights[k]);
    }

    // Irons-Tuck acceleration on alpha[0] - inline computation
    if (iter >= 3) {
      const mat *alpha0_prev = hist[(hist_idx + 1) % 3];
      const mat *alpha0_prev2 = hist[hist_idx];

      const double *curr = alpha[0].memptr();
      const double *prev = alpha0_prev->memptr();
      const double *prev2 = alpha0_prev2->memptr();

      double ssq = 0.0, vprod = 0.0;
      for (uword i = 0; i < total_elem0; ++i) {
        double d1 = prev[i] - prev2[i];
        double d2 = curr[i] - prev[i];
        double dd = d2 - d1;
        ssq += dd * dd;
        vprod += d2 * dd;
      }

      if (ssq > 1e-14) {
        double coef = vprod / ssq;
        if (coef > 0.0 && coef < 2.0) {
          double *a0_ptr = alpha[0].memptr();
          for (uword i = 0; i < total_elem0; ++i) {
            a0_ptr[i] += coef * (a0_ptr[i] - prev[i]);
          }
        }
      }
    }

    // Grand acceleration: second-level IT on overall trajectory
    if (iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        std::memcpy(grand_Y.memptr(), alpha[0].memptr(),
                    total_elem0 * sizeof(double));
        grand_stage = 1;
      } else if (grand_stage == 1) {
        std::memcpy(grand_GY.memptr(), alpha[0].memptr(),
                    total_elem0 * sizeof(double));
        grand_stage = 2;
      } else {
        std::memcpy(grand_GGY.memptr(), alpha[0].memptr(),
                    total_elem0 * sizeof(double));

        const double *Y_ptr = grand_Y.memptr();
        const double *GY_ptr = grand_GY.memptr();
        const double *GGY_ptr = grand_GGY.memptr();

        double ssq = 0.0, vprod = 0.0;
        for (uword i = 0; i < total_elem0; ++i) {
          double delta_GX = GGY_ptr[i] - GY_ptr[i];
          double delta2_X = delta_GX - GY_ptr[i] + Y_ptr[i];
          ssq += delta2_X * delta2_X;
          vprod += delta_GX * delta2_X;
        }

        if (ssq > 1e-14) {
          double coef = vprod / ssq;
          if (coef > 0.0 && coef < 2.0) {
            double *a0_ptr = alpha[0].memptr();
            for (uword i = 0; i < total_elem0; ++i) {
              double delta_GX = GGY_ptr[i] - GY_ptr[i];
              a0_ptr[i] = GGY_ptr[i] - coef * delta_GX;
            }
          }
        }
        grand_stage = 0;
      }
    }

    // Convergence check - inline without temporary
    double diff_sq = 0.0;
    const double *curr = alpha[0].memptr();
    const double *old_ptr = alpha0_old->memptr();
    for (uword i = 0; i < total_elem0; ++i) {
      double d = curr[i] - old_ptr[i];
      diff_sq += d * d;
    }
    if (std::sqrt(diff_sq) < tol)
      break;

    // Rotate ring buffer
    hist_idx = (hist_idx + 1) % 3;
  }

  // Final subtraction - pre-allocate a_ptrs outside loop
  std::vector<std::vector<const double *>> a_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    a_ptrs[k].resize(block_size);
  }

  for (uword p = 0; p < P; p += block_size) {
    uword b_sz = std::min(block_size, P - p);

    double *v_ptrs[4];
    for (uword j = 0; j < b_sz; ++j) {
      v_ptrs[j] = V.colptr(p + j);
    }
    for (uword k = 0; k < K; ++k) {
      for (uword j = 0; j < b_sz; ++j) {
        a_ptrs[k][j] = alpha[k].colptr(p + j);
      }
    }

    for (uword i = 0; i < n_obs; ++i) {
      for (uword j = 0; j < b_sz; ++j) {
        double sum_a = 0.0;
        for (uword k = 0; k < K; ++k) {
          sum_a += a_ptrs[k][j][map_ptrs[k][i]];
        }
        v_ptrs[j][i] -= sum_a;
      }
    }
  }
}

// Main centering dispatch
inline void center_impl(mat &V, const vec &w, const FlatFEMap &map, double tol,
                        uword max_iter, uword grand_acc_period = 10) {
  if (map.K == 2) {
    center_2fe(V, w, map, tol, max_iter, grand_acc_period);
  } else {
    center_kfe(V, w, map, tol, max_iter, grand_acc_period);
  }
}

// Public interface
inline void center_variables(mat &V, const vec &w, const FlatFEMap &map,
                             double tol, uword max_iter,
                             uword grand_acc_period = 10) {
  if (V.is_empty() || map.K == 0)
    return;
  center_impl(V, w, map, tol, max_iter, grand_acc_period);
}

inline void center_variables(mat &V, const vec &w,
                             const field<field<uvec>> &group_indices,
                             double tol, uword max_iter,
                             uword grand_acc_period = 10) {
  if (V.is_empty() || group_indices.n_elem == 0)
    return;
  FlatFEMap map;
  map.build(group_indices);
  map.update_weights(w);
  center_impl(V, w, map, tol, max_iter, grand_acc_period);
}

inline FlatFEMap build_fe_map(const field<field<uvec>> &group_indices,
                              const vec &w) {
  FlatFEMap map;
  map.build(group_indices);
  map.update_weights(w);
  return map;
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
