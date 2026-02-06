// Centering using flat observation-to-group mapping
//
// Key optimization: cell aggregation for 2-FE case reduces O(N) loops to
// O(n_cells) where n_cells << N for typical panel data
//
// Some ideas borrowed from fixest/reghdfe:
// - Cell aggregation for 2-FE case
// - Irons-Tuck acceleration
// - Grand acceleration (second-level IT)
// - Post-acceleration projection
// - fixest-style per-coefficient convergence criterion
// - SSR-based stopping (periodic)

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

// Whereas iteration should continue/stop

inline bool continue_crit(double a, double b, double diffMax) {
  double diff = std::fabs(a - b);
  return (diff > diffMax) && (diff / (0.1 + std::fabs(a)) > diffMax);
}

inline bool stopping_crit(double a, double b, double diffMax) {
  double diff = std::fabs(a - b);
  return (diff < diffMax) || (diff / (0.1 + std::fabs(a)) < diffMax);
}

// Flat FE structure using std::vector for guaranteed contiguous memory
struct FlatFEMap {
  std::vector<std::vector<uword>>
      fe_map;                   // K x n_obs: fe_map[k][i] = group of obs i
  std::vector<vec> inv_weights; // K: precomputed 1/sum(w) per group
  std::vector<uword> n_groups;  // K: number of groups per FE
  uword n_obs;
  uword K;
  bool structure_built; // Flag to indicate structure is ready

  FlatFEMap() : n_obs(0), K(0), structure_built(false) {}

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

    structure_built = true;
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

// Cell-aggregated 2-FE centering
//
// Key idea from fixest/reghdfe: for 2 FEs with n1 and n2 groups,
// many observations share the same (g1, g2) cell. Instead of looping
// over all N observations in the inner iteration, pre-aggregate the
// weighted sums of V into unique cells. The alpha update then iterates
// over n_cells (typically much smaller than N).
//
// The structure (cell_g1, cell_g2, obs_to_cell) is invariant across
// IRLS iterations — only cell_w and cell_wV change with weights.

struct CellAggregated2FE {
  // Cell structure (invariant across weight updates)
  std::vector<uword> cell_g1; // FE1 group for each cell
  std::vector<uword> cell_g2; // FE2 group for each cell
  uword n_cells;

  // Mapping from observations to cells (invariant)
  std::vector<uword> obs_to_cell; // obs_to_cell[i] = cell index

  // Weight-dependent data (changes each IRLS iteration)
  std::vector<double> cell_w; // total weight per cell

  // Pre-aggregated weighted V sums per cell (P columns)
  mat cell_wV; // n_cells x P: sum of w[i]*V[i,p] for obs in cell

  bool structure_built;

  CellAggregated2FE() : n_cells(0), structure_built(false) {}

  // Build cell structure (call once, structure is invariant)
  void build_structure(const uword *g1, const uword *g2, uword n_obs, uword n1,
                       uword n2) {
    const uword max_cells = n1 * n2;
    const uword sentinel = std::numeric_limits<uword>::max();
    std::vector<uword> pair_to_cell(max_cells, sentinel);
    obs_to_cell.resize(n_obs);

    n_cells = 0;
    cell_g1.clear();
    cell_g2.clear();

    cell_g1.reserve(std::min(max_cells, n_obs));
    cell_g2.reserve(std::min(max_cells, n_obs));

    for (uword i = 0; i < n_obs; ++i) {
      const uword pair_id = g1[i] * n2 + g2[i];
      uword cell_id = pair_to_cell[pair_id];
      if (cell_id == sentinel) {
        cell_id = n_cells++;
        pair_to_cell[pair_id] = cell_id;
        cell_g1.push_back(g1[i]);
        cell_g2.push_back(g2[i]);
      }
      obs_to_cell[i] = cell_id;
    }

    structure_built = true;
  }

  // Update weights per cell (call each IRLS iteration)
  void update_cell_weights(const double *w_ptr, uword n_obs) {
    cell_w.assign(n_cells, 0.0);
    for (uword i = 0; i < n_obs; ++i) {
      cell_w[obs_to_cell[i]] += w_ptr[i];
    }
  }

  // Aggregate weighted V values into cells (call each centering invocation)
  void aggregate_wV(const mat &V, const double *w_ptr, uword n_obs) {
    const uword P = V.n_cols;
    cell_wV.zeros(n_cells, P);

    for (uword p = 0; p < P; ++p) {
      const double *v_col = V.colptr(p);
      double *cwv_col = cell_wV.colptr(p);
      for (uword i = 0; i < n_obs; ++i) {
        cwv_col[obs_to_cell[i]] += w_ptr[i] * v_col[i];
      }
    }
  }
};

// Helper: recompute alpha2 from alpha1 using cell-aggregated data
inline void recompute_alpha2(mat &alpha2, const mat &alpha1,
                             const CellAggregated2FE &cells,
                             const FlatFEMap &map, uword P) {
  const uword nc = cells.n_cells;
  const uword *cg1 = cells.cell_g1.data();
  const uword *cg2 = cells.cell_g2.data();
  const double *cw = cells.cell_w.data();

  alpha2.zeros();
  for (uword p = 0; p < P; ++p) {
    const double *a1 = alpha1.colptr(p);
    double *a2 = alpha2.colptr(p);
    const double *cwv = cells.cell_wV.colptr(p);
    for (uword c = 0; c < nc; ++c) {
      a2[cg2[c]] += cwv[c] - cw[c] * a1[cg1[c]];
    }
  }
  scale_rows_inplace(alpha2, map.inv_weights[1]);
}

// Optimized 2FE centering with cell aggregation, Irons-Tuck acceleration,
// grand acceleration, post-acceleration projection, and fixest-style
// convergence
inline void center_2fe(mat &V, const vec &w, const FlatFEMap &map,
                       CellAggregated2FE &cells, double tol, uword max_iter,
                       uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = map.n_groups[0];
  const uword n2 = map.n_groups[1];

  const uword *g1 = map.fe_map[0].data();
  const uword *g2 = map.fe_map[1].data();
  const double *w_ptr = w.memptr();

  // Build cell structure if not already built
  if (!cells.structure_built) {
    cells.build_structure(g1, g2, n_obs, n1, n2);
  }

  // Update weights and aggregate V (weight-dependent, changes each call)
  cells.update_cell_weights(w_ptr, n_obs);
  cells.aggregate_wV(V, w_ptr, n_obs);

  const uword nc = cells.n_cells;
  const uword *cg1 = cells.cell_g1.data();
  const uword *cg2 = cells.cell_g2.data();
  const double *cw = cells.cell_w.data();

  mat alpha1(n1, P, fill::zeros);
  mat alpha2(n2, P, fill::zeros);

  // IT acceleration: X, GX, G2X pattern (fixest-style)
  const uword total_elem = n1 * P;
  mat X_it(n1, P, fill::zeros);  // alpha1 at step n
  mat GX_it(n1, P, fill::zeros); // alpha1 at step n+1
  // alpha1 itself serves as G2X (alpha1 at step n+2)

  // IT acceleration working buffers
  std::vector<double> delta_GX(total_elem);
  std::vector<double> delta2_X(total_elem);

  // Grand acceleration buffers
  mat grand_Y(n1, P, fill::zeros);
  mat grand_GY(n1, P, fill::zeros);
  uword grand_stage = 0;

  // Post-acceleration projection threshold
  constexpr uword iter_proj_after_acc = 40;

  // SSR-based stopping (periodic check)
  constexpr uword ssr_check_period = 40;
  double ssr_old = datum::inf;

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Save current alpha1 as X_it (the "before" state)
    std::memcpy(X_it.memptr(), alpha1.memptr(), total_elem * sizeof(double));

    // === Gauss-Seidel sweep 1: compute GX ===

    // Update alpha1
    alpha1.zeros();
    for (uword p = 0; p < P; ++p) {
      double *a1 = alpha1.colptr(p);
      const double *a2 = alpha2.colptr(p);
      const double *cwv = cells.cell_wV.colptr(p);
      for (uword c = 0; c < nc; ++c) {
        a1[cg1[c]] += cwv[c] - cw[c] * a2[cg2[c]];
      }
    }
    scale_rows_inplace(alpha1, map.inv_weights[0]);

    // Update alpha2
    recompute_alpha2(alpha2, alpha1, cells, map, P);

    // Save GX
    std::memcpy(GX_it.memptr(), alpha1.memptr(), total_elem * sizeof(double));

    // === Gauss-Seidel sweep 2: compute G2X ===

    alpha1.zeros();
    for (uword p = 0; p < P; ++p) {
      double *a1 = alpha1.colptr(p);
      const double *a2 = alpha2.colptr(p);
      const double *cwv = cells.cell_wV.colptr(p);
      for (uword c = 0; c < nc; ++c) {
        a1[cg1[c]] += cwv[c] - cw[c] * a2[cg2[c]];
      }
    }
    scale_rows_inplace(alpha1, map.inv_weights[0]);

    // Update alpha2
    recompute_alpha2(alpha2, alpha1, cells, map, P);

    // alpha1 is now G2X

    // === Irons-Tuck acceleration ===
    // X_it = G2X - coef * delta_GX
    // where delta_GX = G2X - GX, delta2_X = delta_GX - GX + X

    const double *x_ptr = X_it.memptr();
    const double *gx_ptr = GX_it.memptr();
    const double *ggx_ptr = alpha1.memptr();

    double vprod = 0.0, ssq = 0.0;
    for (uword i = 0; i < total_elem; ++i) {
      double dGX = ggx_ptr[i] - gx_ptr[i];
      double d2X = dGX - gx_ptr[i] + x_ptr[i];
      delta_GX[i] = dGX;
      delta2_X[i] = d2X;
      vprod += dGX * d2X;
      ssq += d2X * d2X;
    }

    bool numconv = false;
    if (ssq == 0.0) {
      numconv = true;
    } else {
      double coef = vprod / ssq;
      double *a1_ptr = alpha1.memptr();
      for (uword i = 0; i < total_elem; ++i) {
        a1_ptr[i] = ggx_ptr[i] - coef * delta_GX[i];
      }
    }

    // Recompute alpha2 after IT acceleration
    recompute_alpha2(alpha2, alpha1, cells, map, P);

    // Post-acceleration projection: after iter_proj_after_acc iterations,
    // do an extra Gauss-Seidel sweep to project back into feasible set
    if (iter >= iter_proj_after_acc) {
      alpha1.zeros();
      for (uword p = 0; p < P; ++p) {
        double *a1 = alpha1.colptr(p);
        const double *a2 = alpha2.colptr(p);
        const double *cwv = cells.cell_wV.colptr(p);
        for (uword c = 0; c < nc; ++c) {
          a1[cg1[c]] += cwv[c] - cw[c] * a2[cg2[c]];
        }
      }
      scale_rows_inplace(alpha1, map.inv_weights[0]);
      recompute_alpha2(alpha2, alpha1, cells, map, P);
    }

    if (numconv)
      break;

    // === Grand acceleration ===
    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        std::memcpy(grand_Y.memptr(), alpha1.memptr(),
                    total_elem * sizeof(double));
        grand_stage = 1;
      } else if (grand_stage == 1) {
        std::memcpy(grand_GY.memptr(), alpha1.memptr(),
                    total_elem * sizeof(double));
        grand_stage = 2;
      } else {
        // alpha1 is G2Y
        const double *Y_ptr = grand_Y.memptr();
        const double *GY_ptr = grand_GY.memptr();
        const double *G2Y_ptr = alpha1.memptr();

        double g_vprod = 0.0, g_ssq = 0.0;
        for (uword i = 0; i < total_elem; ++i) {
          double dGX = G2Y_ptr[i] - GY_ptr[i];
          double d2X = dGX - GY_ptr[i] + Y_ptr[i];
          g_ssq += d2X * d2X;
          g_vprod += dGX * d2X;
        }

        if (g_ssq > 1e-14) {
          double coef = g_vprod / g_ssq;
          double *a1_ptr = alpha1.memptr();
          for (uword i = 0; i < total_elem; ++i) {
            double dGX = G2Y_ptr[i] - GY_ptr[i];
            a1_ptr[i] = G2Y_ptr[i] - coef * dGX;
          }
          recompute_alpha2(alpha2, alpha1, cells, map, P);
        }
        grand_stage = 0;
      }
    }

    // === Convergence check: fixest-style per-coefficient criterion ===
    const double *curr = alpha1.memptr();
    const double *old = X_it.memptr();
    bool keep_going = false;
    for (uword i = 0; i < total_elem; ++i) {
      if (continue_crit(curr[i], old[i], tol)) {
        keep_going = true;
        break;
      }
    }
    if (!keep_going)
      break;

    // === SSR-based stopping (periodic, more expensive) ===
    if (iter > 0 && iter % ssr_check_period == 0) {
      // Compute SSR = sum_obs w[i] * (V[i] - alpha1[g1[i]] - alpha2[g2[i]])^2
      double ssr = 0.0;
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        const double *a1 = alpha1.colptr(p);
        const double *a2 = alpha2.colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          double r = v_col[i] - a1[g1[i]] - a2[g2[i]];
          ssr += w_ptr[i] * r * r;
        }
      }
      if (stopping_crit(ssr_old, ssr, tol))
        break;
      ssr_old = ssr;
    }
  }

  // Final subtraction: V[i,p] -= alpha1[g1[i]] + alpha2[g2[i]]
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *a1 = alpha1.colptr(p);
    const double *a2 = alpha2.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      v_col[i] -= a1[g1[i]] + a2[g2[i]];
    }
  }
}

// General K-FE centering with blocked processing
inline void center_kfe(mat &V, const vec &w, const FlatFEMap &map, double tol,
                       uword max_iter, uword grand_acc_period = 4) {
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

  // IT acceleration on alpha[0]: X, GX, G2X pattern
  const uword n0 = map.n_groups[0];
  const uword total_elem0 = n0 * P;
  mat X_it(n0, P, fill::zeros);
  mat GX_it(n0, P, fill::zeros);

  std::vector<double> delta_GX(total_elem0);
  std::vector<double> delta2_X(total_elem0);

  // Grand acceleration buffers
  mat grand_Y(n0, P, fill::zeros);
  mat grand_GY(n0, P, fill::zeros);
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  constexpr uword ssr_check_period = 40;
  double ssr_old = datum::inf;

  // Pre-allocate other_ptrs outside loop
  std::vector<std::vector<const double *>> other_ptrs(K);
  for (uword o = 0; o < K; ++o) {
    other_ptrs[o].resize(block_size);
  }

  // Lambda for a full Gauss-Seidel sweep
  auto gauss_seidel_sweep = [&]() {
    for (uword k = 0; k < K; ++k) {
      mat &alpha_k = alpha[k];
      const uword *gk = map_ptrs[k];

      alpha_k.zeros();

      for (uword p = 0; p < P; p += block_size) {
        uword b_sz = std::min(block_size, P - p);

        double *ak_ptrs[4];
        const double *v_ptrs[4];
        for (uword j = 0; j < b_sz; ++j) {
          ak_ptrs[j] = alpha_k.colptr(p + j);
          v_ptrs[j] = V.colptr(p + j);
        }

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

      scale_rows_inplace(alpha_k, map.inv_weights[k]);
    }
  };

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Save current alpha[0] as X_it
    std::memcpy(X_it.memptr(), alpha[0].memptr(), total_elem0 * sizeof(double));

    // Gauss-Seidel sweep 1 -> GX
    gauss_seidel_sweep();
    std::memcpy(GX_it.memptr(), alpha[0].memptr(),
                total_elem0 * sizeof(double));

    // Gauss-Seidel sweep 2 -> G2X (alpha[0] is G2X)
    gauss_seidel_sweep();

    // Irons-Tuck acceleration on alpha[0]
    const double *x_ptr = X_it.memptr();
    const double *gx_ptr = GX_it.memptr();
    const double *ggx_ptr = alpha[0].memptr();

    double vprod = 0.0, ssq = 0.0;
    for (uword i = 0; i < total_elem0; ++i) {
      double dGX = ggx_ptr[i] - gx_ptr[i];
      double d2X = dGX - gx_ptr[i] + x_ptr[i];
      delta_GX[i] = dGX;
      delta2_X[i] = d2X;
      vprod += dGX * d2X;
      ssq += d2X * d2X;
    }

    bool numconv = false;
    if (ssq == 0.0) {
      numconv = true;
    } else {
      double coef = vprod / ssq;
      double *a0_ptr = alpha[0].memptr();
      for (uword i = 0; i < total_elem0; ++i) {
        a0_ptr[i] = ggx_ptr[i] - coef * delta_GX[i];
      }
    }

    // Post-acceleration projection
    if (iter >= iter_proj_after_acc) {
      gauss_seidel_sweep();
    }

    if (numconv)
      break;

    // Grand acceleration
    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        std::memcpy(grand_Y.memptr(), alpha[0].memptr(),
                    total_elem0 * sizeof(double));
        grand_stage = 1;
      } else if (grand_stage == 1) {
        std::memcpy(grand_GY.memptr(), alpha[0].memptr(),
                    total_elem0 * sizeof(double));
        grand_stage = 2;
      } else {
        const double *Y_ptr = grand_Y.memptr();
        const double *GY_ptr = grand_GY.memptr();
        const double *G2Y_ptr = alpha[0].memptr();

        double g_vprod = 0.0, g_ssq = 0.0;
        for (uword i = 0; i < total_elem0; ++i) {
          double dGX = G2Y_ptr[i] - GY_ptr[i];
          double d2X = dGX - GY_ptr[i] + Y_ptr[i];
          g_ssq += d2X * d2X;
          g_vprod += dGX * d2X;
        }

        if (g_ssq > 1e-14) {
          double coef = g_vprod / g_ssq;
          double *a0_ptr = alpha[0].memptr();
          for (uword i = 0; i < total_elem0; ++i) {
            double dGX = G2Y_ptr[i] - GY_ptr[i];
            a0_ptr[i] = G2Y_ptr[i] - coef * dGX;
          }
        }
        grand_stage = 0;
      }
    }

    // Convergence check: fixest-style per-coefficient criterion
    const double *curr = alpha[0].memptr();
    const double *old = X_it.memptr();
    bool keep_going = false;
    for (uword i = 0; i < total_elem0; ++i) {
      if (continue_crit(curr[i], old[i], tol)) {
        keep_going = true;
        break;
      }
    }
    if (!keep_going)
      break;

    // SSR-based stopping (periodic)
    if (iter > 0 && iter % ssr_check_period == 0) {
      double ssr = 0.0;
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          double r = v_col[i];
          for (uword k = 0; k < K; ++k) {
            r -= alpha[k].at(map_ptrs[k][i], p);
          }
          ssr += w_ptr[i] * r * r;
        }
      }
      if (stopping_crit(ssr_old, ssr, tol))
        break;
      ssr_old = ssr;
    }
  }

  // Final subtraction
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
inline void center_variables(mat &V, const vec &w, FlatFEMap &map,
                             CellAggregated2FE &cells, double tol,
                             uword max_iter, uword grand_acc_period) {
  if (V.is_empty() || map.K == 0)
    return;
  if (map.K == 2) {
    center_2fe(V, w, map, cells, tol, max_iter, grand_acc_period);
  } else {
    center_kfe(V, w, map, tol, max_iter, grand_acc_period);
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
