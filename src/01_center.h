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
//
// Performance design:
// - No branching on P: always use colptr()/memptr() with a single code path
// - Gather/scatter on index arrays, Armadillo dense ops on intermediates
// - Armadillo vectorized mat ops for IT/grand acceleration (accu, square, %)
// - Combined [y | X] centering avoids duplicate iteration overhead

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// In-place row scaling for all columns: A(i,j) *= scale[i] for all j
// Uses colptr() for direct column memory access
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

  // Aggregate weighted V values into cells: cell_wV = S' * diag(w) * V
  // where S is the sparse (n_obs x n_cells) scatter matrix.
  // Single code path using colptr() — no branching on P.
  void aggregate_wV(const mat &V, const double *w_ptr, uword n_obs) {
    const uword P = V.n_cols;
    cell_wV.zeros(n_cells, P);

    const uword *otc = obs_to_cell.data();

    // Per-column scatter: the inner loop is a pure gather-multiply-scatter
    for (uword p = 0; p < P; ++p) {
      const double *v_col = V.colptr(p);
      double *cwv_col = cell_wV.colptr(p);
      for (uword i = 0; i < n_obs; ++i) {
        cwv_col[otc[i]] += w_ptr[i] * v_col[i];
      }
    }
  }
};

// Gauss-Seidel half-step for 2FE centering (gather -> BLAS subtract -> scatter)
//
// Computes alpha_k = diag(inv_w) * G_k' * (cell_wV - diag(cw) * G_other *
// alpha_other)
//
// Decomposed into BLAS-friendly pieces:
//   1. Gather + scale: temp(c, p) = cw[c] * alpha_other(cg_other[c], p)
//   2. Dense subtract:  temp = cell_wV - temp                (Armadillo mat op)
//   3. Scatter-accumulate: alpha_k(cg_k[c], :) += temp(c, :) (index-driven)
//   4. Scale:  alpha_k = diag(inv_w) * alpha_k               (dense)
//
// Single code path — no branching on P.

inline void gauss_seidel_half_step(mat &alpha_k, const mat &alpha_other,
                                   const CellAggregated2FE &cells,
                                   const vec &inv_w, uword P,
                                   bool k_is_first, mat &temp) {
  const uword nc = cells.n_cells;
  const uword *cg_k = k_is_first ? cells.cell_g1.data() : cells.cell_g2.data();
  const uword *cg_other =
      k_is_first ? cells.cell_g2.data() : cells.cell_g1.data();
  const double *cw = cells.cell_w.data();

  // 1. Gather alpha_other into temp, scaled by cell weight
  //    temp(c, p) = cw[c] * alpha_other(cg_other[c], p)
  temp.set_size(nc, P);
  for (uword p = 0; p < P; ++p) {
    double *t_col = temp.colptr(p);
    const double *ao_col = alpha_other.colptr(p);
    for (uword c = 0; c < nc; ++c) {
      t_col[c] = cw[c] * ao_col[cg_other[c]];
    }
  }

  // 2. Dense matrix subtraction: residual = cell_wV - temp
  //    This is a pure BLAS daxpy on (nc × P) dense memory
  temp = cells.cell_wV - temp;

  // 3. Scatter-accumulate residual into alpha_k
  alpha_k.zeros();
  for (uword p = 0; p < P; ++p) {
    double *ak_col = alpha_k.colptr(p);
    const double *t_col = temp.colptr(p);
    for (uword c = 0; c < nc; ++c) {
      ak_col[cg_k[c]] += t_col[c];
    }
  }

  // 4. Scale by inverse weights: dense row-wise scaling
  scale_rows_inplace(alpha_k, inv_w);
}

// Recompute alpha2 from alpha1 using cell-aggregated data
inline void recompute_alpha2(mat &alpha2, const mat &alpha1,
                             const CellAggregated2FE &cells,
                             const FlatFEMap &map, uword P, mat &temp) {
  gauss_seidel_half_step(alpha2, alpha1, cells, map.inv_weights[1], P, false,
                         temp);
}

// 2FE centering with cell aggregation, Irons-Tuck + grand acceleration,
// post-acceleration projection, and fixest-style convergence.
//
// All acceleration uses Armadillo vectorized mat arithmetic (accu, square, %).
// No branching on P anywhere.
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

  if (!cells.structure_built) {
    cells.build_structure(g1, g2, n_obs, n1, n2);
  }

  cells.update_cell_weights(w_ptr, n_obs);
  cells.aggregate_wV(V, w_ptr, n_obs);

  const uword nc = cells.n_cells;

  mat alpha1(n1, P, fill::zeros);
  mat alpha2(n2, P, fill::zeros);

  // Reusable temp buffer for gauss_seidel_half_step (avoids re-alloc)
  mat temp;

  // IT acceleration buffers
  mat X_it(n1, P, fill::zeros);
  mat GX_it(n1, P, fill::zeros);

  // Grand acceleration buffers
  mat grand_Y(n1, P, fill::zeros);
  mat grand_GY(n1, P, fill::zeros);
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (nc > 50000) ? 80 : 40;
  double ssr_old = datum::inf;

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha1;

    // === Gauss-Seidel sweep 1 -> GX ===
    gauss_seidel_half_step(alpha1, alpha2, cells, map.inv_weights[0], P, true,
                           temp);
    recompute_alpha2(alpha2, alpha1, cells, map, P, temp);

    GX_it = alpha1;

    // === Gauss-Seidel sweep 2 -> G2X ===
    gauss_seidel_half_step(alpha1, alpha2, cells, map.inv_weights[0], P, true,
                           temp);
    recompute_alpha2(alpha2, alpha1, cells, map, P, temp);

    // === Irons-Tuck acceleration (Armadillo vectorized mat ops) ===
    const mat delta_GX = alpha1 - GX_it;
    const mat delta2_X = delta_GX - GX_it + X_it;

    const double ssq = accu(square(delta2_X));
    bool numconv = false;

    if (ssq == 0.0) {
      numconv = true;
    } else {
      const double coef = accu(delta_GX % delta2_X) / ssq;
      alpha1 -= coef * delta_GX;
    }

    recompute_alpha2(alpha2, alpha1, cells, map, P, temp);

    if (iter >= iter_proj_after_acc) {
      gauss_seidel_half_step(alpha1, alpha2, cells, map.inv_weights[0], P, true,
                             temp);
      recompute_alpha2(alpha2, alpha1, cells, map, P, temp);
    }

    if (numconv)
      break;

    // === Grand acceleration (Armadillo vectorized mat ops) ===
    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        grand_Y = alpha1;
        grand_stage = 1;
      } else if (grand_stage == 1) {
        grand_GY = alpha1;
        grand_stage = 2;
      } else {
        const mat g_delta = alpha1 - grand_GY;
        const mat g_delta2 = g_delta - grand_GY + grand_Y;
        const double g_ssq = accu(square(g_delta2));

        if (g_ssq > 1e-14) {
          const double coef = accu(g_delta % g_delta2) / g_ssq;
          alpha1 -= coef * g_delta;
          recompute_alpha2(alpha2, alpha1, cells, map, P, temp);
        }
        grand_stage = 0;
      }
    }

    // === Convergence check ===
    const double *curr = alpha1.memptr();
    const double *old = X_it.memptr();
    const uword total_elem = n1 * P;
    bool keep_going = false;
    for (uword i = 0; i < total_elem; ++i) {
      if (continue_crit(curr[i], old[i], tol)) {
        keep_going = true;
        break;
      }
    }
    if (!keep_going)
      break;

    // === SSR-based stopping (periodic) ===
    if (iter > 0 && iter % ssr_check_period == 0) {
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

  // Final subtraction: V -= G1 * alpha1 + G2 * alpha2
  // Single code path, colptr() access
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *a1 = alpha1.colptr(p);
    const double *a2 = alpha2.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      v_col[i] -= a1[g1[i]] + a2[g2[i]];
    }
  }
}

// General K-FE centering — single colptr() code path, Armadillo mat ops
// for acceleration. No block_size branching.
inline void center_kfe(mat &V, const vec &w, const FlatFEMap &map, double tol,
                       uword max_iter, uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  std::vector<mat> alpha(K);
  for (uword k = 0; k < K; ++k) {
    alpha[k].zeros(map.n_groups[k], P);
  }

  const double *w_ptr = w.memptr();

  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

  // IT acceleration on alpha[0]
  const uword n0 = map.n_groups[0];
  const uword total_elem0 = n0 * P;
  mat X_it(n0, P, fill::zeros);
  mat GX_it(n0, P, fill::zeros);

  // Grand acceleration buffers
  mat grand_Y(n0, P, fill::zeros);
  mat grand_GY(n0, P, fill::zeros);
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 100000) ? 80 : 40;
  double ssr_old = datum::inf;

  // Gauss-Seidel sweep: single colptr() code path
  auto gauss_seidel_sweep = [&]() {
    for (uword k = 0; k < K; ++k) {
      mat &alpha_k = alpha[k];
      const uword *gk = map_ptrs[k];

      alpha_k.zeros();

      for (uword p = 0; p < P; ++p) {
        double *ak_col = alpha_k.colptr(p);
        const double *v_col = V.colptr(p);

        for (uword i = 0; i < n_obs; ++i) {
          const double wi = w_ptr[i];
          if (wi > 1e-14) {
            double sum_others = 0.0;
            for (uword o = 0; o < K; ++o) {
              if (o != k) {
                sum_others += alpha[o].at(map_ptrs[o][i], p);
              }
            }
            ak_col[gk[i]] += wi * (v_col[i] - sum_others);
          }
        }
      }

      scale_rows_inplace(alpha_k, map.inv_weights[k]);
    }
  };

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha[0];

    // Gauss-Seidel sweep 1 -> GX
    gauss_seidel_sweep();
    GX_it = alpha[0];

    // Gauss-Seidel sweep 2 -> G2X
    gauss_seidel_sweep();

    // IT acceleration (Armadillo vectorized mat ops)
    const mat delta_GX = alpha[0] - GX_it;
    const mat delta2_X = delta_GX - GX_it + X_it;
    const double ssq = accu(square(delta2_X));

    bool numconv = false;
    if (ssq == 0.0) {
      numconv = true;
    } else {
      const double coef = accu(delta_GX % delta2_X) / ssq;
      alpha[0] -= coef * delta_GX;
    }

    if (iter >= iter_proj_after_acc) {
      gauss_seidel_sweep();
    }

    if (numconv)
      break;

    // Grand acceleration (Armadillo vectorized mat ops)
    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        grand_Y = alpha[0];
        grand_stage = 1;
      } else if (grand_stage == 1) {
        grand_GY = alpha[0];
        grand_stage = 2;
      } else {
        const mat g_delta = alpha[0] - grand_GY;
        const mat g_delta2 = g_delta - grand_GY + grand_Y;
        const double g_ssq = accu(square(g_delta2));

        if (g_ssq > 1e-14) {
          const double coef = accu(g_delta % g_delta2) / g_ssq;
          alpha[0] -= coef * g_delta;
        }
        grand_stage = 0;
      }
    }

    // Convergence check
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

  // Final subtraction — single colptr() code path
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      double sum_a = 0.0;
      for (uword k = 0; k < K; ++k) {
        sum_a += alpha[k].at(map_ptrs[k][i], p);
      }
      v_col[i] -= sum_a;
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