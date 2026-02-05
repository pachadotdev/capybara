// Centering using flat observation-to-group mapping
#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Flat structure for FE mapping:
// - fe_map: umat (n_obs x K) where fe_map(i,k) = group index for obs i in FE k
// - n_groups: uvec (K) number of groups per FE
// - inv_weights: mat (max_groups x K) precomputed 1/sum(w) per group
//
// This is cache-friendly: column-major access for each FE sweep

struct FlatFEMap {
  umat fe_map;      // (n_obs x K): fe_map(i,k) = group of obs i for FE k
  uvec n_groups;    // (K): number of groups per FE
  mat inv_weights;  // (max_groups x K): inverse weights, padded with 0
  uword n_obs;
  uword K;
  uword max_groups;

  // Build from legacy field<field<uvec>> format
  void build(const field<field<uvec>> &group_indices) {
    K = group_indices.n_elem;
    if (K == 0) return;

    // Find n_obs and max_groups
    n_groups.set_size(K);
    max_groups = 0;
    n_obs = 0;

    for (uword k = 0; k < K; ++k) {
      n_groups(k) = group_indices(k).n_elem;
      max_groups = std::max(max_groups, n_groups(k));
      for (uword g = 0; g < n_groups(k); ++g) {
        const uvec &idx = group_indices(k)(g);
        if (!idx.is_empty()) {
          n_obs = std::max(n_obs, idx.max() + 1);
        }
      }
    }

    // Allocate flat structures
    fe_map.set_size(n_obs, K);
    inv_weights.zeros(max_groups, K);

    // Fill fe_map: for each FE k, for each group g, set fe_map(obs, k) = g
    for (uword k = 0; k < K; ++k) {
      for (uword g = 0; g < n_groups(k); ++g) {
        const uvec &idx = group_indices(k)(g);
        fe_map.elem(idx + k * n_obs).fill(g);  // Column-major indexing
      }
    }
  }

  // Update inverse weights given observation weights w
  void update_weights(const vec &w) {
    if (K == 0) return;

    inv_weights.zeros();
    bool use_w = (w.n_elem == n_obs);

    for (uword k = 0; k < K; ++k) {
      vec group_w(n_groups(k), fill::zeros);
      const uword *col_k = fe_map.colptr(k);

      if (use_w) {
        for (uword i = 0; i < n_obs; ++i) {
          group_w(col_k[i]) += w(i);
        }
      } else {
        for (uword i = 0; i < n_obs; ++i) {
          group_w(col_k[i]) += 1.0;
        }
      }

      // Invert with threshold
      double *inv_col = inv_weights.colptr(k);
      for (uword g = 0; g < n_groups(k); ++g) {
        inv_col[g] = (group_w(g) > 1e-12) ? 1.0 / group_w(g) : 0.0;
      }
    }
  }
};

// Compute weighted group means for one FE
// alpha(g, :) = inv_w(g) * sum_{i: fe_map(i,k)=g} w(i) * R(i, :)
inline void compute_group_means(mat &alpha, const mat &R, const vec &w,
                                const uword *fe_col, uword n_grp,
                                const double *inv_w) {
  const uword n_obs = R.n_rows;
  const uword P = R.n_cols;

  alpha.zeros(n_grp, P);

  // Scatter-add weighted rows
  for (uword i = 0; i < n_obs; ++i) {
    alpha.row(fe_col[i]) += w(i) * R.row(i);
  }

  // Apply inverse weights
  for (uword g = 0; g < n_grp; ++g) {
    alpha.row(g) *= inv_w[g];
  }
}

// Expand group means to observation level
inline mat expand_to_obs(const mat &alpha, const uword *fe_col, uword n_obs) {
  mat result(n_obs, alpha.n_cols);
  for (uword i = 0; i < n_obs; ++i) {
    result.row(i) = alpha.row(fe_col[i]);
  }
  return result;
}

// Irons-Tuck acceleration coefficient
inline double irons_tuck_coef(const mat &x2, const mat &x1, const mat &x0) {
  mat D2 = x2 - x1;
  mat DD = D2 - (x1 - x0);
  double ssq = accu(square(DD));
  if (ssq < 1e-14) return 0.0;
  double coef = accu(D2 % DD) / ssq;
  return (coef > 0.0 && coef < 2.0) ? coef : 0.0;
}

// Main centering algorithm
inline void center_impl(mat &V, const vec &w, const FlatFEMap &map,
                        double tol, uword max_iter) {
  const uword K = map.K;
  const uword P = V.n_cols;
  const uword n_obs = map.n_obs;

  // Alpha storage per FE
  field<mat> alpha(K), alpha_prev(K), alpha_prev2(K);
  for (uword k = 0; k < K; ++k) {
    uword ng = map.n_groups(k);
    alpha(k).zeros(ng, P);
    alpha_prev(k).zeros(ng, P);
    alpha_prev2(k).zeros(ng, P);
  }

  mat R = V;  // Working residual

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Store old values for convergence and acceleration
    field<mat> alpha_old(K);
    for (uword k = 0; k < K; ++k) {
      alpha_old(k) = alpha(k);
    }

    // Gauss-Seidel sweep through all FEs
    for (uword k = 0; k < K; ++k) {
      const uword *fe_col = map.fe_map.colptr(k);
      const double *inv_w = map.inv_weights.colptr(k);
      uword ng = map.n_groups(k);

      // Add back current FE contribution
      R += expand_to_obs(alpha(k), fe_col, n_obs);

      // Compute new group means
      compute_group_means(alpha(k), R, w, fe_col, ng, inv_w);

      // Remove updated FE contribution
      R -= expand_to_obs(alpha(k), fe_col, n_obs);
    }

    // Irons-Tuck acceleration after warmup iterations
    if (iter >= 3) {
      bool accelerated = false;
      for (uword k = 0; k < K; ++k) {
        double coef = irons_tuck_coef(alpha(k), alpha_prev(k), alpha_prev2(k));
        if (coef > 0.0) {
          alpha(k) -= coef * (alpha(k) - alpha_prev(k));
          accelerated = true;
        }
      }
      // Recompute residual if acceleration was applied
      if (accelerated) {
        R = V;
        for (uword k = 0; k < K; ++k) {
          R -= expand_to_obs(alpha(k), map.fe_map.colptr(k), n_obs);
        }
      }
    }

    // Update history for acceleration (after full sweep)
    for (uword k = 0; k < K; ++k) {
      alpha_prev2(k) = alpha_prev(k);
      alpha_prev(k) = alpha_old(k);
    }

    // Convergence check on first FE
    double diff = norm(alpha(0) - alpha_old(0), "fro");
    if (diff < tol) break;
  }

  V = R;
}

// Public interface: center with prebuilt FE map
inline void center_variables(mat &V, const vec &w, const FlatFEMap &map,
                             double tol, uword max_iter) {
  if (V.is_empty() || map.K == 0) return;
  center_impl(V, w, map, tol, max_iter);
}

// Public interface: center with group indices (builds map internally)
inline void center_variables(mat &V, const vec &w,
                             const field<field<uvec>> &group_indices,
                             double tol, uword max_iter) {
  if (V.is_empty() || group_indices.n_elem == 0) return;
  FlatFEMap map;
  map.build(group_indices);
  map.update_weights(w);
  center_impl(V, w, map, tol, max_iter);
}

// Build map from group indices (helper for callers who reuse the map)
inline FlatFEMap build_fe_map(const field<field<uvec>> &group_indices,
                              const vec &w) {
  FlatFEMap map;
  map.build(group_indices);
  map.update_weights(w);
  return map;
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
