// Centering using flat observation-to-group mapping
//
// Two centering algorithms are available:
//
// 1. Stammann (default): Alternating projections with Irons-Tuck acceleration.
//    Each iteration performs a full Gauss-Seidel sweep updating each FE
//    dimension in sequence, then applies IT acceleration to the first FE
//    coefficient vector.
//
// 2. Berge: Fixed-point reformulation. For 2-FE: defines F(beta) = f_T(f_I(beta))
//    composing both updates into a single map, then solves beta* = F(beta*) using
//    Irons-Tuck acceleration on the composed iteration.
//
// Both methods use:
// - Precompute in_out[g] = sum_{i: fe[i]==g} w[i] * V[i] once (O(N))
// - Inner iteration operates on coefficient vectors only (O(n_coef))
// - 2-FE special case avoids N-length temporaries entirely
// - Irons-Tuck + grand acceleration on coefficient vectors
// - Post-acceleration projection
// - Warm-starting across IRLS calls
//
// Performance design:
// - No branching on P: always use colptr()/memptr() with a single code path
// - All inner-loop work is scatter/gather on index arrays
// - Armadillo vectorized mat ops for IT/grand acceleration

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Centering algorithm selection
enum CenteringMethod { STAMMANN = 0, BERGE = 1 };

// Convert string to CenteringMethod enum
inline CenteringMethod centering_from_string(const std::string &s) {
  if (s == "berge")
    return BERGE;
  return STAMMANN; // default
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

// Warm-start storage for centering across IRLS iterations
struct CenterWarmStart {
  std::vector<mat> alpha; // K coefficient matrices (n_groups[k] x P)
  uword K;
  uword P;
  bool valid;

  CenterWarmStart() : K(0), P(0), valid(false) {}

  void save(const std::vector<mat> &coeffs, uword n_fe, uword n_cols) {
    K = n_fe;
    P = n_cols;
    alpha.resize(K);
    for (uword k = 0; k < K; ++k) {
      alpha[k] = coeffs[k];
    }
    valid = true;
  }

  bool can_use(uword n_fe, uword n_cols) const {
    if (!valid || n_fe != K || n_cols != P)
      return false;
    return true;
  }
};

// Precompute in_out: in_out_k(g, p) = sum_{i: fe_k[i]==g} w[i] * V(i, p)
// This is done once per centering call (O(N * P * K))
inline void in_out_(std::vector<mat> &in_out, const mat &V, const double *w_ptr,
                    const FlatFEMap &map) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  in_out.resize(K);
  for (uword k = 0; k < K; ++k) {
    in_out[k].zeros(map.n_groups[k], P);
    const uword *gk = map.fe_map[k].data();
    for (uword p = 0; p < P; ++p) {
      const double *v_col = V.colptr(p);
      double *io_col = in_out[k].colptr(p);
      for (uword i = 0; i < n_obs; ++i) {
        io_col[gk[i]] += w_ptr[i] * v_col[i];
      }
    }
  }
}

// 2-FE Gauss-Seidel update: compute alpha_b from alpha_a + in_out
// alpha_b[g2] = (in_out_b[g2] - sum_{i: fe_b[i]==g2} w[i] * alpha_a[fe_a[i]])
//               / sw_b[g2]
//
// no N-length temporaries, operates directly on coefficient vectors. The O(N)
// inner loop is a simple scatter with one array read per observation.
inline void gs_update_2fe(mat &alpha_b, const mat &alpha_a, const mat &in_out_b,
                          const vec &inv_w_b, const uword *ga, const uword *gb,
                          const double *w_ptr, uword n_obs, uword P) {
  const uword n_b = alpha_b.n_rows;

  for (uword p = 0; p < P; ++p) {
    double *ab_col = alpha_b.colptr(p);
    const double *aa_col = alpha_a.colptr(p);
    const double *io_col = in_out_b.colptr(p);

    // Initialize from precomputed in_out
    std::memcpy(ab_col, io_col, n_b * sizeof(double));

    // Subtract weighted contribution of alpha_a
    for (uword i = 0; i < n_obs; ++i) {
      ab_col[gb[i]] -= w_ptr[i] * aa_col[ga[i]];
    }

    // Scale by inverse weights
    const double *iw = inv_w_b.memptr();
    for (uword g = 0; g < n_b; ++g) {
      ab_col[g] *= iw[g];
    }
  }
}

// General K-FE Gauss-Seidel update for one FE dimension k:
// alpha_k[g] = (in_out_k[g] - sum_{i: fe_k[i]==g} w[i] * sum_{j!=k}
// alpha_j[fe_j[i]]) / sw_k[g]
inline void gs_update_kfe(mat &alpha_k, const std::vector<mat> &alpha,
                          const mat &in_out_k, const vec &inv_w_k,
                          const FlatFEMap &map, const double *w_ptr, uword k,
                          uword n_obs, uword P) {
  const uword K = map.K;
  const uword n_k = alpha_k.n_rows;
  const uword *gk = map.fe_map[k].data();

  for (uword p = 0; p < P; ++p) {
    double *ak_col = alpha_k.colptr(p);
    const double *io_col = in_out_k.colptr(p);

    // Initialize from precomputed in_out
    std::memcpy(ak_col, io_col, n_k * sizeof(double));

    // Subtract weighted sum of other FE contributions
    for (uword i = 0; i < n_obs; ++i) {
      double sum_others = 0.0;
      for (uword j = 0; j < K; ++j) {
        if (j != k) {
          sum_others += alpha[j].at(map.fe_map[j].data()[i], p);
        }
      }
      ak_col[gk[i]] -= w_ptr[i] * sum_others;
    }

    // Scale by inverse weights
    const double *iw = inv_w_k.memptr();
    for (uword g = 0; g < n_k; ++g) {
      ak_col[g] *= iw[g];
    }
  }
}

// Irons-Tuck acceleration on coefficient vectors
// Returns true if numerically converged (ssq == 0)
inline bool irons_tuck_acc(mat &X_coef, const mat &GX_coef,
                           const mat &GGX_coef) {
  const mat delta_GX = GGX_coef - GX_coef;
  const mat delta2_X = delta_GX - GX_coef + X_coef;
  const double ssq = accu(square(delta2_X));

  if (ssq == 0.0) {
    return true;
  }

  const double coef = accu(delta_GX % delta2_X) / ssq;
  X_coef = GGX_coef - coef * delta_GX;
  return false;
}

// =========================================================================
// Stammann (alpaca-like): alternating projections with IT acceleration
// =========================================================================

// 2-FE centering (Stammann): alternating projections with Irons-Tuck
// acceleration. Each iteration does a full Gauss-Seidel sweep (update alpha1
// from alpha2, then alpha2 from alpha1). IT acceleration is applied to alpha1.
inline void center_2fe_stammann(mat &V, const vec &w, const FlatFEMap &map,
                               CenterWarmStart &warm, double tol,
                               uword max_iter,
                               uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = map.n_groups[0];
  const uword n2 = map.n_groups[1];

  const uword *g1 = map.fe_map[0].data();
  const uword *g2 = map.fe_map[1].data();
  const double *w_ptr = w.memptr();

  // Step 1: Precompute in_out (O(N * P * 2)) -- done ONCE
  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

  // Step 2: Initialize coefficient vectors
  mat alpha1, alpha2;
  if (warm.can_use(2, P)) {
    alpha1 = warm.alpha[0];
    alpha2 = warm.alpha[1];
  } else {
    alpha1.zeros(n1, P);
    alpha2.zeros(n2, P);
  }

  // Gauss-Seidel sweep: update alpha1 from alpha2, then alpha2 from alpha1
  // This is entirely in coefficient space -- no N-length temporaries
  auto gs_sweep = [&]() {
    gs_update_2fe(alpha1, alpha2, in_out[0], map.inv_weights[0], g2, g1, w_ptr,
                  n_obs, P);
    gs_update_2fe(alpha2, alpha1, in_out[1], map.inv_weights[1], g1, g2, w_ptr,
                  n_obs, P);
  };

  // IT acceleration buffers (coefficient-space, not observation-space)
  mat X_it(n1, P);
  mat GX_it(n1, P);

  // Grand acceleration buffers
  mat grand_Y(n1, P, fill::zeros);
  mat grand_GY(n1, P, fill::zeros);
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 50000) ? 80 : 40;
  double ssr_old = datum::inf;

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha1;

    // GS sweep 1 -> GX
    gs_sweep();
    GX_it = alpha1;

    // GS sweep 2 -> GGX
    gs_sweep();

    // IT acceleration on alpha1 (coefficient space)
    bool numconv = irons_tuck_acc(alpha1, GX_it, alpha1);

    // Recompute alpha2 from accelerated alpha1
    gs_update_2fe(alpha2, alpha1, in_out[1], map.inv_weights[1], g1, g2, w_ptr,
                  n_obs, P);

    if (iter >= iter_proj_after_acc) {
      gs_sweep();
    }

    if (numconv)
      break;

    // Grand acceleration
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
          gs_update_2fe(alpha2, alpha1, in_out[1], map.inv_weights[1], g1, g2,
                        w_ptr, n_obs, P);
        }
        grand_stage = 0;
      }
    }

    // Convergence check (coefficient-space)
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

    // SSR-based stopping (periodic)
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

  // Save warm-start
  std::vector<mat> coeffs = {alpha1, alpha2};
  warm.save(coeffs, 2, P);

  // Final subtraction: V -= G1 * alpha1 + G2 * alpha2
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *a1 = alpha1.colptr(p);
    const double *a2 = alpha2.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      v_col[i] -= a1[g1[i]] + a2[g2[i]];
    }
  }
}

// General K-FE centering (Stammann): alternating projections with Irons-Tuck
inline void center_kfe_stammann(mat &V, const vec &w, const FlatFEMap &map,
                               CenterWarmStart &warm, double tol,
                               uword max_iter,
                               uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  const double *w_ptr = w.memptr();

  // Precompute in_out
  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

  // Initialize coefficient vectors
  std::vector<mat> alpha(K);
  if (warm.can_use(K, P)) {
    for (uword k = 0; k < K; ++k) {
      alpha[k] = warm.alpha[k];
    }
  } else {
    for (uword k = 0; k < K; ++k) {
      alpha[k].zeros(map.n_groups[k], P);
    }
  }

  // Gauss-Seidel sweep: update each FE in sequence
  auto gs_sweep = [&]() {
    for (uword k = 0; k < K; ++k) {
      gs_update_kfe(alpha[k], alpha, in_out[k], map.inv_weights[k], map, w_ptr,
                    k, n_obs, P);
    }
  };

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

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha[0];

    // GS sweep 1 -> GX
    gs_sweep();
    GX_it = alpha[0];

    // GS sweep 2 -> GGX
    gs_sweep();

    // IT acceleration
    bool numconv = irons_tuck_acc(alpha[0], GX_it, alpha[0]);

    if (iter >= iter_proj_after_acc) {
      gs_sweep();
    }

    if (numconv)
      break;

    // Grand acceleration
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
      std::vector<const uword *> map_ptrs(K);
      for (uword k = 0; k < K; ++k) {
        map_ptrs[k] = map.fe_map[k].data();
      }

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

  // Save warm-start
  warm.save(alpha, K, P);

  // Final subtraction
  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

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

// =========================================================================
// Berge (fixest-like): fixed-point reformulation
//
// Instead of alternating projections (update alpha, then beta in sequence),
// compose all FE updates into a single map F = f_T o f_I and solve the
// fixed-point problem beta* = F(beta*) using Irons-Tuck acceleration.
//
// For 2-FE: F(beta) = f_2(f_1(beta)), where f_1 computes alpha from beta
// and f_2 computes beta from alpha.  The Irons-Tuck acceleration is applied
// directly to beta (the last FE dimension's coefficients).
// =========================================================================

// 2-FE centering (Berge): fixed-point F(beta) = f_2(f_1(beta)) with IT
// acceleration on beta. alpha is only reconstructed at the end.
//
// Optimizations vs. naive:
// - All buffers pre-allocated (zero heap allocs in the loop)
// - GX from previous iteration is reused (saves one F evaluation)
// - Grand acceleration uses full IT (3-snapshot Irons-Tuck)
// - Convergence checked on unaccelerated step (X vs GX) for stability

inline void center_2fe_berge(mat &V, const vec &w, const FlatFEMap &map,
                            CenterWarmStart &warm, double tol, uword max_iter,
                            uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = map.n_groups[0];
  const uword n2 = map.n_groups[1];
  const uword total_elem = n2 * P;

  const uword *g1 = map.fe_map[0].data();
  const uword *g2 = map.fe_map[1].data();
  const double *w_ptr = w.memptr();

  // Step 1: Precompute in_out (O(N * P * 2)) -- done ONCE
  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

  // Step 2: Initialize beta (= alpha2, the second FE's coefficients)
  mat beta;
  if (warm.can_use(2, P)) {
    beta = warm.alpha[1];
  } else {
    beta.zeros(n2, P);
  }

  // Pre-allocate ALL buffers outside the loop (zero allocations inside)
  mat alpha1_tmp(n1, P);  // scratch for f_I result
  mat GX(n2, P);          // G(X) = F(beta)
  mat GGX(n2, P);         // G(G(X)) = F(F(beta))
  mat X_it(n2, P);        // saved X for convergence check

  // Grand acceleration buffers (3-snapshot IT)
  mat grand_Y(n2, P, fill::zeros);
  mat grand_GY(n2, P, fill::zeros);
  mat grand_GGY(n2, P, fill::zeros);
  uword grand_stage = 0;

  // Apply F: given src (alpha2), compute alpha1 from src, then new alpha2.
  // Result written into dst (no allocation).
  auto apply_F = [&](const mat &src, mat &dst) {
    gs_update_2fe(alpha1_tmp, src, in_out[0], map.inv_weights[0], g2, g1,
                  w_ptr, n_obs, P);
    gs_update_2fe(dst, alpha1_tmp, in_out[1], map.inv_weights[1], g1, g2,
                  w_ptr, n_obs, P);
  };

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 50000) ? 80 : 40;
  double ssr_old = datum::inf;

  // Bootstrap: compute GX = F(beta) once before the loop
  apply_F(beta, GX);

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Convergence check on unaccelerated step: X vs G(X)
    // This is more stable than checking post-acceleration beta
    {
      const double *curr = GX.memptr();
      const double *old = beta.memptr();
      bool keep_going = false;
      for (uword i = 0; i < total_elem; ++i) {
        if (continue_crit(curr[i], old[i], tol)) {
          keep_going = true;
          break;
        }
      }
      if (!keep_going)
        break;
    }

    X_it = beta;

    // GGX = F(GX) -- only ONE new F evaluation needed (GX already computed)
    apply_F(GX, GGX);

    // IT acceleration: beta = GGX - coef * (GGX - GX)
    bool numconv = irons_tuck_acc(beta, GX, GGX);

    // Post-acceleration projection to stabilize
    if (iter >= iter_proj_after_acc) {
      apply_F(beta, GX);
      beta = GX;
    }

    if (numconv)
      break;

    // Grand acceleration (full 3-snapshot Irons-Tuck)
    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        grand_Y = beta;
        grand_stage = 1;
      } else if (grand_stage == 1) {
        grand_GY = beta;
        grand_stage = 2;
      } else {
        grand_GGY = beta;
        // Apply full IT to the slow sequence
        bool g_numconv = irons_tuck_acc(beta, grand_GY, grand_GGY);
        if (!g_numconv) {
          // Re-stabilize after grand acceleration
          apply_F(beta, GX);
          beta = GX;
        }
        grand_stage = 0;
      }
    }

    // Compute GX = F(beta) for next iteration (reused at top of loop)
    apply_F(beta, GX);

    // SSR-based stopping (periodic)
    if (iter > 0 && iter % ssr_check_period == 0) {
      // alpha1_tmp is already computed from the last apply_F call
      // (it holds f_I(beta)), so just evaluate SSR
      double ssr = 0.0;
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        const double *a1 = alpha1_tmp.colptr(p);
        const double *a2 = beta.colptr(p);
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

  // Reconstruct alpha1 from converged beta
  mat alpha1(n1, P);
  gs_update_2fe(alpha1, beta, in_out[0], map.inv_weights[0], g2, g1, w_ptr,
                n_obs, P);

  // Save warm-start
  std::vector<mat> coeffs = {alpha1, beta};
  warm.save(coeffs, 2, P);

  // Final subtraction: V -= G1 * alpha1 + G2 * beta
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *a1 = alpha1.colptr(p);
    const double *a2 = beta.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      v_col[i] -= a1[g1[i]] + a2[g2[i]];
    }
  }
}

// General K-FE centering (Berge): fixed-point reformulation.
// Compose all K updates into F: given alpha[K-1], compute alpha[0],...,alpha[K-2]
// in sequence, then recompute alpha[K-1]. Track only the last FE coefficient
// vector through the fixed-point iteration with IT acceleration.
//
// Optimizations vs. naive:
// - All buffers pre-allocated (zero heap allocs in the loop)
// - GX from previous iteration is reused (saves one F evaluation)
// - Grand acceleration uses full IT (3-snapshot Irons-Tuck)
// - Convergence checked on unaccelerated step (X vs GX) for stability
inline void center_kfe_berge(mat &V, const vec &w, const FlatFEMap &map,
                            CenterWarmStart &warm, double tol, uword max_iter,
                            uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  const double *w_ptr = w.memptr();

  // Precompute in_out
  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

  // Initialize all coefficient vectors (needed as scratch for F)
  std::vector<mat> alpha(K);
  if (warm.can_use(K, P)) {
    for (uword k = 0; k < K; ++k) {
      alpha[k] = warm.alpha[k];
    }
  } else {
    for (uword k = 0; k < K; ++k) {
      alpha[k].zeros(map.n_groups[k], P);
    }
  }

  // The last FE dimension is the "beta" we track through the fixed point.
  const uword last = K - 1;
  const uword n_last = map.n_groups[last];
  const uword total_elem_last = n_last * P;

  // Pre-allocate ALL buffers outside the loop (zero allocations inside)
  mat GX(n_last, P);           // G(X) = F(beta)
  mat GGX(n_last, P);          // G(G(X)) = F(F(beta))
  mat X_it(n_last, P);         // saved X for convergence check

  // Grand acceleration buffers (3-snapshot IT)
  mat grand_Y(n_last, P, fill::zeros);
  mat grand_GY(n_last, P, fill::zeros);
  mat grand_GGY(n_last, P, fill::zeros);
  uword grand_stage = 0;

  // Apply F: given beta_in, compute alpha[0..last-1] in sequence,
  // then recompute alpha[last] into dst. No allocation.
  auto apply_F = [&](const mat &beta_in, mat &dst) {
    alpha[last] = beta_in;
    // Forward sweep: update alpha[0], alpha[1], ..., alpha[last-1]
    for (uword k = 0; k < last; ++k) {
      gs_update_kfe(alpha[k], alpha, in_out[k], map.inv_weights[k], map, w_ptr,
                    k, n_obs, P);
    }
    // Recompute alpha[last] from updated alpha[0..last-1]
    gs_update_kfe(dst, alpha, in_out[last], map.inv_weights[last], map,
                  w_ptr, last, n_obs, P);
  };

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 100000) ? 80 : 40;
  double ssr_old = datum::inf;

  mat beta = alpha[last];

  // Bootstrap: compute GX = F(beta) once before the loop
  apply_F(beta, GX);

  for (uword iter = 0; iter < max_iter; ++iter) {
    // Convergence check on unaccelerated step: X vs G(X)
    {
      const double *curr = GX.memptr();
      const double *old = beta.memptr();
      bool keep_going = false;
      for (uword i = 0; i < total_elem_last; ++i) {
        if (continue_crit(curr[i], old[i], tol)) {
          keep_going = true;
          break;
        }
      }
      if (!keep_going)
        break;
    }

    X_it = beta;

    // GGX = F(GX) -- only ONE new F evaluation needed (GX already computed)
    apply_F(GX, GGX);

    // IT acceleration: beta = GGX - coef * (GGX - GX)
    bool numconv = irons_tuck_acc(beta, GX, GGX);

    // Post-acceleration projection to stabilize
    if (iter >= iter_proj_after_acc) {
      apply_F(beta, GX);
      beta = GX;
    }

    if (numconv)
      break;

    // Grand acceleration (full 3-snapshot Irons-Tuck)
    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        grand_Y = beta;
        grand_stage = 1;
      } else if (grand_stage == 1) {
        grand_GY = beta;
        grand_stage = 2;
      } else {
        grand_GGY = beta;
        // Apply full IT to the slow sequence
        bool g_numconv = irons_tuck_acc(beta, grand_GY, grand_GGY);
        if (!g_numconv) {
          // Re-stabilize after grand acceleration
          apply_F(beta, GX);
          beta = GX;
        }
        grand_stage = 0;
      }
    }

    // Compute GX = F(beta) for next iteration (reused at top of loop)
    apply_F(beta, GX);

    // SSR-based stopping (periodic)
    if (iter > 0 && iter % ssr_check_period == 0) {
      // alpha[] is up-to-date from the last apply_F call
      double ssr = 0.0;
      std::vector<const uword *> map_ptrs(K);
      for (uword k = 0; k < K; ++k) {
        map_ptrs[k] = map.fe_map[k].data();
      }

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

  // Final reconstruction of all alpha from converged beta
  alpha[last] = beta;
  for (uword k = 0; k < last; ++k) {
    gs_update_kfe(alpha[k], alpha, in_out[k], map.inv_weights[k], map, w_ptr,
                  k, n_obs, P);
  }

  // Save warm-start
  warm.save(alpha, K, P);

  // Final subtraction
  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

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
inline void center_variables(mat &V, const vec &w, FlatFEMap &map, double tol,
                             uword max_iter, uword grand_acc_period,
                             CenterWarmStart *warm = nullptr,
                             CenteringMethod method = STAMMANN) {
  if (V.is_empty() || map.K == 0)
    return;

  CenterWarmStart local_warm;
  CenterWarmStart &ws = warm ? *warm : local_warm;

  if (method == STAMMANN) {
    if (map.K == 2) {
      center_2fe_stammann(V, w, map, ws, tol, max_iter, grand_acc_period);
    } else {
      center_kfe_stammann(V, w, map, ws, tol, max_iter, grand_acc_period);
    }
  } else {
    if (map.K == 2) {
      center_2fe_berge(V, w, map, ws, tol, max_iter, grand_acc_period);
    } else {
      center_kfe_berge(V, w, map, ws, tol, max_iter, grand_acc_period);
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
