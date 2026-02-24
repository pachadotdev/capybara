// Centering using flat observation-to-group mapping
//
// Two centering algorithms are available:
//
// 1. Stammann (default): Alternating projections with Irons-Tuck acceleration.
//    Each iteration performs a full Gauss-Seidel sweep updating each FE
//    dimension in sequence, then applies IT acceleration to the first FE
//    coefficient vector. This works on the coefficient space.
//
// 2. Berge: Fixed-point reformulation. For 2-FE: defines
//    F(alpha) = f_1(f_2(alpha)), composing both updates into a single map,
//    then solves alpha* = F(alpha*) using Irons-Tuck acceleration. For K>=3
//    FE: uses a full backward Gauss-Seidel sweep as the composed map F, with
//    IT acceleration on the first FE's coefficients.
//
// Both methods precompute in_out[g] = sum_{i: fe[i]==g} w[i] * V[i] once,
// then iterate on coefficient vectors only. Warm-starting is used across
// IRLS calls.

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
  bool structure_built;

  FlatFEMap() : n_obs(0), K(0), structure_built(false) {}

  void build(const field<field<uvec>> &group_indices) {
    K = group_indices.n_elem;
    if (K == 0)
      return;

    n_groups.resize(K);
    n_obs = 0;

    for (uword k = 0; k < K; ++k) {
      n_groups[k] = group_indices(k).n_elem;
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        if (idx.n_elem > 0) {
          n_obs = std::max(n_obs, idx.max() + 1);
        }
      }
    }

    fe_map.resize(K);
    for (uword k = 0; k < K; ++k) {
      fe_map[k].assign(n_obs, 0);
    }

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

      if (use_w) {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += w_ptr[i];
        }
      } else {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += 1.0;
        }
      }

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

  // Persistent scratch buffers (sized on first use, reused if dimensions match)
  std::vector<mat> scratch_mats;
  mat scratch_beta; // beta_tmp for 2-FE case
  bool scratch_valid;
  uword scratch_n1, scratch_n2;

  CenterWarmStart()
      : K(0), P(0), valid(false), scratch_valid(false), scratch_n1(0),
        scratch_n2(0) {}

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

  void ensure_scratch_2fe(uword n1, uword n2, uword p) {
    if (scratch_valid && scratch_n1 == n1 && scratch_n2 == n2 &&
        scratch_mats.size() >= 6 && scratch_mats[0].n_cols == p) {
      scratch_mats[3].zeros();
      scratch_mats[4].zeros();
      scratch_mats[5].zeros();
      return;
    }
    scratch_mats.resize(6);
    scratch_mats[0].set_size(n1, p);
    scratch_mats[1].set_size(n1, p);
    scratch_mats[2].set_size(n1, p);
    scratch_mats[3].zeros(n1, p);
    scratch_mats[4].zeros(n1, p);
    scratch_mats[5].zeros(n1, p);
    scratch_beta.set_size(n2, p);
    scratch_n1 = n1;
    scratch_n2 = n2;
    scratch_valid = true;
  }
};

// Precompute weighted group sums: in_out_k(g, p) = sum_{i: fe_k[i]==g} w[i] *
// V(i, p)
inline void in_out_(std::vector<mat> &in_out, const mat &V, const double *w_ptr,
                    const FlatFEMap &map) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  in_out.resize(K);
  for (uword k = 0; k < K; ++k) {
    in_out[k].zeros(map.n_groups[k], P);
    const uword *gk = map.fe_map[k].data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
    for (uword p = 0; p < P; ++p) {
      const double *v_col = V.colptr(p);
      double *io_col = in_out[k].colptr(p);
      for (uword i = 0; i < n_obs; ++i) {
        io_col[gk[i]] += w_ptr[i] * v_col[i];
      }
    }
  }
}

// 2-FE Gauss-Seidel: alpha_b[g] = (in_out_b[g] - sum w[i]*alpha_a[ga[i]]) /
// sw_b[g]
inline void gs_update_2fe(mat &alpha_b, const mat &alpha_a, const mat &in_out_b,
                          const vec &inv_w_b, const uword *__restrict__ ga,
                          const uword *__restrict__ gb,
                          const double *__restrict__ w_ptr, uword n_obs,
                          uword P) {
  const uword n_b = alpha_b.n_rows;
  const double *__restrict__ iw = inv_w_b.memptr();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *__restrict__ ab_col = alpha_b.colptr(p);
    const double *__restrict__ aa_col = alpha_a.colptr(p);
    const double *__restrict__ io_col = in_out_b.colptr(p);

    std::memcpy(ab_col, io_col, n_b * sizeof(double));
    for (uword i = 0; i < n_obs; ++i) {
      ab_col[gb[i]] -= w_ptr[i] * aa_col[ga[i]];
    }

    for (uword g = 0; g < n_b; ++g) {
      ab_col[g] *= iw[g];
    }
  }
}

// K-FE Gauss-Seidel: alpha_k[g] = (in_out_k[g] - sum w[i]*sum_{j!=k}
// alpha_j[fe_j[i]]) / sw_k[g]
inline void gs_update_kfe(mat &alpha_k, const std::vector<mat> &alpha,
                          const mat &in_out_k, const vec &inv_w_k,
                          const FlatFEMap &map,
                          const double *__restrict__ w_ptr, uword k,
                          uword n_obs, uword P) {
  const uword K = map.K;
  const uword n_k = alpha_k.n_rows;
  const uword *__restrict__ gk = map.fe_map[k].data();
  const double *__restrict__ iw = inv_w_k.memptr();

  std::vector<const uword *> fe_ptrs(K);
  for (uword j = 0; j < K; ++j) {
    fe_ptrs[j] = map.fe_map[j].data();
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *__restrict__ ak_col = alpha_k.colptr(p);
    const double *__restrict__ io_col = in_out_k.colptr(p);

    std::vector<const double *> alpha_cols(K);
    for (uword j = 0; j < K; ++j) {
      alpha_cols[j] = alpha[j].colptr(p);
    }

    std::memcpy(ak_col, io_col, n_k * sizeof(double));

    for (uword i = 0; i < n_obs; ++i) {
      double sum_others = 0.0;
      for (uword j = 0; j < K; ++j) {
        if (j != k) {
          sum_others += alpha_cols[j][fe_ptrs[j][i]];
        }
      }
      ak_col[gk[i]] -= w_ptr[i] * sum_others;
    }

    for (uword g = 0; g < n_k; ++g) {
      ak_col[g] *= iw[g];
    }
  }
}

// Irons-Tuck acceleration on coefficient vectors.
// Returns true if numerically converged (ssq == 0).
inline bool irons_tuck_acc(mat &X_coef, const mat &GX_coef,
                           const mat &GGX_coef) {
  const uword n = X_coef.n_elem;
  double *__restrict__ x = X_coef.memptr();
  const double *__restrict__ gx = GX_coef.memptr();
  const double *__restrict__ ggx = GGX_coef.memptr();

  double vprod = 0.0, ssq = 0.0;
  for (uword i = 0; i < n; ++i) {
    const double dg = ggx[i] - gx[i];
    const double d2 = dg - gx[i] + x[i];
    vprod += dg * d2;
    ssq += d2 * d2;
  }

  if (ssq == 0.0) {
    return true;
  }

  const double coef = vprod / ssq;
  for (uword i = 0; i < n; ++i) {
    x[i] = ggx[i] - coef * (ggx[i] - gx[i]);
  }
  return false;
}

// Stammann 2-FE centering: alternating projections with IT acceleration
inline void center_2fe_stammann(mat &V, const vec &w, const FlatFEMap &map,
                                CenterWarmStart &warm, double tol,
                                uword max_iter, uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = map.n_groups[0];
  const uword n2 = map.n_groups[1];

  const uword *g1 = map.fe_map[0].data();
  const uword *g2 = map.fe_map[1].data();
  const double *w_ptr = w.memptr();

  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

  mat alpha1, alpha2;
  if (warm.can_use(2, P)) {
    alpha1 = warm.alpha[0];
    alpha2 = warm.alpha[1];
  } else {
    alpha1.zeros(n1, P);
    alpha2.zeros(n2, P);
  }

  auto gs_sweep = [&]() {
    gs_update_2fe(alpha1, alpha2, in_out[0], map.inv_weights[0], g2, g1, w_ptr,
                  n_obs, P);
    gs_update_2fe(alpha2, alpha1, in_out[1], map.inv_weights[1], g1, g2, w_ptr,
                  n_obs, P);
  };

  mat X_it(n1, P);
  mat GX_it(n1, P);

  mat grand_Y(n1, P, fill::zeros);
  mat grand_GY(n1, P, fill::zeros);
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 50000) ? 80 : 40;
  double ssr_old = datum::inf;

  mat GGX_it(n1, P);

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha1;

    gs_sweep();
    GX_it = alpha1;

    gs_sweep();
    GGX_it = alpha1;

    bool numconv = irons_tuck_acc(alpha1, GX_it, GGX_it);

    gs_update_2fe(alpha2, alpha1, in_out[1], map.inv_weights[1], g1, g2, w_ptr,
                  n_obs, P);

    if (iter >= iter_proj_after_acc) {
      gs_sweep();
    }

    if (numconv)
      break;

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

    if (iter > 0 && iter % ssr_check_period == 0) {
      double ssr = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : ssr) if (P > 1)
#endif
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

  std::vector<mat> coeffs = {alpha1, alpha2};
  warm.save(coeffs, 2, P);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *a1 = alpha1.colptr(p);
    const double *a2 = alpha2.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      v_col[i] -= a1[g1[i]] + a2[g2[i]];
    }
  }
}

// Stammann K-FE centering: alternating projections with IT acceleration
inline void center_kfe_stammann(mat &V, const vec &w, const FlatFEMap &map,
                                CenterWarmStart &warm, double tol,
                                uword max_iter, uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  const double *w_ptr = w.memptr();

  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

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

  auto gs_sweep = [&]() {
    for (uword k = 0; k < K; ++k) {
      gs_update_kfe(alpha[k], alpha, in_out[k], map.inv_weights[k], map, w_ptr,
                    k, n_obs, P);
    }
  };

  const uword n0 = map.n_groups[0];
  const uword total_elem0 = n0 * P;
  mat X_it(n0, P, fill::zeros);
  mat GX_it(n0, P, fill::zeros);
  mat GGX_it(n0, P, fill::zeros);

  mat grand_Y(n0, P, fill::zeros);
  mat grand_GY(n0, P, fill::zeros);
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 100000) ? 80 : 40;
  double ssr_old = datum::inf;

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha[0];

    gs_sweep();
    GX_it = alpha[0];

    gs_sweep();
    GGX_it = alpha[0];

    bool numconv = irons_tuck_acc(alpha[0], GX_it, GGX_it);

    if (iter >= iter_proj_after_acc) {
      gs_sweep();
    }

    if (numconv)
      break;

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

    if (iter > 0 && iter % ssr_check_period == 0) {
      double ssr = 0.0;
      std::vector<const uword *> map_ptrs(K);
      for (uword k = 0; k < K; ++k) {
        map_ptrs[k] = map.fe_map[k].data();
      }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : ssr) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        std::vector<const double *> alpha_cols(K);
        for (uword k = 0; k < K; ++k)
          alpha_cols[k] = alpha[k].colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          double r = v_col[i];
          for (uword k = 0; k < K; ++k) {
            r -= alpha_cols[k][map_ptrs[k][i]];
          }
          ssr += w_ptr[i] * r * r;
        }
      }
      if (stopping_crit(ssr_old, ssr, tol))
        break;
      ssr_old = ssr;
    }
  }

  warm.save(alpha, K, P);

  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    std::vector<const double *> alpha_cols(K);
    for (uword k = 0; k < K; ++k)
      alpha_cols[k] = alpha[k].colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      double sum_a = 0.0;
      for (uword k = 0; k < K; ++k) {
        sum_a += alpha_cols[k][map_ptrs[k][i]];
      }
      v_col[i] -= sum_a;
    }
  }
}

// Berge: backward Gauss-Seidel sweep for K-FE.
// For FE q (K-1 down to 0): uses alpha_src for h < q, alpha_dst for h > q.
inline void gs_sweep_backward_kfe(std::vector<mat> &alpha_dst,
                                  const std::vector<mat> &alpha_src,
                                  const std::vector<mat> &in_out,
                                  const FlatFEMap &map,
                                  const double *__restrict__ w_ptr, uword n_obs,
                                  uword P) {
  const uword K = map.K;

  std::vector<const uword *> fe_ptrs(K);
  for (uword h = 0; h < K; ++h) {
    fe_ptrs[h] = map.fe_map[h].data();
  }

  for (uword q = K; q-- > 0;) {
    const uword n_q = map.n_groups[q];
    const uword *__restrict__ gq = fe_ptrs[q];
    const double *__restrict__ iw = map.inv_weights[q].memptr();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
    for (uword p = 0; p < P; ++p) {
      double *__restrict__ dst_col = alpha_dst[q].colptr(p);
      const double *__restrict__ io_col = in_out[q].colptr(p);

      std::vector<const double *> col_ptrs(K);
      for (uword h = 0; h < K; ++h) {
        if (h == q)
          continue;
        col_ptrs[h] = (h < q) ? alpha_src[h].colptr(p) : alpha_dst[h].colptr(p);
      }

      std::memcpy(dst_col, io_col, n_q * sizeof(double));

      for (uword i = 0; i < n_obs; ++i) {
        double sum_others = 0.0;
        for (uword h = 0; h < K; ++h) {
          if (h == q)
            continue;
          sum_others += col_ptrs[h][fe_ptrs[h][i]];
        }
        dst_col[gq[i]] -= w_ptr[i] * sum_others;
      }

      for (uword g = 0; g < n_q; ++g) {
        dst_col[g] *= iw[g];
      }
    }
  }
}

// Composed map F(alpha) = f_1(f_2(alpha)): beta from alpha_src, then alpha_dst
// from beta.
inline void apply_F_2fe(mat &alpha_dst, mat &beta_tmp, const mat &alpha_src,
                        const std::vector<mat> &in_out, const FlatFEMap &map,
                        const double *__restrict__ w_ptr, uword n_obs,
                        uword P) {
  const uword *__restrict__ g1 = map.fe_map[0].data();
  const uword *__restrict__ g2 = map.fe_map[1].data();
  const double *__restrict__ iw1 = map.inv_weights[0].memptr();
  const double *__restrict__ iw2 = map.inv_weights[1].memptr();
  const uword n1 = alpha_dst.n_rows;
  const uword n2 = beta_tmp.n_rows;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *__restrict__ bt_col = beta_tmp.colptr(p);
    const double *__restrict__ as_col = alpha_src.colptr(p);
    const double *__restrict__ io1_col = in_out[0].colptr(p);
    const double *__restrict__ io2_col = in_out[1].colptr(p);
    double *__restrict__ ad_col = alpha_dst.colptr(p);

    std::memcpy(bt_col, io2_col, n2 * sizeof(double));
    for (uword i = 0; i < n_obs; ++i) {
      bt_col[g2[i]] -= w_ptr[i] * as_col[g1[i]];
    }
    for (uword g = 0; g < n2; ++g) {
      bt_col[g] *= iw2[g];
    }

    std::memcpy(ad_col, io1_col, n1 * sizeof(double));
    for (uword i = 0; i < n_obs; ++i) {
      ad_col[g1[i]] -= w_ptr[i] * bt_col[g2[i]];
    }
    for (uword g = 0; g < n1; ++g) {
      ad_col[g] *= iw1[g];
    }
  }
}

// Berge 2-FE centering: fixed-point F(alpha) = f_1(f_2(alpha)) with IT
// acceleration
inline void center_2fe_berge(mat &V, const vec &w, const FlatFEMap &map,
                             CenterWarmStart &warm, double tol, uword max_iter,
                             uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = map.n_groups[0];
  const uword n2 = map.n_groups[1];

  const uword *g1 = map.fe_map[0].data();
  const uword *g2 = map.fe_map[1].data();
  const double *w_ptr = w.memptr();

  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

  mat alpha;
  if (warm.can_use(2, P)) {
    alpha = warm.alpha[0];
  } else {
    alpha.zeros(n1, P);
  }

  const uword total_elem = n1 * P;

  warm.ensure_scratch_2fe(n1, n2, P);
  mat &beta_tmp = warm.scratch_beta;
  mat &GX = warm.scratch_mats[0];
  mat &GGX = warm.scratch_mats[1];
  mat &X_it = warm.scratch_mats[2];
  mat &grand_Y = warm.scratch_mats[3];
  mat &grand_GY = warm.scratch_mats[4];
  mat &grand_GGY = warm.scratch_mats[5];
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 50000) ? 80 : 40;
  double ssr_old = datum::inf;

  apply_F_2fe(GX, beta_tmp, alpha, in_out, map, w_ptr, n_obs, P);

  {
    bool keep_going = false;
    for (uword i = 0; i < total_elem; ++i) {
      if (continue_crit(alpha.memptr()[i], GX.memptr()[i], tol)) {
        keep_going = true;
        break;
      }
    }
    if (!keep_going) {
      gs_update_2fe(beta_tmp, GX, in_out[1], map.inv_weights[1], g1, g2, w_ptr,
                    n_obs, P);
      std::vector<mat> coeffs = {GX, beta_tmp};
      warm.save(coeffs, 2, P);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        double *v_col = V.colptr(p);
        const double *a1 = GX.colptr(p);
        const double *a2 = beta_tmp.colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          v_col[i] -= a1[g1[i]] + a2[g2[i]];
        }
      }
      return;
    }
  }

  for (uword iter = 0; iter < max_iter; ++iter) {
    apply_F_2fe(GGX, beta_tmp, GX, in_out, map, w_ptr, n_obs, P);

    bool numconv = irons_tuck_acc(alpha, GX, GGX);
    if (numconv)
      break;

    if (iter >= iter_proj_after_acc) {
      X_it = alpha;
      apply_F_2fe(alpha, beta_tmp, X_it, in_out, map, w_ptr, n_obs, P);
    }

    apply_F_2fe(GX, beta_tmp, alpha, in_out, map, w_ptr, n_obs, P);

    {
      const double *__restrict__ curr = alpha.memptr();
      const double *__restrict__ gx = GX.memptr();
      double max_diff = 0.0;
      for (uword i = 0; i < total_elem; ++i) {
        const double d = std::fabs(curr[i] - gx[i]);
        if (d > max_diff)
          max_diff = d;
      }
      if (max_diff < tol)
        break;
      bool keep_going = false;
      for (uword i = 0; i < total_elem; ++i) {
        if (continue_crit(curr[i], gx[i], tol)) {
          keep_going = true;
          break;
        }
      }
      if (!keep_going)
        break;
    }

    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        grand_Y = GX;
        grand_stage = 1;
      } else if (grand_stage == 1) {
        grand_GY = GX;
        grand_stage = 2;
      } else {
        grand_GGY = GX;
        bool grand_numconv = irons_tuck_acc(grand_Y, grand_GY, grand_GGY);
        if (!grand_numconv && grand_Y.is_finite()) {
          apply_F_2fe(GX, beta_tmp, grand_Y, in_out, map, w_ptr, n_obs, P);
        }
        grand_stage = 0;
      }
    }

    if (iter > 0 && iter % ssr_check_period == 0) {
      gs_update_2fe(beta_tmp, alpha, in_out[1], map.inv_weights[1], g1, g2,
                    w_ptr, n_obs, P);

      double ssr = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : ssr) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        const double *a1 = alpha.colptr(p);
        const double *a2 = beta_tmp.colptr(p);
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

  apply_F_2fe(GX, beta_tmp, alpha, in_out, map, w_ptr, n_obs, P);
  alpha = GX;
  gs_update_2fe(beta_tmp, alpha, in_out[1], map.inv_weights[1], g1, g2, w_ptr,
                n_obs, P);

  std::vector<mat> coeffs = {alpha, beta_tmp};
  warm.save(coeffs, 2, P);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *a1 = alpha.colptr(p);
    const double *a2 = beta_tmp.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      v_col[i] -= a1[g1[i]] + a2[g2[i]];
    }
  }
}

// Berge K-FE centering: IT-accelerated backward sweep.
// IT acceleration tracks FEs 0,...,K-2; the last FE is only updated by the
// sweep.
inline void center_kfe_berge(mat &V, const vec &w, const FlatFEMap &map,
                             CenterWarmStart &warm, double tol, uword max_iter,
                             uword grand_acc_period = 4) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  if (K == 2) {
    center_2fe_berge(V, w, map, warm, tol, max_iter, grand_acc_period);
    return;
  }

  const double *w_ptr = w.memptr();

  std::vector<mat> in_out;
  in_out_(in_out, V, w_ptr, map);

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

  std::vector<mat> GX(K), GGX(K);
  for (uword k = 0; k < K; ++k) {
    GX[k].zeros(map.n_groups[k], P);
    GGX[k].zeros(map.n_groups[k], P);
  }

  std::vector<mat> grand_alpha_Y(K - 1), grand_alpha_GY(K - 1);
  for (uword k = 0; k < K - 1; ++k) {
    grand_alpha_Y[k].zeros(map.n_groups[k], P);
    grand_alpha_GY[k].zeros(map.n_groups[k], P);
  }
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 100000) ? 80 : 40;
  double ssr_old = datum::inf;

  gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P);

  {
    bool keep_going = false;
    for (uword k = 0; k < K - 1 && !keep_going; ++k) {
      const uword n_elem = alpha[k].n_elem;
      const double *__restrict__ x = alpha[k].memptr();
      const double *__restrict__ gx_p = GX[k].memptr();
      for (uword i = 0; i < n_elem; ++i) {
        if (continue_crit(x[i], gx_p[i], tol)) {
          keep_going = true;
          break;
        }
      }
    }
    if (!keep_going) {
      for (uword k = 0; k < K; ++k)
        alpha[k] = GX[k];
      warm.save(alpha, K, P);
      std::vector<const uword *> map_ptrs(K);
      for (uword k = 0; k < K; ++k)
        map_ptrs[k] = map.fe_map[k].data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        double *v_col = V.colptr(p);
        std::vector<const double *> alpha_cols(K);
        for (uword k = 0; k < K; ++k)
          alpha_cols[k] = alpha[k].colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          double sum_a = 0.0;
          for (uword k = 0; k < K; ++k)
            sum_a += alpha_cols[k][map_ptrs[k][i]];
          v_col[i] -= sum_a;
        }
      }
      return;
    }
  }

  for (uword iter = 0; iter < max_iter; ++iter) {
    gs_sweep_backward_kfe(GGX, GX, in_out, map, w_ptr, n_obs, P);

    double vprod = 0.0, ssq = 0.0;
    for (uword k = 0; k < K - 1; ++k) {
      const uword n_elem = alpha[k].n_elem;
      const double *__restrict__ x = alpha[k].memptr();
      const double *__restrict__ gx_p = GX[k].memptr();
      const double *__restrict__ ggx_p = GGX[k].memptr();
      for (uword i = 0; i < n_elem; ++i) {
        const double dg = ggx_p[i] - gx_p[i];
        const double d2 = dg - gx_p[i] + x[i];
        vprod += dg * d2;
        ssq += d2 * d2;
      }
    }

    bool numconv = false;
    if (ssq == 0.0) {
      numconv = true;
    } else {
      const double coef = vprod / ssq;
      for (uword k = 0; k < K - 1; ++k) {
        const uword n_elem = alpha[k].n_elem;
        double *__restrict__ x = alpha[k].memptr();
        const double *__restrict__ gx_p = GX[k].memptr();
        const double *__restrict__ ggx_p = GGX[k].memptr();
        for (uword i = 0; i < n_elem; ++i) {
          x[i] = ggx_p[i] - coef * (ggx_p[i] - gx_p[i]);
        }
      }
    }

    alpha[K - 1] = GGX[K - 1];

    if (numconv)
      break;

    if (iter >= iter_proj_after_acc) {
      gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P);
      for (uword k = 0; k < K; ++k)
        alpha[k] = GX[k];
    }

    gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P);

    {
      bool keep_going = false;
      for (uword k = 0; k < K - 1 && !keep_going; ++k) {
        const uword n_elem = alpha[k].n_elem;
        const double *__restrict__ x = alpha[k].memptr();
        const double *__restrict__ gx_p = GX[k].memptr();
        for (uword i = 0; i < n_elem; ++i) {
          if (continue_crit(x[i], gx_p[i], tol)) {
            keep_going = true;
            break;
          }
        }
      }
      if (!keep_going)
        break;
    }

    if (grand_acc_period > 0 && iter > 0 && iter % grand_acc_period == 0) {
      if (grand_stage == 0) {
        for (uword k = 0; k < K - 1; ++k)
          grand_alpha_Y[k] = GX[k];
        grand_stage = 1;
      } else if (grand_stage == 1) {
        for (uword k = 0; k < K - 1; ++k)
          grand_alpha_GY[k] = GX[k];
        grand_stage = 2;
      } else {
        double gvprod = 0.0, gssq = 0.0;
        for (uword k = 0; k < K - 1; ++k) {
          const uword n_elem = GX[k].n_elem;
          const double *__restrict__ y = grand_alpha_Y[k].memptr();
          const double *__restrict__ gy = grand_alpha_GY[k].memptr();
          const double *__restrict__ ggy = GX[k].memptr();
          for (uword i = 0; i < n_elem; ++i) {
            const double dg = ggy[i] - gy[i];
            const double d2 = dg - gy[i] + y[i];
            gvprod += dg * d2;
            gssq += d2 * d2;
          }
        }
        if (gssq > 0.0) {
          const double gcoef = gvprod / gssq;
          bool is_ok = true;
          for (uword k = 0; k < K - 1; ++k) {
            const uword n_elem = GX[k].n_elem;
            double *__restrict__ y = grand_alpha_Y[k].memptr();
            const double *__restrict__ gy = grand_alpha_GY[k].memptr();
            const double *__restrict__ ggy = GX[k].memptr();
            for (uword i = 0; i < n_elem; ++i) {
              y[i] = ggy[i] - gcoef * (ggy[i] - gy[i]);
              if (!std::isfinite(y[i])) {
                is_ok = false;
                break;
              }
            }
            if (!is_ok)
              break;
          }
          if (is_ok) {
            for (uword k = 0; k < K - 1; ++k)
              alpha[k] = grand_alpha_Y[k];
            gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P);
          }
        }
        grand_stage = 0;
      }
    }

    if (iter > 0 && iter % ssr_check_period == 0) {
      double ssr = 0.0;
      std::vector<const uword *> mp(K);
      for (uword k = 0; k < K; ++k)
        mp[k] = map.fe_map[k].data();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : ssr) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        std::vector<const double *> gx_cols(K);
        for (uword k = 0; k < K; ++k)
          gx_cols[k] = GX[k].colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          double r = v_col[i];
          for (uword k = 0; k < K; ++k) {
            r -= gx_cols[k][mp[k][i]];
          }
          ssr += w_ptr[i] * r * r;
        }
      }
      if (stopping_crit(ssr_old, ssr, tol))
        break;
      ssr_old = ssr;
    }
  }

  gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P);
  for (uword k = 0; k < K; ++k)
    alpha[k] = GX[k];

  warm.save(alpha, K, P);

  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    std::vector<const double *> alpha_cols(K);
    for (uword k = 0; k < K; ++k)
      alpha_cols[k] = alpha[k].colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      double sum_a = 0.0;
      for (uword k = 0; k < K; ++k) {
        sum_a += alpha_cols[k][map_ptrs[k][i]];
      }
      v_col[i] -= sum_a;
    }
  }
}

inline void center_variables(mat &V, const vec &w, FlatFEMap &map, double tol,
                             uword max_iter, uword grand_acc_period,
                             CenterWarmStart *warm = nullptr,
                             CenteringMethod method = STAMMANN) {
#ifdef _OPENMP
  set_omp_threads_from_config();
#endif
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
