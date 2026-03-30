#ifndef CAPYBARA_CENTER_BERGE_H
#define CAPYBARA_CENTER_BERGE_H

namespace capybara {

// Berge: backward Gauss-Seidel sweep for K-FE.
// For FE q (K-1 down to 0): uses alpha_src for h < q, alpha_dst for h > q.
inline void gs_sweep_backward_kfe(std::vector<mat> &alpha_dst,
                                  const std::vector<mat> &alpha_src,
                                  const std::vector<mat> &in_out,
                                  const FlatFEMap &map,
                                  const double *__restrict__ w_ptr, uword n_obs,
                                  uword P, CenterWarmStart &warm) {
  const uword K = map.K;

  std::vector<const uword *> fe_ptrs(K);
  for (uword h = 0; h < K; ++h) {
    fe_ptrs[h] = map.fe_map[h].data();
  }

  // Use cached column pointer storage from warm-start to avoid repeated
  // allocations across calls
  std::vector<std::vector<const double *>> &col_ptrs_all =
      warm.ensure_berge_kfe_col_ptrs(P, K);

  for (uword q = K; q-- > 0;) {
    const uword n_q = map.n_groups[q];
    const uword *__restrict__ gq = fe_ptrs[q];
    const double *__restrict__ iw = map.inv_weights[q].memptr();

    // Update column pointers for this q level
    for (uword p = 0; p < P; ++p) {
      for (uword h = 0; h < K; ++h) {
        if (h == q)
          continue;
        col_ptrs_all[p][h] =
            (h < q) ? alpha_src[h].colptr(p) : alpha_dst[h].colptr(p);
      }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
    for (uword p = 0; p < P; ++p) {
      double *__restrict__ dst_col = alpha_dst[q].colptr(p);
      const double *__restrict__ io_col = in_out[q].colptr(p);
      const double *const *__restrict__ col_ptrs = col_ptrs_all[p].data();

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

// Berge 2-FE centering: fixed-point F(alpha) = f_1(f_2(alpha)) with RRE-2
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
  mat &G2X = warm.scratch_mats[1];
  mat &X_it = warm.scratch_mats[2];
  mat &grand_Y = warm.scratch_mats[3];
  mat &grand_GY = warm.scratch_mats[4];
  mat &grand_GGY = warm.scratch_mats[5];
  mat &G3X = warm.scratch_mats[6];
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
    apply_F_2fe(G2X, beta_tmp, GX, in_out, map, w_ptr, n_obs, P);
    apply_F_2fe(G3X, beta_tmp, G2X, in_out, map, w_ptr, n_obs, P);

    bool numconv = rre2_acc(alpha, GX, G2X, G3X);
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

  std::vector<mat> GX(K), G2X(K), G3X(K);
  for (uword k = 0; k < K; ++k) {
    GX[k].zeros(map.n_groups[k], P);
    G2X[k].zeros(map.n_groups[k], P);
    G3X[k].zeros(map.n_groups[k], P);
  }

  // Only allocate grand acceleration matrices if needed
  std::vector<mat> grand_alpha_Y, grand_alpha_GY;
  if (grand_acc_period > 0) {
    grand_alpha_Y.resize(K - 1);
    grand_alpha_GY.resize(K - 1);
    for (uword k = 0; k < K - 1; ++k) {
      grand_alpha_Y[k].zeros(map.n_groups[k], P);
      grand_alpha_GY[k].zeros(map.n_groups[k], P);
    }
  }
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 100000) ? 80 : 40;
  double ssr_old = datum::inf;

  gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P, warm);

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
    gs_sweep_backward_kfe(G2X, GX, in_out, map, w_ptr, n_obs, P, warm);
    gs_sweep_backward_kfe(G3X, G2X, in_out, map, w_ptr, n_obs, P, warm);

    // RRE-2 acceleration across all K-1 FEs jointly
    // Build the 2x2 Gram matrix and RHS across all elements
    double v0v0 = 0.0, v0v1 = 0.0, v1v1 = 0.0;
    double v0u2 = 0.0, v1u2 = 0.0;

    for (uword k = 0; k < K - 1; ++k) {
      const uword n_elem = alpha[k].n_elem;
      const double *__restrict__ x = alpha[k].memptr();
      const double *__restrict__ gx_p = GX[k].memptr();
      const double *__restrict__ G2X_p = G2X[k].memptr();
      const double *__restrict__ G3X_p = G3X[k].memptr();
      for (uword i = 0; i < n_elem; ++i) {
        const double u1 = G2X_p[i] - gx_p[i];
        const double u2 = G3X_p[i] - G2X_p[i];
        const double v0 = u1 - gx_p[i] + x[i]; // G2X - 2*GX + X
        const double v1 = u2 - u1;             // G3X - 2*G2X + GX

        v0v0 += v0 * v0;
        v0v1 += v0 * v1;
        v1v1 += v1 * v1;
        v0u2 += v0 * u2;
        v1u2 += v1 * u2;
      }
    }

    bool numconv = false;
    if (v0v0 + v1v1 < 1e-30) {
      numconv = true;
    } else {
      const double det = v0v0 * v1v1 - v0v1 * v0v1;

      double g0, g1;
      if (std::fabs(det) < 1e-14 * (v0v0 * v1v1 + 1e-30)) {
        // Fall back to single-column (IT-style)
        if (v0v0 < 1e-30) {
          numconv = true;
        } else {
          g0 = 0.0;
          g1 = -v0u2 / v0v0;
        }
      } else {
        const double inv_det = 1.0 / det;
        g0 = (-v0u2 * v1v1 + v1u2 * v0v1) * inv_det;
        g1 = (-v1u2 * v0v0 + v0u2 * v0v1) * inv_det;
      }

      if (!numconv) {
        // Apply extrapolation: alpha_k = G3X_k + g0 * u1_k + g1 * u2_k
        for (uword k = 0; k < K - 1; ++k) {
          const uword n_elem = alpha[k].n_elem;
          double *__restrict__ x = alpha[k].memptr();
          const double *__restrict__ gx_p = GX[k].memptr();
          const double *__restrict__ G2X_p = G2X[k].memptr();
          const double *__restrict__ G3X_p = G3X[k].memptr();
          for (uword i = 0; i < n_elem; ++i) {
            const double u1 = G2X_p[i] - gx_p[i];
            const double u2 = G3X_p[i] - G2X_p[i];
            x[i] = G3X_p[i] + g0 * u1 + g1 * u2;
          }
        }
      }
    }

    alpha[K - 1] = G3X[K - 1];

    if (numconv)
      break;

    if (iter >= iter_proj_after_acc) {
      gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P, warm);
      for (uword k = 0; k < K; ++k)
        alpha[k] = GX[k];
    }

    gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P, warm);

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
            gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P, warm);
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

      // Pre-allocate GX column pointers
      std::vector<std::vector<const double *>> all_gx_cols(
          P, std::vector<const double *>(K));
      for (uword p = 0; p < P; ++p) {
        for (uword k = 0; k < K; ++k)
          all_gx_cols[p][k] = GX[k].colptr(p);
      }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : ssr) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        const double *const *gx_cols = all_gx_cols[p].data();
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

  gs_sweep_backward_kfe(GX, alpha, in_out, map, w_ptr, n_obs, P, warm);
  for (uword k = 0; k < K; ++k)
    alpha[k] = GX[k];

  warm.save(alpha, K, P);

  std::vector<const uword *> map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    map_ptrs[k] = map.fe_map[k].data();
  }

  // Pre-allocate alpha column pointers for final subtraction
  std::vector<std::vector<const double *>> final_alpha_cols(
      P, std::vector<const double *>(K));
  for (uword p = 0; p < P; ++p) {
    for (uword k = 0; k < K; ++k)
      final_alpha_cols[p][k] = alpha[k].colptr(p);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    const double *const *alpha_cols = final_alpha_cols[p].data();
    for (uword i = 0; i < n_obs; ++i) {
      double sum_a = 0.0;
      for (uword k = 0; k < K; ++k) {
        sum_a += alpha_cols[k][map_ptrs[k][i]];
      }
      v_col[i] -= sum_a;
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_BERGE_H
