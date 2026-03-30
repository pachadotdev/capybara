#ifndef CAPYBARA_CENTER_STAMMANN_H
#define CAPYBARA_CENTER_STAMMANN_H

namespace capybara {

// Stammann 2-FE centering: alternating projections with RRE-2 acceleration
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

  // Use warm-start scratch buffers to avoid repeated allocations
  warm.ensure_scratch_stammann_2fe(n1, P);
  mat &X_it = warm.stammann_2fe_scratch[0];
  mat &GX_it = warm.stammann_2fe_scratch[1];
  mat &G2X_it = warm.stammann_2fe_scratch[2];
  mat &G3X_it = warm.stammann_2fe_scratch[3];
  mat &grand_Y = warm.stammann_2fe_grand_Y;
  mat &grand_GY = warm.stammann_2fe_grand_GY;
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 50000) ? 80 : 40;
  double ssr_old = datum::inf;

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha1;

    gs_sweep();
    GX_it = alpha1;

    gs_sweep();
    G2X_it = alpha1;

    gs_sweep();
    G3X_it = alpha1;

    bool numconv = rre2_acc(alpha1, GX_it, G2X_it, G3X_it);

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

// Stammann K-FE centering: alternating projections with RRE-2 acceleration
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

  // Pre-allocate map pointers for SSR computation (used in loop)
  std::vector<const uword *> ssr_map_ptrs(K);
  for (uword k = 0; k < K; ++k) {
    ssr_map_ptrs[k] = map.fe_map[k].data();
  }

  // Use warm-start scratch buffers to avoid repeated allocations
  warm.ensure_scratch_stammann_kfe(n0, P);
  mat &X_it = warm.stammann_kfe_scratch[0];
  mat &GX_it = warm.stammann_kfe_scratch[1];
  mat &G2X_it = warm.stammann_kfe_scratch[2];
  mat &G3X_it = warm.stammann_kfe_scratch[3];
  mat &grand_Y = warm.stammann_kfe_grand_Y;
  mat &grand_GY = warm.stammann_kfe_grand_GY;
  uword grand_stage = 0;

  constexpr uword iter_proj_after_acc = 40;
  const uword ssr_check_period = (n_obs > 100000) ? 80 : 40;
  double ssr_old = datum::inf;

  for (uword iter = 0; iter < max_iter; ++iter) {
    X_it = alpha[0];

    gs_sweep();
    GX_it = alpha[0];

    gs_sweep();
    G2X_it = alpha[0];

    gs_sweep();
    G3X_it = alpha[0];

    bool numconv = rre2_acc(alpha[0], GX_it, G2X_it, G3X_it);

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
      // Compute SSR without temporary allocations - use pre-allocated
      // ssr_map_ptrs and inline pointer access
      double ssr = 0.0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : ssr) if (P > 1)
#endif
      for (uword p = 0; p < P; ++p) {
        const double *v_col = V.colptr(p);
        for (uword i = 0; i < n_obs; ++i) {
          double r = v_col[i];
          for (uword k = 0; k < K; ++k) {
            r -= alpha[k].at(ssr_map_ptrs[k][i], p);
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

  // Reuse ssr_map_ptrs for final subtraction (already allocated outside loop)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *v_col = V.colptr(p);
    for (uword i = 0; i < n_obs; ++i) {
      double sum_a = 0.0;
      for (uword k = 0; k < K; ++k) {
        sum_a += alpha[k].at(ssr_map_ptrs[k][i], p);
      }
      v_col[i] -= sum_a;
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_STAMMANN_H
