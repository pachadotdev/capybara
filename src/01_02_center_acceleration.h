#ifndef CAPYBARA_CENTER_ACCELERATION_H
#define CAPYBARA_CENTER_ACCELERATION_H

namespace capybara {
// Irons-Tuck acceleration on coefficient vectors.
// Returns true if numerically converged (ssq == 0).
inline bool irons_tuck_acc(mat &X_coef, const mat &GX_coef,
                           const mat &G2X_coef) {
  const uword n = X_coef.n_elem;
  double *__restrict__ x = X_coef.memptr();
  const double *__restrict__ gx = GX_coef.memptr();
  const double *__restrict__ G2X = G2X_coef.memptr();

  double vprod = 0.0, ssq = 0.0;
  for (uword i = 0; i < n; ++i) {
    const double dg = G2X[i] - gx[i];
    const double d2 = dg - gx[i] + x[i];
    vprod += dg * d2;
    ssq += d2 * d2;
  }

  if (ssq == 0.0) {
    return true;
  }

  const double coef = vprod / ssq;
  for (uword i = 0; i < n; ++i) {
    x[i] = G2X[i] - coef * (G2X[i] - gx[i]);
  }
  return false;
}

// RRE-2 acceleration using 4 iterates (Ramiere-Helfer "alternate 2-delta
// method"). Uses X, G(X), G^2(X), G^3(X) to compute extrapolation with two
// columns of second differences. Falls back to Irons-Tuck if the 2x2 system is
// singular. Returns true if numerically converged.
inline bool rre2_acc(mat &X_coef, const mat &GX_coef, const mat &G2X_coef,
                     const mat &G3X_coef) {
  const uword n = X_coef.n_elem;
  double *__restrict__ x = X_coef.memptr();
  const double *__restrict__ gx = GX_coef.memptr();
  const double *__restrict__ G2X = G2X_coef.memptr();
  const double *__restrict__ G3X = G3X_coef.memptr();

  // Compute differences and second differences
  // u0 = GX - X, u1 = G2X - GX, u2 = G3X - G2X
  // v0 = u1 - u0 = G2X - 2*GX + X
  // v1 = u2 - u1 = G3X - 2*G2X + GX

  // Build the 2x2 Gram matrix: M = [<v0,v0>, <v0,v1>; <v0,v1>, <v1,v1>]
  // and right-hand side: b = -[<v0,u2>, <v1,u2>]
  double v0v0 = 0.0, v0v1 = 0.0, v1v1 = 0.0;
  double v0u2 = 0.0, v1u2 = 0.0;

  for (uword i = 0; i < n; ++i) {
    const double u1 = G2X[i] - gx[i];
    const double u2 = G3X[i] - G2X[i];
    const double v0 = u1 - gx[i] + x[i]; // G2X - 2*GX + X
    const double v1 = u2 - u1;           // G3X - 2*G2X + GX

    v0v0 += v0 * v0;
    v0v1 += v0 * v1;
    v1v1 += v1 * v1;
    v0u2 += v0 * u2;
    v1u2 += v1 * u2;
  }

  // Check for convergence
  if (v0v0 + v1v1 < 1e-30) {
    return true;
  }

  // Solve 2x2 system: [v0v0, v0v1; v0v1, v1v1] * [g0; g1] = -[v0u2; v1u2]
  const double det = v0v0 * v1v1 - v0v1 * v0v1;

  // If system is nearly singular, fall back to Irons-Tuck (single column)
  if (std::fabs(det) < 1e-14 * (v0v0 * v1v1 + 1e-30)) {
    // Fall back to IT using v0 only
    if (v0v0 < 1e-30) {
      return true;
    }
    const double coef = v0u2 / v0v0;
    for (uword i = 0; i < n; ++i) {
      const double u2 = G3X[i] - G2X[i];
      x[i] = G3X[i] - coef * u2;
    }
    return false;
  }

  // Solve via Cramer's rule
  const double inv_det = 1.0 / det;
  const double g0 = (-v0u2 * v1v1 + v1u2 * v0v1) * inv_det;
  const double g1 = (-v1u2 * v0v0 + v0u2 * v0v1) * inv_det;

  // Extrapolate: X_acc = G3X + g0 * u1 + g1 * u2
  for (uword i = 0; i < n; ++i) {
    const double u1 = G2X[i] - gx[i];
    const double u2 = G3X[i] - G2X[i];
    x[i] = G3X[i] + g0 * u1 + g1 * u2;
  }
  return false;
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

  // Pre-allocate column pointer storage outside parallel region
  std::vector<std::vector<const double *>> all_alpha_cols(
      P, std::vector<const double *>(K));
  for (uword p = 0; p < P; ++p) {
    for (uword j = 0; j < K; ++j) {
      all_alpha_cols[p][j] = alpha[j].colptr(p);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
  for (uword p = 0; p < P; ++p) {
    double *__restrict__ ak_col = alpha_k.colptr(p);
    const double *__restrict__ io_col = in_out_k.colptr(p);
    const double *const *__restrict__ alpha_cols = all_alpha_cols[p].data();

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

} // namespace capybara

#endif // CAPYBARA_CENTER_ACCELERATION_H
