#include "00_main.h"

// Method of alternating projections (Halperin)
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(max_iter), N = V.n_rows, P = V.n_cols,
               K = klist.size(), iint0 = static_cast<size_t>(iter_interrupt),
               isr0 = static_cast<size_t>(iter_ssr);
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t iter, iint, isr, j, jj, k, n, p, J, JJ;
  double num, coef, xbar, ratio, ssr, ssq, ratio0, ssr0;
  vec x(N), x0(N), Gx(N), G2x(N), deltaG(N), delta2(N);

  // Precompute groups into fields
  field<field<uvec>> group_indices(K);
  field<vec> group_inverse_weights(K);
  for (k = 0; k < K; ++k) {
    const list &jlist = klist[k];
    J = jlist.size();
    field<uvec> idxs(J);
    vec invs(J);
    for (j = 0; j < J; ++j) {
      idxs(j) = as_uvec(as_cpp<integers>(jlist[j]));
      invs(j) = 1.0 / accu(w.elem(idxs(j)));
    }
    group_indices(k) = std::move(idxs);
    group_inverse_weights(k) = std::move(invs);
  }

  // Single nested‐field projection helper
  auto project = [&](vec &v) {
    J = group_indices.n_elem;
    for (j = 0; j < J; ++j) {
      auto &idxs = group_indices(j);
      auto &invs = group_inverse_weights(j);
      JJ = idxs.n_elem;
      for (jj = 0; jj < JJ; ++jj) {
        const uvec &coords = idxs(jj);
        xbar = dot(w.elem(coords), v.elem(coords)) * invs(jj);
        v.elem(coords) -= xbar;
      }
    }
  };

  // Column‐wise centering
  for (p = 0; p < P; ++p) {
    x = V.col(p);
    ratio0 = std::numeric_limits<double>::infinity();
    ssr0 = std::numeric_limits<double>::infinity();

    // reset per‐column interrupt
    iint = iint0;
    isr = isr0;

    for (iter = 0; iter < I; ++iter) {
      if (iter == iint) {
        check_user_interrupt();
        iint += iint0;
      }

      x0 = x;

      // 1) main projection
      project(x);
      num = 0.0;
      for (n = 0; n < N; ++n) {
        num += std::abs(x[n] - x0[n]) / (1.0 + std::abs(x0[n])) * w[n];
      }
      ratio = num * inv_sw;
      if (ratio < tol)
        break;

      // 2) acceleration every 5 iters
      if (iter >= 5 && (iter % 5) == 0) {
        Gx = x;
        project(Gx);
        G2x = Gx;
        deltaG = G2x - x;
        delta2 = G2x - 2.0 * x + x0;
        ssq = dot(delta2, delta2);
        if (ssq > 1e-10) {
          coef = dot(deltaG, delta2) / ssq;
          if (coef > 0.0 && coef < 2.0) {
            x = G2x - coef * deltaG;
          } else {
            x = G2x;
          }
        }
      }

      // 3) SSR‐based early exit
      if (iter == isr && iter > 0) {
        check_user_interrupt();
        isr += isr0;
        ssr = dot(x % x, w) * inv_sw;
        if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < tol)
          break;
        ssr0 = ssr;
      }

      // 4) early exit
      if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20)
        break;
      ratio0 = ratio;
    }

    V.col(p) = x;
  }
}

[[cpp11::register]] doubles_matrix<>
center_variables_r_(const doubles_matrix<> &V_r, const doubles &w_r,
                    const list &klist, const double tol, const int max_iter,
                    const int iter_interrupt, const int iter_ssr) {
  mat V = as_mat(V_r);
  center_variables_(V, as_col(w_r), klist, tol, max_iter, iter_interrupt,
                    iter_ssr);
  return as_doubles_matrix(V);
}
