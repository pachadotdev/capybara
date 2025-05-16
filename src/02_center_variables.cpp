// 02_center_variables.cpp (refactored using Armadillo types)
#include "00_main.h"

// Method of alternating projections (Halperin)
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(max_iter), N = V.n_rows, P = V.n_cols,
               K = klist.size(),
               iter_check_interrupt0 = static_cast<size_t>(iter_interrupt),
               iter_check_ssr0 = static_cast<size_t>(iter_ssr);
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t iter, j, k, p, J, iter_check_interrupt = iter_check_interrupt0,
                           iter_check_ssr = iter_check_ssr0;
  double coef, xbar, ratio, ssr, ssq, ratio0, ssr0;
  vec x(N), x0(N), Gx(N), G2x(N), deltaG(N), delta2(N);
  field<field<uvec>> group_indices(K);
  field<vec> group_inverse_weights(K);
  for (k = 0; k < K; ++k) {
    const list &jlist = klist[k];
    J = jlist.size();
    field<uvec> idxs(J);
    vec invs(J);
    for (j = 0; j < J; ++j) {
      idxs(j) = as_uvec(as_cpp<integers>(jlist[j]));
      ;
      invs(j) = 1.0 / accu(w.elem(idxs(j)));
    }
    group_indices(k) = idxs;
    group_inverse_weights(k) = invs;
  }

  for (p = 0; p < P; ++p) {
    x = V.col(p);
    ratio0 = std::numeric_limits<double>::infinity();
    ssr0 = std::numeric_limits<double>::infinity();

    for (iter = 0; iter < I; ++iter) {
      if (iter == iter_check_interrupt) {
        check_user_interrupt();
        iter_check_interrupt += iter_check_interrupt0;
      }

      x0 = x;

      // Halperin projection
      for (k = 0; k < K; ++k) {
        field<uvec> &idxs = group_indices(k);
        J = idxs.n_elem;
        vec &invs = group_inverse_weights(k);
        for (j = 0; j < J; ++j) {
          const uvec &coords = idxs(j);
          xbar = dot(w.elem(coords), x.elem(coords)) * invs(j);
          x.elem(coords) -= xbar;
        }
      }

      // Convergence check
      ratio = dot(abs(x - x0) / (1.0 + abs(x0)), w) * inv_sw;
      if (ratio < tol)
        break;

      // Acceleration every 5 iters
      if (iter > 5 && (iter % 5) == 0) {
        Gx = x;
        // Second projection
        for (size_t k = 0; k < K; ++k) {
          field<uvec> &idxs = group_indices(k);
          vec &invs = group_inverse_weights(k);
          for (j = 0; j < idxs.n_elem; ++j) {
            const uvec &coords = idxs(j);
            xbar = dot(w.elem(coords), Gx.elem(coords)) * invs(j);
            Gx.elem(coords) -= xbar;
          }
        }
        G2x = Gx;

        // Compute deltas
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

      // SSR check
      if (iter == iter_check_ssr && iter > 0) {
        check_user_interrupt();
        iter_check_ssr += iter_check_ssr0;
        ssr = dot(x % x, w) * inv_sw;
        if (fabs(ssr - ssr0) / (1.0 + fabs(ssr0)) < tol)
          break;
        ssr0 = ssr;
      }

      // Early exit
      if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20)
        break;
      ratio0 = ratio;
    }

    V.col(p) = x;
  }
}

[[cpp11::register]] doubles_matrix<>
center_variables_r_(const doubles_matrix<> &V_r, const doubles &w_r,
                    const list &klist, const double &tol, const int &max_iter,
                    const int &iter_interrupt, const int &iter_ssr) {
  mat V = as_mat(V_r);
  center_variables_(V, as_col(w_r), klist, tol, max_iter, iter_interrupt,
                    iter_ssr);
  return as_doubles_matrix(V);
}
