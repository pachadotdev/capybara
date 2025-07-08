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
  size_t iter, iint, isr, j, k, l, p, J, L;
  double coef, xbar, ratio, ssr, ssq, ratio0, ssr0;
  vec x(N, fill::none), x0(N, fill::none), Gx(N, fill::none),
      G2x(N, fill::none), deltaG(N, fill::none), delta2(N, fill::none),
      diff(N, fill::none);

  // Precompute groups into fields
  field<field<uvec>> group_indices(K);
  field<vec> group_inv_w(K);
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
    group_inv_w(k) = std::move(invs);
  }

  // Single projection step (in-place)
  auto project = [&](vec &v) {
    for (k = 0; k < K; ++k) {
      const auto &idxs = group_indices(k);
      const auto &invs = group_inv_w(k);
      L = idxs.n_elem;
      if (L == 0)
        continue;
      for (l = 0; l < L; ++l) {
        const uvec &coords = idxs(l);
        const uword coord_size = coords.n_elem;
        if (coord_size <= 1)
          continue;
        xbar = dot(w.elem(coords), v.elem(coords)) * invs(l);
        v.elem(coords) -= xbar;
      }
    }
  };

  // Column-wise centering with acceleration and SSR checks
  for (p = 0; p < P; ++p) {
    x = V.col(p);
    ratio0 = std::numeric_limits<double>::infinity();
    ssr0 = std::numeric_limits<double>::infinity();
    iint = iint0;
    isr = isr0;

    for (iter = 0; iter < I; ++iter) {
      if (iter == iint) {
        check_user_interrupt();
        iint += iint0;
      }

      x0 = x;
      project(x);

      // 1) convergence via weighted diff
      diff = abs(x - x0) / (1.0 + abs(x0));
      ratio = dot(diff, w) * inv_sw;
      if (ratio < tol)
        break;

      // 2) Irons-Tuck acceleration every 5 iters
      if (iter >= 5 && (iter % 5) == 0) {
        Gx = x;
        project(Gx);
        G2x = Gx;
        deltaG = G2x - x;
        delta2 = G2x - 2.0 * x + x0;
        ssq = dot(delta2, delta2);
        if (ssq > 1e-10) {
          coef = dot(deltaG, delta2) / ssq;
          x = (coef > 0.0 && coef < 2.0) ? (G2x - coef * deltaG) : G2x;
        }
      }

      // 3) SSR-based early exit
      if (iter == isr && iter > 0) {
        check_user_interrupt();
        isr += isr0;
        ssr = dot(x % x, w) * inv_sw;
        if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < tol)
          break;
        ssr0 = ssr;
      }

      // 4) heuristic early exit
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
