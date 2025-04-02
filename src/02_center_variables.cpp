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
  size_t iter, j, k, l, m, p, L, J,
      iter_check_interrupt = iter_check_interrupt0,
      iter_check_ssr = iter_check_ssr0;
  double xbar, ratio, ratio0, ssr, ssr0, vprod, ssq, coef;
  vec x(N), x0(N);
  field<field<uvec>> group_indices(K);
  field<vec> group_inverse_weights(K);

  // Precompute group indices and weights
  for (k = 0; k < K; ++k) {
    const list &jlist = klist[k];
    J = jlist.size();

    field<uvec> indices(J);
    vec inverse_weights(J);

    for (j = 0; j < J; ++j) {
      indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
      inverse_weights(j) = 1.0 / accu(w.elem(indices(j)));
    }

    group_indices(k) = indices;
    group_inverse_weights(k) = inverse_weights;
  }

  // Pre-allocate vectors for acceleration (outside the loop to avoid
  // reallocation)
  vec G_x(N), G2_x(N), delta_G_x(N), delta2_x(N);

  // Halperin projections parallelizing over columns
  for (p = 0; p < P; ++p) {
    x = V.col(p);
    ratio0 = std::numeric_limits<double>::max();
    ssr0 = std::numeric_limits<double>::max();

    for (iter = 0; iter < I; ++iter) {
      // Check for user interrupt less frequently
      if (iter == iter_check_interrupt) {
        check_user_interrupt();
        iter_check_interrupt += iter_check_interrupt0;
      }

      x0 = x;  // Save current x

      // Apply the Halperin projection
      for (l = 0; l < K; ++l) {
        L = group_indices(l).size();
        if (L == 0) continue;

        for (m = 0; m < L; ++m) {
          const uvec &coords = group_indices(l)(m);
          xbar =
              dot(w.elem(coords), x.elem(coords)) * group_inverse_weights(l)(m);
          x.elem(coords) -= xbar;
        }
      }

      // First convergence check
      ratio = dot(abs(x - x0) / (1.0 + abs(x0)), w) * inv_sw;
      if (ratio < tol) break;

      // Apply acceleration less frequently - only every 5 iterations instead of
      // 3 This reduces overhead while still getting acceleration benefits
      if (iter > 5 && iter % 5 == 0) {
        G_x = x;  // G(x) - the result after one projection

        // Apply another projection to get G(G(x))
        for (l = 0; l < K; ++l) {
          L = group_indices(l).size();
          if (L == 0) continue;

          for (m = 0; m < L; ++m) {
            const uvec &coords = group_indices(l)(m);
            xbar = dot(w.elem(coords), G_x.elem(coords)) *
                   group_inverse_weights(l)(m);
            G_x.elem(coords) -= xbar;
          }
        }
        G2_x = G_x;  // G²(x)

        // Irons & Tuck acceleration formula
        delta_G_x = G2_x - x;
        delta2_x = G2_x - 2 * x + x0;

        ssq = dot(delta2_x, delta2_x);
        if (ssq > 1e-10) {  // Add numerical stability threshold
          vprod = dot(delta_G_x, delta2_x);
          coef = vprod / ssq;

          // Limit coefficient to prevent excessive extrapolation
          if (coef > 0 && coef < 2.0) {
            x = G2_x - coef * delta_G_x;
          } else {
            x = G2_x;  // Use G2_x if coefficient is out of bounds
          }
        }
      }

      // Check SSR improvement less frequently
      if (iter == iter_check_ssr && iter > 0) {
        check_user_interrupt();
        iter_check_ssr += iter_check_ssr0;
        ssr = dot(x % x, w) * inv_sw;
        if (fabs(ssr - ssr0) / (1.0 + fabs(ssr0)) < tol) break;
        ssr0 = ssr;
      }

      // Early stopping based on ratio improvement
      if (iter > 3 && ratio0 / ratio < 1.1 && ratio < tol * 20) {
        break;
      }

      ratio0 = ratio;
    }

    V.col(p) = std::move(x);
  }
}

[[cpp11::register]] doubles_matrix<> center_variables_r_(
    const doubles_matrix<> &V_r, const doubles &w_r, const list &klist,
    const double &tol, const int &max_iter, const int &iter_interrupt,
    const int &iter_ssr) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);
  center_variables_(V, w, klist, tol, max_iter, iter_interrupt, iter_ssr);
  return as_doubles_matrix(V);
}
