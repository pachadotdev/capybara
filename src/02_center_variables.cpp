#include "00_main.h"

// Method of alternating projections (Halperin)
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &maxiter) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(maxiter);
  const size_t N = V.n_rows;
  const size_t P = V.n_cols;
  const size_t K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t iter, j, k, l, m, p, L, J;
  vec x(N), x0(N);
  double xbar, ratio;
  size_t iter_check_interrupt = 500;

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

  // Halperin projections parallelizing over columns
  for (p = 0; p < P; ++p) {
    x = V.col(p);

    // Check for user interrupts every 500 iterations
    if (iter == iter_check_interrupt) {
      check_user_interrupt();
      iter_check_interrupt += 500;
    }

    for (iter = 0; iter < I; ++iter) {
      x0 = x;

      for (l = 0; l < K; ++l) {
        L = group_indices(l).size();
        if (L == 0)
          continue;

        for (m = 0; m < L; ++m) {
          const uvec &coords = group_indices(l)(m);
          xbar =
              dot(w.elem(coords), x.elem(coords)) * group_inverse_weights(l)(m);
          x.elem(coords) -= xbar;
        }
      }

      ratio = dot(abs(x - x0) / (1.0 + abs(x0)), w) * inv_sw;
      if (ratio < tol)
        break;
    }
    V.col(p) = std::move(x);
  }
}

[[cpp11::register]] doubles_matrix<>
center_variables_r_(const doubles_matrix<> &V_r, const doubles &w_r,
                    const list &klist, const double &tol, const int &maxiter) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);
  center_variables_(V, w, klist, tol, maxiter);
  return as_doubles_matrix(V);
}
