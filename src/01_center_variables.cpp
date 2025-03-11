#include "00_main.h"

// Method of alternating projections (Halperin)
void center_variables_(Mat<double> &V, const Col<double> &w,
                                const list &klist, const double &tol,
                                const int &maxiter) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(maxiter);
  const size_t N = V.n_rows;
  const size_t P = V.n_cols;
  const size_t K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t j, k, J;

  // Precompute group indices and weights
  field<field<uvec>> group_indices(K);
  field<vec> group_inverse_weights(K);

  for (k = 0; k < K; ++k) {
    list jlist = klist[k];
    J = jlist.size();
    group_indices(k).set_size(J);
    group_inverse_weights(k).set_size(J);
    for (j = 0; j < J; ++j) {
      group_indices(k)(j) = as_uvec(as_cpp<integers>(jlist[j]));
      group_inverse_weights(k)(j) = 1.0 / accu(w.elem(group_indices(k)(j)));
    }
  }

// Halperin projections parallelizing over columns
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t p = 0; p < P; ++p) {
    Col<double> x = V.col(p);
    Col<double> x0(N);
    size_t iter, interrupt_iter = 1000, l, m, L;
    double xbar, ratio;

    for (iter = 0; iter < I; ++iter) {
      if (iter == interrupt_iter) {
        check_user_interrupt();
        interrupt_iter += 1000;
      }

      x0 = x;

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

      ratio = accu(abs(x - x0) / (1.0 + abs(x0)) % w) * inv_sw;
      if (ratio < tol) break;
    }
    V.col(p) = x;
  }
}

[[cpp11::register]] doubles_matrix<> center_variables_r_(
    const doubles_matrix<> &V_r, const doubles &w_r, const list &klist,
    const double &tol, const int &maxiter) {
  Mat<double> V = as_Mat(V_r);
  Col<double> w = as_Col(w_r);
  center_variables_(V, w, klist, tol, maxiter);
  return as_doubles_matrix(V);
}
