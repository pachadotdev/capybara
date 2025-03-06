#include "00_main.h"

// Method of alternating projections (Halperin)
Mat<double> center_variables_(const Mat<double> &V, const Col<double> &w,
                              const list &klist, const double &tol,
                              const int &maxiter) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(maxiter);
  const size_t N = V.n_rows;
  const size_t P = V.n_cols;
  const size_t K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t iter, j, k, p, J, interrupt_iter = 1000;
  double meanj, ratio;
  Mat<double> C(N, P, fill::zeros);
  Col<double> x(N), x0(N);

  // Precompute group indices and weights
  field<field<uvec>> group_indices(K);
  field<vec> group_weights(K);

  for (k = 0; k < K; ++k) {
    list jlist = klist[k];
    J = jlist.size();
    group_indices(k).set_size(J);
    group_weights(k).set_size(J);
    for (j = 0; j < J; ++j) {
      group_indices(k)(j) = as_uvec(as_cpp<integers>(jlist[j]));
      group_weights(k)(j) = accu(w.elem(group_indices(k)(j)));
    }
  }

// Halperin projections
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(x, x0, iter, k, j, meanj, J, ratio)
#endif
  for (p = 0; p < P; ++p) {
    // Center each variable
    x = V.col(p);

    for (iter = 0; iter < I; ++iter) {
      if (iter == interrupt_iter) {
        check_user_interrupt();
        interrupt_iter += 1000;
      }

      // Store centered vector from the last iteration
      x0 = x;

      // Alternate between categories
      for (k = 0; k < K; ++k) {
        // Substract the weighted group means of category 'k'
        J = group_indices(k).size();
        if (J == 0) continue;  // Skip empty groups

        for (j = 0; j < J; ++j) {
          // Subset j-th group of category 'k'
          const uvec &coords = group_indices(k)(j);
          meanj = dot(w.elem(coords), x.elem(coords)) / group_weights(k)(j);
          x.elem(coords) -= meanj;
        }
      }

      // Break loop if convergence is reached
      // ratio = accu(abs(x - x0) / (1.0 + abs(x0)) % w) * inv_sw;
      ratio = norm(x - x0, 2) * inv_sw;
      if (ratio < tol) break;
    }
    C.col(p) = x;
  }

  return C;
}

[[cpp11::register]] doubles_matrix<> center_variables_r_(
    const doubles_matrix<> &V_r, const doubles &w_r, const list &klist,
    const double &tol, const int &maxiter) {
  return as_doubles_matrix(
      center_variables_(as_Mat(V_r), as_Mat(w_r), klist, tol, maxiter));
}
