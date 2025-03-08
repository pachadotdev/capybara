#include "00_main.h"

// Method of alternating projections (Halperin) with Irons & Tuck acceleration
void center_variables_(Mat<double> &V, const Col<double> &w, const list &klist,
                       const double &tol, const int &maxiter) {
  // Auxiliary variables (fixed)
  const size_t I = static_cast<size_t>(maxiter);
  const size_t N = V.n_rows;
  const size_t P = V.n_cols;
  const size_t K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  size_t iter, j, k, J, interrupt_iter = 1000;
  double meanj, ratio;

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

  // Halperin projections with Irons & Tuck acceleration
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(iter, k, j, J, meanj,       \
                                                   ratio) \
    shared(V, w, group_indices, group_inverse_weights)
#endif
  for (size_t p = 0; p < P; ++p) {
    // Center each variable
    Col<double> x = V.col(p);
    Col<double> x0(N);

    for (iter = 0; iter < I; ++iter) {
      if (iter == interrupt_iter) {
#pragma omp critical
        check_user_interrupt();
        interrupt_iter += 1000;
      }

      // Store centered vector from the last iteration
      x0 = x;

      // Alternate between categories
      for (k = 0; k < K; ++k) {
        // Subtract the weighted group means of category 'k'
        J = group_indices(k).size();
        if (J == 0)
          continue; // Skip empty groups

        for (j = 0; j < J; ++j) {
          // Subset j-th group of category 'k'
          const uvec &coords = group_indices(k)(j);
          meanj = dot(w.elem(coords), x.elem(coords)) * group_inverse_weights(k)(j);
          x.elem(coords) -= meanj;
        }
      }

      // Break loop if convergence is reached
      ratio = accu(abs(x - x0) / (1.0 + abs(x0)) % w) * inv_sw;
      if (ratio < tol)
        break;
    }
    V.col(p) = x;
  }
}

[[cpp11::register]] doubles_matrix<>
center_variables_r_(const doubles_matrix<> &V_r, const doubles &w_r,
                    const list &klist, const double &tol, const int &maxiter) {
  Mat<double> V = as_Mat(V_r);
  Col<double> w = as_Col(w_r);
  center_variables_(V, w, klist, tol, maxiter);
  return as_doubles_matrix(V);
}
