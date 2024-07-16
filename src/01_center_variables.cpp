#include "00_main.h"

// Method of alternating projections (Halperin)
[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &v_sum_r,
                  const doubles &w_r, const list &klist, const double &tol,
                  const int &maxiter, const bool &sum_v) {
  // Type conversion
  Mat<double> V = as_Mat(V_r);
  Mat<double> w = as_Mat(w_r);

  if (sum_v) {
    Mat<double> v_sum = as_Mat(v_sum_r);
    V.each_col() += v_sum;
    v_sum.reset();
  }
  
  // Auxiliary variables (fixed)
  const int N = V.n_rows;
  const int P = V.n_cols;
  const int K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  int iter, j, k, p, J;
  double delta, meanj;
  Mat<double> x(N, 1);
  Mat<double> x0(N, 1);

  // Precompute group indices and weights
  field<field<uvec>> group_indices(K);
  field<vec> group_weights(K);

  // #ifdef _OPENMP
  // #pragma omp parallel for private(indices, j, J) schedule(static)
  // #endif
  for (k = 0; k < K; ++k) {
    list jlist = klist[k];
    J = jlist.size();
    group_indices(k) = field<uvec>(J);
    group_weights(k) = vec(J);
    for (j = 0; j < J; ++j) {
      group_indices(k)(j) = as_uvec(as_cpp<integers>(jlist[j]));
      group_weights(k)(j) = accu(w(group_indices(k)(j)));
    }
  }

  // Halperin projections
  // #ifdef _OPENMP 
  // #pragma omp parallel for private(x, x0, iter, j, k, J, meanj, delta) schedule(static)
  // #endif
  for (p = 0; p < P; ++p) {
    // Center each variable
    x = V.col(p);
    for (iter = 0; iter < maxiter; ++iter) {
      if ((iter % 1000) == 0) {
        check_user_interrupt();
      }
      // Store centered vector from the last iteration
      x0 = x;

      // Alternate between categories
      for (k = 0; k < K; ++k) {
        // Substract the weighted group means of category 'k'
        J = group_indices(k).size();
        for (j = 0; j < J; ++j) {
          // Subset j-th group of category 'k'
          const uvec &coords = group_indices(k)(j);
          meanj = dot(w(coords), x(coords)) / group_weights(k)(j);
          x.elem(coords) -= meanj;
        }
      }

      // Break loop if convergence is reached
      delta =
          accu(abs(x - x0) / (1.0 + abs(x0)) % w) * inv_sw;
      if (delta < tol) {
        break;
      }
    }
    V.col(p) = x;
  }

  // Return matrix with centered variables
  return as_doubles_matrix(V);
}
