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
  size_t iter, j, k, p, J;
  double delta, meanj;
  Mat<double> C(N, P);
  Col<double> x(N);
  Col<double> x0(N);

  // Precompute group indices and weights
  field<field<uvec>> group_indices(K);
  field<vec> group_weights(K);

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
  for (p = 0; p < P; ++p) {
    // Center each variable
    x = V.col(p);

    for (iter = 0; iter < I; ++iter) {
      if (iter % 1000 == 0) {
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
      delta = accu(abs(x - x0) / (1.0 + abs(x0)) % w) * inv_sw;
      if (delta < tol) {
        break;
      }
    }
    C.col(p) = x;
  }

  // Return matrix with centered variables
  return C;
}

[[cpp11::register]] doubles_matrix<>
center_variables_r_(const doubles_matrix<> &V_r, const doubles &w_r,
                    const list &klist, const double &tol, const int &maxiter) {
  return as_doubles_matrix(
      center_variables_(as_Mat(V_r), as_Mat(w_r), klist, tol, maxiter));
}
