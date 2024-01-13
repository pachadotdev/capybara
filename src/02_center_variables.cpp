#include "00_main.h"
#include "01_cpp11_to_from_arma.h"

// Method of alternating projections (Halperin)
[[cpp11::register]] doubles_matrix<> center_variables_(
    const doubles_matrix<>& V0, const doubles& w0, const list& klist,
    const double tol) {
  // Cast to Armadillo types
  Mat<double> V = doubles_matrix_to_Mat_(V0);
  Col<double> w = doubles_to_Vec_(w0);

  // Auxiliary variables (fixed)
  const int n = V.n_rows;
  const int K = klist.size();
  const int P = V.n_cols;
  const double sw = accu(w);

  // Auxiliary variables (storage)
  double delta, denom, meanj, num, wt;
  int index, iter, j, k, p, t, J, T;
  Mat<double> M(n, P);
  Col<double> x(n);
  Col<double> y(n);

  // Halperin projections
  for (p = 0; p < P; ++p) {
    // Center each variable
    x = V.col(p);
    for (iter = 0; iter < 100000; ++iter) {
      // Check user interrupt
      check_user_interrupt();

      // Store centered vector from the last iteration
      y = x;

      // Alternate between categories
      for (k = 0; k < K; ++k) {
        // Compute all weighted group means of category 'k' and subtract them
        writable::list jlist = klist[k];
        J = jlist.size();
        for (j = 0; j < J; ++j) {
          // Subset j-th group of category 'k'

          // integers indexes = jlist[j];
          // In cpp11, you can't directly assign a list element to an
          // integers object as you could in `Rcpp`
          integers indexes = as_cpp<integers>(jlist[j]);

          T = indexes.size();

          // Compute numerator and denominator of the weighted group mean
          num = 0.0;
          denom = 0.0;
          for (t = 0; t < T; ++t) {
            index = indexes[t];
            wt = w(index);
            num += wt * x(index);
            denom += wt;
          }

          // Subtract weighted group mean
          meanj = num / denom;
          for (t = 0; t < T; ++t) {
            index = indexes[t];
            x(index) -= meanj;
          }
        }
      }

      // Check convergence
      delta = accu(abs(x - y) / (1.0 + abs(y)) % w) / sw;
      if (delta < tol) {
        break;
      }
    }
    M.col(p) = x;
  }

  // Return matrix with centered variables
  return Mat_to_doubles_matrix(M);
}
