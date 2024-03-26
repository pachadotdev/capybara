#include "00_main.h"

// Method of alternating projections (Halperin)
[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &v_sum_r,
                  const doubles &w_r, const list &klist, const double tol,
                  const int maxiter, bool sum_v) {
  // Types conversion
  Mat<double> V = as_Mat(V_r);
  Mat<double> w = as_Mat(w_r);

  if (sum_v) {
    Mat<double> v_sum = as_Mat(v_sum_r);
    V.each_col() += v_sum;
  }

  // Auxiliary variables (fixed)
  const int N = V.n_rows;
  const int P = V.n_cols;
  const int K = klist.size();
  const double sw = accu(w);

  // Auxiliary variables (storage)
  double delta, denom, meanj, num, wt;
  int index, iter, i, j, k, p, I, J;
  Mat<double> C(N, P);
  Mat<double> x(N, 1);
  Mat<double> x0(N, 1);

  // Halperin projections
  for (p = 0; p < P; p++) {
    // Center each variable
    x = V.col(p);

    int interruptCheckCounter = 0;

    for (iter = 0; iter < 100000; ++iter) {
      // Check user interrupt
      if (++interruptCheckCounter == 1000) {
        check_user_interrupt();
        interruptCheckCounter = 0;
      }

      // Store centered vector from the last iteration
      x0 = x;

      // Alternate between categories
      for (k = 0; k < K; k++) {
        // Compute all weighted group means of category 'k' and subtract them
        writable::list jlist = klist[k];
        J = jlist.size();
        for (j = 0; j < J; j++) {
          // Subset j-th group of category 'k'
          integers indexes = as_cpp<integers>(jlist[j]);
          I = indexes.size();

          // Compute numerator and denominator of the weighted group mean
          num = 0.0;
          denom = 0.0;
          for (i = 0; i < I; i++) {
            index = indexes[i];
            wt = w(index);
            num += wt * x(index);
            denom += wt;
          }

          // Subtract weighted group mean
          meanj = num / denom;
          for (i = 0; i < I; i++) {
            index = indexes[i];
            x(index) -= meanj;
          }
        }
      }

      // Check convergence
      delta = accu(abs(x - x0) / (1.0 + abs(x0)) % w) / sw;
      if (delta < tol) {
        break;
      }
    }
    C.col(p) = x;
  }

  // Return matrix with centered variables
  return as_doubles_matrix(C);
}
