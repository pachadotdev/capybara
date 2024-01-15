#include "00_main.h"

// Method of alternating projections (Halperin)
[[cpp11::register]] doubles_matrix<> center_variables_(
    const doubles_matrix<>& V, const doubles& w, const list& klist,
    const double tol, const int maxiter) {
  // Auxiliary variables (fixed)
  const int N = V.nrow();
  const int P = V.ncol();
  const int K = klist.size();

  double sw = 0.0;
  for (double val : w) {
    sw += val;
  }

  // Auxiliary variables (storage)
  double delta, denom, meanj, num, wt;
  int index, iter, i, j, k, n, p, I, J;
  writable::doubles_matrix<> C(N, P);
  writable::doubles_matrix<> x(N, 1);
  writable::doubles_matrix<> y(N, 1);

  // Halperin projections
  for (p = 0; p < P; p++) {
    // Center each variable
    for (n = 0; n < N; n++) {
      x(n, 0) = V(n, p);
    }

    int interruptCheckCounter = 0;

    for (iter = 0; iter < maxiter; iter++) {
      // Check user interrupt
      if (++interruptCheckCounter == 1000) {
        check_user_interrupt();
        interruptCheckCounter = 0;
      }

      // Store centered vector from the last iteration
      doubles_matrix<> y = x;

      // Alternate between categories
      for (k = 0; k < K; k++) {
        // Compute all weighted group means of category 'k' and subtract them
        writable::list jlist = klist[k];
        J = jlist.size();
        for (j = 0; j < J; j++) {
          // Subset j-th group of category 'k'

          // integers indexes = jlist[j];
          // In cpp11, you can't directly assign a list element to an
          // integers object as you could in `Rcpp`
          integers indexes = as_cpp<integers>(jlist[j]);

          I = indexes.size();

          // Compute numerator and denominator of the weighted group mean
          num = 0.0;
          denom = 0.0;
          for (i = 0; i < I; i++) {
            index = indexes[i];
            wt = w[index];
            num += wt * x(index, 0);
            denom += wt;
          }

          // Subtract weighted group mean
          meanj = num / denom;
          for (i = 0; i < I; i++) {
            index = indexes[i];
            x(index, 0) -= meanj;
          }
        }
      }

      // Check convergence
      delta = 0.0;
      for (n = 0; n < N; n++) {
        delta += abs(x(n, 0) - y(n, 0)) / (1.0 + abs(y(n, 0))) * w[n];
      }
      delta /= sw;

      if (delta < tol) {
        break;
      }
    }

    for (n = 0; n < N; n++) {
      C(n, p) = static_cast<double>(x(n, 0));
    }
  }

  return C;
}
