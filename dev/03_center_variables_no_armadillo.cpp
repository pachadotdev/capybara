#include "00_main.hpp"
#include "02_cpp11_to_from_arma.hpp"

// Method of alternating projections (Halperin)
[[cpp11::register]] doubles_matrix<> center_variables_(
    const doubles_matrix<>& V, const doubles& w, const list& klist,
    const double tol) {
  // Auxiliary variables (fixed)
  const int n = V.nrow();
  const int K = klist.size();
  const int P = V.ncol();
  const double sw = accumulate(w.begin(), w.end(), 0.0);

  // Auxiliary variables (storage)
  double delta, denom, meanj, num, wt;
  int index, iter, j, k, p, t, J, T;
  writable::doubles_matrix<> M(n, P);
  writable::doubles x(n);
  writable::doubles y(n);

  // Halperin projections
  for (p = 0; p < P; ++p) {
    // Center each variable
    for (int i = 0; i < n; ++i) {
      x[i] = V(i, p);
    }

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
            wt = w[index];
            num += wt * x[index];
            denom += wt;
          }

          // Subtract weighted group mean
          meanj = num / denom;
          for (t = 0; t < T; ++t) {
            index = indexes[t];
            x[index] -= meanj;
          }
        }
      }

      // Check convergence
      delta = 0.0;
      for (int i = 0; i < n; ++i) {
        delta += abs(x[i] - y[i]) / (1.0 + abs(y[i])) * w[i];
      }
      delta /= sw;

      if (delta < tol) {
        break;
      }
    }

    for (int i = 0; i < n; ++i) {
      M(i, p) = static_cast<double>(x[i]);
    }
  }

  // Return matrix with centered variables
  return M;
}
