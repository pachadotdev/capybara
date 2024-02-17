#include <algorithm>
#include <cmath>
#include <cpp11.hpp>
#include <iostream>
#include <vector>

using namespace cpp11;
using namespace std;

[[cpp11::register]] doubles_matrix<>
center_variables2_(const doubles_matrix<> &V, const doubles &w,
                   const list &klist, const double tol, const int maxiter) {
  // Auxiliary variables (fixed)
  const int N = V.nrow();
  const int P = V.ncol();
  const int K = klist.size();

  double sw = 0.0;
  for (int i = 0; i < w.size(); i++) {
    sw += w[i];
  }

  // Auxiliary variables (storage)
  double delta, denom, meanj, num, wt;
  int index, iter, i, j, k, n, p, I, J;
  writable::doubles_matrix<> C(N, P);
  writable::doubles x(N);
  writable::doubles x0(N);

  // Halperin projections
  for (p = 0; p < P; p++) {
    // Center each variable
    for (n = 0; n < N; n++) {
      x[n] = V(n, p);
    }

    for (iter = 0; iter < maxiter; iter++) {
      // Store centered vector from the last iteration
      writable::doubles x0(N);
      for (n = 0; n < N; n++) {
        x0[n] = static_cast<double>(x[n]);
      }

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
            num += wt * x[index];
            denom += wt;
          }

          // Subtract weighted group mean
          meanj = num / denom;
          for (i = 0; i < I; i++) {
            index = indexes[i];
            x[index] -= meanj;
          }
        }
      }

      // Check convergence
      delta = 0.0;
      for (n = 0; n < N; n++) {
        delta += abs(x[n] - x0[n]) / (1.0 + abs(x0[n])) * w[n];
      }
      delta /= sw;

      if (delta < tol) {
        break;
      }
    }

    for (n = 0; n < N; n++) {
      C(n, p) = static_cast<double>(x[n]);
    }
  }

  return C;
}
