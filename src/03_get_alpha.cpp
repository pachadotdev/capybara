#include "00_main.h"

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const double tol) {
  // Types conversion
  Mat<double> p = as_Mat(p_r);

  // Auxiliary variables (fixed)
  const int N = p.n_rows;
  const int K = klist.size();

  // Auxiliary variables (storage)
  double crit, denom, num, sum;
  int iter, i, j, k, l, I, J;
  Mat<double> y(N, 1);

  // Generate starting guess
  arma::field<arma::vec> Alpha(K);
  for (k = 0; k < K; k++) {
    J = as_cpp<list>(klist[k]).size();
    Alpha(k) = arma::zeros(J);
  }

  // Start alternating between normal equations
  arma::field<arma::vec> Alpha0(arma::size(Alpha));

  // Create vector for alpha
  arma::vec alpha(J);

  int interruptCheckCounter = 0;

  for (iter = 0; iter < 10000; iter++) {
    // Check user interrupt
    if (++interruptCheckCounter == 1000) {
      check_user_interrupt();
      interruptCheckCounter = 0;
    }
    // Store \alpha_{0} of the previous iteration
    Alpha0 = Alpha;

    // Solve normal equations of category k
    for (k = 0; k < K; k++) {
      // Compute adjusted dependent variable
      y = p;

      for (l = 0; l < K; l++) {
        if (l != k) {
          J = as_cpp<list>(klist[l]).size();
          for (j = 0; j < J; j++) {
            integers indexes = as_cpp<list>(klist[l])[j];
            I = indexes.size();
            for (i = 0; i < I; i++) {
              y(indexes[i]) -= Alpha(j)(j);
            }
          }
        }
      }

      // Compute group mean
      J = as_cpp<list>(klist[k]).size();
      for (j = 0; j < J; j++) {
        // Subset the j-th group of category k
        integers indexes = as_cpp<list>(klist[k])[j];
        I = indexes.size();

        // Compute group sum
        sum = 0.0;
        for (i = 0; i < I; i++) {
          sum += y(indexes[i]);
        }

        // Store group mean
        alpha(j) = sum / I;
      }

      // Update \alpha_{k}
      Alpha(k) = alpha;
    }

    // Compute termination criterion and check convergence
    num = 0.0;
    denom = 0.0;
    for (k = 0; k < K; k++) {
      num += arma::accu(arma::pow(Alpha(k) - Alpha0(k), 2));
      denom += arma::accu(arma::pow(Alpha0(k), 2));
    }
    crit = sqrt(num / denom);
    if (crit < tol) {
      break;
    }
  }

  // Return alpha
  writable::list Alpha_r(K);
  for (k = 0; k < K; k++) {
    Alpha_r[k] = as_doubles_matrix(Alpha(k));
  }
  return Alpha_r;
}
