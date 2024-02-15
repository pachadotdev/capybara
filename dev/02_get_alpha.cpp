#include "00_main.h"

[[cpp11::register]] list get_alpha_(const doubles_matrix<>& p,
                                    const list& klist, const double tol) {
  // Auxiliary variables (fixed)
  const int N = p.nrow();
  const int K = klist.size();

  // Auxiliary variables (storage)
  double crit, denom, num, sum;
  int iter, i, j, k, l, I, J;
  writable::doubles_matrix<> y(N, 1);

  // Generate starting guess
  writable::list Alpha(K);
  for (k = 0; k < K; k++) {
    list jlist = klist[k];
    J = jlist.size();
    writable::doubles_matrix<> zeros(J, 1);
    for (j = 0; j < J; j++) {
      zeros(j, 0) = 0.0;
    }
    Alpha[k] = zeros;
  }

  // Start alternating between normal equations
  writable::list Alpha0(Alpha.size());

  int interruptCheckCounter = 0;

  for (iter = 0; iter < 10000; iter++) {
    if (++interruptCheckCounter == 1000) {
      check_user_interrupt();
      interruptCheckCounter = 0;
    }

    // Store \alpha_{0} of the previous iteration
    Alpha0 = Alpha;

    // Solve normal equations of category k
    for (k = 0; k < K; k++) {
      // Compute adjusted dependent variable
      writable::doubles_matrix<> y(N, 1);
      for (i = 0; i < N; i++) {
        y(i, 0) = p(i, 0);
      }

      for (l = 0; l < K; l++) {
        if (l != k) {
          doubles_matrix<> Alpha_l = as_cpp<doubles_matrix<>>(Alpha[l]);
          list jlist = klist[l];
          J = jlist.size();
          for (j = 0; j < J; j++) {
            integers indexes = jlist[j];
            I = indexes.size();
            for (i = 0; i < I; i++) {
              y(indexes[i], 0) -= Alpha_l(j, 0);
            }
          }
        }
      }

      // Compute group mean
      list jlist = klist[k];
      J = jlist.size();
      writable::doubles_matrix<> alpha(J, 1);
      for (j = 0; j < J; j++) {
        // Subset the j-th group of category k
        integers indexes = jlist[j];
        I = indexes.size();

        // Compute group sum
        sum = 0.0;
        for (i = 0; i < I; i++) {
          sum += y(indexes[i], 0);
        }

        // Store group mean
        alpha(j, 0) = sum / I;
      }

      // Update \alpha_{k}
      Alpha[k] = alpha;
    }

    // Compute termination criterion and check convergence
    num = 0.0;
    denom = 0.0;
    for (k = 0; k < K; k++) {
      doubles_matrix<> Alpha_k = as_cpp<doubles_matrix<>>(Alpha[k]);
      doubles_matrix<> Alpha0_k = as_cpp<doubles_matrix<>>(Alpha0[k]);
      for (j = 0; j < J; j++) {
        num += pow(Alpha_k(j, 0) - Alpha0_k(j, 0), 2);
        denom += pow(Alpha0_k(j, 0), 2);
      }
    }
    crit = sqrt(num / denom);
    if (crit < tol) {
      break;
    }
  }

  return Alpha;
}
