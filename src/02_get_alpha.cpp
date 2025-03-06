#include "00_main.h"

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const double &tol) {
  // Types conversion
  Mat<double> p = as_Mat(p_r);

  // Auxiliary variables (fixed)
  const size_t N = p.n_rows;
  const size_t K = klist.size();
  const size_t max_iter = 10000;

  // Auxiliary variables (storage)
  double crit, denom, num;
  size_t iter, k, l;
  Col<double> y(N);
  size_t interrupt_iter = 1000;

  // Pre-compute list sizes
  field<int> list_sizes(K);
  for (k = 0; k < K; ++k) {
    list_sizes(k) = as_cpp<list>(klist[k]).size();
  }

  // Generate starting guess
  field<Mat<double>> Alpha(K);
  for (k = 0; k < K; ++k) {
    Alpha(k) = zeros<Col<double>>(list_sizes[k]);
  }

  // Start alternating between normal equations
  field<Mat<double>> Alpha0(K);

  for (iter = 0; iter < max_iter; ++iter) {
    if (iter == interrupt_iter) {
      check_user_interrupt();
      interrupt_iter += 1000;
    }
    // Store alpha_0 of the previous iteration
    Alpha0 = Alpha;

    for (k = 0; k < K; ++k) {
      // Compute adjusted dependent variable
      y = p;

      for (l = 0; l < K; ++l) {
        if (l != k) {
          const list &klist_l = klist[l];
          for (int j = 0; j < list_sizes[l]; ++j) {
            uvec indexes = as_uvec(as_cpp<integers>(klist_l[j]));
            y(indexes) -= Alpha(l)(j);
          }
        }
      }

      const list &klist_k = as_cpp<list>(klist[k]);
      Mat<double> &alpha = Alpha(k);

      for (int j = 0; j < list_sizes[k]; ++j) {
        // Subset the j-th group of category k
        uvec indexes = as_uvec(as_cpp<integers>(klist_k[j]));

        // Store group mean
        alpha(j) = mean(y(indexes));
      }
    }

    // Compute termination criterion and check convergence
    num = 0.0;
    denom = 0.0;
    for (k = 0; k < K; ++k) {
      const Mat<double> &diff = Alpha(k) - Alpha0(k);
      num += accu(diff % diff);
      denom += accu(Alpha0(k) % Alpha0(k));
    }

    crit = sqrt(num / denom);
    if (crit < tol) {
      break;
    }
  }

  // Return alpha
  writable::list Alpha_r(K);
  for (k = 0; k < K; ++k) {
    Alpha_r[k] = as_doubles_matrix(Alpha(k));
  }

  return Alpha_r;
}
