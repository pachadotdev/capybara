#include "00_main.h"

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const double &tol) {
  // Types conversion
  Mat<double> p = as_Mat(p_r);

  // Auxiliary variables (fixed)
  const int N = p.n_rows;
  const int K = klist.size();

  // Auxiliary variables (storage)
  double crit, denom, num;
  int iter, k, l;
  Mat<double> y(N, 1);

  // Pre-compute list sizes
  std::vector<int> list_sizes(K);
  for (k = 0; k < K; ++k) {
    list_sizes[k] = as_cpp<list>(klist[k]).size();
  }

  // Generate starting guess
  field<Mat<double>> Alpha(K);
  for (k = 0; k < K; ++k) {
    Alpha(k) = zeros<Mat<double>>(list_sizes[k], 1);
  }

  // Start alternating between normal equations
  field<Mat<double>> Alpha0(size(Alpha));

  for (iter = 0; iter < 10000; ++iter) {
    // Check user interrupt
    if ((iter % 1000) == 0) {
      check_user_interrupt();
    }

    // Store alpha_0 of the previous iteration
    Alpha0 = Alpha;

    for (k = 0; k < K; ++k) {
      // Compute adjusted dependent variable
      y = p;

      for (l = 0; l < K; ++l) {
        if (l != k) {
          list klist_l = klist[l];
          for (int j = 0; j < list_sizes[l]; ++j) {
            uvec indexes =
                conv_to<uvec>::from(as_cpp<std::vector<int>>(klist_l[j]));
            y(indexes) -= Alpha(l)(j);
          }
        }
      }

      list klist_k = as_cpp<list>(klist[k]);
      Mat<double> alpha(list_sizes[k], 1);

      for (int j = 0; j < list_sizes[k]; ++j) {
        // Subset the j-th group of category k
        uvec indexes =
            conv_to<uvec>::from(as_cpp<std::vector<int>>(klist_k[j]));

        // Store group mean
        alpha(j) = mean(y(indexes));
      }

      // Update alpha_k
      Alpha(k) = alpha;
    }

    // Compute termination criterion and check convergence
    num = 0.0;
    denom = 0.0;
    for (k = 0; k < K; ++k) {
      Mat<double> diff = Alpha(k) - Alpha0(k);
      num += accu(diff % diff);
      denom += accu(Alpha0(k) % Alpha0(k));
      Alpha0(k) = Alpha(k);
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
