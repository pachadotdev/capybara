#include "00_main.h"

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const double &tol) {
  // Types conversion
  Col<double> p = as_Mat(p_r);

  // Auxiliary variables (fixed)
  const size_t K = klist.size(), max_iter = 10000;

  // Auxiliary variables (storage)
  size_t j, k, l, iter, interrupt_iter = 1000, J, J1, J2;
  double num, denom;
  Col<double> y(p.n_elem);

  // Pre-compute list sizes
  field<int> list_sizes(K);
  field<field<uvec>> group_indices(K);

  for (k = 0; k < K; ++k) {
    const list &jlist = as_cpp<list>(klist[k]);
    J = jlist.size();
    list_sizes(k) = J;
    group_indices(k).set_size(J);
    for (j = 0; j < J; ++j) {
      group_indices(k)(j) = as_uvec(as_cpp<integers>(jlist[j]));
    }
  }

  // Generate starting guess
  field<Col<double>> Alpha(K), Alpha0(K);
  for (k = 0; k < K; ++k) {
    if (list_sizes(k) > 0) {
      Alpha(k).zeros(list_sizes(k));
      Alpha0(k).zeros(list_sizes(k));
    }
  }

  // Start alternating between normal equations
  for (iter = 0; iter < max_iter; ++iter) {
    if (iter == interrupt_iter) {
      check_user_interrupt();
      interrupt_iter += 1000;
    }

    // Store alpha_0 of the previous iteration
    Alpha0 = Alpha;

    for (k = 0; k < K; ++k) {
      if (list_sizes(k) == 0) continue;  // Skip empty groups

      // Compute adjusted dependent variable
      y = p;
      for (l = 0; l < K; ++l) {
        J1 = list_sizes(l);
        if (l == k || J1 == 0) continue;
        for (j = 0; j < J1; ++j) {
          const uvec &indexes = group_indices(l)(j);
          if (!indexes.is_empty()) {
            y.elem(indexes) -= Alpha0(l)(j);
          }
        }
      }

      J2 = list_sizes(k);
      for (j = 0; j < J2; ++j) {
        // Subset the j-th group of category k
        const uvec &indexes = group_indices(k)(j);

        // Store group mean
        if (!indexes.is_empty()) {
          Alpha(k)(j) = mean(y.elem(indexes));
        }
      }
    }

    // Compute termination criterion and check convergence
    num = 0.0, denom = 0.0;
    for (k = 0; k < K; ++k) {
      if (list_sizes(k) == 0) continue;  // Skip empty groups
      const Col<double> &diff = Alpha(k) - Alpha0(k);
      num += dot(diff, diff);
      denom += dot(Alpha0(k), Alpha0(k));
    }

    if (sqrt(num / denom) < tol) {
      break;
    }
  }

  // Return alpha
  writable::list Alpha_r(K);
  for (k = 0; k < K; ++k) {
    Alpha_r[k] = as_doubles_matrix(Alpha(k).eval());  // Ensure materialization
  }

  return Alpha_r;
}
