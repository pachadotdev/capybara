#include "00_main.hpp"
#include "02_cpp11_to_from_arma.hpp"

field<Col<double>> get_alpha_field_(const Col<double>& pi, const list& klist,
                                    const double tol) {
  // Auxiliary variables (fixed)
  const int n = pi.n_rows;
  const int K = klist.size();

  // Auxiliary variables (storage)
  double crit, denom, num, sum;
  int iter, j, k, kk, t, J, T;
  Col<double> y(n);

  // Generate starting guess
  field<Col<double>> Alpha(K);
  for (k = 0; k < K; ++k) {
    list jlist = klist[k];
    J = jlist.size();
    Alpha(k) = zeros(J);
  }

  // Start alternating between normal equations
  field<Col<double>> Alpha0(Alpha.size());
  for (iter = 0; iter < 10000; ++iter) {
    check_user_interrupt();

    // Store \alpha_{0} of the previous iteration
    Alpha0 = Alpha;

    // Solve normal equations of category k
    for (k = 0; k < K; ++k) {
      // Compute adjusted dependent variable
      y = pi;
      for (kk = 0; kk < K; ++kk) {
        if (kk != k) {
          list jlist = klist[kk];
          J = jlist.size();
          for (j = 0; j < J; ++j) {
            integers indexes = jlist[j];
            T = indexes.size();
            for (t = 0; t < T; ++t) {
              y[indexes[t]] -= Alpha(kk)(j);
            }
          }
        }
      }

      // Compute group mean
      list jlist = klist[k];
      J = jlist.size();
      Col<double> alpha(J);
      for (j = 0; j < J; ++j) {
        // Subset the j-th group of category k
        integers indexes = jlist[j];
        T = indexes.size();

        // Compute group sum
        sum = 0.0;
        for (t = 0; t < T; ++t) {
          sum += y(indexes[t]);
        }

        // Store group mean
        alpha(j) = sum / T;
      }

      // Update \alpha_{k}
      Alpha(k) = alpha;
    }

    // Compute termination criterion and check convergence
    num = 0.0;
    denom = 0.0;
    for (k = 0; k < K; ++k) {
      num += accu(pow(Alpha(k) - Alpha0(k), 2));
      denom += accu(pow(Alpha0(k), 2));
    }
    crit = sqrt(num / denom);
    if (crit < tol) {
      break;
    }
  }

  // Return \alpha
  return Alpha;
}

[[cpp11::register]] list get_alpha_(const doubles& pi0, const list& klist,
                                    const double tol) {
  // Cast pi0 to Col<double>
  Col<double> pi = doubles_to_Vec_(pi0);

  field<Col<double>> Alpha0 = get_alpha_field_(pi, klist, tol);

  int Alpha0_len = Alpha0.n_rows;

  writable::list Alpha(Alpha0_len);
  for (int k = 0; k < Alpha0_len; ++k) {
    Col<double> Alpha0_k = Alpha0[k];

    // doubles Alpha_k = Vec_to_doubles_(Alpha0_k);

    // 100% equivalent to alpaca
    doubles_matrix<> Alpha_k = Vec_to_doubles_matrix_(Alpha0_k);

    Alpha[k] = Alpha_k;
  }

  return Alpha;
}
