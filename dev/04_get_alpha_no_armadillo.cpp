#include <cpp11.hpp>
#include <vector>
#include <numeric>

using namespace cpp11;
using namespace std;

[[cpp11::register]] list get_alpha(const doubles& pi, const list& klist, const double tol) {
  // Auxiliary variables (fixed)
  const int n = pi.size();
  const int K = klist.size();

  // Auxiliary variables (storage)
  double crit, denom, num, sum;
  int iter, j, k, kk, t, J, T;
  vector<double> y(n);

  // Generate starting guess
  vector<vector<double>> Alpha(K);
  for (k = 0; k < K; ++k) {
    list jlist = klist[k];
    J = jlist.size();
    vector<double> Alpha_k(J, 0.0);
    Alpha[k] = Alpha_k;
  }

  // Start alternating between normal equations
  vector<vector<double>> Alpha0(K);
  for (iter = 0; iter < 10000; ++iter) {
    check_user_interrupt();

    // Store \alpha_{0} of the previous iteration
    Alpha0 = Alpha;

    // Solve normal equations of category k
    for (k = 0; k < K; ++k) {
      // Compute adjusted dependent variable
      y = vector<double>(pi.begin(), pi.end());
      for (kk = 0; kk < K; ++kk) {
        if (kk != k) {
          list jlist = klist[kk];
          J = jlist.size();
          for (j = 0; j < J; ++j) {
            integers jlist_j = as_cpp<integers>(jlist[j]);
            vector<int> indexes(jlist_j.begin(), jlist_j.end());
            T = indexes.size();
            for (t = 0; t < T; ++t) {
              y[indexes[t]] -= Alpha[kk][j];
            }
          }
        }
      }

      // Compute group mean
      list jlist = klist[k];
      J = jlist.size();
      vector<double> alpha(J);
      for (j = 0; j < J; ++j) {
        // Subset the j-th group of category k
        integers indexes = jlist[j];
        T = indexes.size();

        // Compute group sum
        sum = 0.0;
        for (t = 0; t < T; ++t) {
          sum += y[indexes[t]];
        }

        // Store group mean
        alpha[j] = sum / T;
      }

      // Update \alpha_{k}
      Alpha[k] = alpha;
    }

    // Compute termination criterion and check convergence
    num = 0.0;
    denom = 0.0;
    for (k = 0; k < K; ++k) {
      vector<double> Alpha_k = Alpha[k];
      vector<double> Alpha0_k = Alpha0[k];
      num +=
          inner_product(Alpha_k.begin(), Alpha_k.end(), Alpha_k.begin(), 0.0);
      denom += inner_product(Alpha0_k.begin(), Alpha0_k.end(), Alpha0_k.begin(),
                             0.0);
    }
    crit = sqrt(num / denom);
    if (crit < tol) {
      break;
    }
  }

  // Return \alpha
  writable::list Alpha_list(K);
  for (k = 0; k < K; ++k) {
    Alpha_list[k] = writable::doubles(Alpha[k].begin(), Alpha[k].end());
  }

  return Alpha_list;
}
