#include "00_main.h"

// Method of alternating projections (Halperin)
[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &v_sum_r,
                  const doubles &w_r, const list &klist, const double &tol,
                  const int &maxiter, const bool &sum_v) {
  // Type conversion
  Mat<double> V = as_Mat(V_r);
  Mat<double> w = as_Mat(w_r);

  if (sum_v) {
    Mat<double> v_sum = as_Mat(v_sum_r);
    V.each_col() += v_sum;
    v_sum.reset();
  }

  // Auxiliary variables (fixed)
  const int N = V.n_rows;
  const int P = V.n_cols;
  const int K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Auxiliary variables (storage)
  Mat<double> C(N, P);

  // Precompute group weights and values
  std::vector<std::vector<uvec>> group_indices(K);
  std::vector<std::vector<double>> denom_groups(K);

  for (int k = 0; k < K; ++k) {
    list jlist = klist[k];
    int J = jlist.size();
    group_indices[k].resize(J);
    denom_groups[k].resize(J);
    for (int j = 0; j < J; ++j) {
      std::vector<int> indices = as_cpp<std::vector<int>>(jlist[j]);
      group_indices[k][j] = conv_to<uvec>::from(indices);
      denom_groups[k][j] = accu(w(group_indices[k][j]));
    }
  }

  // Halperin projections
  for (int p = 0; p < P; ++p) {
    // Auxiliary variables for each parallel thread
    Mat<double> x(N, 1), x0(N, 1);
    double group_mean, delta;
    int iter, j, k, J;

    // Center each variable
    x = V.col(p);

    for (iter = 0; iter < maxiter; ++iter) {
      // Temporarily remove user interrupt check for debugging
      // if ((iter % 1000) == 0) {
      //   check_user_interrupt();
      // }

      // Store centered vector from the last iteration
      x0 = x;

      // Alternate between categories
      for (k = 0; k < K; ++k) {
        J = group_indices[k].size();

        for (j = 0; j < J; ++j) {
          // Subset j-th group of category 'k'
          const uvec &g_indices = group_indices[k][j];
          const Col<double> &x_group = x(g_indices);
          const Col<double> &w_group = w(g_indices);

          // Subtract weighted group mean
          group_mean = dot(w_group, x_group) / denom_groups[k][j];
          x(g_indices) -= group_mean;
        }
      }

      // Break loop if convergence is reached
      delta = accu(abs(x - x0) / (1.0 + abs(x0)) % w) * inv_sw;

      if (delta < tol) {
        break;
      }
    }

    C.col(p) = x;
  }

  // Return matrix with centered variables
  return as_doubles_matrix(C);
}
