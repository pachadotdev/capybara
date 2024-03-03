#include "00_main.h"

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  // Types conversion
  Mat<double> M = as_Mat(M_r);
  Mat<double> w = as_Mat(w_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  Mat<double> num(P, 1);
  std::vector<int> indexes;

  // Compute sum of weighted group sums
  double denom = 0.0;

  for (int j = 0; j < J; j++) {
    // Subset j-th group
    indexes = as_cpp<std::vector<int>>(jlist[j]);
    int I = indexes.size();

    // Compute numerator of the weighted group sum
    num.zeros();
    for (int p = 0; p < P; ++p) {
      for (int i = 0; i < I; i++) {
        num(p, 0) += M(indexes[i], p);
      }
    }

    // Compute denominator of the weighted group sum
    for (int i = 0; i < I; i++) {
      denom += w(indexes[i]);
    }
  }

  num = num / denom;

  return as_doubles_matrix(num);
}

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const int K,
                     const list &jlist) {
  // Types conversion
  Mat<double> M = as_Mat(M_r);
  Mat<double> v = as_Mat(v_r);
  Mat<double> w = as_Mat(w_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  Mat<double> num(P, 1);

  // Compute sum of weighted group sums
  double denom = 0.0;

  // Precompute weights and values
  std::vector<double> weights(J);
  std::vector<std::vector<double>> values(J, std::vector<double>(P));

  for (int j = 0; j < J; j++) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    int I = indexes.size();

    for (int i = 0; i < I; i++) {
      weights[i] = w[indexes[i]];
      for (int p = 0; p < P; p++) {
        values[i][p] = M(indexes[i], p);
      }
    }

    // Compute numerator of the weighted group sum
    num.zeros();
    for (int p = 0; p < P; ++p) {
      for (int i = 0; i < I; i++) {
        num(p, 0) += values[i][p];
      }
    }

    // Compute denominator of the weighted group sum
    for (int i = 0; i < I; i++) {
      denom += weights[i];
    }
  }

  num = num / denom;

  return as_doubles_matrix(num);
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  // Types conversion
  Mat<double> M = as_Mat(M_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  Mat<double> v(P, 1);
  Mat<double> V(P, P);

  // Precompute values
  std::vector<std::vector<double>> values(J, std::vector<double>(P));

  // Compute covariance matrix
  V.zeros();
  for (int j = 0; j < J; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    int I = indexes.size();

    for (int i = 0; i < I; i++) {
      for (int p = 0; p < P; p++) {
        values[j][p] = M(indexes[i], p);
      }
    }

    // Compute group sum
    v.zeros();
    for (int p = 0; p < P; p++) {
      for (int i = 0; i < I; ++i) {
        v[p] += values[j][p];
      }
    }

    // Add to covariance matrix
    V += v * v.t();
  }

  return as_doubles_matrix(V);
}

[[cpp11::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  // Types conversion
  Mat<double> M = as_Mat(M_r);
  Mat<double> N = as_Mat(N_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;
  const int I = as_cpp<integers>(jlist[0])
                    .size(); // assuming all groups have the same size

  // Auxiliary variables (storage)
  Mat<double> V(P, P);

  // Precompute values
  std::vector<std::vector<double>> M_values(I, std::vector<double>(P));
  std::vector<std::vector<double>> N_values(I, std::vector<double>(P));

  // Compute covariance matrix
  V.zeros();
  for (int j = 0; j < J; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);

    for (int i = 0; i < I; i++) {
      for (int p = 0; p < P; p++) {
        M_values[i][p] = M(indexes[i], p);
        N_values[i][p] = N(indexes[i], p);
      }
    }

    // Add to covariance matrix
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < P; q++) {
        for (int i = 0; i < I; i++) {
          for (int s = i + 1; s < I; s++) {
            V(q, p) += M_values[i][q] * N_values[s][p];
          }
        }
      }
    }
  }

  // Return matrix
  return as_doubles_matrix(V);
}
