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
  double denom;
  int i, j, p, I;
  Mat<double> num(P, 1);

  // Compute sum of weighted group sums
  for (j = 0; j < J; j++) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Compute numerator of the weighted group sum
    num.zeros();
    for (p = 0; p < P; ++p) {
      for (i = 0; i < I; i++) {
        num(p, 0) += M(indexes[i], p);
      }
    }

    // Compute denominator of the weighted group sum
    denom = 0.0;
    for (i = 0; i < I; i++) {
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
  double denom;
  int i, j, k, p, I;
  Mat<double> num(P, 1);

  // Compute sum of weighted group sums
  for (j = 0; j < J; j++) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Compute numerator of the weighted group sum given bandwidth 'L'
    num.zeros();
    for (p = 0; p < P; p++) {
      for (k = 1; k <= K; k++) {
        for (i = k; i < I; i++) {
          num(p, 0) += M(indexes[i], p) * v(indexes[i - k]) * I / (I - k);
        }
      }
    }

    // Compute denominator of the weighted group sum
    denom = 0.0;
    for (i = 0; i < I; i++) {
      denom += w(indexes[i]);
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
  int i, j, p, I;
  Mat<double> v(P, 1);
  Mat<double> V(P, P);

  // Compute covariance matrix
  V.zeros();
  for (j = 0; j < J; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Compute group sum
    v.zeros();
    for (p = 0; p < P; p++) {
      for (i = 0; i < I; ++i) {
        v(p) += M(indexes[i], p);
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

  // Auxiliary variables (storage)
  int i, j, p, q, s, I;
  Mat<double> V(P, P);

  // Compute covariance matrix
  V.zeros();
  for (j = 0; j < J; ++j) {
    // Subset j-th group
    integers indexes = as_cpp<integers>(jlist[j]);
    I = indexes.size();

    // Add to covariance matrix
    for (p = 0; p < P; p++) {
      for (q = 0; q < P; q++) {
        for (i = 0; i < I; i++) {
          for (s = i + 1; s < I; s++) {
            V(q, p) += M(indexes[i], q) * N(indexes[s], p);
          }
        }
      }
    }
  }

  // Return matrix
  return as_doubles_matrix(V);
}
