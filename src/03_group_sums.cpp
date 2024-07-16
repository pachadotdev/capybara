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
  int i, j, I;
  Mat<double> b(P, 1, fill::zeros);
  Mat<double> num(P, 1);
  uvec indexes;
  double denom;

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    denom = 0.0;

    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    I = indexes.size();

    num.zeros();
    for (i = 0; i < I; ++i) {
      num += M.row(indexes[i]).t();
      denom += w(indexes[i]);
    }

    b += num / denom;
  }

  return as_doubles_matrix(b);
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
  int i, j, k, I;
  Mat<double> b(P, 1, fill::zeros);
  Mat<double> num(P, 1);
  double denom;

  // Compute sum of weighted group sums
  for (j = 0; j < J; j++) {
    uvec indexes = as_uvec(as_cpp<integers>(jlist[j]));
    I = indexes.size();

    num.zeros();
    denom = 0.0;

    for (i = 1; i < I; ++i) {
      for (k = 1; k <= K; ++k) {
        num += M.row(indexes[i]) * v(indexes[i - k], 0) * I / (I - 1);
      }
    }

    for (i = 0; i < I; ++i) {
      denom += w(indexes[i]);
    }

    b += num / denom;
  }

  return as_doubles_matrix(b);
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  // Types conversion
  Mat<double> M = as_Mat(M_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  int j;
  Mat<double> v(P, 1);
  Mat<double> V(P, P, fill::zeros);

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    uvec indexes = as_uvec(as_cpp<integers>(jlist[j]));

    Mat<double> M_sub = M.rows(indexes);
    v = sum(M_sub, 0).t();
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
  int j;
  size_t i, k, I;
  uvec indexes;
  Mat<double> V(P, P, fill::zeros);

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    I = indexes.n_elem;

    for (i = 0; i < I; ++i) {
      for (k = i + 1; k < I; ++k) {
        V += M.row(indexes[i]).t() * N.row(indexes[k]);
      }
    }
  }

  // Return matrix
  return as_doubles_matrix(V);
}
