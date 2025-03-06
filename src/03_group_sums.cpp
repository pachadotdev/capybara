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
  int j;
  uvec indexes;
  Mat<double> b(P, 1, fill::zeros);

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    Row<double> groupSum = sum(M.rows(indexes), 0);
    double denom = accu(w.elem(indexes));

    b += groupSum.t() / denom;
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
  int j, I;
  Mat<double> b(P, 1, fill::zeros);
  double denom;

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    uvec indexes = as_uvec(as_cpp<integers>(jlist[j]));
    I = indexes.n_elem;

    if (I <= 1)
      continue;

    Col<double> num(P, fill::zeros);
    denom = accu(w.elem(indexes));

    Col<double> v_shifted(I, fill::zeros);
    for (int k = 1; k <= K && k < I; ++k) {
      v_shifted.subvec(k, I - 1) += v.elem(indexes.subvec(0, I - k - 1));
    }

    num = M.rows(indexes).t() * (v_shifted * (I / (I - 1.0)));

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

    Col<double> v = sum(M.rows(indexes), 0).t();

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
  uvec indexes;
  Mat<double> V(P, P, fill::zeros);

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));

    if (indexes.n_elem < 2) {
      continue;
    }

    V += M.rows(indexes).t() * N.rows(indexes);
  }

  return as_doubles_matrix(V);
}
