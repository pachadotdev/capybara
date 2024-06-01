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
  int i, j, p;
  Mat<double> num(P, 1, fill::zeros);
  integers indexes;

  // Compute sum of weighted group sums
  double denom = 0.0;
  for (j = 0; j < J; j++) {
    uvec arma_indexes = as_uvec(as_cpp<integers>(jlist[j]));
    int I = arma_indexes.size();

    num.zeros();
    for (p = 0; p < P; ++p) {
      for (i = 0; i < I; i++) {
        num(p, 0) += M(arma_indexes[i], p);
      }
    }

    for (i = 0; i < I; i++) {
      denom += w(arma_indexes[i]);
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
  int j;
  Mat<double> num(P, 1, fill::zeros);

  // Compute sum of weighted group sums
  double denom = 0.0;
  for (j = 0; j < J; j++) {
    uvec arma_indexes = as_uvec(as_cpp<integers>(jlist[j]));
    // arma_indexes -= 1;

    Mat<double> M_sub = M.rows(arma_indexes);
    Mat<double> w_sub = w.rows(arma_indexes);

    num += sum(M_sub.each_col() % w_sub, 0).t();
    denom += accu(w_sub);
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
  int j;
  Mat<double> V(P, P, fill::zeros);

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    uvec arma_indexes = as_uvec(as_cpp<integers>(jlist[j]));
    // arma_indexes -= 1;

    Mat<double> M_sub = M.rows(arma_indexes);
    vec v = sum(M_sub, 0).t();
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
  int j, p, q;
  Mat<double> V(P, P, fill::zeros);

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    uvec arma_indexes = as_uvec(as_cpp<integers>(jlist[j]));

    Mat<double> M_sub = M.rows(arma_indexes);
    Mat<double> N_sub = N.rows(arma_indexes);

    for (p = 0; p < P; p++) {
      for (q = 0; q < P; q++) {
        V(q, p) += accu(M_sub.col(q) * N_sub.col(p).t());
      }
    }
  }

  // Return matrix
  return as_doubles_matrix(V);
}
