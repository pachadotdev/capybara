#include "00_main.h"

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);
  const mat w = as_Mat(w_r);

  // Auxiliary variables (fixed)
  const size_t J = jlist.size(), P = M.n_cols;

  // Auxiliary variables (storage)
  size_t j;
  uvec indexes;
  Row<double> groupSum(P, fill::none);
  double denom;
  mat b(P, 1, fill::zeros);

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    groupSum = sum(M.rows(indexes), 0);
    denom = accu(w.elem(indexes));

    b += groupSum.t() / denom;
  }

  return as_doubles_matrix(b);
}

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const int K,
                     const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);
  const mat v = as_Mat(v_r);
  const mat w = as_Mat(w_r);

  // Auxiliary variables (fixed)
  const size_t J = jlist.size(), K1 = K, P = M.n_cols;

  // Auxiliary variables (storage)
  size_t i, j, k, I;
  uvec indexes;
  vec num(P, fill::none), v_shifted;
  mat b(P, 1, fill::zeros);
  double denom;

  // Compute sum of weighted group sums
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    I = indexes.n_elem;

    if (I <= 1)
      continue;

    num.fill(0.0);

    denom = accu(w.elem(indexes));

    v_shifted.zeros(I);
    for (k = 1; k <= K1 && k < I; ++k) {
      for (i = 0; i < I - k; ++i) {
        v_shifted(i + k) += v(indexes(i));
      }
    }

    num = M.rows(indexes).t() * (v_shifted * (I / (I - 1.0)));

    b += num / denom;
  }

  return as_doubles_matrix(b);
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  int j;
  mat v(P, 1, fill::none), V(P, P, fill::zeros);
  uvec indexes;

  // Compute covariance matrix
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));

    v = sum(M.rows(indexes), 0).t();

    V += v * v.t();
  }

  return as_doubles_matrix(V);
}

[[cpp11::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  // Types conversion
  const mat M = as_Mat(M_r);
  const mat N = as_Mat(N_r);

  // Auxiliary variables (fixed)
  const int J = jlist.size();
  const int P = M.n_cols;

  // Auxiliary variables (storage)
  int j;
  uvec indexes;
  mat V(P, P, fill::zeros);

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
