#ifndef CAPYBARA_GROUP_SUMS
#define CAPYBARA_GROUP_SUMS

struct GroupSumsResult {
  mat result;
  cpp11::doubles_matrix<> to_matrix() const {
    return as_doubles_matrix(result);
  }
};

inline GroupSumsResult group_sums(const mat &M, const mat &w,
                                  const list &jlist) {
  const size_t J = jlist.size(), P = M.n_cols;
  size_t j;
  uvec indexes;
  Row<double> groupSum(P, fill::none);
  double denom;
  mat b(P, 1, fill::zeros);
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    groupSum = sum(M.rows(indexes), 0);
    denom = accu(w.elem(indexes));
    b += groupSum.t() / denom;
  }
  GroupSumsResult res;
  res.result = b;
  return res;
}

inline GroupSumsResult group_sums_spectral(const mat &M, const mat &v,
                                           const mat &w, int K,
                                           const list &jlist) {
  const size_t J = jlist.size(), K1 = K, P = M.n_cols;
  size_t i, j, k, I;
  uvec indexes;
  vec num(P, fill::none), v_shifted;
  mat b(P, 1, fill::zeros);
  double denom;
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
  GroupSumsResult res;
  res.result = b;
  return res;
}

inline GroupSumsResult group_sums_var(const mat &M, const list &jlist) {
  const int J = jlist.size();
  const int P = M.n_cols;
  int j;
  mat v(P, 1, fill::none), V(P, P, fill::zeros);
  uvec indexes;
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    v = sum(M.rows(indexes), 0).t();
    V += v * v.t();
  }
  GroupSumsResult res;
  res.result = V;
  return res;
}

inline GroupSumsResult group_sums_cov(const mat &M, const mat &N,
                                      const list &jlist) {
  const int J = jlist.size();
  const int P = M.n_cols;
  int j;
  uvec indexes;
  mat V(P, P, fill::zeros);
  for (j = 0; j < J; ++j) {
    indexes = as_uvec(as_cpp<integers>(jlist[j]));
    if (indexes.n_elem < 2)
      continue;
    V += M.rows(indexes).t() * N.rows(indexes);
  }
  GroupSumsResult res;
  res.result = V;
  return res;
}

#endif // CAPYBARA_GROUP_SUMS
