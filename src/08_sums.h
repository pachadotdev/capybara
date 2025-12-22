#ifndef CAPYBARA_SUMS_H
#define CAPYBARA_SUMS_H

namespace capybara {

mat group_sums(const mat &M, const mat &w, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem, P = M.n_cols;

  Row<double> groupSum(P, fill::none);
  double denom;
  mat b(P, 1, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    groupSum = sum(M.rows(indexes), 0);
    denom = accu(w.elem(indexes));

    b += groupSum.t() / denom;
  }

  return b;
}

mat group_sums_spectral(const mat &M, const mat &v, const mat &w, const size_t K,
                        const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem, K1 = K, P = M.n_cols;

  vec num(P, fill::none), v_shifted;
  mat b(P, 1, fill::zeros);
  double denom;

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    const uword I = indexes.n_elem;

    if (I <= 1)
      continue;

    num.fill(0.0);
    denom = accu(w.elem(indexes));

    v_shifted.zeros(I);
    for (uword k = 1; k <= K1 && k < I; ++k) {
      for (uword i = 0; i < I - k; ++i) {
        v_shifted(i + k) += v(indexes(i));
      }
    }

    num = M.rows(indexes).t() * (v_shifted * (I / (I - 1.0)));
    b += num / denom;
  }

  return b;
}

mat group_sums_var(const mat &M, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat v(P, 1, fill::none), V(P, P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    v = sum(M.rows(indexes), 0).t();
    V += v * v.t();
  }

  return V;
}

mat group_sums_cov(const mat &M, const mat &N,
                   const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat V(P, P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);

    if (indexes.n_elem < 2) {
      continue;
    }

    V += M.rows(indexes).t() * N.rows(indexes);
  }

  return V;
}

} // namespace capybara

#endif // CAPYBARA_SUMS_H
