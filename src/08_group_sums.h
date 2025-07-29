// Group sum operations for various statistical computations

#ifndef CAPYBARA_GROUP_SUMS_H
#define CAPYBARA_GROUP_SUMS_H

namespace capybara {

struct GroupSums {
  mat result;

  GroupSums() = default;

  explicit GroupSums(size_t rows, size_t cols)
      : result(rows, cols, fill::zeros) {}

  cpp11::doubles_matrix<> to_matrix() const {
    return as_doubles_matrix(result);
  }
};

inline GroupSums group_sums(const mat &M, const vec &w,
                            const field<uvec> &group_indices) {
  const size_t J = group_indices.n_elem;
  const size_t P = M.n_cols;

  GroupSums result(P, 1);

  for (size_t j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    Row<double> groupSum = sum(M.rows(indexes), 0);
    double denom = accu(w.elem(indexes));
    result.result += groupSum.t() / denom;
  }

  return result;
}

inline GroupSums group_sums_spectral(const mat &M, const mat &v, const mat &w,
                                     size_t K,
                                     const field<uvec> &group_indices) {
  const size_t J = group_indices.n_elem;
  const size_t P = M.n_cols;

  GroupSums result(P, 1);
  vec num(P, fill::none);
  vec v_shifted;

  for (size_t j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    size_t I = indexes.n_elem;

    if (I <= 1) {
      continue;
    }

    num.fill(0.0);
    double denom = accu(w.elem(indexes));
    v_shifted.zeros(I);
    vec v_indexed = v.elem(indexes);

    for (size_t k = 1; k <= K && k < I; ++k) {

      v_shifted.subvec(k, I - 1) += v_indexed.subvec(0, I - k - 1);
    }

    num = M.rows(indexes).t() * (v_shifted * (I / (I - 1.0)));
    result.result += num / denom;
  }

  return result;
}

inline GroupSums group_sums_var(const mat &M,
                                const field<uvec> &group_indices) {
  const size_t J = group_indices.n_elem;
  const size_t P = M.n_cols;

  GroupSums result(P, P);
  vec v(P, fill::none);

  for (size_t j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    v = sum(M.rows(indexes), 0).t();
    result.result += v * v.t();
  }

  return result;
}

inline GroupSums group_sums_cov(const mat &M, const mat &N,
                                const field<uvec> &group_indices) {
  const size_t J = group_indices.n_elem;
  const size_t P = M.n_cols;

  GroupSums result(P, P);

  for (size_t j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);

    if (indexes.n_elem < 2) {
      continue;
    }

    result.result += M.rows(indexes).t() * N.rows(indexes);
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_GROUP_SUMS_H
