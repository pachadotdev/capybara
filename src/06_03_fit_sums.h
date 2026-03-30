#ifndef CAPYBARA_GLM_SUMS_H
#define CAPYBARA_GLM_SUMS_H

namespace capybara {

///////////////////////////////////////////////////////////////////////////
// Group aggregation functions
///////////////////////////////////////////////////////////////////////////

inline vec group_sums(const mat &M, const vec &w,
                      const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  vec b(P, fill::zeros);
  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    if (idx.n_elem == 0)
      continue;

    const double denom = accu(w.elem(idx));
    if (denom == 0.0)
      continue;

    b += sum(M.rows(idx), 0).t() / denom;
  }
  return b;
}

inline vec group_sums_spectral(const mat &M, const vec &v, const vec &w,
                               const uword K,
                               const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  vec b(P, fill::zeros);
  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    const uword I = idx.n_elem;
    if (I <= 1)
      continue;

    const double denom = accu(w.elem(idx));
    if (denom == 0.0)
      continue;

    const vec v_group = v.elem(idx);
    const vec v_cumsum = cumsum(v_group);
    const uword max_k = std::min(K, I - 1);

    // Compute shifted sum: v_shifted[i] = sum_{k=1}^{min(K,i)} v_group[i-k]
    vec v_shifted(I, fill::zeros);
    for (uword i = 1; i < I; ++i) {
      const uword start = (i > max_k) ? i - max_k : 0;
      v_shifted(i) = v_cumsum(i - 1) - (start > 0 ? v_cumsum(start - 1) : 0.0);
    }

    const double scale = static_cast<double>(I) / ((I - 1.0) * denom);
    b += M.rows(idx).t() * v_shifted * scale;
  }
  return b;
}

inline mat group_sums_var(const mat &M, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat V(P, P, fill::zeros);
  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    if (idx.n_elem == 0)
      continue;

    const rowvec row_sum = sum(M.rows(idx), 0);
    V += row_sum.t() * row_sum;
  }
  return V;
}

inline mat group_sums_cov(const mat &M, const mat &N,
                          const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat V(P, P, fill::zeros);
  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    if (idx.n_elem < 2)
      continue;

    V += M.rows(idx).t() * N.rows(idx);
  }
  return V;
}

} // namespace capybara

#endif // CAPYBARA_GLM_SUMS_H
