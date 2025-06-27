#ifndef CAPYBARA_GROUPS_H
#define CAPYBARA_GROUPS_H

// Indices of groups with at least min_size members
inline uvec get_valid_groups(const single_fe_indices &indices,
                             size_t min_size = 1) {
  return find(indices.group_sizes >= min_size);
}

// Weighted group means
mat group_sums(const mat &M, const vec &w, const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat result(P, 1, fill::zeros);

  const uvec valid = get_valid_groups(indices, 1);
  if (valid.is_empty())
    return result;

  vec weight_sums(valid.n_elem);
  for (uword i = 0; i < valid.n_elem; ++i) {
    const uword g = valid[i];
    const uvec grp = indices.get_group(g);
    weight_sums[i] = accu(w.elem(grp));
  }

  const uvec pos = find(weight_sums > 0);
  for (uword i = 0; i < pos.n_elem; ++i) {
    const uword idx = pos[i];
    const uword g = valid[idx];
    const uvec grp = indices.get_group(g);
    result += sum(M.rows(grp), 0).t() / weight_sums[idx];
  }

  return result;
}

// Group-wise spectral/lagged sums
mat group_sums_spectral(const mat &M, const vec &v, const vec &w, size_t K,
                        const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat result(P, 1, fill::zeros);

  const uvec valid = get_valid_groups(indices, 2);
  if (valid.is_empty())
    return result;

  const uword max_size = indices.group_sizes.elem(valid).max();
  vec v_group(max_size);
  vec v_shifted(max_size);

  for (uword i = 0; i < valid.n_elem; ++i) {
    const uword g = valid[i];
    const uvec grp = indices.get_group(g);
    const uword I = grp.n_elem;
    const double sum_w = accu(w.elem(grp));
    if (sum_w <= 0)
      continue;

    v_group.head(I) = v.elem(grp);
    v_shifted.head(I).zeros();
    const uword m = std::min<uword>(K, I - 1);
    for (uword k = 1; k <= m; ++k) {
      v_shifted.subvec(k, I - 1) += v_group.subvec(0, I - k - 1);
    }

    const double scale = I / ((I - 1.0) * sum_w);
    result += scale * (M.rows(grp).t() * v_shifted.head(I));
  }

  return result;
}

// Sum of outer products of group means (group variance)
mat group_sums_var(const mat &M, const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat V(P, P, fill::zeros);

  const uvec valid = get_valid_groups(indices, 1);
  for (uword i = 0; i < valid.n_elem; ++i) {
    const uword g = valid[i];
    const uvec grp = indices.get_group(g);
    const vec s = sum(M.rows(grp), 0).t();
    V += s * s.t();
  }
  return V;
}

// Sum of cross-products within each group (group covariance)
mat group_sums_cov(const mat &M, const mat &N,
                   const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat V(P, P, fill::zeros);

  const uvec valid = get_valid_groups(indices, 2);
  for (uword i = 0; i < valid.n_elem; ++i) {
    const uword g = valid[i];
    const uvec grp = indices.get_group(g);
    V += M.rows(grp).t() * N.rows(grp);
  }
  return V;
}

#endif // CAPYBARA_GROUPS_H
