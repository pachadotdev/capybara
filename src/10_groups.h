#ifndef CAPYBARA_GROUPS_H
#define CAPYBARA_GROUPS_H

#include "01_types.h"

// Returns indices of groups whose size is >= min_size
inline uvec get_valid_groups(const single_fe_indices &indices,
                             size_t min_size = 1) {
  return find(indices.group_sizes >= min_size);
}

// Sum over groups, weighted by w, only for groups with at least one member and
// positive weight
mat group_sums(const mat &M, const vec &w, const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat result(P, 1, fill::zeros);

  uvec valid = get_valid_groups(indices, 1);
  if (valid.is_empty())
    return result;

  // Compute total weight per group
  vec weight_sums(valid.n_elem);
  for (uword i = 0; i < valid.n_elem; ++i) {
    uword g = valid[i];
    uvec grp = indices.get_group(g);
    weight_sums[i] = accu(w.elem(grp));
  }

  // Only use groups with positive total weight
  uvec pos = find(weight_sums > 0);
  for (uword i = 0; i < pos.n_elem; ++i) {
    uword idx = pos[i];
    uword g = valid[idx];
    uvec grp = indices.get_group(g);
    result += sum(M.rows(grp), 0).t() / weight_sums[idx];
  }

  return result;
}

// Spectral group sums: only groups with size >= 2
mat group_sums_spectral(const mat &M, const vec &v, const vec &w, size_t K,
                        const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat result(P, 1, fill::zeros);

  uvec valid = get_valid_groups(indices, 2);
  if (valid.is_empty())
    return result;

  // Pre-allocate for maximum group length
  uword max_size = indices.group_sizes.elem(valid).max();
  vec v_group(max_size);
  vec v_shifted(max_size);

  for (uword i = 0; i < valid.n_elem; ++i) {
    uword g = valid[i];
    uvec grp = indices.get_group(g);
    uword I = grp.n_elem;
    double sum_w = accu(w.elem(grp));
    if (sum_w <= 0)
      continue;

    v_group.head(I) = v.elem(grp);
    v_shifted.head(I).zeros();
    uword m = std::min<uword>(K, I - 1);
    for (uword k = 1; k <= m; ++k) {
      v_shifted.subvec(k, I - 1) += v_group.subvec(0, I - k - 1);
    }

    double scale = I / ((I - 1.0) * sum_w);
    result += scale * (M.rows(grp).t() * v_shifted.head(I));
  }

  return result;
}

// Variance for M groups (size >=1)
mat group_sums_var(const mat &M, const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat V(P, P, fill::zeros);

  uvec valid = get_valid_groups(indices, 1);
  for (uword i = 0; i < valid.n_elem; ++i) {
    uword g = valid[i];
    uvec grp = indices.get_group(g);
    vec s = sum(M.rows(grp), 0).t();
    V += s * s.t();
  }
  return V;
}

// Covariance between M and N over groups (size >=2)
mat group_sums_cov(const mat &M, const mat &N,
                   const single_fe_indices &indices) {
  const size_t P = M.n_cols;
  mat V(P, P, fill::zeros);

  uvec valid = get_valid_groups(indices, 2);
  for (uword i = 0; i < valid.n_elem; ++i) {
    uword g = valid[i];
    uvec grp = indices.get_group(g);
    V += M.rows(grp).t() * N.rows(grp);
  }
  return V;
}

#endif // CAPYBARA_GROUPS_H
