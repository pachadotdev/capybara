#ifndef CAPYBARA_GROUP_OPS
#define CAPYBARA_GROUP_OPS

inline mat calc_group_weights(const vec &sample_weights,
                              const umat &group_ids) {
  const size_t n_samples = group_ids.n_rows;
  const size_t n_factors = group_ids.n_cols;

  uword n_groups = 0;
  for (size_t j = 0; j < n_factors; ++j) {
    uword max_id = max(group_ids.col(j));
    n_groups = std::max(n_groups, max_id + 1);
  }

  // Group weights matrix: (n_groups, n_factors)
  mat group_weights(n_groups, n_factors, fill::zeros);

  for (size_t j = 0; j < n_factors; ++j) {
    for (size_t i = 0; i < n_samples; ++i) {
      uword group_id = group_ids(i, j);
      group_weights(group_id, j) += sample_weights(i);
    }
  }

  return group_weights;
}

inline void subtract_weighted_group_mean(vec &x, const vec &sample_weights,
                                         const uvec &group_ids,
                                         const vec &group_weights_factor,
                                         vec &group_weighted_sums) {
  group_weighted_sums.zeros();

  // First pass: compute weighted sums for each group
  for (size_t i = 0; i < x.n_elem; ++i) {
    uword group_id = group_ids(i);
    group_weighted_sums(group_id) += sample_weights(i) * x(i);
  }

  // Second pass: subtract group means
  for (size_t i = 0; i < x.n_elem; ++i) {
    uword group_id = group_ids(i);
    if (group_weights_factor(group_id) > 0.0) {
      x(i) -= group_weighted_sums(group_id) / group_weights_factor(group_id);
    }
  }
}

void add_group_effects(vec &y, const vec &group_effects,
                       const field<uvec> &group_indices) {
  const size_t n_groups = group_indices.n_elem;

  for (size_t g = 0; g < n_groups; ++g) {
    const uvec &idx = group_indices(g);
    if (idx.n_elem == 0)
      continue;

    const double effect = group_effects(g);
    y.elem(idx) += effect;
  }
}

#endif // CAPYBARA_GROUP_OPS