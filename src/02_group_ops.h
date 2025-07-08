#ifndef CAPYBARA_GROUP_OPS
#define CAPYBARA_GROUP_OPS

void weighted_group_sums(const vec &values, const vec &weights,
                         const field<uvec> &group_indices, vec &group_sums,
                         vec &group_weights) {
  const size_t n_groups = group_indices.n_elem;

  // Initialize outputs
  group_sums.zeros(n_groups);
  group_weights.zeros(n_groups);

  // Single pass through groups (better cache locality)
  for (size_t g = 0; g < n_groups; ++g) {
    const uvec &idx = group_indices(g);
    if (idx.n_elem == 0)
      continue;

    // Use Armadillo's optimized dot products
    group_sums(g) = dot(weights.elem(idx), values.elem(idx));
    group_weights(g) = accu(weights.elem(idx));
  }
}

void group_means_precomputed(const vec &values, const vec &weights,
                             const field<uvec> &group_indices,
                             const vec &inv_weights, vec &group_means) {
  const size_t n_groups = group_indices.n_elem;
  group_means.set_size(n_groups);

  for (size_t g = 0; g < n_groups; ++g) {
    const uvec &idx = group_indices(g);
    if (idx.n_elem == 0) {
      group_means(g) = 0.0;
      continue;
    }

    // Single pass with precomputed inverse
    group_means(g) = dot(weights.elem(idx), values.elem(idx)) * inv_weights(g);
  }
}

void compute_residuals(const vec &y, const vec &group_effects,
                       const field<uvec> &group_indices, vec &residuals) {
  const size_t n_groups = group_indices.n_elem;

  residuals = y; // Start with original values

  // Subtract group effects (optimized memory access)
  for (size_t g = 0; g < n_groups; ++g) {
    const uvec &idx = group_indices(g);
    if (idx.n_elem == 0)
      continue;

    const double effect = group_effects(g);
    residuals.elem(idx) -= effect;
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

void two_way_residuals(const vec &y_orig, const vec &alpha0,
                       const field<uvec> &indices0, const vec &alpha1,
                       const field<uvec> &indices1, vec &residuals) {
  residuals = y_orig;

  // Subtract first FE
  const size_t n0 = indices0.n_elem;
  for (size_t g = 0; g < n0; ++g) {
    const uvec &idx = indices0(g);
    if (idx.n_elem == 0)
      continue;
    residuals.elem(idx) -= alpha0(g);
  }

  // Subtract second FE
  const size_t n1 = indices1.n_elem;
  for (size_t g = 0; g < n1; ++g) {
    const uvec &idx = indices1(g);
    if (idx.n_elem == 0)
      continue;
    residuals.elem(idx) -= alpha1(g);
  }
}

void block_group_sums(const vec &values, const vec &weights,
                      const field<uvec> &group_indices, vec &group_sums,
                      vec &group_weights, size_t block_size = 8192) {
  const size_t n = values.n_elem;
  const size_t n_groups = group_indices.n_elem;

  group_sums.zeros(n_groups);
  group_weights.zeros(n_groups);

  // Process in blocks for better cache utilization
  for (size_t start = 0; start < n; start += block_size) {
    size_t end = std::min(start + block_size, n);

    // For each group, process only observations in current block
    for (size_t g = 0; g < n_groups; ++g) {
      const uvec &idx = group_indices(g);
      if (idx.n_elem == 0)
        continue;

      // Find indices in current block
      for (size_t i = 0; i < idx.n_elem; ++i) {
        size_t obs = idx(i);
        if (obs >= start && obs < end) {
          group_sums(g) += weights(obs) * values(obs);
          group_weights(g) += weights(obs);
        }
      }
    }
  }
}

bool convergence_check(const vec &x_new, const vec &x_old, const vec &weights,
                       double tol, const std::string &family = "gaussian") {
  if (family == "poisson") {
    double ssr_new = dot(weights, square(x_new));
    double ssr_old = dot(weights, square(x_old));
    return std::abs(ssr_new - ssr_old) / (0.1 + std::abs(ssr_new)) < tol;
  } else {
    // Gaussian and others: weighted absolute difference
    double diff = dot(abs(x_new - x_old), weights);
    double total_weight = accu(weights);
    return (diff / total_weight) < tol;
  }
}

#endif // CAPYBARA_GROUP_OPS