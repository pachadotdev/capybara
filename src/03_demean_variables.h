#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// PPML weighted demeaning helper
struct WeightedDemeanResult {
  mat demeaned_data;
  bool success;
};

// Flat array group information structure (keep existing optimized version)
struct GroupInfo {
  // Flat storage for cache efficiency
  uvec group_starts;      // Start index for each group in flat_indices
  uvec group_sizes;       // Size of each group
  uvec flat_indices;      // All group indices stored contiguously
  vec sum_weights;        // Precomputed sum of weights per group
  vec inv_weights;        // Precomputed inverse weights per group
  vec cached_group_means; // Cache for group means
  size_t n_groups;
  size_t max_group_size;
  bool has_cached_means = false;

  GroupInfo() = default;

  GroupInfo(const field<uvec> &group_field, const vec &w) {
    n_groups = group_field.n_elem;
    group_starts.set_size(n_groups);
    group_sizes.set_size(n_groups);
    sum_weights.set_size(n_groups);
    inv_weights.set_size(n_groups);
    cached_group_means.set_size(n_groups);
    max_group_size = 0;

    // First pass: compute total size and group sizes
    size_t total_size = 0;
    for (size_t j = 0; j < n_groups; ++j) {
      const uvec &group_obs = group_field(j);
      group_sizes(j) = group_obs.n_elem;
      max_group_size =
          std::max(max_group_size, static_cast<size_t>(group_sizes(j)));
      total_size += group_sizes(j);
    }

    // Allocate flat array
    flat_indices.set_size(total_size);

    // Second pass: populate flat array and compute starts
    size_t current_pos = 0;
    for (size_t j = 0; j < n_groups; ++j) {
      const uvec &group_obs = group_field(j);
      group_starts(j) = current_pos;

      // Copy indices to flat array
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        flat_indices(current_pos + i) = group_obs(i);
      }

      // Precompute weights with efficient summation using Armadillo
      sum_weights(j) = accu(w.elem(group_obs));
      inv_weights(j) = 1.0 / sum_weights(j);

      current_pos += group_obs.n_elem;
    }
  }

  // Compute group means
  void compute_group_means(const vec &values, const vec &weights) {
    for (size_t j = 0; j < n_groups; ++j) {
      const size_t start = group_starts(j);
      const size_t size = group_sizes(j);

      if (size == 0) {
        cached_group_means(j) = 0.0;
        continue;
      }

      double weighted_sum = 0.0;
      // Vectorized computation through flat array
      for (size_t i = 0; i < size; ++i) {
        const size_t idx = flat_indices(start + i);
        weighted_sum += weights(idx) * values(idx);
      }
      cached_group_means(j) = weighted_sum * inv_weights(j);
    }
    has_cached_means = true;
  }

  // Vectorized group subtraction
  void subtract_group_effects(vec &values, const vec &group_effects) const {
    for (size_t j = 0; j < n_groups; ++j) {
      const size_t start = group_starts(j);
      const size_t size = group_sizes(j);
      const double effect = group_effects(j);

      // Sequential memory access
      for (size_t i = 0; i < size; ++i) {
        values(flat_indices(start + i)) -= effect;
      }
    }
  }

  // Vectorized group addition
  void add_group_effects(vec &values, const vec &group_effects) const {
    for (size_t j = 0; j < n_groups; ++j) {
      const size_t start = group_starts(j);
      const size_t size = group_sizes(j);
      const double effect = group_effects(j);

      // Sequential memory access
      for (size_t i = 0; i < size; ++i) {
        values(flat_indices(start + i)) += effect;
      }
    }
  }
};

inline bool check_convergence(const vec &x_curr, const vec &x_prev,
                              double tol) {
  for (size_t i = 0; i < x_curr.n_elem; ++i) {
    if (std::abs(x_curr(i) - x_prev(i)) >= tol) {
      return false;
    }
  }
  return true;
}

inline WeightedDemeanResult
demean_variables(const mat &data, const umat &fe, const vec &weights,
                 double tol, int max_iter,
                 const std::string &family = "gaussian") {
  WeightedDemeanResult result;
  result.success = false;

  if (fe.n_cols == 0) {
    // No fixed effects - return original data
    result.demeaned_data = data;
    result.success = true;
    return result;
  }

  const size_t n_samples = data.n_rows;
  const size_t n_features = data.n_cols;
  const size_t n_factors = fe.n_cols;

  mat group_weights = calc_group_weights(weights, fe);
  const size_t n_groups = group_weights.n_rows;

  result.demeaned_data.set_size(n_samples, n_features);

  bool converged = false;
  size_t not_converged = 0;

  vec group_weighted_sums(n_groups);

  for (size_t k = 0; k < n_features; ++k) {
    vec xk_curr = data.col(k);
    vec xk_prev = xk_curr - 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
      for (size_t j = 0; j < n_factors; ++j) {
        subtract_weighted_group_mean(xk_curr, weights, fe.col(j),
                                     group_weights.col(j), group_weighted_sums);
      }

      converged = check_convergence(xk_curr, xk_prev, tol);
      if (converged)
        break;

      xk_prev = xk_curr;
    }

    if (!converged) {
      not_converged++;
    }

    result.demeaned_data.col(k) = xk_curr;
  }

  result.success = (not_converged == 0);
  return result;
}

#endif // CAPYBARA_CENTER
