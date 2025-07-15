#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// PPML weighted demeaning helper
struct WeightedDemeanResult {
  mat demeaned_data;
  bool success;
};

// Flat array group information structure
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

inline WeightedDemeanResult demean_variables(const mat &data, const umat &fe,
                                             const vec &weights, double tol,
                                             int max_iter,
                                             const std::string &family = "gaussian") {
  WeightedDemeanResult result;
  result.success = false;

  if (fe.n_cols == 0) {
    // No fixed effects - return original data
    result.demeaned_data = data;
    result.success = true;
    return result;
  }

  // Convert umat fixed effects to internal format for processing
  const size_t n_vars = data.n_cols;
  const size_t n_factors = fe.n_cols;

  // Precompute group information for all factors
  std::vector<std::vector<uvec>> factor_groups(n_factors);
  std::vector<vec> group_weights(n_factors);

  for (size_t k = 0; k < n_factors; ++k) {
    uvec fe_col = fe.col(k);
    uvec unique_levels = unique(fe_col);
    size_t n_groups = unique_levels.n_elem;

    factor_groups[k].resize(n_groups);
    group_weights[k].zeros(n_groups);

    // Build group membership and compute group weights
    for (size_t g = 0; g < n_groups; ++g) {
      uvec group_members = find(fe_col == unique_levels(g));
      factor_groups[k][g] = group_members;

      // Sum of weights for this group
      group_weights[k](g) = accu(weights.elem(group_members));
    }
  }

  // Convergence check using vectorized operation
  auto converged_check = [](const vec &a, const vec &b, double tol) -> bool {
    return max(abs(a - b)) < tol;
  };

  // Process each variable separately (matching Python parallel structure)
  result.demeaned_data = data; // Initialize with original data
  bool all_converged = true;

  for (size_t p = 0; p < n_vars; ++p) {
    vec x_curr = data.col(p);
    vec x_prev = x_curr - 1.0; // Initialize differently to ensure first iteration runs

    bool converged = false;
    // Alternating projections loop (matching Python demean function)
    for (int iter = 0; iter < max_iter; ++iter) {
      // For each fixed effect factor (matching Python factor iteration)
      for (size_t k = 0; k < n_factors; ++k) {
        const std::vector<uvec> &groups_k = factor_groups[k];
        const vec &weights_k = group_weights[k];

        // Apply factor k (matching Python _subtract_weighted_group_mean)
        for (size_t g = 0; g < groups_k.size(); ++g) {
          const uvec &group_members = groups_k[g];
          if (group_members.n_elem == 0 || weights_k(g) == 0.0)
            continue;

          // Compute weighted group sum using vectorized operation
          double group_weighted_sum = accu(weights.elem(group_members) % x_curr.elem(group_members));

          // Compute and subtract group mean using vectorized operation
          double group_mean = group_weighted_sum / weights_k(g);
          x_curr.elem(group_members) -= group_mean;
        }
      }

      // Check convergence (matching Python convergence logic)
      if (converged_check(x_curr, x_prev, tol)) {
        converged = true;
        break;
      }

      x_prev = x_curr;
    }

    if (!converged) {
      all_converged = false;
    }

    result.demeaned_data.col(p) = x_curr;
  }

  result.success = all_converged;
  return result;
}

#endif // CAPYBARA_CENTER
