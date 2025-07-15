#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// PPML weighted demeaning helper
struct WeightedDemeanResult {
  mat demeaned_data;
  bool success;
};

double update_negbin(const arma::vec &y, const arma::vec &mu,
                     const arma::vec &w, double theta, double prev_alpha,
                     double tol, int max_iter) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double mui = mu(i) * exp(x1);
      f += w(i) * (y(i) - (mui + theta) * (y(i) / mui));
      df += w(i) * (-mui * (y(i) / (mui * mui)));
    }
    if (std::abs(df) < 1e-12)
      break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper)
      x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol)
      break;
    if (f > 0)
      lower = x1;
    else
      upper = x1;
  }
  return x1;
}

double update_logit(const arma::vec &y, const arma::vec &mu, const arma::vec &w,
                    double prev_alpha, double tol, int max_iter) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double eta = x1 + mu(i);
      double p = 1.0 / (1.0 + std::exp(-eta));
      f += w(i) * (y(i) - p);
      df += w(i) * (-p * (1 - p));
    }
    if (std::abs(df) < 1e-12)
      break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper)
      x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol)
      break;
    if (f > 0)
      lower = x1;
    else
      upper = x1;
  }
  return x1;
}

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

      // Precompute weights with efficient summation
      sum_weights(j) = 0.0;
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        sum_weights(j) += w(group_obs(i));
      }
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
      // Sequential memory access through flat array
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

inline void demean_variables(mat &V, const vec &weights,
                             const field<field<uvec>> &group_indices,
                             double tol, int max_iter,
                             const std::string &family = "gaussian",
                             bool use_acceleration = false) {
  const size_t P = V.n_cols, K = group_indices.n_elem;

  if (K == 0) {
    return; // No fixed effects to demean
  }

  // Convergence check matching Python: abs(a[i] - b[i]) >= tol for any i
  auto converged_check = [](const vec &a, const vec &b, double tol) -> bool {
    for (size_t i = 0; i < a.n_elem; ++i) {
      if (std::abs(a(i) - b(i)) >= tol) {
        return false;
      }
    }
    return true;
  };

  // Compute group weights once
  field<vec> group_weights(K);
  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &groups_k = group_indices(k);
    group_weights[k].zeros(groups_k.n_elem);

    for (size_t g = 0; g < groups_k.n_elem; ++g) {
      const uvec &group_g = groups_k(g);
      for (size_t i = 0; i < group_g.n_elem; ++i) {
        group_weights[k](g) += weights(group_g(i));
      }
    }
  }

  // Process each variable separately (matching Python parallel structure)
  for (size_t p = 0; p < P; ++p) {
    vec x_curr = V.col(p);
    vec x_prev =
        x_curr - 1.0; // Initialize as different to ensure first iteration runs

    // Alternating projections loop (matching Python demean function)
    for (int iter = 0; iter < max_iter; ++iter) {
      // For each fixed effect factor
      for (size_t k = 0; k < K; ++k) {
        const field<uvec> &groups_k = group_indices(k);

        // For each group in this fixed effect
        for (size_t g = 0; g < groups_k.n_elem; ++g) {
          const uvec &group_g = groups_k(g);
          if (group_g.n_elem == 0 || group_weights[k](g) == 0)
            continue;

          // Compute weighted group sum (matching Python
          // _subtract_weighted_group_mean)
          double group_weighted_sum = 0.0;
          for (size_t i = 0; i < group_g.n_elem; ++i) {
            group_weighted_sum += weights(group_g(i)) * x_curr(group_g(i));
          }

          // Compute group mean and subtract from all group members
          double group_mean = group_weighted_sum / group_weights[k](g);
          for (size_t i = 0; i < group_g.n_elem; ++i) {
            x_curr(group_g(i)) -= group_mean;
          }
        }
      }

      // Check convergence (matching Python _sad_converged)
      if (converged_check(x_curr, x_prev, tol)) {
        break;
      }

      x_prev = x_curr;
    }

    V.col(p) = x_curr;
  }
}

WeightedDemeanResult weighted_demean(const mat &data, const umat &fe,
                                     const vec &weights, double tol,
                                     int maxiter) {
  WeightedDemeanResult result;
  result.success = false;

  if (fe.n_cols == 0) {
    // No fixed effects - return original data
    result.demeaned_data = data;
    result.success = true;
    return result;
  }

  // Convert umat fixed effects to field<field<uvec>> format expected by
  // demean_variables
  field<field<uvec>> group_indices(fe.n_cols);

  for (size_t k = 0; k < fe.n_cols; k++) {
    uvec fe_col = fe.col(k);
    uvec unique_levels = unique(fe_col);

    group_indices(k).set_size(unique_levels.n_elem);

    for (size_t g = 0; g < unique_levels.n_elem; g++) {
      uvec level_indices = find(fe_col == unique_levels(g));
      group_indices(k)(g) = level_indices;
    }
  }

  // Apply weighted demeaning using the existing infrastructure
  result.demeaned_data = data;
  try {
    demean_variables(result.demeaned_data, weights, group_indices, tol, maxiter,
                     "gaussian");
    result.success = true;
  } catch (...) {
    result.success = false;
  }

  return result;
}

#endif // CAPYBARA_CENTER
