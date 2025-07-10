#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

double update_negbin(const arma::vec &y, const arma::vec &mu, const arma::vec &w, double theta, double prev_alpha, double tol = 1e-8, int max_iter = 100) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double mui = mu(i) * exp(x1);
      f += w(i) * (y(i) - (mui + theta) * (y(i) / mui));
      df += w(i) * (-mui * (y(i) / (mui * mui)));
    }
    if (std::abs(df) < 1e-12) break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper) x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol) break;
    if (f > 0) lower = x1; else upper = x1;
  }
  return x1;
}

double update_logit(const arma::vec &y, const arma::vec &mu, const arma::vec &w, double prev_alpha, double tol = 1e-8, int max_iter = 100) {
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
    if (std::abs(df) < 1e-12) break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper) x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol) break;
    if (f > 0) lower = x1; else upper = x1;
  }
  return x1;
}

// Bucket sort for efficient group processing
inline std::vector<int> bucket_argsort(const std::vector<int>& group_ids, int max_groups) {
  std::vector<int> result;
  result.reserve(group_ids.size());
  
  // Count occurrences
  std::vector<int> counts(max_groups + 1, 0);
  for (int id : group_ids) {
    counts[id]++;
  }
  
  // Create sorted indices
  for (int g = 0; g <= max_groups; ++g) {
    for (size_t i = 0; i < group_ids.size(); ++i) {
      if (group_ids[i] == g) {
        result.push_back(i);
      }
    }
  }
  return result;
}

// Enhanced group information structure with flat memory layout and bucket sort
struct GroupInfo {
  // Flat storage for cache efficiency
  std::vector<std::vector<size_t>> indices;  // Use std::vector for better cache locality
  vec sum_weights;
  vec inv_weights;
  vec cached_group_means;  // Cache for group means
  size_t n_groups;
  size_t max_group_size;
  bool has_cached_means = false;
  
  GroupInfo() = default;
  
  // Portable constructor using pure Armadillo types
  GroupInfo(const field<uvec> &group_field, const vec &w) {
    n_groups = group_field.n_elem;
    indices.resize(n_groups);
    sum_weights.set_size(n_groups);
    inv_weights.set_size(n_groups);
    cached_group_means.set_size(n_groups);
    max_group_size = 0;
    
    for (size_t j = 0; j < n_groups; ++j) {
      const uvec &group_obs = group_field(j);
      indices[j].resize(group_obs.n_elem);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        indices[j][i] = group_obs(i);
      }
      max_group_size = std::max(max_group_size, indices[j].size());
      
      // Precompute weights with efficient summation
      sum_weights(j) = 0.0;
      const auto& coords = indices[j];
      for (size_t idx : coords) {
        sum_weights(j) += w(idx);
      }
      inv_weights(j) = 1.0 / sum_weights(j);
    }
  }
  
  
  // Compute group means efficiently with caching
  void compute_group_means(const vec& values, const vec& weights) {
    for (size_t j = 0; j < n_groups; ++j) {
      const auto& coords = indices[j];
      if (coords.empty()) {
        cached_group_means(j) = 0.0;
        continue;
      }
      
      double weighted_sum = 0.0;
      for (size_t idx : coords) {
        weighted_sum += weights(idx) * values(idx);
      }
      cached_group_means(j) = weighted_sum * inv_weights(j);
    }
    has_cached_means = true;
  }
  
  // Vectorized group subtraction
  void subtract_group_effects(vec& values, const vec& group_effects) const {
    for (size_t j = 0; j < n_groups; ++j) {
      const auto& coords = indices[j];
      const double effect = group_effects(j);
      for (size_t idx : coords) {
        values(idx) -= effect;
      }
    }
  }
  
  // Vectorized group addition
  void add_group_effects(vec& values, const vec& group_effects) const {
    for (size_t j = 0; j < n_groups; ++j) {
      const auto& coords = indices[j];
      const double effect = group_effects(j);
      for (size_t idx : coords) {
        values(idx) += effect;
      }
    }
  }
};

// Group mean subtraction - portable version using pure Armadillo types
inline void demean_variables(mat &V, const vec &weights,
                                      const field<field<uvec>> &group_indices,
                                      double tol = 1e-8, int max_iter = 1000,
                                      const std::string &family = "gaussian") {
  const size_t N = V.n_rows, P = V.n_cols, K = group_indices.n_elem;
  const double inv_sw = 1.0 / accu(weights);

  // Build GroupInfo structures for each fixed effect using portable constructor
  std::vector<GroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info[k] = GroupInfo(group_indices(k), weights);
  }

  // Use existing convergence check lambda
  auto convergence_check = [&](const vec &v, const vec &v0, const vec &w,
                               double tol) {
    if (family == "poisson") {
      double ssr = dot(w, square(v));
      double ssr0 = dot(w, square(v0));
      return std::abs(ssr - ssr0) / (0.1 + std::abs(ssr)) < tol;
    } else {
      return dot(abs(v - v0), w) * inv_sw < tol;
    }
  };

  // Main demeaning loop (reuse existing algorithm)
  for (size_t p = 0; p < P; ++p) {
    const vec v_orig = V.col(p);
    vec v = v_orig;
    field<vec> alpha(K);
    for (size_t k = 0; k < K; ++k) {
      alpha(k).zeros(group_info[k].n_groups);
    }
    vec alpha_sum = zeros<vec>(N);
    vec v0(N, fill::none);

    if (K == 2) {
      // K=2 specialization
      for (int iter = 0; iter < max_iter; ++iter) {
        v0 = v;
        for (size_t k = 0; k < 2; ++k) {
          const auto &gi = group_info[k];

          // Remove current FE from alpha_sum
          gi.subtract_group_effects(alpha_sum, alpha(k));

          // Compute residual
          v = v_orig - alpha_sum;

          // Update alpha
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;

            double w_sum = 0.0;
            for (size_t idx : coords) {
              w_sum += weights(idx) * v(idx);
            }
            alpha(k)(l) = w_sum * gi.inv_weights(l);
          }

          // Add back new FE
          gi.add_group_effects(alpha_sum, alpha(k));
        }
        v = v_orig - alpha_sum;
        if (convergence_check(v, v0, weights, tol)) break;
      }
    } else {
      // K > 2 case with acceleration (simplified version)
      for (int iter = 0; iter < max_iter; ++iter) {
        v0 = v;
        for (size_t k = 0; k < K; ++k) {
          const auto &gi = group_info[k];

          // Remove current FE from alpha_sum
          gi.subtract_group_effects(alpha_sum, alpha(k));

          // Compute residual
          v = v_orig - alpha_sum;

          // Update alpha
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;

            double w_sum = 0.0;
            for (size_t idx : coords) {
              w_sum += weights(idx) * v(idx);
            }
            alpha(k)(l) = w_sum * gi.inv_weights(l);
          }

          // Add back new FE
          gi.add_group_effects(alpha_sum, alpha(k));
        }
        v = v_orig - alpha_sum;
        if (convergence_check(v, v0, weights, tol)) break;
      }
    }
    V.col(p) = v;
  }
}

// Portable GLM step demeaning
inline void demean_glm_step(mat &X, vec &y, const vec &weights, const field<field<uvec>> &group_indices,
                           double tol = 1e-8, int max_iter = 1000,
                           const std::string &family = "gaussian") {
  const size_t N = X.n_rows, P = X.n_cols, K = group_indices.n_elem;
  const double inv_sw = 1.0 / accu(weights);

  // Build GroupInfo structures for each fixed effect using portable constructor
  std::vector<GroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info[k] = GroupInfo(group_indices(k), weights);
  }

  // Convergence check lambda
  auto convergence_check = [&](const vec &x, const vec &x0, const vec &w, double tol) {
    if (family == "poisson") {
      double ssr = dot(w, square(x));
      double ssr0 = dot(w, square(x0));
      return std::abs(ssr - ssr0) / (0.1 + std::abs(ssr)) < tol;
    } else {
      return dot(abs(x - x0), w) * inv_sw < tol;
    }
  };

  // Process all columns of X and then y in the same alternating projection loop
  const size_t total_vars = P + 1; // X columns + y
  
  for (size_t var_idx = 0; var_idx < total_vars; ++var_idx) {
    vec var_orig, v;
    if (var_idx == P) {
      // Process y
      var_orig = y;
      v = var_orig;
    } else {
      // Process X column
      var_orig = X.col(var_idx);
      v = var_orig;
    }
    
    field<vec> alpha(K);
    for (size_t k = 0; k < K; ++k) {
      alpha(k).zeros(group_info[k].n_groups);
    }
    vec alpha_sum = zeros<vec>(N);
    vec v0(N, fill::none);

    if (K == 2) {
      // K=2 specialization
      for (int iter = 0; iter < max_iter; ++iter) {
        v0 = v;
        for (size_t k = 0; k < 2; ++k) {
          const auto &gi = group_info[k];

          // Remove current FE from alpha_sum
          gi.subtract_group_effects(alpha_sum, alpha(k));

          // Compute residual
          v = var_orig - alpha_sum;

          // Update alpha
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;

            double w_sum = 0.0;
            for (size_t idx : coords) {
              w_sum += weights(idx) * v(idx);
            }
            alpha(k)(l) = w_sum * gi.inv_weights(l);
          }

          // Add back new FE
          gi.add_group_effects(alpha_sum, alpha(k));
        }
        v = var_orig - alpha_sum;
        if (convergence_check(v, v0, weights, tol)) break;
      }
    } else {
      // K > 2 case
      for (int iter = 0; iter < max_iter; ++iter) {
        v0 = v;
        for (size_t k = 0; k < K; ++k) {
          const auto &gi = group_info[k];

          // Remove current FE from alpha_sum
          gi.subtract_group_effects(alpha_sum, alpha(k));

          // Compute residual
          v = var_orig - alpha_sum;

          // Update alpha
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;

            double w_sum = 0.0;
            for (size_t idx : coords) {
              w_sum += weights(idx) * v(idx);
            }
            alpha(k)(l) = w_sum * gi.inv_weights(l);
          }

          // Add back new FE
          gi.add_group_effects(alpha_sum, alpha(k));
        }
        v = var_orig - alpha_sum;
        if (convergence_check(v, v0, weights, tol)) break;
      }
    }
    
    // Assign result back to the appropriate variable
    if (var_idx == P) {
      y = v;
    } else {
      X.col(var_idx) = v;
    }
  }
}

#endif // CAPYBARA_CENTER
