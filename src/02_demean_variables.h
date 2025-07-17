#ifndef CAPYBARA_DEMEAN
#define CAPYBARA_DEMEAN

using namespace arma;

// High-performance joint demeaning for multiple variables
// Follows fixest approach exactly but optimized for Armadillo

struct DemeanResult {
  mat demeaned_data;
  vec fixed_effects;
  uvec iterations;
  bool success;

  DemeanResult(size_t n_obs, size_t n_vars)
      : demeaned_data(n_obs, n_vars, fill::none),
        fixed_effects(0, fill::none),
        iterations(n_vars, fill::none),
        success(false) {}
};

// Optimized FE structure - no conversions, direct storage
class FEClass {
 private:
  const size_t n_obs;
  const size_t Q;
  size_t nb_coef_T;

  // Pre-allocated working memory
  vec workspace_means;
  vec workspace_coef;
  vec workspace_contrib;

  // Fix: Use proper member name
  const field<uvec>& fe_ids;
  const vec& weights;
  const bool has_weights;

  // Pre-computed sizes and offsets
  const uvec nb_coef_Q;
  const uvec coef_start_Q;
  uvec nb_id_Q;

  // Pre-computed weight sums for each FE coefficient
  field<vec> sum_weights_Q;

 public:
  FEClass(size_t n_obs, const field<uvec>& fe_ids, const vec& weights)
      : n_obs(n_obs),
        Q(fe_ids.n_elem),
        fe_ids(fe_ids),
        weights(weights),
        has_weights(weights.n_elem > 1) {
    // Compute sizes efficiently
    nb_id_Q.set_size(Q);
    uvec nb_coef_Q_tmp(Q);
    size_t total_coef = 0;

    for (size_t q = 0; q < Q; ++q) {
      nb_id_Q(q) = fe_ids(q).max() + 1; // 0-based group IDs: max gives us the highest index
      nb_coef_Q_tmp(q) = nb_id_Q(q);
      total_coef += nb_coef_Q_tmp(q);
    }

    const_cast<uvec&>(nb_coef_Q) = std::move(nb_coef_Q_tmp);
    nb_coef_T = total_coef;

    // Compute coefficient starting positions
    uvec coef_start_Q_tmp(Q);
    coef_start_Q_tmp(0) = 0;
    for (size_t q = 1; q < Q; ++q) {
      coef_start_Q_tmp(q) = coef_start_Q_tmp(q - 1) + nb_coef_Q(q - 1);
    }
    const_cast<uvec&>(coef_start_Q) = std::move(coef_start_Q_tmp);

    // Pre-allocate workspace
    workspace_means.set_size(n_obs);
    workspace_coef.set_size(nb_coef_Q.max());
    workspace_contrib.set_size(n_obs);

    // Pre-compute weight sums
    precompute_weight_sums();
  }

 private:
  void precompute_weight_sums() {
    sum_weights_Q.set_size(Q);

    for (size_t q = 0; q < Q; ++q) {
      const size_t n_coef = nb_coef_Q(q);
      sum_weights_Q(q).zeros(n_coef);

      const uvec& fe_id = fe_ids(q);

      if (has_weights) {
        for (size_t i = 0; i < n_obs; ++i) {
          sum_weights_Q(q)(fe_id(i)) += weights(i); // Use 0-based indexing directly
        }
        // Handle zero weights
        for (size_t m = 0; m < n_coef; ++m) {
          if (sum_weights_Q(q)(m) == 0.0) {
            sum_weights_Q(q)(m) = 1.0;
          }
        }
      } else {
        // Compute counts efficiently
        for (size_t i = 0; i < n_obs; ++i) {
          sum_weights_Q(q)(fe_id(i)) += 1.0; // Use 0-based indexing directly
        }
      }
    }
  }

 public:
  // Single FE case - closed form solution
  void compute_single_fe(vec& fe_coef, const vec& variable) {
    const uvec& fe_id = fe_ids(0);
    const vec& sum_weights = sum_weights_Q(0);

    fe_coef.zeros();

    // Accumulate contributions
    if (has_weights) {
      for (size_t i = 0; i < n_obs; ++i) {
        fe_coef(fe_id(i)) += weights(i) * variable(i); // Use 0-based indexing directly
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        fe_coef(fe_id(i)) += variable(i); // Use 0-based indexing directly
      }
    }

    // Normalize by weight sums
    fe_coef /= sum_weights;
  }

  // Add FE contributions to output - template to handle both vec and subview
  template <typename VecType>
  void add_fe_to_output(vec& output, const VecType& fe_coef, size_t q) {
    const uvec& fe_id = fe_ids(q);

    if (has_weights) {
      for (size_t i = 0; i < n_obs; ++i) {
        output(i) += fe_coef(fe_id(i)) * weights(i); // Use 0-based indexing directly
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        output(i) += fe_coef(fe_id(i)); // Use 0-based indexing directly
      }
    }
  }

  // Compute FE coefficients for multi-FE case
  void compute_fe_coef(vec& fe_coef_dest, const vec& variable, size_t q,
                       const vec& sum_other_means) {
    const uvec& fe_id = fe_ids(q);
    const vec& sum_weights = sum_weights_Q(q);
    const size_t start_idx = coef_start_Q(q);
    const size_t n_coef = nb_coef_Q(q);

    // Get subview for this FE's coefficients
    subview<double> my_fe_coef =
        fe_coef_dest.subvec(start_idx, start_idx + n_coef - 1);
    my_fe_coef.zeros();

    // Accumulate: sum(variable - sum_other_means) for each FE level
    if (has_weights) {
      for (size_t i = 0; i < n_obs; ++i) {
        my_fe_coef(fe_id(i)) += weights(i) * (variable(i) - sum_other_means(i)); // Use 0-based indexing directly
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        my_fe_coef(fe_id(i)) += variable(i) - sum_other_means(i); // Use 0-based indexing directly
      }
    }

    // Normalize
    my_fe_coef /= sum_weights;
  }

  // Get FE coefficient subview
  subview<double> get_fe_coef_subview(vec& fe_coef, size_t q) {
    const size_t start_idx = coef_start_Q(q);
    return fe_coef.subvec(start_idx, start_idx + nb_coef_Q(q) - 1);
  }

  // Getters
  size_t get_nb_coef_T() const { return nb_coef_T; }
  size_t get_n_obs() const { return n_obs; }
  size_t get_Q() const { return Q; }
};

struct WeightedDemeanResult {
  mat demeaned_data;
  bool success;

  WeightedDemeanResult() : success(false) {}
};

// Forward declaration
DemeanResult joint_demean(const mat& YX_combined,
                         const field<uvec>& fe_ids,
                         const vec& weights, double tol,
                         size_t max_iter);

// Add the missing demean_variables function
inline WeightedDemeanResult demean_variables(
    const mat& data, const umat& fe_matrix, const vec& weights = vec(),
    double tol = 1e-8, size_t max_iter = 10000,
    const std::string& family = "gaussian") {
  WeightedDemeanResult result;

  if (fe_matrix.n_cols == 0) {
    result.demeaned_data = data;
    result.success = true;
    return result;
  }

  // Convert umat to field<uvec> format for joint_demean
  const size_t Q = fe_matrix.n_cols;
  field<uvec> fe_ids(Q);

  for (size_t q = 0; q < Q; ++q) {
    fe_ids(q) = fe_matrix.col(q);
  }

  // Use joint_demean
  DemeanResult demean_result =
      joint_demean(data, fe_ids, weights, tol, max_iter);

  result.demeaned_data = demean_result.demeaned_data;
  result.success = demean_result.success;

  return result;
}

// Optimized Irons-Tuck acceleration
inline bool irons_tuck_update(vec& X, const vec& GX, const vec& GGX,
                                   vec& delta_GX, vec& delta2_X) {
  // Vectorized computation - single pass
  delta_GX = GGX - GX;
  delta2_X = delta_GX - GX + X;

  const double vprod = dot(delta_GX, delta2_X);
  const double ssq = dot(delta2_X, delta2_X);

  if (ssq == 0.0) {
    return true;  // Failed
  }

  const double coef = vprod / ssq;
  X = GGX - coef * delta_GX;  // Vectorized update

  return false;  // Success
}

//  convergence checking
inline bool check_convergence(const vec& X, const vec& GX, double tol) {
  const vec diff = abs(GX - X);
  const vec rel_diff = diff / (0.1 + abs(X));
  // Change from AND to OR logic like fixest
  return all(diff <= tol) || all(rel_diff <= tol);  // Changed from && to ||
}

// Main joint demeaning function - optimized for speed
DemeanResult joint_demean(const mat& YX_combined,
                                   const field<uvec>& fe_ids,
                                   const vec& weights, double tol = 1e-6,
                                   size_t max_iter = 10000) {
  const size_t n_obs = YX_combined.n_rows;
  const size_t n_vars = YX_combined.n_cols;
  const size_t Q = fe_ids.n_elem;

  DemeanResult result(n_obs, n_vars);

  if (Q == 0) {
    // No fixed effects - return original data
    result.demeaned_data = YX_combined;
    result.success = true;
    return result;
  }

  // Initialize FE class
  FEClass fe_info(n_obs, fe_ids, weights);

  if (Q == 1) {
    // Single FE - closed form solution
    const size_t nb_coef = fe_info.get_nb_coef_T();
    vec fe_coef(nb_coef, fill::none);

    for (size_t v = 0; v < n_vars; ++v) {
      const vec variable = YX_combined.col(v);

      fe_info.compute_single_fe(fe_coef, variable);

      vec output(n_obs, fill::zeros);
      fe_info.add_fe_to_output(output, fe_coef, 0);

      result.demeaned_data.col(v) = variable - output;
      result.iterations(v) = 1;
    }

    result.success = true;
    return result;
  }

  // Multiple FE case - iterative algorithm with acceleration
  const size_t nb_coef_T = fe_info.get_nb_coef_T();

  // Pre-allocate all working vectors - avoid repeated allocation
  vec fe_coef_X(nb_coef_T, fill::none);
  vec fe_coef_GX(nb_coef_T, fill::none);
  vec fe_coef_GGX(nb_coef_T, fill::none);
  vec sum_other_means(n_obs, fill::none);
  vec delta_GX(nb_coef_T, fill::none);
  vec delta2_X(nb_coef_T, fill::none);

  // Process each variable
  for (size_t v = 0; v < n_vars; ++v) {
    const vec variable = YX_combined.col(v);

    // Initialize
    fe_coef_X.zeros();

    size_t iter = 0;
    bool converged = false;

    // First iteration
    for (size_t q = 0; q < Q; ++q) {
      // Compute sum of other FE contributions
      sum_other_means.zeros();
      for (size_t h = 0; h < Q; ++h) {
        if (h != q) {
          subview<double> other_coef =
              fe_info.get_fe_coef_subview(fe_coef_X, h);
          fe_info.add_fe_to_output(sum_other_means, other_coef, h);
        }
      }

      // Update this FE's coefficients
      fe_info.compute_fe_coef(fe_coef_GX, variable, q, sum_other_means);
    }

    // Check initial convergence
    converged = check_convergence(fe_coef_X, fe_coef_GX, tol);

    // Main iteration loop with acceleration
    while (!converged && iter < max_iter) {
      ++iter;

      // Second projection for acceleration
      fe_coef_X = fe_coef_GX;

      for (size_t q = 0; q < Q; ++q) {
        sum_other_means.zeros();
        for (size_t h = 0; h < Q; ++h) {
          if (h != q) {
            subview<double> other_coef =
                fe_info.get_fe_coef_subview(fe_coef_GX, h);
            fe_info.add_fe_to_output(sum_other_means, other_coef, h);
          }
        }
        fe_info.compute_fe_coef(fe_coef_GGX, variable, q, sum_other_means);
      }

      // Irons-Tuck acceleration
      if (irons_tuck_update(fe_coef_X, fe_coef_GX, fe_coef_GGX, delta_GX,
                                 delta2_X)) {
        break;  // Numerical failure
      }

      // Next iteration
      for (size_t q = 0; q < Q; ++q) {
        sum_other_means.zeros();
        for (size_t h = 0; h < Q; ++h) {
          if (h != q) {
            subview<double> other_coef =
                fe_info.get_fe_coef_subview(fe_coef_X, h);
            fe_info.add_fe_to_output(sum_other_means, other_coef, h);
          }
        }
        fe_info.compute_fe_coef(fe_coef_GX, variable, q, sum_other_means);
      }

      converged = check_convergence(fe_coef_X, fe_coef_GX, tol);
    }

    // Apply final fixed effects
    vec output(n_obs, fill::zeros);
    for (size_t q = 0; q < Q; ++q) {
      subview<double> final_coef = fe_info.get_fe_coef_subview(fe_coef_GX, q);
      fe_info.add_fe_to_output(output, final_coef, q);
    }

    result.demeaned_data.col(v) = variable - output;
    result.iterations(v) = iter;
  }

  result.success = true;
  return result;
}

#endif  // CAPYBARA_DEMEAN
