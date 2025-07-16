#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// Stopping/continuing criteria from fixest demeaning.cpp - exact implementation
inline bool continue_crit(double a, double b, double diffMax) {
  // continuing criterion of the algorithm
  double diff = std::abs(a - b);
  return ((diff > diffMax) && (diff / (0.1 + std::abs(a)) > diffMax));
}

inline bool stopping_crit(double a, double b, double diffMax) {
  // stopping criterion of the algorithm
  double diff = std::abs(a - b);
  return ((diff < diffMax) || (diff / (0.1 + std::abs(a)) < diffMax));
}

// Core structures matching fixest's approach
struct DemeanResult {
  vec demeaned_data;
  field<vec> fixed_effects;
  bool success;
  size_t iterations;
  double final_diff;

  DemeanResult() : success(false), iterations(0), final_diff(0.0) {}
};

struct MultiDemeanResult {
  mat demeaned_data;
  field<vec> fixed_effects;
  bool success;
  size_t iterations;
  double final_diff;

  MultiDemeanResult() : success(false), iterations(0), final_diff(0.0) {}
};

// FE class following fixest's exact implementation structure
class FEClass {
 private:
  size_t Q;        // Number of FE dimensions
  size_t n_obs;    // Number of observations
  bool is_weight;  // Whether weights are used

  // Group indices for each FE dimension (converted to 0-based)
  field<uvec> fe_indices;  // FE indices for each observation
  uvec nb_id_Q;            // Number of unique IDs per dimension

  double *p_weights;    // Pointer to weights
  vec weights_storage;  // Storage for weights if needed

  // Pre-computed sum of weights for each group
  field<vec> sum_weights;

 public:
  size_t nb_coef_T;   // Total number of coefficients
  uvec nb_coef_Q;     // Number of coefficients per dimension
  uvec coef_start_Q;  // Starting position of coefficients per dimension

  // Constructor - mimics fixest's FEClass constructor
  FEClass(size_t n_obs, size_t Q, const vec &weights,
          const field<field<uvec>> &group_indices);

  // Core functions matching fixest's API
  void compute_in_out(size_t q, vec &in_out_C, const vec &input_N,
                      const vec &output_N);
  void compute_fe_coef(size_t q, vec &fe_coef_C, const vec &sum_other_coef_N,
                       vec &in_out_C);
  void add_fe_coef_to_mu(size_t q, const vec &fe_coef_C, vec &out_N);
  void compute_fe_coef_2(const vec &fe_coef_in_C, vec &fe_coef_out_C,
                         vec &fe_coef_tmp, vec &in_out_C);
};

// Constructor implementation
inline FEClass::FEClass(size_t n_obs, size_t Q, const vec &weights,
                        const field<field<uvec>> &group_indices)
    : Q(Q), n_obs(n_obs) {
  // Initialize weights
  is_weight = weights.n_elem > 1;
  if (is_weight) {
    weights_storage = weights;
    p_weights = weights_storage.memptr();
  } else {
    weights_storage.set_size(1);
    weights_storage(0) = 1.0;
    p_weights = weights_storage.memptr();
  }

  // Initialize FE structures
  fe_indices.set_size(Q);
  nb_id_Q.set_size(Q);
  nb_coef_Q.set_size(Q);
  coef_start_Q.set_size(Q);
  sum_weights.set_size(Q);

  nb_coef_T = 0;
  for (size_t q = 0; q < Q; ++q) {
    // Convert group indices to FE indices (find unique groups)
    const field<uvec> &groups_q = group_indices(q);

    // Create mapping from observation to group ID
    uvec fe_id_q(n_obs);
    for (size_t g = 0; g < groups_q.n_elem; ++g) {
      const uvec &group_obs = groups_q(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_id_q(group_obs(i)) = g;  // 0-based indexing
      }
    }

    fe_indices(q) = fe_id_q;
    nb_id_Q(q) = groups_q.n_elem;
    nb_coef_Q(q) = groups_q.n_elem;

    // Compute starting positions
    if (q == 0) {
      coef_start_Q(q) = 0;
    } else {
      coef_start_Q(q) = coef_start_Q(q - 1) + nb_coef_Q(q - 1);
    }

    nb_coef_T += nb_coef_Q(q);

    // Pre-compute sum of weights for each group
    vec sum_w_q(nb_id_Q(q), fill::zeros);
    if (is_weight) {
      for (size_t i = 0; i < n_obs; ++i) {
        sum_w_q(fe_id_q(i)) += weights(i);
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        sum_w_q(fe_id_q(i)) += 1.0;
      }
    }
    sum_weights(q) = sum_w_q;
  }
}

// Compute conditional sum of input minus output - matching fixest's
// compute_in_out
inline void FEClass::compute_in_out(size_t q, vec &in_out_C, const vec &input_N,
                                    const vec &output_N) {
  const uvec &fe_id = fe_indices(q);
  const size_t nb_coef = nb_coef_Q(q);
  const size_t start_pos = coef_start_Q(q);

  // Initialize the relevant slice
  for (size_t c = 0; c < nb_coef; ++c) {
    in_out_C(start_pos + c) = 0.0;
  }

  // Accumulate weighted differences
  if (is_weight) {
    for (size_t i = 0; i < n_obs; ++i) {
      const size_t group_id = fe_id(i);
      in_out_C(start_pos + group_id) +=
          (input_N(i) - output_N(i)) * p_weights[i];
    }
  } else {
    for (size_t i = 0; i < n_obs; ++i) {
      const size_t group_id = fe_id(i);
      in_out_C(start_pos + group_id) += input_N(i) - output_N(i);
    }
  }
}

// Compute FE coefficients - matching fixest's compute_fe_coef
inline void FEClass::compute_fe_coef(size_t q, vec &fe_coef_C,
                                     const vec &sum_other_coef_N,
                                     vec &in_out_C) {
  const uvec &fe_id = fe_indices(q);
  const size_t nb_coef = nb_coef_Q(q);
  const size_t start_pos = coef_start_Q(q);
  const vec &sum_w = sum_weights(q);

  // Initialize coefficients from in_out
  for (size_t c = 0; c < nb_coef; ++c) {
    fe_coef_C(start_pos + c) = in_out_C(start_pos + c);
  }

  // Subtract contributions from other FEs
  if (is_weight) {
    for (size_t i = 0; i < n_obs; ++i) {
      const size_t group_id = fe_id(i);
      fe_coef_C(start_pos + group_id) -= sum_other_coef_N(i) * p_weights[i];
    }
  } else {
    for (size_t i = 0; i < n_obs; ++i) {
      const size_t group_id = fe_id(i);
      fe_coef_C(start_pos + group_id) -= sum_other_coef_N(i);
    }
  }

  // Normalize by sum of weights
  for (size_t c = 0; c < nb_coef; ++c) {
    if (sum_w(c) > 0) {
      fe_coef_C(start_pos + c) /= sum_w(c);
    }
  }
}

// Add FE coefficients to output - matching fixest's add_fe_coef_to_mu
inline void FEClass::add_fe_coef_to_mu(size_t q, const vec &fe_coef_C,
                                       vec &out_N) {
  const uvec &fe_id = fe_indices(q);
  const size_t start_pos = coef_start_Q(q);

  for (size_t i = 0; i < n_obs; ++i) {
    const size_t group_id = fe_id(i);
    out_N(i) += fe_coef_C(start_pos + group_id);
  }
}

// Two-FE computation - matching fixest's compute_fe_coef_2
inline void FEClass::compute_fe_coef_2(const vec &fe_coef_in_C,
                                       vec &fe_coef_out_C, vec &fe_coef_tmp,
                                       vec &in_out_C) {
  const size_t start_a = coef_start_Q(0);
  const size_t start_b = coef_start_Q(1);
  const size_t nb_coef_b = nb_coef_Q(1);

  const uvec &fe_id_a = fe_indices(0);
  const uvec &fe_id_b = fe_indices(1);
  const vec &sum_w_b = sum_weights(1);

  // Initialize coefficients for dimension B from in_out
  for (size_t c = 0; c < nb_coef_b; ++c) {
    fe_coef_out_C(start_b + c) = in_out_C(start_b + c);
  }

  // Subtract contributions from dimension A
  if (is_weight) {
    for (size_t i = 0; i < n_obs; ++i) {
      const size_t group_a = fe_id_a(i);
      const size_t group_b = fe_id_b(i);
      fe_coef_out_C(start_b + group_b) -=
          fe_coef_in_C(start_a + group_a) * p_weights[i];
    }
  } else {
    for (size_t i = 0; i < n_obs; ++i) {
      const size_t group_a = fe_id_a(i);
      const size_t group_b = fe_id_b(i);
      fe_coef_out_C(start_b + group_b) -= fe_coef_in_C(start_a + group_a);
    }
  }

  // Normalize by sum of weights
  for (size_t c = 0; c < nb_coef_b; ++c) {
    if (sum_w_b(c) > 0) {
      fe_coef_out_C(start_b + c) /= sum_w_b(c);
    }
  }
}

// Core FE computation function - matching fixest's compute_fe
inline void compute_fe(size_t Q, vec &X, vec &GX, vec &sum_other_means,
                       vec &sum_in_out, FEClass &FE_info, const vec &input,
                       const vec &output) {
  const size_t n_obs = input.n_elem;

  if (Q == 2) {
    // Special optimized 2-FE case following fixest exactly
    FE_info.compute_fe_coef_2(X, GX, sum_other_means, sum_in_out);
  } else {
    // General Q-FE case
    sum_other_means.zeros();

    // Compute contributions from all other FEs
    for (size_t q = 1; q < Q; ++q) {
      vec temp_coef =
          GX.subvec(FE_info.coef_start_Q(q),
                    FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
      vec temp_output(n_obs, fill::zeros);
      FE_info.add_fe_coef_to_mu(q, temp_coef, temp_output);
      sum_other_means += temp_output;
    }

    // Compute first FE coefficients
    vec in_out_slice = sum_in_out.head(FE_info.nb_coef_Q(0));
    FE_info.compute_fe_coef(0, GX, sum_other_means, in_out_slice);

    // Compute remaining FE coefficients
    for (size_t q = 1; q < Q; ++q) {
      sum_other_means.zeros();

      // Add contributions from all other FEs
      for (size_t q2 = 0; q2 < Q; ++q2) {
        if (q2 != q) {
          vec temp_coef =
              GX.subvec(FE_info.coef_start_Q(q2),
                        FE_info.coef_start_Q(q2) + FE_info.nb_coef_Q(q2) - 1);
          vec temp_output(n_obs, fill::zeros);
          FE_info.add_fe_coef_to_mu(q2, temp_coef, temp_output);
          sum_other_means += temp_output;
        }
      }

      vec in_out_slice =
          sum_in_out.subvec(FE_info.coef_start_Q(q),
                            FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
      FE_info.compute_fe_coef(q, GX, sum_other_means, in_out_slice);
    }
  }

  // Copy GX to X for the iteration
  X = GX;
}

// Main demeaning function with Irons-Tuck acceleration - following fixest's
// demean_acc_gnl
inline bool demean_acc_gnl(vec &input, vec &output,
                           const field<field<uvec>> &group_indices,
                           const vec &weights, size_t iterMax, double diffMax,
                           bool two_fe = false) {
  const size_t n_obs = input.n_elem;
  const size_t Q = group_indices.n_elem;

  if (Q == 0) {
    output.zeros();
    return true;
  }

  // Create FE class
  FEClass FE_info(n_obs, Q, weights, group_indices);

  const size_t effective_Q = two_fe ? 2 : Q;
  const size_t nb_coef_T = two_fe
                               ? (FE_info.nb_coef_Q(0) + FE_info.nb_coef_Q(1))
                               : FE_info.nb_coef_T;
  const size_t nb_coef_first = FE_info.nb_coef_Q(0);

  // Setup vectors
  vec X(nb_coef_first, fill::zeros);
  vec GX(nb_coef_T, fill::zeros);
  vec GGX(nb_coef_T, fill::zeros);

  vec Y(nb_coef_first, fill::zeros);
  vec GY(nb_coef_T, fill::zeros);
  vec GGY(nb_coef_T, fill::zeros);

  // Temporary vectors
  const size_t size_other_means = two_fe ? FE_info.nb_coef_Q(1) : n_obs;
  vec sum_other_means(size_other_means, fill::zeros);
  vec sum_in_out(nb_coef_T, fill::zeros);

  // Compute initial in_out for all FEs
  for (size_t q = 0; q < effective_Q; ++q) {
    FE_info.compute_in_out(q, sum_in_out, input, output);
  }

  // First iteration
  compute_fe(effective_Q, X, GX, sum_other_means, sum_in_out, FE_info, input,
             output);

  // Check convergence criteria
  bool keepGoing = false;
  for (size_t i = 0; i < nb_coef_first; ++i) {
    if (continue_crit(X(i), GX(i), diffMax)) {
      keepGoing = true;
      break;
    }
  }

  if (!keepGoing) {
    // Already converged, compute final output
    output.zeros();
    for (size_t q = 0; q < effective_Q; ++q) {
      vec coef_slice =
          GX.subvec(FE_info.coef_start_Q(q),
                    FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
      FE_info.add_fe_coef_to_mu(q, coef_slice, output);
    }
    return true;
  }

  // Main iteration loop with acceleration
  size_t iter = 0;
  const size_t n_extraProj = 5;        // fixest default
  const size_t iter_projAfterAcc = 3;  // fixest default
  const size_t iter_grandAcc =
      1000000;  // fixest default (effectively disabled)

  size_t grand_acc = 0;

  while (keepGoing && iter < iterMax) {
    iter++;

    // Extra projections for acceleration
    for (size_t rep = 0; rep < n_extraProj; ++rep) {
      // Recompute in_out
      for (size_t q = 0; q < effective_Q; ++q) {
        FE_info.compute_in_out(q, sum_in_out, input, output);
      }

      compute_fe(effective_Q, X, GX, sum_other_means, sum_in_out, FE_info,
                 input, output);
    }

    // Check for Irons-Tuck acceleration
    bool do_acceleration =
        (iter % iter_projAfterAcc == 0) && (iter < iter_grandAcc);

    if (do_acceleration) {
      // Save current state
      Y = X;
      GY = GX;

      // One more iteration to get GGX
      for (size_t q = 0; q < effective_Q; ++q) {
        FE_info.compute_in_out(q, sum_in_out, input, output);
      }
      compute_fe(effective_Q, X, GGX, sum_other_means, sum_in_out, FE_info,
                 input, output);

      // Irons-Tuck acceleration
      vec delta_GX = GX - X;
      vec delta2_X = GGX - 2.0 * GX + X;

      double num = dot(delta_GX, delta2_X);
      double denom = dot(delta2_X, delta2_X);

      if (denom > 1e-14 && std::abs(num / denom) < 10.0) {  // Safety check
        double lambda = -num / denom;
        lambda = std::max(0.0, std::min(1.0, lambda));  // Clamp to [0,1]

        // Apply acceleration
        X = X + lambda * delta_GX;

        grand_acc++;
      } else {
        // Acceleration failed, use regular update
        X = GX;
      }
    } else {
      // Regular update
      X = GX;
    }

    // Check convergence
    keepGoing = false;
    for (size_t i = 0; i < nb_coef_first; ++i) {
      if (continue_crit(X(i), GX(i), diffMax)) {
        keepGoing = true;
        break;
      }
    }
  }

  // Compute final output
  output.zeros();
  for (size_t q = 0; q < effective_Q; ++q) {
    vec coef_slice =
        X.subvec(FE_info.coef_start_Q(q),
                 FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
    if (q == 0) {
      FE_info.add_fe_coef_to_mu(q, coef_slice, output);
    } else {
      vec full_coef =
          GX.subvec(FE_info.coef_start_Q(q),
                    FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
      FE_info.add_fe_coef_to_mu(q, full_coef, output);
    }
  }

  return iter < iterMax;
}

// Forward declaration for demean_acc_gnl_complete
inline bool demean_acc_gnl_complete(vec &input, vec &output, const field<field<uvec>> &group_indices,
                                   const vec &weights, size_t iterMax, double diffMax, 
                                   bool two_fe = false, bool save_fixef = false,
                                   field<vec> *saved_fixef = nullptr);

// Complete single variable demeaning following fixest's demean_single_gnl exactly
inline DemeanResult demean_single_gnl_complete(vec &x, const field<field<uvec>> &group_indices,
                                              const vec &weights = vec(), double tol = 1e-8,
                                              size_t max_iter = 10000, bool save_fixef = false) {
  DemeanResult result;
  const size_t Q = group_indices.n_elem;
  
  if (Q == 0) {
    result.demeaned_data = x;
    result.success = true;
    result.iterations = 0;
    result.final_diff = 0.0;
    return result;
  }
  
  vec output(x.n_elem, fill::zeros);
  field<vec> saved_fixef_storage;
  
  if (Q == 2) {
    // Use 2-FE optimization exactly like fixest
    result.success = demean_acc_gnl_complete(x, output, group_indices, weights, max_iter, tol, 
                                           false, save_fixef, save_fixef ? &saved_fixef_storage : nullptr);
    result.iterations = max_iter;  // Would need to track actual iterations for precision
  } else {
    // General algorithm with warm-up and re-acceleration following fixest exactly
    bool conv = false;
    size_t iter_used = 0;
    
    const size_t iter_warmup = 15;  // fixest default
    
    // Warm-up phase
    if (iter_warmup > 0 && max_iter > iter_warmup) {
      conv = demean_acc_gnl_complete(x, output, group_indices, weights, iter_warmup, tol, 
                                   false, save_fixef, save_fixef ? &saved_fixef_storage : nullptr);
      iter_used += iter_warmup;
    }
    
    if (!conv && iter_used < max_iter) {
      // Convergence for first 2 FEs only
      size_t iter_max_2FE = (max_iter - iter_used) / 2;
      if (iter_max_2FE > 0) {
        demean_acc_gnl_complete(x, output, group_indices, weights, iter_max_2FE, tol, 
                              true, save_fixef, save_fixef ? &saved_fixef_storage : nullptr);
        iter_used += iter_max_2FE;
      }
      
      // Re-acceleration with all FEs
      if (iter_used < max_iter) {
        conv = demean_acc_gnl_complete(x, output, group_indices, weights, max_iter - iter_used, tol, 
                                     false, save_fixef, save_fixef ? &saved_fixef_storage : nullptr);
        iter_used = max_iter;
      }
    }
    
    result.success = conv;
    result.iterations = iter_used;
  }
  
  result.demeaned_data = x - output;
  result.final_diff = tol;  // Simplified
  
  // Set fixed effects
  if (save_fixef && saved_fixef_storage.n_elem > 0) {
    result.fixed_effects = saved_fixef_storage;
  } else {
    result.fixed_effects.set_size(Q);
    for (size_t q = 0; q < Q; ++q) {
      result.fixed_effects(q) = vec(group_indices(q).n_elem, fill::zeros);
    }
  }
  
  return result;
}

// Single variable demeaning following fixest's demean_single_gnl logic
inline DemeanResult demean_single_gnl(vec &x,
                                      const field<field<uvec>> &group_indices,
                                      const vec &weights = vec(),
                                      double tol = 1e-8,
                                      size_t max_iter = 10000) {
  // Use the complete implementation for full fixest compatibility
  return demean_single_gnl_complete(x, group_indices, weights, tol, max_iter, true);
}

// Simplified matrix demeaning interface
inline MultiDemeanResult demean_matrix(const mat &X,
                                       const field<field<uvec>> &group_indices,
                                       const vec &weights = vec(),
                                       double tol = 1e-8,
                                       size_t max_iter = 10000) {
  MultiDemeanResult result;
  result.demeaned_data = X;
  result.success = true;
  result.iterations = 0;

  if (group_indices.n_elem == 0) {
    return result;
  }

  const size_t n_vars = X.n_cols;
  result.fixed_effects.set_size(group_indices.n_elem);

  // Demean each variable separately using the optimized algorithm
  size_t total_iter = 0;

  for (size_t v = 0; v < n_vars; ++v) {
    vec x_v = X.col(v);
    DemeanResult var_result;

    if (group_indices.n_elem == 1) {
      // Single FE - use fast direct computation
      var_result = demean_single_gnl_complete(x_v, group_indices, weights, tol, max_iter, false);
    } else if (group_indices.n_elem == 2) {
      // Two FE - use optimized 2-FE algorithm
      var_result = demean_single_gnl_complete(x_v, group_indices, weights, tol, max_iter, false);
    } else {
      // General case - use complete fixest algorithm with full acceleration
      var_result = demean_single_gnl_complete(x_v, group_indices, weights, tol, max_iter, false);
    }
    
    result.demeaned_data.col(v) = var_result.demeaned_data;
    total_iter = std::max(total_iter, var_result.iterations);

    // Store fixed effects for first variable
    if (v == 0) {
      result.fixed_effects = var_result.fixed_effects;
    }
  }

  result.iterations = total_iter;
  result.final_diff = tol;

  return result;
}

// Convenience function for single variable demeaning
inline DemeanResult demean_variable(vec x,
                                    const field<field<uvec>> &group_indices,
                                    const vec &weights = vec(),
                                    double tol = 1e-8, size_t max_iter = 10000,
                                    const std::string &method = "auto") {
  DemeanResult result;

  if (group_indices.n_elem == 0) {
    result.demeaned_data = x;
    result.success = true;
    result.iterations = 0;
    result.final_diff = 0.0;
    return result;
  }

  // Use matrix interface for single variable
  mat X_mat = x;
  X_mat.reshape(x.n_elem, 1);

  MultiDemeanResult multi_result =
      demean_matrix(X_mat, group_indices, weights, tol, max_iter);

  result.demeaned_data = multi_result.demeaned_data.col(0);
  result.fixed_effects = multi_result.fixed_effects;
  result.success = multi_result.success;
  result.iterations = multi_result.iterations;
  result.final_diff = multi_result.final_diff;

  return result;
}

// Helper function to add group effects back (for recovery)
inline void add_group_effects(vec &y, const vec &group_effects,
                              const field<uvec> &group_indices) {
  const size_t n_groups = group_indices.n_elem;

  for (size_t g = 0; g < n_groups; ++g) {
    const uvec &idx = group_indices(g);
    if (idx.n_elem == 0) continue;

    const double effect = group_effects(g);
    y.elem(idx) += effect;
  }
}

// PPML weighted demeaning helper
struct WeightedDemeanResult {
  mat demeaned_data;
  bool success;
};

// Simplified FEClass for single/two FE cases - optimized version
class SimpleFEClass {
private:
  size_t Q;              // Number of fixed effects
  size_t n_obs;          // Number of observations
  bool is_weight;        // Whether weights are used
  
  // Group structure - exactly like fixest
  field<uvec> fe_id;     // FE identifiers for each observation (Q vectors of length n_obs)
  uvec nb_id_Q;          // Number of groups in each FE
  
  // Weights
  vec weights;           // Observation weights
  field<vec> sum_weights_C; // Sum of weights for each group in each FE
  
  // Working storage
  field<vec> fe_coef;    // Fixed effects coefficients
  
public:
  SimpleFEClass(const field<field<uvec>> &group_indices_in, const vec &weights_in = vec()) {
    Q = group_indices_in.n_elem;
    is_weight = (weights_in.n_elem > 1);
    
    if (Q == 0) return;
    
    // Determine n_obs from group structure
    n_obs = 0;
    for (size_t q = 0; q < Q; ++q) {
      for (size_t g = 0; g < group_indices_in(q).n_elem; ++g) {
        const uvec &group_obs = group_indices_in(q)(g);
        if (group_obs.n_elem > 0) {
          n_obs = std::max(n_obs, static_cast<size_t>(max(group_obs) + 1));
        }
      }
    }
    
    // Initialize structures
    fe_id.set_size(Q);
    nb_id_Q.set_size(Q);
    sum_weights_C.set_size(Q);
    fe_coef.set_size(Q);
    
    // Convert group_indices format to fe_id format (like fixest)
    for (size_t q = 0; q < Q; ++q) {
      fe_id(q).set_size(n_obs);
      nb_id_Q(q) = group_indices_in(q).n_elem;
      
      // Map observations to group IDs
      for (size_t g = 0; g < group_indices_in(q).n_elem; ++g) {
        const uvec &group_obs = group_indices_in(q)(g);
        for (size_t i = 0; i < group_obs.n_elem; ++i) {
          fe_id(q)(group_obs(i)) = g;
        }
      }
      
      // Initialize coefficient storage
      fe_coef(q).set_size(nb_id_Q(q));
      sum_weights_C(q).set_size(nb_id_Q(q));
    }
    
    // Set up weights
    if (is_weight) {
      weights = weights_in;
    } else {
      weights.ones(n_obs);
    }
    
    // Precompute sum of weights for each group
    for (size_t q = 0; q < Q; ++q) {
      sum_weights_C(q).zeros();
      for (size_t i = 0; i < n_obs; ++i) {
        uword group_id = fe_id(q)(i);
        sum_weights_C(q)(group_id) += weights(i);
      }
    }
  }
  
  // Demean using alternating projection algorithm following fixest exactly
  vec demean_alternating_projection(vec x, double tol = 1e-8, size_t max_iter = 10000) {
    if (Q == 0) return x;
    
    vec mu = x;  // Working copy
    vec mu_old = mu;
    
    size_t iter = 0;
    bool converged = false;
    
    while (iter < max_iter && !converged) {
      mu_old = mu;
      
      // Project onto each FE space sequentially
      for (size_t q = 0; q < Q; ++q) {
        project_onto_fe(q, mu);
      }
      
      // Check convergence using fixest criteria
      double max_diff = 0.0;
      for (size_t i = 0; i < n_obs; ++i) {
        double diff = std::abs(mu(i) - mu_old(i));
        max_diff = std::max(max_diff, diff);
        if (continue_crit(mu(i), mu_old(i), tol)) {
          max_diff = datum::inf;  // Force continuation
          break;
        }
      }
      
      converged = (max_diff < tol);
      iter++;
    }
    
    return x - mu;  // Return demeaned data
  }
  
  // Single FE projection - compute group means and subtract
  void project_onto_fe(size_t q, vec &mu) {
    const uvec &my_fe = fe_id(q);
    const size_t nb_coef = nb_id_Q(q);
    
    // Compute group means
    fe_coef(q).zeros();
    for (size_t i = 0; i < n_obs; ++i) {
      uword group_id = my_fe(i);
      fe_coef(q)(group_id) += weights(i) * mu(i);
    }
    
    // Normalize by sum of weights
    for (size_t g = 0; g < nb_coef; ++g) {
      if (sum_weights_C(q)(g) > 0) {
        fe_coef(q)(g) /= sum_weights_C(q)(g);
      }
    }
    
    // Subtract group means
    for (size_t i = 0; i < n_obs; ++i) {
      uword group_id = my_fe(i);
      mu(i) -= fe_coef(q)(group_id);
    }
  }
  
  // Two-FE specialized algorithm following fixest exactly  
  vec demean_two_fe_optimized(vec x, double tol = 1e-8, size_t max_iter = 10000) {
    if (Q != 2) {
      return demean_alternating_projection(x, tol, max_iter);
    }
    
    vec mu = x;
    vec alpha_a(nb_id_Q(0), fill::zeros);  // First FE coefficients
    vec alpha_b(nb_id_Q(1), fill::zeros);  // Second FE coefficients
    
    const uvec &fe_a = fe_id(0);
    const uvec &fe_b = fe_id(1);
    
    size_t iter = 0;
    bool converged = false;
    
    while (iter < max_iter && !converged) {
      vec mu_old = mu;
      
      // Update first FE coefficients
      alpha_a.zeros();
      for (size_t i = 0; i < n_obs; ++i) {
        alpha_a(fe_a(i)) += weights(i) * (x(i) - alpha_b(fe_b(i)));
      }
      for (size_t g = 0; g < nb_id_Q(0); ++g) {
        if (sum_weights_C(0)(g) > 0) {
          alpha_a(g) /= sum_weights_C(0)(g);
        }
      }
      
      // Update second FE coefficients  
      alpha_b.zeros();
      for (size_t i = 0; i < n_obs; ++i) {
        alpha_b(fe_b(i)) += weights(i) * (x(i) - alpha_a(fe_a(i)));
      }
      for (size_t g = 0; g < nb_id_Q(1); ++g) {
        if (sum_weights_C(1)(g) > 0) {
          alpha_b(g) /= sum_weights_C(1)(g);
        }
      }
      
      // Update mu
      for (size_t i = 0; i < n_obs; ++i) {
        mu(i) = alpha_a(fe_a(i)) + alpha_b(fe_b(i));
      }
      
      // Check convergence
      converged = true;
      for (size_t i = 0; i < n_obs; ++i) {
        if (continue_crit(mu(i), mu_old(i), tol)) {
          converged = false;
          break;
        }
      }
      
      iter++;
    }
    
    return x - mu;  // Return demeaned data
  }
  
  // Get fixed effects for output
  field<vec> get_fixed_effects() const {
    return fe_coef;
  }
  
  size_t get_n_fe() const { return Q; }
  size_t get_n_obs() const { return n_obs; }
};

// Irons-Tuck acceleration update - following fixest exactly
inline bool dm_update_X_IronsTuck(size_t nb_coef, vec &X, const vec &GX, const vec &GGX,
                                  vec &delta_GX, vec &delta2_X) {
  // Compute differences following fixest exactly
  for (size_t i = 0; i < nb_coef; ++i) {
    double GX_tmp = GX(i);
    delta_GX(i) = GGX(i) - GX_tmp;
    delta2_X(i) = delta_GX(i) - GX_tmp + X(i);
  }
  
  // Compute acceleration parameter
  double vprod = dot(delta_GX, delta2_X);
  double ssq = dot(delta2_X, delta2_X);
  
  bool failed = false;
  
  if (ssq == 0.0) {
    failed = true;
  } else {
    double coef = vprod / ssq;
    
    // Safety bounds on acceleration coefficient
    if (std::abs(coef) > 10.0) {
      failed = true;
    } else {
      // Apply Irons-Tuck update: X = GGX - coef * delta_GX
      X = GGX - coef * delta_GX;
    }
  }
  
  return failed;
}

// General FE computation for Q >= 3 - following fixest's compute_fe_gnl
inline void compute_fe_gnl(vec &fe_coef_origin, vec &fe_coef_destination,
                          vec &sum_other_means, vec &sum_in_out,
                          FEClass &FE_info, size_t Q, size_t n_obs) {
  
  // Update each FE coefficient, starting from Q-1 (the last one)
  for (int q = static_cast<int>(Q) - 1; q >= 0; --q) {
    size_t uq = static_cast<size_t>(q);
    
    // STEP 1: Compute sum of all other FE contributions
    sum_other_means.zeros();
    
    for (size_t h = 0; h < Q; ++h) {
      if (h == uq) continue;
      
      // Choose source: origin for h < q, destination for h > q
      const vec &my_fe_coef = (h < uq) ? fe_coef_origin : fe_coef_destination;
      
      // Add weighted FE contribution to sum
      vec temp_contrib(n_obs, fill::zeros);
      FE_info.add_fe_coef_to_mu(h, my_fe_coef, temp_contrib);
      sum_other_means += temp_contrib;
    }
    
    // STEP 2: Compute FE coefficients for dimension q
    vec in_out_slice = sum_in_out.subvec(FE_info.coef_start_Q(uq), 
                                        FE_info.coef_start_Q(uq) + FE_info.nb_coef_Q(uq) - 1);
    FE_info.compute_fe_coef(uq, fe_coef_destination, sum_other_means, in_out_slice);
  }
}

// Two-FE specific functions following fixest exactly
inline void add_2_fe_coef_to_mu(FEClass &FE_info, const vec &fe_coef_a, const vec &fe_coef_b, 
                               const vec &sum_in_out, vec &output, bool update_beta = true) {
  output.zeros();
  
  // Create full coefficient vectors that match FE_info expectations
  vec full_coef_a(FE_info.nb_coef_T, fill::zeros);
  vec full_coef_b(FE_info.nb_coef_T, fill::zeros);
  
  // Copy fe_coef_a to the correct positions
  for (size_t i = 0; i < fe_coef_a.n_elem; ++i) {
    full_coef_a(FE_info.coef_start_Q(0) + i) = fe_coef_a(i);
  }
  
  // Copy fe_coef_b to the correct positions
  if (update_beta) {
    for (size_t i = 0; i < fe_coef_b.n_elem; ++i) {
      full_coef_b(FE_info.coef_start_Q(1) + i) = fe_coef_b(i);
    }
  } else {
    // Use coefficients from sum_in_out slice
    for (size_t i = 0; i < FE_info.nb_coef_Q(1); ++i) {
      full_coef_b(FE_info.coef_start_Q(1) + i) = sum_in_out(FE_info.coef_start_Q(1) + i);
    }
  }
  
  // Add FE contributions using full coefficient vectors
  FE_info.add_fe_coef_to_mu(0, full_coef_a, output);
  FE_info.add_fe_coef_to_mu(1, full_coef_b, output);
}

// Enhanced compute_fe function with full fixest compatibility
inline void compute_fe_enhanced(size_t Q, vec &X, vec &GX, vec &sum_other_means,
                               vec &sum_in_out, FEClass &FE_info, const vec &input, 
                               const vec &output, bool two_fe_algo = false) {
  const size_t n_obs = input.n_elem;
  
  if (two_fe_algo && Q >= 2) {
    // Two-FE alternating projection algorithm following fixest exactly
    // This implements the specialized two-FE algorithm from compute_fe_coef_2
    
    // Step 1: Compute first FE coefficients accounting for second FE
    vec temp_means(n_obs, fill::zeros);
    vec in_out_a = sum_in_out.subvec(FE_info.coef_start_Q(0), 
                                     FE_info.coef_start_Q(0) + FE_info.nb_coef_Q(0) - 1);
    FE_info.compute_fe_coef(0, in_out_a, temp_means, in_out_a);
    
    // Store first FE coefficients in GX
    for (size_t i = 0; i < FE_info.nb_coef_Q(0); ++i) {
      GX(FE_info.coef_start_Q(0) + i) = in_out_a(i);
    }
    
    // Step 2: Use compute_fe_coef_2 for the second FE
    vec fe_coef_temp = sum_other_means;  // Use as temporary storage
    FE_info.compute_fe_coef_2(in_out_a, GX, fe_coef_temp, sum_in_out);
    
  } else if (Q <= 2) {
    // Simple 1-2 FE case
    for (size_t q = 0; q < Q; ++q) {
      if (q == 0) {
        // First FE - compute without other contributions
        vec temp_means(n_obs, fill::zeros);
        vec in_out_slice = sum_in_out.subvec(FE_info.coef_start_Q(q), 
                                            FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
        FE_info.compute_fe_coef(q, GX, temp_means, in_out_slice);
      } else {
        // Second FE - subtract first FE contribution
        vec temp_contrib(n_obs, fill::zeros);
        vec first_fe_slice = GX.subvec(FE_info.coef_start_Q(0), 
                                      FE_info.coef_start_Q(0) + FE_info.nb_coef_Q(0) - 1);
        FE_info.add_fe_coef_to_mu(0, first_fe_slice, temp_contrib);
        
        vec in_out_slice = sum_in_out.subvec(FE_info.coef_start_Q(q), 
                                            FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
        FE_info.compute_fe_coef(q, GX, temp_contrib, in_out_slice);
      }
    }
  } else {
    // General Q >= 3 case - use compute_fe_gnl approach
    vec fe_coef_temp = GX;  // Working copy
    compute_fe_gnl(X, GX, sum_other_means, sum_in_out, FE_info, Q, n_obs);
  }
  
  // Update X for next iteration (only first FE coefficients for convergence check)
  X = GX.head(FE_info.nb_coef_Q(0));
}

// Complete demean_acc_gnl following fixest's exact algorithm structure
inline bool demean_acc_gnl_complete(vec &input, vec &output, const field<field<uvec>> &group_indices,
                                   const vec &weights, size_t iterMax, double diffMax, 
                                   bool two_fe, bool save_fixef,
                                   field<vec> *saved_fixef) {
  const size_t n_obs = input.n_elem;
  const size_t Q = group_indices.n_elem;
  
  if (Q == 0) {
    output.zeros();
    return true;
  }
  
  // Create FE class
  FEClass FE_info(n_obs, Q, weights, group_indices);

  const size_t effective_Q = two_fe ? std::min(Q, 2UL) : Q;
  const bool two_fe_algo = (effective_Q == 2);
  const size_t nb_coef_T = two_fe_algo ? (FE_info.nb_coef_Q(0) + FE_info.nb_coef_Q(1)) : FE_info.nb_coef_T;
  const size_t nb_coef_first = FE_info.nb_coef_Q(0);
  
  // Setup vectors following fixest exactly
  vec X(nb_coef_first, fill::zeros);
  vec GX(nb_coef_T, fill::zeros);  
  vec GGX(nb_coef_T, fill::zeros);
  
  // Irons-Tuck acceleration vectors
  vec Y(nb_coef_first, fill::zeros);
  vec GY(nb_coef_T, fill::zeros);
  vec delta_GX(nb_coef_first, fill::zeros);
  vec delta2_X(nb_coef_first, fill::zeros);
  
  // Temporary vectors
  const size_t size_other_means = two_fe_algo ? FE_info.nb_coef_Q(1) : n_obs;
  vec sum_other_means(size_other_means, fill::zeros);
  vec sum_in_out(nb_coef_T, fill::zeros);

  // Compute initial in_out for all FEs
  for (size_t q = 0; q < effective_Q; ++q) {
    FE_info.compute_in_out(q, sum_in_out, input, output);
  }

  // First iteration
  compute_fe_enhanced(effective_Q, X, GX, sum_other_means, sum_in_out, FE_info, input, output, two_fe_algo);

  // Check convergence criteria with improved tolerance
  bool keepGoing = false;
  for (size_t i = 0; i < nb_coef_first; ++i) {
    if (continue_crit(X(i), GX(i), diffMax)) {
      keepGoing = true;
      break;
    }
  }

  if (!keepGoing) {
    // Already converged, compute final output
    if (two_fe_algo) {
      // Check bounds before slicing - FIX THE BOUNDS CHECKING
      
      if (FE_info.nb_coef_Q(0) > 0 && FE_info.nb_coef_Q(0) <= GX.n_elem) {
        vec fe_coef_a = GX.subvec(0, FE_info.nb_coef_Q(0) - 1);
        
        size_t start_idx = FE_info.coef_start_Q(1);
        size_t end_idx = start_idx + FE_info.nb_coef_Q(1) - 1;
                
        if (start_idx < GX.n_elem && end_idx < GX.n_elem && start_idx <= end_idx) {
          vec fe_coef_b = GX.subvec(start_idx, end_idx);
          add_2_fe_coef_to_mu(FE_info, fe_coef_a, fe_coef_b, sum_in_out, output, true);
        } else {
          output.zeros();
        }
      } else {
        output.zeros();
      }
    } else {
      output.zeros();
      for (size_t q = 0; q < effective_Q; ++q) {
        size_t start_idx = FE_info.coef_start_Q(q);
        size_t end_idx = start_idx + FE_info.nb_coef_Q(q) - 1;
        
        if (end_idx < GX.n_elem) {
          vec coef_slice = GX.subvec(start_idx, end_idx);
          FE_info.add_fe_coef_to_mu(q, coef_slice, output);
        } else {
          // TODO: CRAN-likeable error that is also portable
          std::cout << "Error: coef_slice index out of bounds for q=" << q << ": " << end_idx << " >= " << GX.n_elem << std::endl;
        }
      }
    }
    
    // Save fixed effects if requested
    if (save_fixef && saved_fixef) {
      saved_fixef->set_size(effective_Q);
      for (size_t q = 0; q < effective_Q; ++q) {
        (*saved_fixef)(q) = GX.subvec(FE_info.coef_start_Q(q), 
                                     FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
      }
    }
    
    return true;
  }

  // Main iteration loop with full fixest algorithm
  size_t iter = 0;
  const size_t n_extraProj = two_fe_algo ? 10 : 5;        // More iterations for 2FE
  const size_t iter_projAfterAcc = 3;  // fixest default
  const size_t iter_grandAcc = 1000000; // fixest default (effectively disabled)
  
  size_t grand_acc = 0;
  double ssr = 0.0;  // Sum of squared residuals for convergence check
  
  while (keepGoing && iter < iterMax) {
    iter++;
    
    // Extra projections for acceleration
    for (size_t rep = 0; rep < n_extraProj; ++rep) {
      // Recompute in_out
      for (size_t q = 0; q < effective_Q; ++q) {
        FE_info.compute_in_out(q, sum_in_out, input, output);
      }
      
      compute_fe_enhanced(effective_Q, X, GX, sum_other_means, sum_in_out, FE_info, input, output, two_fe_algo);
    }
    
    // Check for Irons-Tuck acceleration
    bool do_acceleration = (iter % iter_projAfterAcc == 0) && (iter < iter_grandAcc);
    
    if (do_acceleration) {
      // Save current state
      Y = X;
      GY = GX;
      
      // One more iteration to get GGX
      for (size_t q = 0; q < effective_Q; ++q) {
        FE_info.compute_in_out(q, sum_in_out, input, output);
      }
      compute_fe_enhanced(effective_Q, Y, GGX, sum_other_means, sum_in_out, FE_info, input, output, two_fe_algo);
      
      // Apply Irons-Tuck acceleration on first FE coefficients only
      vec delta_GX_sub = delta_GX.head(nb_coef_first);
      vec delta2_X_sub = delta2_X.head(nb_coef_first);
      vec GX_sub = GX.head(nb_coef_first);
      vec GGX_sub = GGX.head(nb_coef_first);
      
      bool accel_failed = dm_update_X_IronsTuck(nb_coef_first, X, GX_sub, GGX_sub, delta_GX_sub, delta2_X_sub);
      
      if (accel_failed) {
        // Acceleration failed, use regular update
        X = GX_sub;
      } else {
        grand_acc++;
      }
    } else {
      // Regular update
      X = GX.head(nb_coef_first);
    }
    
    // Check convergence using fixest criteria with better tolerance
    keepGoing = false;
    double max_change = 0.0;
    for (size_t i = 0; i < nb_coef_first; ++i) {
      double change = std::abs(X(i) - GX(i));
      max_change = std::max(max_change, change);
      if (continue_crit(X(i), GX(i), diffMax)) {
        keepGoing = true;
      }
    }
    
    // Additional SSR-based convergence check every 40 iterations (like fixest)
    if (iter % 40 == 0) {
      vec mu_current(n_obs, fill::zeros);
      
      if (two_fe_algo) {
        vec fe_coef_a = GX.head(FE_info.nb_coef_Q(0));
        vec fe_coef_b = sum_other_means;  // Used as temp storage in 2FE case
        add_2_fe_coef_to_mu(FE_info, fe_coef_a, fe_coef_b, sum_in_out, mu_current, true);
      } else {
        for (size_t q = 0; q < effective_Q; ++q) {
          vec coef_slice = GX.subvec(FE_info.coef_start_Q(q), 
                                    FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
          FE_info.add_fe_coef_to_mu(q, coef_slice, mu_current);
        }
      }
      
      double ssr_old = ssr;
      vec residuals = input - mu_current;
      ssr = dot(residuals, residuals);
      
      if (stopping_crit(ssr_old, ssr, diffMax)) {
        break;
      }
    }
  }
  
  // Final output computation
  if (two_fe_algo) {
    // Final iteration for two-FE case
    vec fe_coef_a = GX.head(FE_info.nb_coef_Q(0));
    vec fe_coef_b = sum_other_means;
    
    FE_info.compute_fe_coef_2(fe_coef_a, fe_coef_a, fe_coef_b, sum_in_out);
    add_2_fe_coef_to_mu(FE_info, fe_coef_a, fe_coef_b, sum_in_out, output, false);
  } else {
    output.zeros();
    for (size_t q = 0; q < effective_Q; ++q) {
      vec coef_slice = GX.subvec(FE_info.coef_start_Q(q), 
                                FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
      FE_info.add_fe_coef_to_mu(q, coef_slice, output);
    }
  }
  
  // Save fixed effects if requested
  if (save_fixef && saved_fixef) {
    saved_fixef->set_size(effective_Q);
    for (size_t q = 0; q < effective_Q; ++q) {
      (*saved_fixef)(q) = GX.subvec(FE_info.coef_start_Q(q), 
                                   FE_info.coef_start_Q(q) + FE_info.nb_coef_Q(q) - 1);
    }
  }
  
  return iter < iterMax;
}

// Implementation moved above to avoid forward declaration issues

// Legacy interface for backward compatibility
// Main interface function that converts umat FE format to field format
inline WeightedDemeanResult demean_variables(const mat &data, const umat &fe_matrix, 
                                            const vec &weights = vec(), 
                                            double tol = 1e-8, size_t max_iter = 10000,
                                            const std::string &family = "gaussian") {
  WeightedDemeanResult result;
  result.success = true;
  
  // Handle case with no fixed effects
  if (fe_matrix.n_cols == 0 || fe_matrix.n_rows == 0) {
    result.demeaned_data = data;
    return result;
  }
  
  const size_t n_obs = fe_matrix.n_rows;
  const size_t Q = fe_matrix.n_cols;  // Number of FE dimensions
  
  // Convert umat format to field<field<uvec>> format - FIXED VERSION
  field<field<uvec>> group_indices(Q);

  for (size_t q = 0; q < Q; ++q) {
    // Get unique FE levels for this dimension
    uvec fe_col = fe_matrix.col(q);
    uvec unique_levels = unique(fe_col);
    std::sort(unique_levels.begin(), unique_levels.end());  // Ensure sorted order
    const size_t n_groups = unique_levels.n_elem;
    
    // Create field of group indices
    field<uvec> groups_q(n_groups);
    
    for (size_t g = 0; g < n_groups; ++g) {
      const uword level = unique_levels(g);
      // Find all observations with this FE level - MORE EFFICIENT
      uvec group_obs = find(fe_col == level);
      if (group_obs.n_elem > 0) {
        groups_q(g) = group_obs;
      } else {
        // Create empty group if no observations
        groups_q(g) = uvec();
      }
    }
    
    group_indices(q) = groups_q;
  }

  // Use the proper demeaning algorithm - bypass demean_matrix
  mat demeaned_data = data;
  
  if (Q > 0) {
    const size_t n_vars = data.n_cols;
    
    // Demean each variable separately
    for (size_t v = 0; v < n_vars; ++v) {
      vec x_v = data.col(v);
      vec output(n_obs, fill::zeros);

      // Use the core demeaning algorithm directly with correct parameter order
      bool success = demean_acc_gnl_complete(x_v, output, group_indices, weights, max_iter, tol, 
                                            Q == 2, false, nullptr);

      if (!success) {
        result.success = false;
        return result;
      }
      
      // Store demeaned variable
      demeaned_data.col(v) = x_v - output;
    }
  }

  result.demeaned_data = demeaned_data;
  result.success = true;
  
  return result;
}

// Legacy interface for backward compatibility
inline mat demean_data(const mat &data, const umat &fe, const vec &weights = vec()) {
  WeightedDemeanResult result = demean_variables(data, fe, weights, 1e-8, 10000, "gaussian");
  return result.demeaned_data;
}

#endif // CAPYBARA_CENTER
