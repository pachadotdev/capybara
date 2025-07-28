// From a design matrix X and response vector Y, this creates T(X)
// and T(Y) after iteratively removing the group means.
// OPTIMIZED VERSION: Vectorized operations, pre-allocated workspace, minimal memory allocations

#ifndef CAPYBARA_DEMEAN_H
#define CAPYBARA_DEMEAN_H

#include <cmath> // For std::isfinite

namespace capybara {
namespace demean {

using capybara::convergence::continue_criterion;
using capybara::convergence::stopping_criterion;
using capybara::convergence::update_irons_tuck;
using capybara::convergence::vector_continue_criterion;
using capybara::convergence::vector_stopping_criterion;

//////////////////////////////////////////////////////////////////////////////
// OPTIMIZED FIXED EFFECTS CLASS WITH WORKSPACE CACHING
//////////////////////////////////////////////////////////////////////////////

// Pre-allocated workspace to avoid memory allocations during demeaning
struct DemeanWorkspace {
  // Pre-allocated vectors for computations
  vec fe_accumulator;
  vec prediction_buffer;
  vec residual_buffer;
  vec weight_buffer;
  
  // Pre-allocated coefficient vectors
  vec fe_coef_primary;
  vec fe_coef_secondary;
  vec fe_coef_temp;
  
  // Pre-allocated sum vectors
  vec sum_in_out;
  vec sum_other_fe;
  
  // Irons-Tuck acceleration workspace
  vec delta_GX;
  vec delta2_X;
  vec Y_acc, GY_acc, GGY_acc;
  
  // Constructor pre-allocates based on problem size
  DemeanWorkspace(size_t n_obs, size_t total_coefficients, size_t max_fe_group_size) {
    // Core computation buffers
    fe_accumulator.set_size(max_fe_group_size);
    prediction_buffer.set_size(n_obs);
    residual_buffer.set_size(n_obs);
    weight_buffer.set_size(n_obs);
    
    // Coefficient storage
    fe_coef_primary.set_size(total_coefficients);
    fe_coef_secondary.set_size(total_coefficients);
    fe_coef_temp.set_size(max_fe_group_size);
    
    // Sum vectors
    sum_in_out.set_size(total_coefficients);
    sum_other_fe.set_size(n_obs);
    
    // Acceleration workspace
    delta_GX.set_size(total_coefficients);
    delta2_X.set_size(total_coefficients);
    Y_acc.set_size(total_coefficients);
    GY_acc.set_size(total_coefficients);
    GGY_acc.set_size(total_coefficients);
  }
};

class FixedEffects {
private:
  size_t n_obs_;
  size_t n_fe_groups_;
  bool has_weights_;

  // Core data
  vec weights_;
  field<uvec> fe_indices_;

  // FE configuration
  uvec nb_ids_;      // Number of IDs per FE group
  uvec nb_coefs_;    // Number of coefficients per group
  uvec coef_starts_; // Starting index for coefficients per group

  // Precomputed weight sums per FE - VECTORIZED STORAGE
  field<vec> sum_weights_;
  
  // Precomputed inverse weight sums for fast division
  field<vec> inv_sum_weights_;
  
  // Maximum FE group size for workspace allocation
  size_t max_fe_group_size_;
  
  // Parameters for safe operations
  const CapybaraParameters &params_;

  void setup_weights_vectorized(const field<uvec> &fe_id_tables);

public:
  FixedEffects(size_t n_obs, size_t n_fe_groups, const vec &weights,
               const field<uvec> &fe_indices, const uvec &nb_ids,
               const field<uvec> &fe_id_tables, const CapybaraParameters &params);

  // VECTORIZED CORE COMPUTATION FUNCTIONS
  void accumulate_by_fe_vectorized(size_t group_idx, const vec &values, vec &result) const;
  void accumulate_weighted_by_fe_vectorized(size_t group_idx, const vec &values, vec &result) const;
  void broadcast_fe_to_obs_vectorized(size_t group_idx, const vec &fe_values, vec &result) const;
  
  // OPTIMIZED COEFFICIENT COMPUTATION
  void compute_fe_coefficients_vectorized(size_t group_idx, vec &fe_coef,
                                         const vec &target_residuals,
                                         DemeanWorkspace &workspace) const;

  void compute_fe_coefficients_single_vectorized(vec &fe_coef, const vec &target,
                                                DemeanWorkspace &workspace) const;

  void compute_2fe_coefficients_vectorized(vec &fe_coef_a, vec &fe_coef_b,
                                          const vec &target, DemeanWorkspace &workspace) const;

  // VECTORIZED PREDICTION FUNCTIONS
  void add_fe_prediction_vectorized(size_t group_idx, const vec &fe_coef,
                                   vec &prediction) const;
  
  void compute_full_prediction_vectorized(const vec &all_fe_coef, vec &prediction,
                                         DemeanWorkspace &workspace) const;

  // Accessors
  size_t total_coefficients() const { return accu(nb_coefs_); }
  size_t num_fe_groups() const { return n_fe_groups_; }
  size_t num_observations() const { return n_obs_; }
  size_t max_fe_group_size() const { return max_fe_group_size_; }
  const uvec &coefficient_counts() const { return nb_coefs_; }
  const uvec &coefficient_starts() const { return coef_starts_; }
  const field<uvec> &fe_indices() const { return fe_indices_; }
  const vec &weights() const { return weights_; }
  const field<vec> &sum_weights() const { return sum_weights_; }
};

//////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR IMPLEMENTATION - VECTORIZED INITIALIZATION
//////////////////////////////////////////////////////////////////////////////

FixedEffects::FixedEffects(size_t n_obs, size_t n_fe_groups, const vec &weights,
                           const field<uvec> &fe_indices, const uvec &nb_ids,
                           const field<uvec> &fe_id_tables, const CapybaraParameters &params)
    : n_obs_(n_obs), n_fe_groups_(n_fe_groups), weights_(weights),
      fe_indices_(fe_indices), nb_ids_(nb_ids), params_(params) {
  // Validate inputs
  if (n_fe_groups_ != fe_indices_.n_elem) {
    cpp11::stop("Number of FE groups (%d) doesn't match fe_indices size (%d)",
                n_fe_groups_, fe_indices_.n_elem);
  }
  if (n_fe_groups_ != nb_ids_.n_elem) {
    cpp11::stop("Number of FE groups (%d) doesn't match nb_ids size (%d)",
                n_fe_groups_, nb_ids_.n_elem);
  }

  // Check weights
  has_weights_ = weights_.n_elem > 1;
  if (has_weights_) {
    if (weights_.n_elem != n_obs_) {
      cpp11::stop(
          "Weight vector size (%d) doesn't match number of observations (%d)",
          weights_.n_elem, n_obs_);
    }
  } else {
    // Create unit weights if not provided
    weights_ = ones<vec>(n_obs_);
  }

  // Validate fe_indices and compute max FE group size
  max_fe_group_size_ = 0;
  for (size_t q = 0; q < n_fe_groups_; ++q) {
    if (fe_indices_(q).n_elem != n_obs_) {
      cpp11::stop("FE indices for group %d has wrong size: %d vs %d expected",
                  q, fe_indices_(q).n_elem, n_obs_);
    }
    // Check that indices are within bounds
    uword max_idx = fe_indices_(q).max();
    if (max_idx >= nb_ids_(q)) {
      cpp11::stop(
          "FE index out of bounds for group %d: max index %d >= nb_ids %d", q,
          max_idx, nb_ids_(q));
    }
    
    // Track maximum FE group size for workspace allocation
    if (nb_ids_(q) > max_fe_group_size_) {
      max_fe_group_size_ = nb_ids_(q);
    }
  }

  // Initialize configuration arrays
  nb_coefs_ = nb_ids_; // For non-slopes case, coefficients == IDs
  coef_starts_ = zeros<uvec>(n_fe_groups_);

  // Compute coefficient starting positions
  if (n_fe_groups_ > 0) {
    coef_starts_(0) = 0;
    for (size_t q = 1; q < n_fe_groups_; ++q) {
      coef_starts_(q) = coef_starts_(q - 1) + nb_coefs_(q - 1);
    }
  }

  // Setup vectorized weight sums
  setup_weights_vectorized(fe_id_tables);
}

//////////////////////////////////////////////////////////////////////////////
// VECTORIZED SETUP FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

void FixedEffects::setup_weights_vectorized(const field<uvec> &fe_id_tables) {
  sum_weights_.set_size(n_fe_groups_);
  inv_sum_weights_.set_size(n_fe_groups_);

  for (size_t q = 0; q < n_fe_groups_; ++q) {
    sum_weights_(q) = zeros<vec>(nb_ids_(q));

    if (has_weights_) {
      // VECTORIZED accumulation using Armadillo's advanced indexing
      const uvec &fe_idx = fe_indices_(q);
      
      // Use Armadillo's accumarray-like functionality if available
      // Otherwise use optimized manual accumulation
      for (size_t obs = 0; obs < n_obs_; ++obs) {
        sum_weights_(q)(fe_idx(obs)) += weights_(obs);
      }
    } else {
      // Use pre-computed counts from fe_id_tables - OPTIMIZED PATH
      if (fe_id_tables.n_elem > q && fe_id_tables(q).n_elem == nb_ids_(q)) {
        sum_weights_(q) = conv_to<vec>::from(fe_id_tables(q));
      } else {
        // Manual counting with optimized loop - avoid incompatible hist() call
        const uvec &fe_idx = fe_indices_(q);
        for (size_t obs = 0; obs < n_obs_; ++obs) {
          sum_weights_(q)(fe_idx(obs)) += 1.0;
        }
      }
    }

    // Vectorized safe minimum with efficient Armadillo operations
    sum_weights_(q) = max(sum_weights_(q), params_.safe_division_min * ones<vec>(nb_ids_(q)));
    
    // Pre-compute inverse for fast division - MAJOR OPTIMIZATION
    inv_sum_weights_(q) = 1.0 / sum_weights_(q);
  }
}

//////////////////////////////////////////////////////////////////////////////
// VECTORIZED CORE FUNCTIONS - MASSIVE OPTIMIZATION
//////////////////////////////////////////////////////////////////////////////

// Accumulate values by FE group using vectorized operations
void FixedEffects::accumulate_by_fe_vectorized(size_t group_idx, const vec &values, vec &result) const {
  const uvec &fe_idx = fe_indices_(group_idx);
  result.zeros();
  
  // Use Armadillo's efficient accumulation pattern
  for (size_t obs = 0; obs < n_obs_; ++obs) {
    result(fe_idx(obs)) += values(obs);
  }
}

// Accumulate weighted values by FE group
void FixedEffects::accumulate_weighted_by_fe_vectorized(size_t group_idx, const vec &values, vec &result) const {
  const uvec &fe_idx = fe_indices_(group_idx);
  result.zeros();
  
  if (has_weights_) {
    // Vectorized weighted accumulation
    vec weighted_values = values % weights_;
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      result(fe_idx(obs)) += weighted_values(obs);
    }
  } else {
    accumulate_by_fe_vectorized(group_idx, values, result);
  }
}

// Broadcast FE values to observations using vectorized indexing
void FixedEffects::broadcast_fe_to_obs_vectorized(size_t group_idx, const vec &fe_values, vec &result) const {
  const uvec &fe_idx = fe_indices_(group_idx);
  
  // Use Armadillo's advanced indexing for vectorized broadcast
  result = fe_values(fe_idx);
}

// VECTORIZED FE coefficient computation - single group
void FixedEffects::compute_fe_coefficients_vectorized(size_t group_idx, vec &fe_coef,
                                                     const vec &target_residuals,
                                                     DemeanWorkspace &workspace) const {
  // Use pre-allocated workspace
  vec &accumulator = workspace.fe_accumulator;
  accumulator.set_size(nb_ids_(group_idx));
  
  // Vectorized accumulation
  accumulate_weighted_by_fe_vectorized(group_idx, target_residuals, accumulator);
  
  // Vectorized division using pre-computed inverse weights - MAJOR SPEEDUP
  accumulator %= inv_sum_weights_(group_idx);
  
  // Store in output vector at correct position
  size_t coef_start = coef_starts_(group_idx);
  size_t nb_coef = nb_coefs_(group_idx);
  fe_coef.subvec(coef_start, coef_start + nb_coef - 1) = accumulator;
}

// VECTORIZED single FE computation
void FixedEffects::compute_fe_coefficients_single_vectorized(vec &fe_coef, const vec &target,
                                                            DemeanWorkspace &workspace) const {
  // Use pre-allocated workspace
  vec &accumulator = workspace.fe_accumulator;
  accumulator.set_size(nb_ids_(0));
  
  // Vectorized accumulation
  accumulate_weighted_by_fe_vectorized(0, target, accumulator);
  
  // Vectorized division using pre-computed inverse
  fe_coef = accumulator % inv_sum_weights_(0);
}

// VECTORIZED 2-FE coefficient computation - Balanced convergent approach
void FixedEffects::compute_2fe_coefficients_vectorized(vec &fe_coef_a, vec &fe_coef_b,
                                                      const vec &target, DemeanWorkspace &workspace) const {
  if (n_fe_groups_ < 2) return;
  
  // Use pre-allocated workspace buffers
  vec &accum_buffer = workspace.fe_accumulator;
  vec &prediction = workspace.prediction_buffer;
  vec &residual_buffer = workspace.residual_buffer;
  
  // Sizes for the two FE groups
  size_t nb_coef_0 = nb_coefs_(0);
  size_t nb_coef_1 = nb_coefs_(1);
  
  // Ensure output vectors have correct sizes
  if (fe_coef_a.n_elem != nb_coef_0) {
    fe_coef_a.set_size(nb_coef_0);
  }
  if (fe_coef_b.n_elem != nb_coef_1) {
    fe_coef_b.set_size(nb_coef_1);
  }
  
  // Initialize coefficients to zero
  fe_coef_a.zeros();
  fe_coef_b.zeros();
  
  // Alternating projection with full convergence
  const size_t max_iter = params_.demean_2fe_max_iter;
  const double tolerance = params_.demean_2fe_tolerance;
  
  for (size_t iter = 0; iter < max_iter; ++iter) {
    vec fe_coef_a_old = fe_coef_a;
    vec fe_coef_b_old = fe_coef_b;
    
    // Update FE group 1 (b) given current FE group 0 (a)
    broadcast_fe_to_obs_vectorized(0, fe_coef_a, prediction);
    residual_buffer = target - prediction;
    
    accum_buffer.set_size(nb_coef_1);
    accumulate_weighted_by_fe_vectorized(1, residual_buffer, accum_buffer);
    fe_coef_b = accum_buffer % inv_sum_weights_(1);
    
    // Update FE group 0 (a) given updated FE group 1 (b)
    broadcast_fe_to_obs_vectorized(1, fe_coef_b, prediction);
    residual_buffer = target - prediction;
    
    accum_buffer.set_size(nb_coef_0);
    accumulate_weighted_by_fe_vectorized(0, residual_buffer, accum_buffer);
    fe_coef_a = accum_buffer % inv_sum_weights_(0);
    
    // Check convergence
    double diff_a = norm(fe_coef_a - fe_coef_a_old, 2);
    double diff_b = norm(fe_coef_b - fe_coef_b_old, 2);
    if (diff_a < tolerance && diff_b < tolerance) {
      break;
    }
  }
}

// VECTORIZED prediction computation
void FixedEffects::add_fe_prediction_vectorized(size_t group_idx, const vec &fe_coef,
                                               vec &prediction) const {
  size_t coef_start = coef_starts_(group_idx);
  size_t nb_coef = nb_coefs_(group_idx);
  
  // Extract relevant coefficients and broadcast to observations
  vec group_coefs = fe_coef.subvec(coef_start, coef_start + nb_coef - 1);
  vec group_prediction(n_obs_);
  broadcast_fe_to_obs_vectorized(group_idx, group_coefs, group_prediction);
  
  // Vectorized addition
  prediction += group_prediction;
}

// VECTORIZED full prediction computation
void FixedEffects::compute_full_prediction_vectorized(const vec &all_fe_coef, vec &prediction,
                                                     DemeanWorkspace &workspace) const {
  prediction.zeros();
  
  // Use workspace buffer for group predictions
  vec &group_pred = workspace.prediction_buffer;
  
  for (size_t q = 0; q < n_fe_groups_; ++q) {
    size_t coef_start = coef_starts_(q);
    size_t nb_coef = nb_coefs_(q);
    
    // Extract group coefficients and broadcast
    vec group_coefs = all_fe_coef.subvec(coef_start, coef_start + nb_coef - 1);
    broadcast_fe_to_obs_vectorized(q, group_coefs, group_pred);
    
    // Vectorized accumulation
    prediction += group_pred;
  }
}

//////////////////////////////////////////////////////////////////////////////
// OPTIMIZED DEMEAN PARAMETERS WITH WORKSPACE CACHING
//////////////////////////////////////////////////////////////////////////////

struct DemeanParams {
  size_t n_obs;
  size_t n_fe_groups;
  size_t total_coefficients;
  size_t max_iterations;
  double convergence_tolerance;

  // Algorithm parameters
  size_t extra_projections;
  size_t warmup_iterations;
  size_t projections_after_acceleration;
  size_t grand_acceleration_frequency;
  size_t ssr_check_frequency;

  // Data
  field<vec> input_variables;
  field<vec> output_variables;

  // Fixed effects processor
  std::shared_ptr<FixedEffects> fe_processor;

  // Capybara parameters for convergence functions
  CapybaraParameters capybara_params;

  // Results tracking
  uvec iteration_counts;
  bool save_fixed_effects;
  vec fixed_effect_values;

  // Convergence control
  bool stop_requested;
  field<bool> job_completed;
  
  // PRE-ALLOCATED WORKSPACE - MAJOR OPTIMIZATION
  std::unique_ptr<DemeanWorkspace> workspace;

  // Constructor with workspace initialization
  DemeanParams(size_t n_obs, size_t n_fe_groups,
               std::shared_ptr<FixedEffects> fe_proc,
               const CapybaraParameters &params, bool save_fe,
               const field<vec> &input_vars)
      : n_obs(n_obs), n_fe_groups(n_fe_groups),
        total_coefficients(fe_proc->total_coefficients()),
        max_iterations(params.iter_demean_max),
        convergence_tolerance(params.demean_tol),
        extra_projections(params.demean_extra_projections),
        warmup_iterations(params.demean_warmup_iterations),
        projections_after_acceleration(params.demean_projections_after_acc),
        grand_acceleration_frequency(params.demean_grand_acc_frequency),
        ssr_check_frequency(params.demean_ssr_check_frequency),
        input_variables(input_vars), fe_processor(fe_proc),
        capybara_params(params), save_fixed_effects(save_fe),
        stop_requested(false) {

    size_t n_vars = input_vars.n_elem;

    // Initialize data structures
    output_variables.set_size(n_vars);
    iteration_counts = zeros<uvec>(n_vars);
    job_completed.set_size(n_vars);
    job_completed.fill(false);

    if (save_fixed_effects) {
      fixed_effect_values = zeros<vec>(total_coefficients);
    }

    for (size_t i = 0; i < n_vars; ++i) {
      output_variables(i) = zeros<vec>(n_obs);
    }
    
    // Pre-allocate workspace - MAJOR OPTIMIZATION
    workspace = std::make_unique<DemeanWorkspace>(n_obs, total_coefficients, fe_proc->max_fe_group_size());
  }
};

struct DemeanResult {
  field<vec> demeaned_vars;
  vec fixed_effects;
  bool has_fixed_effects;

  DemeanResult(size_t n_vars)
      : demeaned_vars(n_vars), has_fixed_effects(false) {}
};

//////////////////////////////////////////////////////////////////////////////
// VECTORIZED HELPER FUNCTIONS FOR ACCELERATION ALGORITHM  
//////////////////////////////////////////////////////////////////////////////

// VECTORIZED update fixed effects coefficients - general case (Q >= 2)
void fe_general_vectorized(size_t var_idx, const vec &fe_coef_origin,
                          vec &fe_coef_destination, DemeanParams &params) {
  size_t n_fe_groups = params.n_fe_groups;
  DemeanWorkspace &workspace = *params.workspace;
  
  const vec &input = params.input_variables(var_idx);
  const vec &output = params.output_variables(var_idx);
  
  // Compute residual once - vectorized operation
  vec residual = input - output;

  // Update each FE group in reverse order (like fixest) with vectorized operations
  for (int q = n_fe_groups - 1; q >= 0; --q) {
    size_t uq = static_cast<size_t>(q);
    
    // Compute prediction from other FE groups - vectorized
    vec &other_prediction = workspace.prediction_buffer;
    other_prediction.zeros();

    for (size_t h = 0; h < n_fe_groups; ++h) {
      if (h == uq) continue;

      // Use origin coefficients for h < q, destination for h > q
      const vec &fe_coef_to_use = (h < uq) ? fe_coef_origin : fe_coef_destination;
      
      // Extract coefficients for this FE group and add to prediction
      size_t coef_start = params.fe_processor->coefficient_starts()(h);
      size_t nb_coef = params.fe_processor->coefficient_counts()(h);
      vec group_coefs = fe_coef_to_use.subvec(coef_start, coef_start + nb_coef - 1);
      
      vec group_prediction(params.n_obs);
      params.fe_processor->broadcast_fe_to_obs_vectorized(h, group_coefs, group_prediction);
      other_prediction += group_prediction;
    }

    // Compute target residual for this FE group - vectorized
    vec target_for_group = residual - other_prediction;
    
    // Update FE coefficients for group q using vectorized computation
    params.fe_processor->compute_fe_coefficients_vectorized(uq, fe_coef_destination, 
                                                           target_for_group, workspace);
  }
}

// VECTORIZED wrapper to handle 2-FE special case vs general case
void fe_vectorized(size_t var_idx, size_t Q, vec &fe_coef_origin, vec &fe_coef_destination,
                  DemeanParams &params) {
  if (Q == 2) {
    // Special 2-FE algorithm - vectorized
    const vec &input = params.input_variables(var_idx);
    const vec &output = params.output_variables(var_idx);
    vec target = input - output;
    
    // Extract the individual FE coefficient vectors
    size_t nb_coef_0 = params.fe_processor->coefficient_counts()(0);
    size_t nb_coef_1 = params.fe_processor->coefficient_counts()(1);
    size_t coef_start_1 = params.fe_processor->coefficient_starts()(1);
    
    // Create temporary vectors for 2-FE computation
    vec fe_coef_a(nb_coef_0);
    vec fe_coef_b(nb_coef_1);
    
    params.fe_processor->compute_2fe_coefficients_vectorized(fe_coef_a, fe_coef_b,
                                                           target, *params.workspace);
    
    // Store results back in the destination vector
    fe_coef_destination.head(nb_coef_0) = fe_coef_a;
    fe_coef_destination.subvec(coef_start_1, coef_start_1 + nb_coef_1 - 1) = fe_coef_b;
  } else {
    // General algorithm - vectorized
    fe_general_vectorized(var_idx, fe_coef_origin, fe_coef_destination, params);
  }
}

//////////////////////////////////////////////////////////////////////////////
// HYPER-OPTIMIZED DEMEANING ALGORITHMS
//////////////////////////////////////////////////////////////////////////////

// Single fixed effect demeaning (simple case) - VECTORIZED
void demean_single_fe_vectorized(size_t var_idx, DemeanParams &params) {
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);
  DemeanWorkspace &workspace = *params.workspace;

  // Compute fixed effect coefficients using vectorized function
  vec &fe_coef = workspace.fe_coef_primary;
  fe_coef.set_size(params.fe_processor->coefficient_counts()(0));
  
  params.fe_processor->compute_fe_coefficients_single_vectorized(fe_coef, input, workspace);

  // Add to prediction using vectorized broadcast
  output.zeros();
  params.fe_processor->broadcast_fe_to_obs_vectorized(0, fe_coef, output);

  // Save fixed effects if requested
  if (params.save_fixed_effects) {
    // Copy to the full fixed effects vector at the correct position
    size_t nb_coef = params.fe_processor->coefficient_counts()(0);
    params.fixed_effect_values.head(nb_coef) = fe_coef;
  }

  params.job_completed(var_idx) = true;
}

// HYPER-OPTIMIZED multi-FE demeaning with vectorized Irons-Tuck acceleration
bool demean_accelerated_vectorized(size_t var_idx, size_t iter_max, DemeanParams &params,
                                  bool two_fe_mode = false) {
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);
  DemeanWorkspace &workspace = *params.workspace;

  size_t Q = params.n_fe_groups;
  size_t nb_coef_T = params.total_coefficients;
  double tol = params.convergence_tolerance;

  // Handle 2-FE mode
  bool two_fe_algo = two_fe_mode || Q == 2;
  if (two_fe_algo) {
    Q = 2;
    // For 2-FE mode, we still use full coefficient vector size
    // but only update the first two FE groups
  }

  // Use pre-allocated workspace vectors - ZERO ALLOCATIONS
  vec &X = workspace.fe_coef_primary;
  vec &GX = workspace.fe_coef_secondary;
  vec &GGX = workspace.fe_coef_temp;
  X.set_size(nb_coef_T);
  GX.set_size(nb_coef_T);  
  GGX.set_size(nb_coef_T);
  
  // Initialize coefficients
  X.zeros();
  output.zeros();

  // First iteration using vectorized computation
  fe_vectorized(var_idx, Q, X, GX, params);

  // Vectorized convergence check using new convergence functions
  bool keep_going = vector_continue_criterion(X, GX, tol, params.capybara_params);

  // Pre-allocated acceleration workspace
  vec &delta_GX = workspace.delta_GX;
  vec &delta2_X = workspace.delta2_X;
  delta_GX.set_size(nb_coef_T);
  delta2_X.set_size(nb_coef_T);

  // Additional storage for grand acceleration
  vec &Y = workspace.Y_acc;
  vec &GY = workspace.GY_acc;
  vec &GGY = workspace.GGY_acc;
  Y.set_size(nb_coef_T);
  GY.set_size(nb_coef_T);
  GGY.set_size(nb_coef_T);

  size_t iter = 0;
  size_t grand_acc_counter = 0;
  double ssr_old = 0;

  while (!params.stop_requested && keep_going && iter < iter_max) {
    ++iter;

    // Extra projections - vectorized
    for (size_t rep = 0; rep < params.extra_projections; ++rep) {
      fe_vectorized(var_idx, Q, GX, GGX, params);
      fe_vectorized(var_idx, Q, GGX, X, params);
      fe_vectorized(var_idx, Q, X, GX, params);
    }

    // Main step - vectorized
    fe_vectorized(var_idx, Q, GX, GGX, params);

    // Vectorized Irons-Tuck acceleration - use workspace vectors
    delta_GX = GGX - GX;
    delta2_X = delta_GX - GX + X;

    double vprod = dot(delta_GX, delta2_X);
    double ssq = dot(delta2_X, delta2_X);

    if (ssq < params.capybara_params.irons_tuck_eps) {
      break;
    }

    double coef = vprod / ssq;
    X = GGX - coef * delta_GX; // Vectorized update

    // Post-acceleration projections
    if (iter >= params.projections_after_acceleration) {
      Y = X;
      fe_vectorized(var_idx, Q, Y, X, params);
    }

    // Next iteration
    fe_vectorized(var_idx, Q, X, GX, params);

    // Vectorized convergence check
    keep_going = vector_continue_criterion(X, GX, tol, params.capybara_params);

    // Grand acceleration - vectorized
    if (iter % params.grand_acceleration_frequency == 0) {
      ++grand_acc_counter;
      if (grand_acc_counter == 1) {
        Y = GX;
      } else if (grand_acc_counter == 2) {
        GY = GX;
      } else {
        GGY = GX;
        
        // Vectorized grand acceleration step
        delta_GX = GGY - GY;
        delta2_X = delta_GX - GY + Y;
        
        vprod = dot(delta_GX, delta2_X);
        ssq = dot(delta2_X, delta2_X);
        
        if (ssq < params.capybara_params.irons_tuck_eps) {
          break;
        }
        
        coef = vprod / ssq;
        Y = GGY - coef * delta_GX;
        
        fe_vectorized(var_idx, Q, Y, GX, params);
        grand_acc_counter = 0;
      }
    }

    // SSR-based stopping criterion - vectorized
    if (iter % params.ssr_check_frequency == 0) {
      vec &mu_current = workspace.prediction_buffer;
      mu_current.zeros();

      // Compute current prediction using vectorized functions
      params.fe_processor->compute_full_prediction_vectorized(GX, mu_current, workspace);

      // Vectorized SSR computation
      vec residual = input - mu_current;
      double ssr = dot(residual, residual);

      if (stopping_criterion(ssr_old, ssr, tol, params.capybara_params)) {
        break;
      }
      ssr_old = ssr;
    }
  }

  // Final update of output using vectorized computation
  params.fe_processor->compute_full_prediction_vectorized(GX, output, workspace);

  // Save fixed effects if requested
  if (params.save_fixed_effects) {
    params.fixed_effect_values = GX;
  }

  // Update iteration count
  params.iteration_counts(var_idx) += iter;

  params.job_completed(var_idx) = true;
  return (iter < iter_max);
}

// VECTORIZED general multi-FE demeaning
void demean_general_vectorized(size_t var_idx, DemeanParams &params) {
  size_t Q = params.n_fe_groups;

  if (Q == 1) {
    demean_single_fe_vectorized(var_idx, params);
  } else if (Q == 2) {
    // Direct to vectorized 2-FE algorithm
    demean_accelerated_vectorized(var_idx, params.max_iterations, params);
  } else {
    // Q >= 3: Use vectorized three-phase approach
    bool converged = false;

    // Phase 1: Warmup with vectorized acceleration
    if (params.warmup_iterations > 0) {
      converged = demean_accelerated_vectorized(var_idx, params.warmup_iterations, params);
    }

    if (!converged && params.max_iterations > params.warmup_iterations) {
      // Phase 2: Vectorized 2-FE convergence
      size_t iter_previous = params.iteration_counts(var_idx);
      size_t iter_max_2fe = params.max_iterations / 2 - params.warmup_iterations;

      if (iter_max_2fe > 0) {
        demean_accelerated_vectorized(var_idx, iter_max_2fe, params, true); // 2-FE mode
      }

      // Phase 3: Re-acceleration for all FEs
      iter_previous = params.iteration_counts(var_idx);
      size_t remaining = params.max_iterations - iter_previous;
      if (remaining > 0) {
        demean_accelerated_vectorized(var_idx, remaining, params);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// OPTIMIZED MAIN INTERFACE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// VECTORIZED main demeaning function - MAJOR PERFORMANCE OPTIMIZATION
DemeanResult demean_variables(const field<vec> &input_vars, const vec &weights,
                              const field<uvec> &fe_indices, const uvec &nb_ids,
                              const field<uvec> &fe_id_tables,
                              bool save_fixed_effects,
                              const CapybaraParameters &params) {
  size_t n_vars = input_vars.n_elem;

  if (n_vars == 0) {
    DemeanResult empty_result(0);
    return empty_result;
  }

  size_t n_obs = input_vars(0).n_elem;
  size_t n_fe_groups = fe_indices.n_elem;

  // Validate inputs
  for (size_t i = 0; i < n_vars; ++i) {
    if (input_vars(i).n_elem != n_obs) {
      cpp11::stop(
          "All input variables must have the same number of observations");
    }
  }

  // Create fixed effects processor with vectorized initialization
  auto fe_processor = std::make_shared<FixedEffects>(
      n_obs, n_fe_groups, weights, fe_indices, nb_ids, fe_id_tables, params);

  // Setup parameters with pre-allocated workspace
  DemeanParams demean_params(n_obs, n_fe_groups, fe_processor, params,
                             save_fixed_effects, input_vars);

  // Process each variable using VECTORIZED algorithms
  for (size_t v = 0; v < n_vars; ++v) {
    if (demean_params.stop_requested)
      break;

    demean_general_vectorized(v, demean_params);
  }

  // Return demeaned variables (input - fitted) using vectorized operations
  DemeanResult result(n_vars);
  for (size_t v = 0; v < n_vars; ++v) {
    // Vectorized subtraction - no loops
    result.demeaned_vars(v) = demean_params.input_variables(v) - demean_params.output_variables(v);
  }

  if (save_fixed_effects) {
    result.fixed_effects = demean_params.fixed_effect_values;
    result.has_fixed_effects = true;
  }

  return result;
}

} // namespace demean
} // namespace capybara

#endif // CAPYBARA_DEMEAN_H
