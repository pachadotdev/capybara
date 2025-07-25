// From a design matrix X and response vector Y, this creates T(X)
// and T(Y) after iteratively removing the group means.

#ifndef CAPYBARA_DEMEAN_H
#define CAPYBARA_DEMEAN_H

#include <cmath> // For std::isfinite

namespace capybara {
namespace demean {

using convergence::continue_criterion;
using convergence::stopping_criterion;
using convergence::update_irons_tuck;

//////////////////////////////////////////////////////////////////////////////
// WORKSPACE STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// Workspace for demean algorithms to avoid repeated allocations
struct DemeanWorkspace {
  // Core vectors for acceleration algorithms
  vec sum_other_fe_or_tmp;
  vec sum_in_out;
  vec X, GX, GGX;
  vec Y, GY, GGY;
  vec delta_GX, delta2_X;

  // Working vectors for convergence checking
  vec mu_current;

  void resize_for_demean(size_t n_obs, size_t total_coefficients,
                        size_t max_other_size, size_t Q, size_t sum_in_out_size = 0) {
    // Resize only if needed to avoid unnecessary allocations
    if (sum_other_fe_or_tmp.n_elem != max_other_size) {
      sum_other_fe_or_tmp.set_size(max_other_size);
    }
    // sum_in_out should be sized according to the specific algorithm needs
    size_t actual_sum_in_out_size = (sum_in_out_size > 0) ? sum_in_out_size : total_coefficients;
    if (sum_in_out.n_elem != actual_sum_in_out_size) {
      sum_in_out.set_size(actual_sum_in_out_size);
    }
    if (X.n_elem != total_coefficients) {
      X.set_size(total_coefficients);
      GX.set_size(total_coefficients);
      GGX.set_size(total_coefficients);
      Y.set_size(total_coefficients);
      GY.set_size(total_coefficients);
      GGY.set_size(total_coefficients);
      // Pre-allocate to maximum possible size to avoid reallocations in loops
      delta_GX.set_size(total_coefficients);
      delta2_X.set_size(total_coefficients);
    }
    if (mu_current.n_elem != n_obs) {
      mu_current.set_size(n_obs);
    }
  }
};

// Overload for update_irons_tuck that accepts subviews and temporary vectors
bool update_irons_tuck_subvec(vec &X_subvec, vec &GX_subvec, vec &GGX_subvec,
                              vec &delta_GX, vec &delta2_X,
                              const CapybaraParameters &params) {
  delta_GX = GGX_subvec - GX_subvec;
  delta2_X = delta_GX - GX_subvec + X_subvec;

  double vprod = dot(delta_GX, delta2_X);
  double ssq = dot(delta2_X, delta2_X);

  if (ssq < params.irons_tuck_eps) {
    return true; // numerical convergence
  }

  double coef = vprod / ssq;
  X_subvec = GGX_subvec - coef * delta_GX;

  return false;
}

//////////////////////////////////////////////////////////////////////////////
// FIXED EFFECTS CLASS
//////////////////////////////////////////////////////////////////////////////

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

  // Precomputed weight sums per FE
  field<vec> sum_weights_;

  void setup_weights(const field<uvec> &fe_id_tables);

public:
  FixedEffects(size_t n_obs, size_t n_fe_groups, const vec &weights,
               const field<uvec> &fe_indices, const uvec &nb_ids,
               const field<uvec> &fe_id_tables);

  // Core computation functions
  void compute_fe_coefficients(size_t group_idx, vec &fe_coef, const vec &sum_other_fe,
                       const vec &sum_in_out);

  void compute_fe_coefficients_single(vec &fe_coef, const vec &target);

  void compute_fe_coef_2(vec &fe_coef_in, vec &fe_coef_out, vec &fe_coef_tmp,
                 const vec &sum_in_out);

  void add_fe_to_prediction(size_t group_idx, const vec &fe_coef,
                            vec &prediction);

  void add_weighted_fe_to_prediction(size_t group_idx, const vec &fe_coef,
                                     vec &prediction);

  void add_2_fe_to_prediction(const vec &fe_coef_a, const vec &fe_coef_b,
                              vec &prediction, const vec &sum_in_out,
                              bool update_beta);

  void compute_in_out(size_t group_idx, vec &in_out_sum, const vec &input,
              const vec &output);

  // Accessors
  size_t total_coefficients() const { return accu(nb_coefs_); }
  size_t num_fe_groups() const { return n_fe_groups_; }
  size_t num_observations() const { return n_obs_; }
  const uvec &coefficient_counts() const { return nb_coefs_; }
  const uvec &coefficient_starts() const { return coef_starts_; }
  const field<uvec> &fe_indices() const { return fe_indices_; }
  const vec &weights() const { return weights_; }
  const field<vec> &sum_weights() const { return sum_weights_; }
  size_t nb_coef_group(size_t q) const { return nb_coefs_(q); }
};

//////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////

FixedEffects::FixedEffects(size_t n_obs, size_t n_fe_groups, const vec &weights,
                           const field<uvec> &fe_indices, const uvec &nb_ids,
                           const field<uvec> &fe_id_tables)
    : n_obs_(n_obs), n_fe_groups_(n_fe_groups), weights_(weights),
      fe_indices_(fe_indices), nb_ids_(nb_ids) {
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

  // Validate fe_indices
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
  }

  // Initialize configuration arrays
  nb_coefs_ = nb_ids_; // For non-slopes case, coefficients == IDs
  coef_starts_ = uvec(n_fe_groups_, fill::none);
  coef_starts_.zeros();

  // Compute coefficient starting positions
  if (n_fe_groups_ > 0) {
    coef_starts_(0) = 0;
    for (size_t q = 1; q < n_fe_groups_; ++q) {
      coef_starts_(q) = coef_starts_(q - 1) + nb_coefs_(q - 1);
    }
  }

  // Setup weight sums
  setup_weights(fe_id_tables);
}

//////////////////////////////////////////////////////////////////////////////
// PRIVATE SETUP FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

void FixedEffects::setup_weights(const field<uvec> &fe_id_tables) {
  sum_weights_.set_size(n_fe_groups_);

  for (size_t q = 0; q < n_fe_groups_; ++q) {
    sum_weights_(q) = vec(nb_ids_(q), fill::none);
    sum_weights_(q).zeros();

    if (has_weights_) {
      // Accumulate weights for each FE ID
      const uvec &fe_idx = fe_indices_(q);
      for (size_t obs = 0; obs < n_obs_; ++obs) {
        uword idx = fe_idx(obs);
        if (idx >= nb_ids_(q)) {
          cpp11::stop("Index out of bounds in setup_weights: %d >= %d", idx,
                      nb_ids_(q));
        }
        sum_weights_(q)(idx) += weights_(obs);
      }
    } else {
      // Use pre-computed counts from fe_id_tables if available
      if (fe_id_tables.n_elem > q && fe_id_tables(q).n_elem == nb_ids_(q)) {
        sum_weights_(q) = conv_to<vec>::from(fe_id_tables(q));
      } else {
        // Count occurrences manually
        const uvec &fe_idx = fe_indices_(q);
        for (size_t obs = 0; obs < n_obs_; ++obs) {
          uword idx = fe_idx(obs);
          if (idx >= nb_ids_(q)) {
            cpp11::stop("Index out of bounds in setup_weights: %d >= %d", idx,
                        nb_ids_(q));
          }
          sum_weights_(q)(idx) += 1.0;
        }
      }
    }

    // Avoid division by zero - check for any zero weights
    for (size_t i = 0; i < sum_weights_(q).n_elem; ++i) {
      if (sum_weights_(q)(i) < 1e-12) {
        sum_weights_(q)(i) = 1e-12;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// CORE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

void FixedEffects::compute_fe_coefficients(size_t group_idx, vec &fe_coef,
                                   const vec &sum_other_fe,
                                   const vec &sum_in_out) {
  const uvec &fe_idx = fe_indices_(group_idx);
  size_t coef_start = coef_starts_(group_idx);
  size_t nb_coef = nb_coefs_(group_idx);

  // Initialize with sum_in_out for this group
  vec group_coefs = sum_in_out.subvec(coef_start, coef_start + nb_coef - 1);

  // Subtract sum of other FEs
  for (size_t obs = 0; obs < n_obs_; ++obs) {
    group_coefs(fe_idx(obs)) -= sum_other_fe(obs);
  }

  // Divide by weights to get coefficients - use safe_divide like reference
  group_coefs = convergence::utils::safe_divide(group_coefs, sum_weights_(group_idx), 1e-12);

  // Store in output vector
  fe_coef.subvec(coef_start, coef_start + nb_coef - 1) = group_coefs;
}

void FixedEffects::compute_fe_coefficients_single(vec &fe_coef, const vec &target) {
  fe_coef.zeros();

  const uvec &fe_idx = fe_indices_(0);

  if (has_weights_) {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      fe_coef(fe_idx(obs)) += weights_(obs) * target(obs);
    }
  } else {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      fe_coef(fe_idx(obs)) += target(obs);
    }
  }

  // Use safe division like reference
  fe_coef = convergence::utils::safe_divide(fe_coef, sum_weights_(0), 1e-12);
}

void FixedEffects::add_fe_to_prediction(size_t group_idx, const vec &fe_coef,
                                        vec &prediction) {
  const uvec &fe_idx = fe_indices_(group_idx);
  size_t coef_start = coef_starts_(group_idx);

  // Add bounds checking
  if (group_idx >= n_fe_groups_) {
    cpp11::stop("add_fe_to_prediction: group_idx=%zu >= n_fe_groups=%zu", group_idx, n_fe_groups_);
  }

  for (size_t obs = 0; obs < n_obs_; ++obs) {
    size_t access_idx = coef_start + fe_idx(obs);
    
    // Bounds check for fe_coef access
    if (access_idx >= fe_coef.n_elem) {
      cpp11::stop("add_fe_to_prediction bounds error: coef_start=%zu, fe_id=%zu, access_idx=%zu, fe_coef.size=%zu", 
                  coef_start, (size_t)fe_idx(obs), access_idx, fe_coef.n_elem);
    }
    
    prediction(obs) += fe_coef(access_idx);
  }
}

void FixedEffects::add_weighted_fe_to_prediction(size_t group_idx,
                                                 const vec &fe_coef,
                                                 vec &prediction) {
  if (!has_weights_) {
    add_fe_to_prediction(group_idx, fe_coef, prediction);
    return;
  }

  const uvec &fe_idx = fe_indices_(group_idx);
  size_t coef_start = coef_starts_(group_idx);

  for (size_t obs = 0; obs < n_obs_; ++obs) {
    size_t access_idx = coef_start + fe_idx(obs);
    
    // Bounds check for fe_coef access
    if (access_idx >= fe_coef.n_elem) {
      cpp11::stop("add_weighted_fe_to_prediction bounds error: coef_start=%zu, fe_id=%zu, access_idx=%zu, fe_coef.size=%zu", 
                  coef_start, (size_t)fe_idx(obs), access_idx, fe_coef.n_elem);
    }
    
    prediction(obs) += fe_coef(access_idx) * weights_(obs);
  }
}

void FixedEffects::compute_in_out(size_t group_idx, vec &in_out_sum, const vec &input,
                          const vec &output) {
  const uvec &fe_idx = fe_indices_(group_idx);
  size_t coef_start = coef_starts_(group_idx);
  size_t nb_coef = nb_coefs_(group_idx);

  // Validate bounds
  if (coef_start + nb_coef > in_out_sum.n_elem) {
    cpp11::stop("in_out bounds error: coef_start=%zu, nb_coef=%zu, in_out_sum.size=%zu", 
                coef_start, nb_coef, in_out_sum.n_elem);
  }

  // Zero out this group's section
  in_out_sum.subvec(coef_start, coef_start + nb_coef - 1).zeros();

  // Accumulate sum of (input - output) for each FE
  if (has_weights_) {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      size_t fe_id = fe_idx(obs);
      size_t access_idx = coef_start + fe_id;
      
      // Bounds check
      if (access_idx >= in_out_sum.n_elem) {
        cpp11::stop("in_out access bounds error: coef_start=%zu, fe_id=%zu, access_idx=%zu, in_out_sum.size=%zu", 
                    coef_start, fe_id, access_idx, in_out_sum.n_elem);
      }
      
      in_out_sum(access_idx) += (input(obs) - output(obs)) * weights_(obs);
    }
  } else {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      size_t fe_id = fe_idx(obs);
      size_t access_idx = coef_start + fe_id;
      
      // Bounds check
      if (access_idx >= in_out_sum.n_elem) {
        cpp11::stop("in_out access bounds error: coef_start=%zu, fe_id=%zu, access_idx=%zu, in_out_sum.size=%zu", 
                    coef_start, fe_id, access_idx, in_out_sum.n_elem);
      }
      
      in_out_sum(access_idx) += input(obs) - output(obs);
    }
  }
}

// Special 2-FE algorithm implementation
void FixedEffects::compute_fe_coef_2(vec &fe_coef_in, vec &fe_coef_out,
                             vec &fe_coef_tmp, const vec &sum_in_out) {
  // This implements the 2-FE special case from fixest
  // fe_coef_in: coefficients for first FE (input and output)
  // fe_coef_tmp: coefficients for second FE
  // sum_in_out: precomputed sums

  if (n_fe_groups_ < 2) {
    return;
  }

  const size_t nb_coef_0 = nb_coefs_(0);
  const size_t nb_coef_1 = nb_coefs_(1);
  const size_t coef_start_1 = coef_starts_(1);

  const uvec &fe_idx_0 = fe_indices_(0);
  const uvec &fe_idx_1 = fe_indices_(1);

  // Step 1: Update second FE based on first FE
  fe_coef_tmp = sum_in_out.subvec(coef_start_1, coef_start_1 + nb_coef_1 - 1);

  // Subtract contribution from first FE
  if (has_weights_) {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      fe_coef_tmp(fe_idx_1(obs)) -= fe_coef_in(fe_idx_0(obs)) * weights_(obs);
    }
  } else {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      fe_coef_tmp(fe_idx_1(obs)) -= fe_coef_in(fe_idx_0(obs));
    }
  }

  // Divide by weights
  fe_coef_tmp = convergence::utils::safe_divide(fe_coef_tmp, sum_weights_(1), 1e-12);

  // Step 2: Update first FE based on updated second FE
  fe_coef_out = sum_in_out.subvec(0, nb_coef_0 - 1);

  // Subtract contribution from second FE
  if (has_weights_) {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      fe_coef_out(fe_idx_0(obs)) -= fe_coef_tmp(fe_idx_1(obs)) * weights_(obs);
    }
  } else {
    for (size_t obs = 0; obs < n_obs_; ++obs) {
      fe_coef_out(fe_idx_0(obs)) -= fe_coef_tmp(fe_idx_1(obs));
    }
  }

  // Divide by weights
  fe_coef_out = convergence::utils::safe_divide(fe_coef_out, sum_weights_(0), 1e-12);
}

void FixedEffects::add_2_fe_to_prediction(const vec &fe_coef_a,
                                          const vec &fe_coef_b, vec &prediction,
                                          const vec &sum_in_out,
                                          bool update_beta) {
  if (update_beta) {
    // Need to update coefficients - use temporary storage
    vec fe_coef_a_new = fe_coef_a;
    vec fe_coef_b_new = fe_coef_b;
    compute_fe_coef_2(const_cast<vec &>(fe_coef_a), fe_coef_a_new, fe_coef_b_new,
              sum_in_out);

    // Add both FE contributions
    const uvec &fe_idx_0 = fe_indices_(0);
    const uvec &fe_idx_1 = fe_indices_(1);

    for (size_t obs = 0; obs < n_obs_; ++obs) {
      prediction(obs) +=
          fe_coef_a_new(fe_idx_0(obs)) + fe_coef_b_new(fe_idx_1(obs));
    }
  } else {
    // Just add without updating
    const uvec &fe_idx_0 = fe_indices_(0);
    const uvec &fe_idx_1 = fe_indices_(1);

    for (size_t obs = 0; obs < n_obs_; ++obs) {
      prediction(obs) += fe_coef_a(fe_idx_0(obs)) + fe_coef_b(fe_idx_1(obs));
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// DEMEAN PARAMETERS STRUCTURES
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

  // Constructor
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
    iteration_counts = uvec(n_vars, fill::none);
    iteration_counts.zeros();
    job_completed.set_size(n_vars);
    job_completed.fill(false);

    if (save_fixed_effects) {
      fixed_effect_values = vec(total_coefficients, fill::none);
      fixed_effect_values.zeros();
    }

    for (size_t i = 0; i < n_vars; ++i) {
      output_variables(i) = vec(n_obs, fill::none);
      output_variables(i).zeros();
    }
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
// HELPER FUNCTIONS FOR ACCELERATION ALGORITHM
//////////////////////////////////////////////////////////////////////////////

// Update fixed effects coefficients - general case (Q >= 2)
void compute_fe_general(size_t var_idx, const vec &fe_coef_origin,
                vec &fe_coef_destination, vec &sum_other_fe,
                const vec &sum_in_out, DemeanParams &params) {
  size_t n_fe_groups = params.n_fe_groups;

  // Update each FE group in reverse order (like fixest)
  for (int q = n_fe_groups - 1; q >= 0; --q) {
    // Compute sum of other FE coefficients for each observation
    sum_other_fe.zeros();

    for (size_t h = 0; h < n_fe_groups; ++h) {
      if (h == (size_t)q)
        continue;

      // Use origin coefficients for h < q, destination for h > q
      const vec &fe_coef_to_use =
          (h < (size_t)q) ? fe_coef_origin : fe_coef_destination;
      params.fe_processor->add_weighted_fe_to_prediction(h, fe_coef_to_use,
                                                         sum_other_fe);
    }

    // Update FE coefficients for group q
    params.fe_processor->compute_fe_coefficients(q, fe_coef_destination, sum_other_fe,
                                         sum_in_out);
  }
}

// Wrapper to handle 2-FE special case vs general case
void compute_fe(size_t var_idx, size_t Q, vec &fe_coef_origin, vec &fe_coef_destination,
        vec &sum_other_fe_or_tmp, const vec &sum_in_out, DemeanParams &params) {
  if (Q == 2) {
    // Special 2-FE algorithm
    params.fe_processor->compute_fe_coef_2(fe_coef_origin, fe_coef_destination,
                                           sum_other_fe_or_tmp, sum_in_out);
  } else {
    // General algorithm
    compute_fe_general(var_idx, fe_coef_origin, fe_coef_destination,
                       sum_other_fe_or_tmp, sum_in_out, params);
  }
}

//////////////////////////////////////////////////////////////////////////////
// MAIN DEMEANING ALGORITHMS
//////////////////////////////////////////////////////////////////////////////

// Single fixed effect demeaning (simple case)
void demean_single_fe(size_t var_idx, DemeanParams &params) {
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);

  // Initialize workspace - ensure proper zeros for numerical accuracy
  vec fe_coef(params.total_coefficients, fill::zeros);
  params.fe_processor->compute_fe_coefficients_single(fe_coef, input);

  // Add to prediction
  output.zeros();
  params.fe_processor->add_fe_to_prediction(0, fe_coef, output);

  // Save fixed effects if requested
  if (params.save_fixed_effects) {
    params.fixed_effect_values = fe_coef;
  }

  params.job_completed(var_idx) = true;
}

// Multi-FE demeaning with Irons-Tuck acceleration
bool demean_accelerated(size_t var_idx, size_t iter_max, DemeanParams &params,
                        bool two_fe_mode = false) {
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);

  size_t n_obs = params.n_obs;
  size_t Q = params.n_fe_groups;
  size_t nb_coef_T = params.total_coefficients;
  double tol = params.convergence_tolerance;

  // Handle 2-FE mode
  size_t nb_coef_all = nb_coef_T;
  bool two_fe_algo = two_fe_mode || Q == 2;
  if (two_fe_algo) {
    Q = 2;
    nb_coef_T =
        params.fe_processor->nb_coef_group(0); // Only first FE for main vectors
    nb_coef_all = params.fe_processor->nb_coef_group(0) +
                  params.fe_processor->nb_coef_group(1);
  }

  // Working vector - either sum_other_fe (general) or second FE coefs (2-FE)
  size_t size_other = two_fe_algo ? params.fe_processor->nb_coef_group(1) : n_obs;
  vec sum_other_fe_or_tmp(size_other, fill::zeros);

  // Compute sum of (input - output) for each FE group ONCE at the beginning
  vec sum_in_out(nb_coef_all, fill::zeros);
  for (size_t q = 0; q < Q; ++q) {
    params.fe_processor->compute_in_out(q, sum_in_out, input, output);
  }

  // Initialize coefficient vectors with proper zeros
  vec X(nb_coef_T, fill::zeros);
  vec GX(nb_coef_T, fill::zeros);
  vec GGX(nb_coef_T, fill::zeros);

  // First iteration
  compute_fe(var_idx, Q, X, GX, sum_other_fe_or_tmp, sum_in_out, params);

  // Check if we need to iterate
  bool keep_going = false;
  for (size_t i = 0; i < nb_coef_T; ++i) {
    if (continue_criterion(X(i), GX(i), tol, params.capybara_params)) {
      keep_going = true;
      break;
    }
  }

  // Temp vectors for acceleration (exclude last FE group)
  size_t nb_coef_no_Q = 0;
  for (size_t q = 0; q < Q - 1; ++q) {
    nb_coef_no_Q += params.fe_processor->nb_coef_group(q);
  }
  vec delta_GX(nb_coef_no_Q);
  vec delta2_X(nb_coef_no_Q);

  // Additional storage for grand acceleration
  vec Y(nb_coef_T);
  vec GY(nb_coef_T);
  vec GGY(nb_coef_T);

  size_t iter = 0;
  bool numerical_convergence = false;
  size_t grand_acc_counter = 0;
  double ssr_old = 0;

  while (!params.stop_requested && keep_going && iter < iter_max) {
    ++iter;

    // Extra projections
    for (size_t rep = 0; rep < params.extra_projections; ++rep) {
      compute_fe(var_idx, Q, GX, GGX, sum_other_fe_or_tmp, sum_in_out, params);
      compute_fe(var_idx, Q, GGX, X, sum_other_fe_or_tmp, sum_in_out, params);
      compute_fe(var_idx, Q, X, GX, sum_other_fe_or_tmp, sum_in_out, params);
    }

    // Main step
    compute_fe(var_idx, Q, GX, GGX, sum_other_fe_or_tmp, sum_in_out, params);

    // Irons-Tuck acceleration (only if we have coefficients to accelerate)
    if (nb_coef_no_Q > 0) {
      vec X_subvec = X.subvec(0, nb_coef_no_Q - 1);
      vec GX_subvec = GX.subvec(0, nb_coef_no_Q - 1);
      vec GGX_subvec = GGX.subvec(0, nb_coef_no_Q - 1);
      numerical_convergence =
          update_irons_tuck_subvec(X_subvec, GX_subvec, GGX_subvec, delta_GX,
                                   delta2_X, params.capybara_params);
      // Update the original vectors
      X.subvec(0, nb_coef_no_Q - 1) = X_subvec;
      if (numerical_convergence)
        break;
    }

    // Post-acceleration projections
    if (iter >= params.projections_after_acceleration) {
      Y = X;
      compute_fe(var_idx, Q, Y, X, sum_other_fe_or_tmp, sum_in_out, params);
    }

    // Next iteration
    compute_fe(var_idx, Q, X, GX, sum_other_fe_or_tmp, sum_in_out, params);

    // Check convergence
    keep_going = false;
    for (size_t i = 0; i < nb_coef_no_Q; ++i) {
      if (continue_criterion(X(i), GX(i), tol, params.capybara_params)) {
        keep_going = true;
        break;
      }
    }

    // Grand acceleration
    if (iter % params.grand_acceleration_frequency == 0) {
      ++grand_acc_counter;
      if (grand_acc_counter == 1) {
        Y = GX;
      } else if (grand_acc_counter == 2) {
        GY = GX;
      } else {
        GGY = GX;
        if (nb_coef_no_Q > 0) {
          vec Y_subvec = Y.subvec(0, nb_coef_no_Q - 1);
          vec GY_subvec = GY.subvec(0, nb_coef_no_Q - 1);
          vec GGY_subvec = GGY.subvec(0, nb_coef_no_Q - 1);
          numerical_convergence =
              update_irons_tuck_subvec(Y_subvec, GY_subvec, GGY_subvec, delta_GX,
                                       delta2_X, params.capybara_params);
          // Update the original vectors
          Y.subvec(0, nb_coef_no_Q - 1) = Y_subvec;
          if (numerical_convergence)
            break;
        }
        compute_fe(var_idx, Q, Y, GX, sum_other_fe_or_tmp, sum_in_out, params);
        grand_acc_counter = 0;
      }
    }

    // SSR-based stopping criterion
    if (iter % params.ssr_check_frequency == 0) {
      vec mu_current(n_obs, fill::none);
      mu_current.zeros();

      if (two_fe_algo) {
        // Recompute sum_in_out for SSR check
        vec sum_in_out_ssr(nb_coef_all, fill::none);
        sum_in_out_ssr.zeros();
        for (size_t q = 0; q < 2; ++q) {
          params.fe_processor->compute_in_out(q, sum_in_out_ssr, input, output);
        }

        params.fe_processor->add_2_fe_to_prediction(
            GX, sum_other_fe_or_tmp, mu_current, sum_in_out_ssr, false);
      } else {
        for (size_t q = 0; q < Q; ++q) {
          params.fe_processor->add_fe_to_prediction(q, GX, mu_current);
        }
      }

      // Compute SSR
      double ssr = accu(square(input - mu_current));

      if (stopping_criterion(ssr_old, ssr, tol, params.capybara_params)) {
        break;
      }
      ssr_old = ssr;
    }
  }

  // Final update of output
  output.zeros();

  if (two_fe_algo) {
    // Final iteration for 2-FE
    vec sum_in_out_final(nb_coef_all, fill::none);
    sum_in_out_final.zeros();
    for (size_t q = 0; q < 2; ++q) {
      params.fe_processor->compute_in_out(q, sum_in_out_final, input, output);
    }

    params.fe_processor->add_2_fe_to_prediction(GX, sum_other_fe_or_tmp, output,
                                                sum_in_out_final, true);

    // Save fixed effects if requested
    if (params.save_fixed_effects) {
      // First FE
      size_t nb_coef_0 = params.fe_processor->nb_coef_group(0);
      params.fixed_effect_values.subvec(0, nb_coef_0 - 1) = GX;

      // Second FE
      size_t coef_start_1 = params.fe_processor->coefficient_starts()(1);
      size_t nb_coef_1 = sum_other_fe_or_tmp.n_elem;
      params.fixed_effect_values.subvec(
          coef_start_1, coef_start_1 + nb_coef_1 - 1) = sum_other_fe_or_tmp;
    }
  } else {
    for (size_t q = 0; q < Q; ++q) {
      params.fe_processor->add_fe_to_prediction(q, GX, output);
    }

    if (params.save_fixed_effects) {
      vec final_fe_coef(params.total_coefficients, fill::none);
      final_fe_coef.zeros();

      vec sum_other_fe_final(n_obs, fill::none);
      sum_other_fe_final.zeros();

      compute_fe_general(var_idx, GX, final_fe_coef, sum_other_fe_final, sum_in_out,
                 params);

      params.fixed_effect_values = final_fe_coef;
    }
  }

  // Update iteration count
  params.iteration_counts(var_idx) += iter;

  params.job_completed(var_idx) = true;
  return (iter < iter_max);
}

// Optimized multi-FE demeaning with workspace to avoid allocations
bool demean_accelerated_fast(size_t var_idx, size_t iter_max, DemeanParams &params,
                            DemeanWorkspace &workspace, bool two_fe_mode = false) {
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);
  
  size_t n_obs = input.n_elem;
  size_t Q = params.n_fe_groups;
  bool two_fe_algo = two_fe_mode || (Q == 2);
  
  double tol = params.convergence_tolerance;
  bool keep_going = true;
  
  size_t nb_coef_T = params.total_coefficients;
  size_t nb_coef_all = nb_coef_T;
  
  // Handle 2-FE mode
  if (two_fe_algo) {
    Q = 2;
    nb_coef_T = params.fe_processor->nb_coef_group(0); // Only first FE for main vectors
    nb_coef_all = params.fe_processor->nb_coef_group(0) +
                  params.fe_processor->nb_coef_group(1);
  }
  
  // Determine workspace sizing
  size_t size_other = two_fe_algo ? params.fe_processor->nb_coef_group(1) : n_obs;
  // size_t max_workspace_size = std::max({nb_coef_T, nb_coef_all, size_other, n_obs});
  
  // Resize workspace efficiently - pass nb_coef_all for sum_in_out size
  workspace.resize_for_demean(n_obs, nb_coef_T, size_other, Q, nb_coef_all);
  
  // Get references to workspace vectors - ZERO allocations
  vec &sum_other_fe_or_tmp = workspace.sum_other_fe_or_tmp;
  vec &sum_in_out = workspace.sum_in_out;
  vec &X = workspace.X;
  vec &GX = workspace.GX;
  vec &GGX = workspace.GGX;
  vec &Y = workspace.Y;
  vec &GY = workspace.GY;
  vec &GGY = workspace.GGY;
  vec &delta_GX = workspace.delta_GX;
  vec &delta2_X = workspace.delta2_X;
  vec &mu_current = workspace.mu_current;
  
  // Resize workspace vectors to the actual sizes needed for this algorithm
  if (X.n_elem != nb_coef_T) {
    X.set_size(nb_coef_T);
    GX.set_size(nb_coef_T);
    GGX.set_size(nb_coef_T);
    Y.set_size(nb_coef_T);
    GY.set_size(nb_coef_T);
    GGY.set_size(nb_coef_T);
  }
  if (sum_in_out.n_elem != nb_coef_all) {
    sum_in_out.set_size(nb_coef_all);
  }
  
  // Initialize vectors efficiently
  sum_other_fe_or_tmp.zeros();
  sum_in_out.zeros();
  X.zeros();
  GX.zeros();
  
  // Compute sum of (input - output) for each FE group ONCE
  for (size_t q = 0; q < Q; ++q) {
    params.fe_processor->compute_in_out(q, sum_in_out, input, output);
  }
  
  size_t iter = 0;
  bool numerical_convergence = false;
  size_t grand_acc_counter = 0;
  double ssr_old = 0;
  
  while (!params.stop_requested && keep_going && iter < iter_max) {
    ++iter;
    
    // Extra projections
    for (size_t rep = 0; rep < params.extra_projections; ++rep) {
      compute_fe(var_idx, Q, GX, GGX, sum_other_fe_or_tmp, sum_in_out, params);
      compute_fe(var_idx, Q, GGX, X, sum_other_fe_or_tmp, sum_in_out, params);
      compute_fe(var_idx, Q, X, GX, sum_other_fe_or_tmp, sum_in_out, params);
    }
    
    // Main projection iteration
    compute_fe(var_idx, Q, GX, GGX, sum_other_fe_or_tmp, sum_in_out, params);
    
    GGX = GX;
    
    // Acceleration step - use pre-allocated workspace vectors
    if (iter > params.warmup_iterations) {
      // Calculate nb_coef_no_Q the same way as the original version
      size_t nb_coef_no_Q = 0;
      for (size_t q = 0; q < Q - 1; ++q) {
        nb_coef_no_Q += params.fe_processor->nb_coef_group(q);
      }
      
      // Only apply acceleration if we have coefficients to accelerate
      if (nb_coef_no_Q > 0) {
        // Use subviews for efficiency - no allocation needed since workspace is pre-sized
        vec X_subvec = X.subvec(0, nb_coef_no_Q - 1);
        vec GX_subvec = GX.subvec(0, nb_coef_no_Q - 1);
        vec GGX_subvec = GGX.subvec(0, nb_coef_no_Q - 1);
        
        // Use pre-allocated workspace vectors - just create subviews, no allocation
        vec delta_GX_sub = delta_GX.subvec(0, nb_coef_no_Q - 1);
        vec delta2_X_sub = delta2_X.subvec(0, nb_coef_no_Q - 1);
        
        numerical_convergence = update_irons_tuck_subvec(
            X_subvec, GX_subvec, GGX_subvec, delta_GX_sub, delta2_X_sub, params.capybara_params);
        
        if (numerical_convergence) {
          break;
        }
      }
    }
    
    // Grand acceleration
    ++grand_acc_counter;
    if (grand_acc_counter >= params.grand_acceleration_frequency) {
      grand_acc_counter = 0;
      
      Y = X;
      GY = GX;
      
      compute_fe(var_idx, Q, Y, GY, sum_other_fe_or_tmp, sum_in_out, params);
      
      GGY = GY;
      
      // Apply acceleration to Y
      size_t nb_coef_no_Q = nb_coef_T;
      if (Q > 1) {
        nb_coef_no_Q = nb_coef_T - params.fe_processor->nb_coef_group(Q - 1);
      }
      
      vec Y_subvec = Y.subvec(0, nb_coef_no_Q - 1);
      vec GY_subvec = GY.subvec(0, nb_coef_no_Q - 1);
      vec GGY_subvec = GGY.subvec(0, nb_coef_no_Q - 1);
      
      // Use pre-allocated workspace vectors - just create subviews
      vec delta_GX_sub = delta_GX.subvec(0, nb_coef_no_Q - 1);
      vec delta2_X_sub = delta2_X.subvec(0, nb_coef_no_Q - 1);
      
      update_irons_tuck_subvec(Y_subvec, GY_subvec, GGY_subvec, delta_GX_sub, delta2_X_sub, 
                              params.capybara_params);
      
      X = Y;
      GX = GY;
    }
    
    X = GX;
    
    // Next iteration
    compute_fe(var_idx, Q, X, GX, sum_other_fe_or_tmp, sum_in_out, params);
    
    // Check convergence with early termination optimization
    keep_going = false;
    size_t nb_coef_no_Q = 0;
    for (size_t q = 0; q < Q - 1; ++q) {
      nb_coef_no_Q += params.fe_processor->nb_coef_group(q);
    }
    
    // Only check convergence on coefficients that matter
    for (size_t i = 0; i < nb_coef_no_Q; ++i) {
      if (continue_criterion(X(i), GX(i), tol, params.capybara_params)) {
        keep_going = true;
        break; // Early exit for efficiency
      }
    }
    
    // SSR check for convergence
    if (iter % params.ssr_check_frequency == 0) {
      // Compute current prediction efficiently using workspace vector
      mu_current.zeros();
      if (two_fe_algo) {
        // For 2-FE, use direct coefficient access for better performance
        const uvec &fe_idx_0 = params.fe_processor->fe_indices()(0);
        const uvec &fe_idx_1 = params.fe_processor->fe_indices()(1);
        
        // Vectorized computation
        for (size_t obs = 0; obs < n_obs; ++obs) {
          mu_current(obs) = X(fe_idx_0(obs)) + sum_other_fe_or_tmp(fe_idx_1(obs));
        }
      } else {
        for (size_t q = 0; q < Q; ++q) {
          params.fe_processor->add_fe_to_prediction(q, X, mu_current);
        }
      }
      
      // Efficient SSR computation using Armadillo's optimized functions
      double ssr = accu(square(input - mu_current));
      
      if (stopping_criterion(ssr_old, ssr, tol, params.capybara_params)) {
        break;
      }
      ssr_old = ssr;
    }
  }
  
  // Final update of output - use efficient computation
  output.zeros();
  if (two_fe_algo) {
    // For 2-FE, final computation without redundant sum_in_out recalculation
    const uvec &fe_idx_0 = params.fe_processor->fe_indices()(0);
    const uvec &fe_idx_1 = params.fe_processor->fe_indices()(1);
    
    // Direct vectorized computation for best performance
    for (size_t obs = 0; obs < n_obs; ++obs) {
      output(obs) = X(fe_idx_0(obs)) + sum_other_fe_or_tmp(fe_idx_1(obs));
    }
  } else {
    // General case - use the existing prediction infrastructure
    for (size_t q = 0; q < Q; ++q) {
      params.fe_processor->add_fe_to_prediction(q, X, output);
    }
  }
  
  // Save fixed effects if requested
  if (params.save_fixed_effects) {
    if (two_fe_algo) {
      // For 2-FE, combine coefficients from both groups
      size_t nb_coef_0 = params.fe_processor->nb_coef_group(0);
      size_t nb_coef_1 = params.fe_processor->nb_coef_group(1);
      size_t coef_start_1 = params.fe_processor->coefficient_starts()(1);
      
      params.fixed_effect_values.set_size(nb_coef_0 + nb_coef_1);
      params.fixed_effect_values.subvec(0, nb_coef_0 - 1) = X;
      params.fixed_effect_values.subvec(coef_start_1, coef_start_1 + nb_coef_1 - 1) = sum_other_fe_or_tmp;
    } else {
      params.fixed_effect_values = X;
    }
  }
  
  params.iteration_counts(var_idx) += iter;
  params.job_completed(var_idx) = true;
  
  return numerical_convergence;
}
//////////////////////////////////////////////////////////////////////////////
// MAIN INTERFACE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Main demeaning function
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

  // Create fixed effects processor
  auto fe_processor = std::make_shared<FixedEffects>(
      n_obs, n_fe_groups, weights, fe_indices, nb_ids, fe_id_tables);

  // Setup parameters using constructor
  DemeanParams demean_params(n_obs, n_fe_groups, fe_processor, params,
                             save_fixed_effects, input_vars);

  // Process each variable
  for (size_t v = 0; v < n_vars; ++v) {
    if (demean_params.stop_requested)
      break;

    demean_accelerated(v, demean_params.max_iterations, demean_params);
  }

  // Return demeaned variables (input - fitted)
  DemeanResult result(n_vars);
  for (size_t v = 0; v < n_vars; ++v) {
    result.demeaned_vars(v) =
        demean_params.input_variables(v) - demean_params.output_variables(v);
  }

  if (save_fixed_effects) {
    result.fixed_effects = demean_params.fixed_effect_values;
    result.has_fixed_effects = true;
  }

  return result;
}

// Optimized demean function with workspace reuse
DemeanResult demean_variables_fast(const field<vec> &input_vars, const vec &weights,
                                  const field<uvec> &fe_indices, const uvec &nb_ids,
                                  const field<uvec> &fe_id_tables,
                                  bool save_fixed_effects,
                                  const CapybaraParameters &params,
                                  DemeanWorkspace &workspace) {
  size_t n_vars = input_vars.n_elem;
  if (n_vars == 0) {
    return DemeanResult(0);
  }

  size_t n_obs = input_vars(0).n_elem;
  size_t n_fe_groups = fe_indices.n_elem;

  // Create fixed effects processor once
  auto fe_processor = std::make_shared<FixedEffects>(
      n_obs, n_fe_groups, weights, fe_indices, nb_ids, fe_id_tables);

  // Setup parameters using constructor
  DemeanParams demean_params(n_obs, n_fe_groups, fe_processor, params,
                             save_fixed_effects, input_vars);

  // Pre-size workspace to avoid repeated allocations
  size_t total_coef = fe_processor->total_coefficients();
  size_t max_other_size = (n_fe_groups == 2) ? fe_processor->nb_coef_group(1) : n_obs;
  workspace.resize_for_demean(n_obs, total_coef, max_other_size, n_fe_groups, total_coef);

  // Process each variable with optimized algorithm
  for (size_t v = 0; v < n_vars; ++v) {
    if (n_fe_groups == 1) {
      demean_single_fe(v, demean_params);
    } else {
      demean_accelerated_fast(v, params.iter_demean_max, demean_params, workspace);
    }
  }

  // Return demeaned variables (input - fitted)
  DemeanResult result(n_vars);
  for (size_t v = 0; v < n_vars; ++v) {
    result.demeaned_vars(v) = demean_params.input_variables(v) - demean_params.output_variables(v);
  }

  if (save_fixed_effects) {
    result.fixed_effects = demean_params.fixed_effect_values;
    result.has_fixed_effects = true;
  }

  return result;
}

// Fast wrapper without workspace parameter - uses thread-local workspace
DemeanResult demean_variables_fast(const field<vec> &input_vars, const vec &weights,
                                  const field<uvec> &fe_indices, const uvec &nb_ids,
                                  const field<uvec> &fe_id_tables,
                                  bool save_fixed_effects,
                                  const CapybaraParameters &params) {
  // Use thread-local workspace for better performance in multi-threaded contexts
  static thread_local DemeanWorkspace workspace;
  return demean_variables_fast(input_vars, weights, fe_indices, nb_ids, fe_id_tables,
                              save_fixed_effects, params, workspace);
}

} // namespace demean
} // namespace capybara

#endif // CAPYBARA_DEMEAN_H
