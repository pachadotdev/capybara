// From a design matrix X and response vector Y, this creates T(X)
// and T(Y) after iteratively removing the group means.

#ifndef CAPYBARA_DEMEAN_H
#define CAPYBARA_DEMEAN_H

namespace capybara {
namespace demean {

using capybara::convergence::continue_criterion;
using capybara::convergence::stopping_criterion;
using capybara::convergence::vector_continue_criterion;
using capybara::convergence::vector_stopping_criterion;

struct DemeanMemoryPool {
  vec work_vec1, work_vec2, work_vec3;
  vec accum_vec1, accum_vec2;
  uvec index_vec1, index_vec2;

  DemeanMemoryPool(size_t max_size) {
    work_vec1.set_size(max_size);
    work_vec2.set_size(max_size);
    work_vec3.set_size(max_size);
    accum_vec1.set_size(max_size);
    accum_vec2.set_size(max_size);
    index_vec1.set_size(max_size);
    index_vec2.set_size(max_size);
  }
};

struct DemeanWorkspace {
  vec fe_accumulator;
  vec prediction_buffer;
  vec residual_buffer;
  vec weight_buffer;

  vec fe_coef_primary;
  vec fe_coef_secondary;
  vec fe_coef_temp;

  vec sum_in_out;
  vec sum_other_fe;

  vec delta_GX, delta2_X;
  vec Y_acc, GY_acc, GGY_acc;

  DemeanMemoryPool pool;

  field<uvec> sorted_indices;
  field<uvec> inverse_indices;

  DemeanWorkspace(size_t n_obs, size_t total_coefficients,
                  size_t max_fe_group_size)
      : pool(std::max(n_obs, total_coefficients)) {
    fe_accumulator.set_size(max_fe_group_size);
    prediction_buffer.set_size(n_obs);
    residual_buffer.set_size(n_obs);
    weight_buffer.set_size(n_obs);

    fe_coef_primary.set_size(total_coefficients);
    fe_coef_secondary.set_size(total_coefficients);
    fe_coef_temp.set_size(total_coefficients);

    sum_in_out.set_size(total_coefficients);
    sum_other_fe.set_size(n_obs);

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

  vec weights_;
  field<uvec> fe_indices_;

  uvec nb_ids_;
  uvec nb_coefs_;
  uvec coef_starts_;

  field<vec> sum_weights_;
  field<vec> inv_sum_weights_;

  field<uvec> sorted_obs_by_fe_;
  field<uvec> fe_group_starts_;
  field<uvec> fe_group_sizes_;

  size_t max_fe_group_size_;
  const CapybaraParameters &params_;

  void setup(const field<uvec> &fe_id_tables);

public:
  FixedEffects(size_t n_obs, size_t n_fe_groups, const vec &weights,
               const field<uvec> &fe_indices, const uvec &nb_ids,
               const field<uvec> &fe_id_tables,
               const CapybaraParameters &params);

  // TODO: check these
  void accumulate(size_t group_idx, const vec &values, vec &result,
                  DemeanWorkspace &ws, bool use_weights = true) const;
  void broadcast(size_t group_idx, const vec &fe_values, vec &result) const;

  void compute_fe_coef_single(vec &fe_coef, const vec &target,
                              DemeanWorkspace &ws) const;
  void compute_fe_coef_two(vec &fe_coef_a, vec &fe_coef_b, const vec &target,
                           DemeanWorkspace &ws) const;
  void compute_fe_coef_general(size_t group_idx, vec &fe_coef,
                               const vec &target_residuals,
                               DemeanWorkspace &ws) const;

  void add_fe_prediction(size_t group_idx, const vec &fe_coef,
                         vec &prediction) const;
  void compute_full_prediction(const vec &all_fe_coef, vec &prediction,
                               DemeanWorkspace &ws) const;

  size_t total_coefficients() const { return accu(nb_coefs_); }
  size_t num_fe_groups() const { return n_fe_groups_; }
  size_t num_observations() const { return n_obs_; }
  size_t max_fe_group_size() const { return max_fe_group_size_; }
  const uvec &coefficient_counts() const { return nb_coefs_; }
  const uvec &coefficient_starts() const { return coef_starts_; }
  const field<uvec> &fe_indices() const { return fe_indices_; }
  const vec &weights() const { return weights_; }
};

FixedEffects::FixedEffects(size_t n_obs, size_t n_fe_groups, const vec &weights,
                           const field<uvec> &fe_indices, const uvec &nb_ids,
                           const field<uvec> &fe_id_tables,
                           const CapybaraParameters &params)
    : n_obs_(n_obs), n_fe_groups_(n_fe_groups), weights_(weights),
      fe_indices_(fe_indices), nb_ids_(nb_ids), params_(params) {
  CAPYBARA_TIME_FUNCTION("FixedEffects::FixedEffects");

  has_weights_ = weights_.n_elem > 1 &&
                 !approx_equal(weights_, ones<vec>(n_obs_), "absdiff", 1e-14);
  if (!has_weights_) {
    weights_ = ones<vec>(n_obs_);
  }

  nb_coefs_ = nb_ids_;
  coef_starts_.set_size(n_fe_groups_);

  if (n_fe_groups_ > 0) {
    coef_starts_(0) = 0;
    for (size_t q = 1; q < n_fe_groups_; ++q) {
      coef_starts_(q) = coef_starts_(q - 1) + nb_coefs_(q - 1);
    }
  }

  max_fe_group_size_ = nb_ids_.max();

  setup(fe_id_tables);
}

void FixedEffects::setup(const field<uvec> &fe_id_tables) {
  CAPYBARA_TIME_FUNCTION("FixedEffects::setup");
  
  sum_weights_.set_size(n_fe_groups_);
  inv_sum_weights_.set_size(n_fe_groups_);
  sorted_obs_by_fe_.set_size(n_fe_groups_);
  fe_group_starts_.set_size(n_fe_groups_);
  fe_group_sizes_.set_size(n_fe_groups_);

  for (size_t q = 0; q < n_fe_groups_; ++q) {
    size_t nb_groups = nb_ids_(q);

    sum_weights_(q).set_size(nb_groups);
    sum_weights_(q).zeros();
    uvec sort_idx = sort_index(fe_indices_(q));
    sorted_obs_by_fe_(q) = sort_idx;

    fe_group_starts_(q).set_size(nb_groups + 1);
    fe_group_starts_(q).zeros();
    fe_group_sizes_(q).set_size(nb_groups);
    fe_group_sizes_(q).zeros();

    const uvec &fe_idx = fe_indices_(q);
    for (size_t i = 0; i < n_obs_; ++i) {
      uword group = fe_idx(i);
      fe_group_sizes_(q)(group)++;
      sum_weights_(q)(group) += weights_(i);
    }

    for (size_t g = 1; g <= nb_groups; ++g) {
      fe_group_starts_(q)(g) =
          fe_group_starts_(q)(g - 1) + fe_group_sizes_(q)(g - 1);
    }

    // TODO: do we really need this?
    // Element-wise max to avoid division by zero
    // for (size_t g = 0; g < nb_groups; ++g) {
    //   if (sum_weights_(q)(g) < params_.safe_division_min) {
    //     sum_weights_(q)(g) = params_.safe_division_min;
    //   }
    // }

    inv_sum_weights_(q).set_size(nb_groups);
    inv_sum_weights_(q) = 1.0 / sum_weights_(q);
  }
}

void FixedEffects::accumulate(size_t group_idx, const vec &values, vec &result,
                              DemeanWorkspace &ws, bool use_weights) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::accumulate");
  
  result.zeros();

  const uvec &sorted_idx = sorted_obs_by_fe_(group_idx);
  const uvec &group_starts = fe_group_starts_(group_idx);
  const uvec &group_sizes = fe_group_sizes_(group_idx);

  for (size_t g = 0; g < nb_ids_(group_idx); ++g) {
    if (group_sizes(g) == 0)
      continue;

    uword start = group_starts(g);
    uword end = group_starts(g + 1);

    if (use_weights && has_weights_) {
      // More efficient: use Armadillo's advanced indexing without temporary
      // vectors
      result(g) = accu(values.elem(sorted_idx.subvec(start, end - 1)) %
                       weights_.elem(sorted_idx.subvec(start, end - 1)));
    } else {
      // More efficient: use Armadillo's advanced indexing without temporary
      // vectors
      result(g) = accu(values.elem(sorted_idx.subvec(start, end - 1)));
    }
  }
}

void FixedEffects::broadcast(size_t group_idx, const vec &fe_values,
                             vec &result) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::broadcast");
  
  const uvec &fe_idx = fe_indices_(group_idx);

  result = fe_values(fe_idx);
}

void FixedEffects::compute_fe_coef_single(vec &fe_coef, const vec &target,
                                          DemeanWorkspace &ws) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::compute_fe_coef_single");
  
  vec accumulator(nb_ids_(0));

  accumulate(0, target, accumulator, ws, true);

  if (inv_sum_weights_(0).n_elem == accumulator.n_elem) {
    fe_coef = accumulator % inv_sum_weights_(0);
  } else if (inv_sum_weights_(0).n_elem == 1) {
    fe_coef = accumulator * inv_sum_weights_(0)(0);
  } else {
    if (accumulator.n_elem > inv_sum_weights_(0).n_elem) {
      throw std::runtime_error("Accumulator size mismatch in demean");
    }
    vec inv_weights = inv_sum_weights_(0).subvec(0, accumulator.n_elem - 1);
    fe_coef = accumulator % inv_weights;
  }
}

// Gauss-Seidel iteration
void FixedEffects::compute_fe_coef_two(vec &fe_coef_a, vec &fe_coef_b,
                                       const vec &target,
                                       DemeanWorkspace &ws) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::compute_fe_coef_two");
  
  fe_coef_a.set_size(nb_coefs_(0));
  fe_coef_a.zeros();
  fe_coef_b.set_size(nb_coefs_(1));
  fe_coef_b.zeros();

  vec &pred = ws.prediction_buffer;
  vec &resid = ws.residual_buffer;

  const size_t max_iter = params_.demean_2fe_max_iter;
  const double tol = params_.demean_2fe_tolerance;

  for (size_t iter = 0; iter < max_iter; ++iter) {
    vec fe_a_old = fe_coef_a;
    vec fe_b_old = fe_coef_b;

    // Update FE group 1 given FE group 0
    broadcast(0, fe_coef_a, pred);
    resid = target - pred;

    vec accum_1(nb_ids_(1));
    accum_1.zeros();
    accumulate(1, resid, accum_1, ws, true);

    fe_coef_b = accum_1 % inv_sum_weights_(1);

    // Update FE group 0 given FE group 1
    broadcast(1, fe_coef_b, pred);
    resid = target - pred;

    vec accum_0(nb_ids_(0));
    accum_0.zeros();
    accumulate(0, resid, accum_0, ws, true);

    fe_coef_a = accum_0 % inv_sum_weights_(0);

    if (norm(fe_coef_a - fe_a_old, 2) < tol &&
        norm(fe_coef_b - fe_b_old, 2) < tol) {
      break;
    }
  }
}

void FixedEffects::compute_fe_coef_general(size_t group_idx, vec &fe_coef,
                                           const vec &target_residuals,
                                           DemeanWorkspace &ws) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::compute_fe_coef_general");
  
  vec accum(nb_ids_(group_idx));

  accumulate(group_idx, target_residuals, accum, ws, true);

  size_t coef_start = coef_starts_(group_idx);
  size_t nb_coef = nb_coefs_(group_idx);
  fe_coef.subvec(coef_start, coef_start + nb_coef - 1) =
      accum % inv_sum_weights_(group_idx);
}

// TODO: Check if this can be optimized further
void FixedEffects::add_fe_prediction(size_t group_idx, const vec &fe_coef,
                                     vec &prediction) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::add_fe_prediction");
  
  size_t coef_start = coef_starts_(group_idx);
  size_t nb_coef = nb_coefs_(group_idx);

  vec group_pred(n_obs_);
  broadcast(group_idx, fe_coef.subvec(coef_start, coef_start + nb_coef - 1),
            group_pred);

  prediction += group_pred;
}

void FixedEffects::compute_full_prediction(const vec &all_fe_coef,
                                           vec &prediction,
                                           DemeanWorkspace &ws) const {
  CAPYBARA_TIME_FUNCTION("FixedEffects::compute_full_prediction");
  
  prediction.zeros();

  vec group_pred(n_obs_);

  for (size_t q = 0; q < n_fe_groups_; ++q) {
    size_t coef_start = coef_starts_(q);
    size_t nb_coef = nb_coefs_(q);

    broadcast(q, all_fe_coef.subvec(coef_start, coef_start + nb_coef - 1),
              group_pred);
    prediction += group_pred;
  }
}

struct DemeanParams {
  size_t n_obs;
  size_t n_fe_groups;
  size_t total_coefficients;
  size_t max_iterations;
  double convergence_tolerance;

  size_t extra_projections;
  size_t warmup_iterations;
  size_t projections_after_acceleration;
  size_t grand_acceleration_frequency;
  size_t ssr_check_frequency;

  field<vec> input_variables;
  field<vec> output_variables;

  std::shared_ptr<FixedEffects> fe_processor;

  CapybaraParameters capybara_params;

  uvec iteration_counts;
  bool save_fixed_effects;
  vec fixed_effect_values;

  bool stop_requested;
  field<bool> job_completed;

  std::unique_ptr<DemeanWorkspace> workspace;

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
    CAPYBARA_TIME_FUNCTION("DemeanParams::DemeanParams");

    size_t n_vars = input_vars.n_elem;

    output_variables.set_size(n_vars);
    iteration_counts.set_size(n_vars);
    iteration_counts.zeros();
    job_completed.set_size(n_vars);
    job_completed.fill(false);

    if (save_fixed_effects) {
      fixed_effect_values.set_size(total_coefficients);
      fixed_effect_values.zeros();
    }

    for (size_t i = 0; i < n_vars; ++i) {
      output_variables(i).set_size(n_obs);
      output_variables(i).zeros();
    }

    workspace = std::make_unique<DemeanWorkspace>(n_obs, total_coefficients,
                                                  fe_proc->max_fe_group_size());
  }
};

void fe_general(size_t var_idx, const vec &fe_coef_origin,
                vec &fe_coef_destination, DemeanParams &params) {
  CAPYBARA_TIME_FUNCTION("fe_general");
  
  size_t n_fe_groups = params.n_fe_groups;
  DemeanWorkspace &ws = *params.workspace;

  const vec &input = params.input_variables(var_idx);
  const vec &output = params.output_variables(var_idx);

  vec &residual = ws.residual_buffer;
  residual = input - output;

  for (int q = n_fe_groups - 1; q >= 0; --q) {
    size_t uq = static_cast<size_t>(q);

    vec &other_pred = ws.prediction_buffer;
    other_pred.zeros();

    for (size_t h = 0; h < n_fe_groups; ++h) {
      if (h == uq)
        continue;

      const vec &fe_coef = (h < uq) ? fe_coef_origin : fe_coef_destination;
      params.fe_processor->add_fe_prediction(h, fe_coef, other_pred);
    }

    vec target = residual - other_pred;

    params.fe_processor->compute_fe_coef_general(uq, fe_coef_destination,
                                                 target, ws);
  }
}

void fe(size_t var_idx, size_t Q, vec &fe_coef_origin, vec &fe_coef_destination,
        DemeanParams &params) {
  CAPYBARA_TIME_FUNCTION("fe");
  
  DemeanWorkspace &ws = *params.workspace;
  const vec &input = params.input_variables(var_idx);
  const vec &output = params.output_variables(var_idx);

  vec target = input - output;

  if (Q == 2) {
    size_t nb_coef_0 = params.fe_processor->coefficient_counts()(0);
    size_t nb_coef_1 = params.fe_processor->coefficient_counts()(1);
    size_t coef_start_1 = params.fe_processor->coefficient_starts()(1);

    vec fe_coef_a(nb_coef_0);
    vec fe_coef_b(nb_coef_1);

    params.fe_processor->compute_fe_coef_two(fe_coef_a, fe_coef_b, target, ws);

    fe_coef_destination.subvec(0, nb_coef_0 - 1) = fe_coef_a;
    fe_coef_destination.subvec(coef_start_1, coef_start_1 + nb_coef_1 - 1) =
        fe_coef_b;
  } else {
    fe_general(var_idx, fe_coef_origin, fe_coef_destination, params);
  }
}

void demean_single_fe(size_t var_idx, DemeanParams &params) {
  CAPYBARA_TIME_FUNCTION("demean_single_fe");
  
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);
  DemeanWorkspace &ws = *params.workspace;

  vec fe_coef(params.fe_processor->coefficient_counts()(0));
  params.fe_processor->compute_fe_coef_single(fe_coef, input, ws);

  params.fe_processor->broadcast(0, fe_coef, output);

  if (params.save_fixed_effects) {
    params.fixed_effect_values.subvec(0, fe_coef.n_elem - 1) = fe_coef;
  }

  params.job_completed(var_idx) = true;
}

bool demean_accelerated(size_t var_idx, size_t iter_max, DemeanParams &params,
                        bool two_fe_mode = false) {
  CAPYBARA_TIME_FUNCTION("demean_accelerated");
  
  const vec &input = params.input_variables(var_idx);
  vec &output = params.output_variables(var_idx);
  DemeanWorkspace &ws = *params.workspace;

  size_t Q = two_fe_mode ? 2 : params.n_fe_groups;
  size_t nb_coef_T = params.total_coefficients;
  double tol = params.convergence_tolerance;

  vec &X = ws.fe_coef_primary;
  vec &GX = ws.fe_coef_secondary;
  vec &GGX = ws.fe_coef_temp;

  if (nb_coef_T > X.n_elem || nb_coef_T > GX.n_elem || nb_coef_T > GGX.n_elem) {
    throw std::runtime_error("Workspace vectors too small for subvec access");
  }

  // Use the original approach but avoid repeated subvec operations
  X.subvec(0, nb_coef_T - 1).zeros();
  GX.subvec(0, nb_coef_T - 1).zeros();
  GGX.subvec(0, nb_coef_T - 1).zeros();

  output.zeros();

  fe(var_idx, Q, X, GX, params);

  bool keep_going = vector_continue_criterion(X.subvec(0, nb_coef_T - 1),
                                              GX.subvec(0, nb_coef_T - 1), tol,
                                              params.capybara_params);

  vec &delta_GX = ws.delta_GX;
  vec &delta2_X = ws.delta2_X;
  vec &Y = ws.Y_acc;
  vec &GY = ws.GY_acc;
  vec &GGY = ws.GGY_acc;

  size_t iter = 0;
  size_t grand_acc_counter = 0;
  double ssr_old = 0;

  while (!params.stop_requested && keep_going && iter < iter_max) {
    ++iter;

    for (size_t rep = 0; rep < params.extra_projections; ++rep) {
      fe(var_idx, Q, GX, GGX, params);
      fe(var_idx, Q, GGX, X, params);
      fe(var_idx, Q, X, GX, params);
    }

    fe(var_idx, Q, GX, GGX, params);

    delta_GX.subvec(0, nb_coef_T - 1) =
        GGX.subvec(0, nb_coef_T - 1) - GX.subvec(0, nb_coef_T - 1);
    delta2_X.subvec(0, nb_coef_T - 1) = delta_GX.subvec(0, nb_coef_T - 1) -
                                        GX.subvec(0, nb_coef_T - 1) +
                                        X.subvec(0, nb_coef_T - 1);

    double vprod = dot(delta_GX.subvec(0, nb_coef_T - 1),
                       delta2_X.subvec(0, nb_coef_T - 1));
    double ssq = dot(delta2_X.subvec(0, nb_coef_T - 1),
                     delta2_X.subvec(0, nb_coef_T - 1));

    if (ssq < params.capybara_params.irons_tuck_eps) {
      break;
    }

    X.subvec(0, nb_coef_T - 1) =
        GGX.subvec(0, nb_coef_T - 1) -
        (vprod / ssq) * delta_GX.subvec(0, nb_coef_T - 1);

    if (iter >= params.projections_after_acceleration) {
      Y.subvec(0, nb_coef_T - 1) = X.subvec(0, nb_coef_T - 1);
      fe(var_idx, Q, Y, X, params);
    }

    fe(var_idx, Q, X, GX, params);

    keep_going = vector_continue_criterion(X.subvec(0, nb_coef_T - 1),
                                           GX.subvec(0, nb_coef_T - 1), tol,
                                           params.capybara_params);

    if (iter % params.grand_acceleration_frequency == 0) {
      ++grand_acc_counter;
      if (grand_acc_counter == 1) {
        Y.subvec(0, nb_coef_T - 1) = GX.subvec(0, nb_coef_T - 1);
      } else if (grand_acc_counter == 2) {
        GY.subvec(0, nb_coef_T - 1) = GX.subvec(0, nb_coef_T - 1);
      } else {
        GGY.subvec(0, nb_coef_T - 1) = GX.subvec(0, nb_coef_T - 1);

        delta_GX.subvec(0, nb_coef_T - 1) =
            GGY.subvec(0, nb_coef_T - 1) - GY.subvec(0, nb_coef_T - 1);
        delta2_X.subvec(0, nb_coef_T - 1) = delta_GX.subvec(0, nb_coef_T - 1) -
                                            GY.subvec(0, nb_coef_T - 1) +
                                            Y.subvec(0, nb_coef_T - 1);

        vprod = dot(delta_GX.subvec(0, nb_coef_T - 1),
                    delta2_X.subvec(0, nb_coef_T - 1));
        ssq = dot(delta2_X.subvec(0, nb_coef_T - 1),
                  delta2_X.subvec(0, nb_coef_T - 1));

        if (ssq < params.capybara_params.irons_tuck_eps) {
          break;
        }

        Y.subvec(0, nb_coef_T - 1) =
            GGY.subvec(0, nb_coef_T - 1) -
            (vprod / ssq) * delta_GX.subvec(0, nb_coef_T - 1);
        fe(var_idx, Q, Y, GX, params);
        grand_acc_counter = 0;
      }
    }

    if (iter % params.ssr_check_frequency == 0) {
      vec &mu_current = ws.prediction_buffer;
      params.fe_processor->compute_full_prediction(GX.subvec(0, nb_coef_T - 1),
                                                   mu_current, ws);

      vec resid = input - mu_current;
      double ssr = dot(resid, resid);

      if (stopping_criterion(ssr_old, ssr, tol, params.capybara_params)) {
        break;
      }
      ssr_old = ssr;
    }
  }

  params.fe_processor->compute_full_prediction(GX.subvec(0, nb_coef_T - 1),
                                               output, ws);

  if (params.save_fixed_effects) {
    params.fixed_effect_values = GX.subvec(0, nb_coef_T - 1);
  }

  params.iteration_counts(var_idx) += iter;
  params.job_completed(var_idx) = true;

  return (iter < iter_max);
}

void demean_general(size_t var_idx, DemeanParams &params) {
  CAPYBARA_TIME_FUNCTION("demean_general");
  
  size_t Q = params.n_fe_groups;

  if (Q == 1) {
    demean_single_fe(var_idx, params);
  } else if (Q == 2) {
    demean_accelerated(var_idx, params.max_iterations, params);
  } else {
    // Three-phase approach for Q >= 3
    bool converged = false;

    // Phase 1: Warmup
    if (params.warmup_iterations > 0) {
      converged = demean_accelerated(var_idx, params.warmup_iterations, params);
    }

    if (!converged && params.max_iterations > params.warmup_iterations) {
      // Phase 2: 2-FE convergence
      size_t iter_max_2fe =
          params.max_iterations / 2 - params.warmup_iterations;
      if (iter_max_2fe > 0) {
        demean_accelerated(var_idx, iter_max_2fe, params, true);
      }

      // Phase 3: Re-acceleration
      size_t remaining =
          params.max_iterations - params.iteration_counts(var_idx);
      if (remaining > 0) {
        demean_accelerated(var_idx, remaining, params);
      }
    }
  }
}

struct DemeanResult {
  field<vec> demeaned_vars;
  vec fixed_effects;
  bool has_fixed_effects;

  DemeanResult(size_t n_vars)
      : demeaned_vars(n_vars), has_fixed_effects(false) {}
};

DemeanResult demean_variables(const field<vec> &input_vars, const vec &weights,
                              const field<uvec> &fe_indices, const uvec &nb_ids,
                              const field<uvec> &fe_id_tables,
                              bool save_fixed_effects,
                              const CapybaraParameters &params) {
  CAPYBARA_TIME_FUNCTION("demean_variables");
  
  size_t n_vars = input_vars.n_elem;

  if (n_vars == 0) {
    return DemeanResult(0);
  }

  size_t n_obs = input_vars(0).n_elem;
  size_t n_fe_groups = fe_indices.n_elem;

  auto fe_processor = std::make_shared<FixedEffects>(
      n_obs, n_fe_groups, weights, fe_indices, nb_ids, fe_id_tables, params);

  DemeanParams demean_params(n_obs, n_fe_groups, fe_processor, params,
                             save_fixed_effects, input_vars);

  for (size_t v = 0; v < n_vars; ++v) {
    if (demean_params.stop_requested)
      break;
    demean_general(v, demean_params);
  }

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

} // namespace demean
} // namespace capybara

#endif // CAPYBARA_DEMEAN_H
