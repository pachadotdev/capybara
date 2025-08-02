// From a design matrix X and response vector Y, this creates T(X)
// and T(Y) after iteratively removing the group means.

#ifndef CAPYBARA_DEMEAN_H
#define CAPYBARA_DEMEAN_H

namespace capybara {

struct DemeanWorkspace {
  vec X, GX, GGX;           // Irons-Tuck vectors
  vec Y, GY, GGY;           // Grand acceleration vectors
  vec delta_GX, delta2_X;   // Acceleration deltas
  vec mu_current;           // Current prediction
  vec residual;             // Working residual
  vec fe_coef_a, fe_coef_b; // Two FE coefficients
  vec sum_in_out;           // Accumulated sums
  vec sum_other_fe;         // Other FE contributions

  DemeanWorkspace(size_t n_obs, size_t max_coef) {
    X.set_size(max_coef);
    GX.set_size(max_coef);
    GGX.set_size(max_coef);
    Y.set_size(max_coef);
    GY.set_size(max_coef);
    GGY.set_size(max_coef);
    delta_GX.set_size(max_coef);
    delta2_X.set_size(max_coef);

    mu_current.set_size(n_obs);
    residual.set_size(n_obs);
    fe_coef_a.set_size(max_coef);
    fe_coef_b.set_size(max_coef);
    sum_in_out.set_size(max_coef);
    sum_other_fe.set_size(n_obs);
  }
};

class FEProcessor {
private:
  size_t n_obs_;
  size_t n_fe_groups_;
  bool has_weights_;

  vec weights_;
  field<uvec> fe_indices_; // FE group assignments for each observation
  uvec nb_coefs_;          // Number of coefficients per FE group
  uvec coef_starts_;       // Starting index for each FE group
  vec sum_weights_;        // Sum of weights per FE coefficient (flat)

  size_t total_coefficients_;

public:
  FEProcessor(size_t n_obs, size_t n_fe_groups, const vec &weights,
              const field<uvec> &fe_indices, const uvec &nb_ids)
      : n_obs_(n_obs), n_fe_groups_(n_fe_groups), weights_(weights),
        fe_indices_(fe_indices), nb_coefs_(nb_ids) {

    has_weights_ = weights_.n_elem > 1 &&
                   !approx_equal(weights_, ones<vec>(n_obs_), "absdiff", 1e-14);
    if (!has_weights_) {
      weights_ = ones<vec>(n_obs_);
    }

    coef_starts_.set_size(n_fe_groups_);
    total_coefficients_ = 0;
    for (size_t q = 0; q < n_fe_groups_; ++q) {
      coef_starts_(q) = total_coefficients_;
      total_coefficients_ += nb_coefs_(q);
    }

    sum_weights_.set_size(total_coefficients_);
    compute_sum_weights();
  }

  void compute_sum_weights() {

    sum_weights_.zeros();
    const double *w_ptr = weights_.memptr();
    double *sum_ptr = sum_weights_.memptr();

    for (size_t q = 0; q < n_fe_groups_; ++q) {
      const uword *fe_ptr = fe_indices_(q).memptr();
      size_t coef_start = coef_starts_(q);

      for (size_t i = 0; i < n_obs_; ++i) {
        sum_ptr[coef_start + fe_ptr[i]] += w_ptr[i];
      }
    }
  }

  void compute_fe_coef_single(const vec &target, vec &fe_coef) const {

    fe_coef.set_size(nb_coefs_(0));
    fe_coef.zeros();

    const double *target_ptr = target.memptr();
    const double *w_ptr = weights_.memptr();
    const uword *fe_ptr = fe_indices_(0).memptr();
    double *coef_ptr = fe_coef.memptr();
    const double *sum_w_ptr = sum_weights_.memptr();

    for (size_t i = 0; i < n_obs_; ++i) {
      coef_ptr[fe_ptr[i]] += target_ptr[i] * w_ptr[i];
    }

    for (size_t c = 0; c < nb_coefs_(0); ++c) {
      coef_ptr[c] /= sum_w_ptr[c];
    }
  }

  void compute_fe_coef_two(const vec &target, vec &fe_coef_a, vec &fe_coef_b,
                           DemeanWorkspace &ws,
                           const CapybaraParameters &params) const {
    fe_coef_a.set_size(nb_coefs_(0));
    fe_coef_b.set_size(nb_coefs_(1));
    fe_coef_a.zeros();
    fe_coef_b.zeros();

    // Use pre-allocated workspace vectors
    vec &prediction = ws.mu_current;
    vec &residual = ws.residual;

    const double tol = params.demean_2fe_tolerance;
    const size_t max_iter = params.demean_2fe_max_iter;

    // Raw pointer optimization for hot 2-FE algorithm
    const double *target_ptr = target.memptr();
    const double *weights_ptr = weights_.memptr();
    const uword *fe_a_ptr = fe_indices_(0).memptr();
    const uword *fe_b_ptr = fe_indices_(1).memptr();
    double *pred_ptr = prediction.memptr();
    double *resid_ptr = residual.memptr();
    double *coef_a_ptr = fe_coef_a.memptr();
    double *coef_b_ptr = fe_coef_b.memptr();
    const double *sum_weights_a_ptr = sum_weights_.memptr();
    const double *sum_weights_b_ptr = sum_weights_.memptr() + nb_coefs_(0);

    for (size_t iter = 0; iter < max_iter; ++iter) {
      // Store old coefficients for convergence check using workspace
      vec &fe_a_old = ws.fe_coef_a;
      vec &fe_b_old = ws.fe_coef_b;
      fe_a_old = fe_coef_a;
      fe_b_old = fe_coef_b;

      // Broadcast FE A to prediction - vectorized
      for (size_t i = 0; i < n_obs_; ++i) {
        pred_ptr[i] = coef_a_ptr[fe_a_ptr[i]];
      }

      // Compute residual - vectorized
      for (size_t i = 0; i < n_obs_; ++i) {
        resid_ptr[i] = target_ptr[i] - pred_ptr[i];
      }

      // Compute FE B from residual - vectorized accumulation
      std::fill(coef_b_ptr, coef_b_ptr + nb_coefs_(1), 0.0);
      for (size_t i = 0; i < n_obs_; ++i) {
        coef_b_ptr[fe_b_ptr[i]] += resid_ptr[i] * weights_ptr[i];
      }
      for (size_t c = 0; c < nb_coefs_(1); ++c) {
        coef_b_ptr[c] /= sum_weights_b_ptr[c];
      }

      // Broadcast FE B to prediction - vectorized
      for (size_t i = 0; i < n_obs_; ++i) {
        pred_ptr[i] = coef_b_ptr[fe_b_ptr[i]];
      }

      // Compute residual - vectorized
      for (size_t i = 0; i < n_obs_; ++i) {
        resid_ptr[i] = target_ptr[i] - pred_ptr[i];
      }

      // Compute FE A from residual - vectorized accumulation
      std::fill(coef_a_ptr, coef_a_ptr + nb_coefs_(0), 0.0);
      for (size_t i = 0; i < n_obs_; ++i) {
        coef_a_ptr[fe_a_ptr[i]] += resid_ptr[i] * weights_ptr[i];
      }
      for (size_t c = 0; c < nb_coefs_(0); ++c) {
        coef_a_ptr[c] /= sum_weights_a_ptr[c];
      }

      // Check convergence using vectorized norm
      if (norm(fe_coef_a - fe_a_old, 2) < tol &&
          norm(fe_coef_b - fe_b_old, 2) < tol) {
        break;
      }
    }
  }

  void broadcast_fe_to_prediction(size_t group_idx, const vec &fe_coef,
                                  vec &prediction) const {
    // Raw pointer optimization for broadcast operation
    const uword *fe_ptr = fe_indices_(group_idx).memptr();
    const double *coef_ptr = fe_coef.memptr();
    double *pred_ptr = prediction.memptr();

    // Vectorized broadcast using raw pointers
    for (size_t i = 0; i < n_obs_; ++i) {
      pred_ptr[i] = coef_ptr[fe_ptr[i]];
    }
  }

  void compute_fe_from_residual(size_t group_idx, const vec &residual,
                                vec &fe_coef) const {
    // Use vectorized zeroing
    fe_coef.zeros();

    // Raw pointer optimization for accumulation
    const double *resid_ptr = residual.memptr();
    const double *w_ptr = weights_.memptr();
    const uword *fe_ptr = fe_indices_(group_idx).memptr();
    double *coef_ptr = fe_coef.memptr();
    size_t coef_start = coef_starts_(group_idx);
    const double *sum_w_ptr = sum_weights_.memptr() + coef_start;

    // Vectorized accumulation loop
    for (size_t i = 0; i < n_obs_; ++i) {
      coef_ptr[fe_ptr[i]] += resid_ptr[i] * w_ptr[i];
    }

    // Vectorized division loop
    for (size_t c = 0; c < nb_coefs_(group_idx); ++c) {
      coef_ptr[c] /= sum_w_ptr[c];
    }
  }

  void compute_fe_coef_general(size_t group_idx, const vec &target_residual,
                               vec &all_fe_coef) const {

    size_t coef_start = coef_starts_(group_idx);
    size_t nb_coef = nb_coefs_(group_idx);

    // Raw pointer optimization for general case
    const double *resid_ptr = target_residual.memptr();
    const double *w_ptr = weights_.memptr();
    const uword *fe_ptr = fe_indices_(group_idx).memptr();
    double *coef_ptr = all_fe_coef.memptr() + coef_start;
    const double *sum_w_ptr = sum_weights_.memptr() + coef_start;

    // Vectorized zeroing
    std::fill(coef_ptr, coef_ptr + nb_coef, 0.0);

    // Vectorized accumulation
    for (size_t i = 0; i < n_obs_; ++i) {
      coef_ptr[fe_ptr[i]] += resid_ptr[i] * w_ptr[i];
    }

    // Vectorized division
    for (size_t c = 0; c < nb_coef; ++c) {
      coef_ptr[c] /= sum_w_ptr[c];
    }
  }

  void add_fe_to_prediction(size_t group_idx, const vec &all_fe_coef,
                            vec &prediction) const {

    size_t coef_start = coef_starts_(group_idx);
    const uword *fe_ptr = fe_indices_(group_idx).memptr();
    const double *coef_ptr = all_fe_coef.memptr() + coef_start;
    double *pred_ptr = prediction.memptr();

    for (size_t i = 0; i < n_obs_; ++i) {
      pred_ptr[i] += coef_ptr[fe_ptr[i]];
    }
  }

  void compute_full_prediction(const vec &all_fe_coef, vec &prediction) const {
    // Vectorized zeroing
    prediction.zeros();

    // Raw pointers for maximum performance
    const double *coef_ptr = all_fe_coef.memptr();
    double *pred_ptr = prediction.memptr();

    // Vectorized accumulation across all FE groups
    for (size_t q = 0; q < n_fe_groups_; ++q) {
      size_t coef_start = coef_starts_(q);
      const uword *fe_ptr = fe_indices_(q).memptr();
      const double *group_coef_ptr = coef_ptr + coef_start;

      // Accumulate contributions from this FE group
      for (size_t i = 0; i < n_obs_; ++i) {
        pred_ptr[i] += group_coef_ptr[fe_ptr[i]];
      }
    }
  }

  size_t num_observations() const { return n_obs_; }
  size_t num_fe_groups() const { return n_fe_groups_; }
  size_t total_coefficients() const { return total_coefficients_; }
  const uvec &coefficient_counts() const { return nb_coefs_; }
  const uvec &coefficient_starts() const { return coef_starts_; }
};

inline bool irons_tuck_update(const vec &X_subvec, const vec &GX_subvec,
                              const vec &GGX_subvec, vec &delta_GX,
                              vec &delta2_X, const CapybaraParameters &params) {
  delta_GX = GGX_subvec - GX_subvec;
  delta2_X = delta_GX - GX_subvec + X_subvec;
  double ssq = dot(delta2_X, delta2_X);
  if (ssq < params.irons_tuck_eps) {
    return true;
  }
  return false;
}

struct DemeanResult {
  field<vec> demeaned_vars;
  vec fe_coefficients;      // All FE coefficients (for warm-starting)
  field<vec> fixed_effects; // Per-dimension FE (for final results)
  bool has_fixed_effects;

  DemeanResult(size_t n_vars)
      : demeaned_vars(n_vars), has_fixed_effects(false) {}
};

DemeanResult demean_variables(const field<vec> &variables, const vec &weights,
                              const field<uvec> &fe_indices, const uvec &nb_ids,
                              const field<uvec> &fe_id_tables,
                              bool save_fixed_effects,
                              const CapybaraParameters &params,
                              const vec &init_fe_coef) {
  size_t n_vars = variables.n_elem;
  size_t n_obs = variables(0).n_elem;
  size_t n_fe_groups = fe_indices.n_elem;

  FEProcessor fe_proc(n_obs, n_fe_groups, weights, fe_indices, nb_ids);
  DemeanWorkspace ws(n_obs, fe_proc.total_coefficients());
  DemeanResult result(n_vars);

  for (size_t i = 0; i < n_vars; ++i) {
    result.demeaned_vars(i).set_size(n_obs);
  }

  vec all_fixed_effects;
  if (save_fixed_effects) {
    all_fixed_effects.set_size(fe_proc.total_coefficients());
  }

  // Track if we should save FE coefficients for warm-starting
  bool first_var_processed = false;

  for (size_t i = 0; i < n_vars; ++i) {
    if (n_fe_groups == 1) {
      vec fe_coef;
      if (init_fe_coef.n_elem == fe_proc.total_coefficients() && i == 0) {
        fe_coef = init_fe_coef.subvec(0, fe_proc.coefficient_counts()(0) - 1);
        vec temp_pred = ws.mu_current;
        fe_proc.broadcast_fe_to_prediction(0, fe_coef, temp_pred);
        vec residual = variables(i) - temp_pred;
        fe_proc.compute_fe_from_residual(0, residual, fe_coef);
      } else {
        fe_proc.compute_fe_coef_single(variables(i), fe_coef);
      }
      fe_proc.broadcast_fe_to_prediction(0, fe_coef, result.demeaned_vars(i));
      result.demeaned_vars(i) = variables(i) - result.demeaned_vars(i);
      // Save FE coefficients from first variable for warm-starting
      if (!first_var_processed) {
        result.fe_coefficients.set_size(fe_proc.total_coefficients());
        result.fe_coefficients.subvec(0, fe_coef.n_elem - 1) = fe_coef;
        first_var_processed = true;
      }

      if (save_fixed_effects) {
        all_fixed_effects.subvec(0, fe_coef.n_elem - 1) = fe_coef;
      }

    } else if (n_fe_groups == 2) {
      vec fe_coef_a, fe_coef_b;
      if (init_fe_coef.n_elem == fe_proc.total_coefficients() && i == 0) {
        size_t n_a = fe_proc.coefficient_counts()(0);
        size_t n_b = fe_proc.coefficient_counts()(1);
        fe_coef_a = init_fe_coef.subvec(0, n_a - 1);
        fe_coef_b = init_fe_coef.subvec(n_a, n_a + n_b - 1);
        CapybaraParameters warm_params = params;
        warm_params.demean_2fe_max_iter =
            std::max(size_t(3), params.demean_2fe_max_iter / 5);
        fe_proc.compute_fe_coef_two(variables(i), fe_coef_a, fe_coef_b, ws,
                                    warm_params);
      } else {
        fe_proc.compute_fe_coef_two(variables(i), fe_coef_a, fe_coef_b, ws,
                                    params);
      }
      vec prediction = ws.mu_current;
      fe_proc.broadcast_fe_to_prediction(0, fe_coef_a, prediction);
      vec temp_pred = ws.residual;
      fe_proc.broadcast_fe_to_prediction(1, fe_coef_b, temp_pred);
      prediction += temp_pred;
      result.demeaned_vars(i) = variables(i) - prediction;
      // Save FE coefficients from first variable
      if (!first_var_processed) {
        result.fe_coefficients.set_size(fe_proc.total_coefficients());
        result.fe_coefficients.subvec(0, fe_coef_a.n_elem - 1) = fe_coef_a;
        result.fe_coefficients.subvec(fe_coef_a.n_elem,
                                      fe_coef_a.n_elem + fe_coef_b.n_elem - 1) =
            fe_coef_b;
        first_var_processed = true;
      }

      if (save_fixed_effects) {
        all_fixed_effects.subvec(0, fe_coef_a.n_elem - 1) = fe_coef_a;
        all_fixed_effects.subvec(fe_coef_a.n_elem,
                                 fe_coef_a.n_elem + fe_coef_b.n_elem - 1) =
            fe_coef_b;
      }

    } else {
      // General case: 3+ FE groups - use Irons-Tuck acceleration like fixest
      vec &all_fe_coef = ws.sum_in_out; // X vector
      vec &GX = ws.X;                   // GX vector  
      vec &GGX = ws.GX;                 // GGX vector
      vec &delta_GX = ws.delta_GX;      // Delta vectors
      vec &delta2_X = ws.delta2_X;
      
      all_fe_coef.set_size(fe_proc.total_coefficients());
      GX.set_size(fe_proc.total_coefficients());
      GGX.set_size(fe_proc.total_coefficients());
      delta_GX.set_size(fe_proc.total_coefficients());
      delta2_X.set_size(fe_proc.total_coefficients());

      if (init_fe_coef.n_elem == fe_proc.total_coefficients() && i == 0) {
        all_fe_coef = init_fe_coef;
      } else {
        all_fe_coef.zeros();
      }

      vec &prediction = ws.mu_current; // Use workspace vector
      vec &residual = ws.residual;     // Use workspace vector

      size_t max_iter =
          (init_fe_coef.n_elem > 0 && i == 0)
              ? std::max(size_t(3), params.demean_2fe_max_iter / 5)
              : params.demean_2fe_max_iter;

      // Step 1: Compute GX (first projection from X)
      for (size_t q = 0; q < n_fe_groups; ++q) {
        prediction.zeros();
        for (size_t other_q = 0; other_q < n_fe_groups; ++other_q) {
          if (other_q != q) {
            fe_proc.add_fe_to_prediction(other_q, all_fe_coef, prediction);
          }
        }
        residual = variables(i) - prediction;
        fe_proc.compute_fe_coef_general(q, residual, GX);
      }
      
      // Check initial convergence
      vec &old_coef = ws.fe_coef_a; // Reuse workspace for convergence check
      old_coef.set_size(fe_proc.total_coefficients());
      
      bool keepGoing = norm(all_fe_coef - GX, 2) > params.demean_2fe_tolerance;
      
      size_t iter = 0;
      
      while (keepGoing && iter < max_iter) {
        ++iter;
        old_coef = all_fe_coef;

        // Step 2: Compute GGX (second projection from GX) 
        for (size_t q = 0; q < n_fe_groups; ++q) {
          prediction.zeros();
          for (size_t other_q = 0; other_q < n_fe_groups; ++other_q) {
            if (other_q != q) {
              fe_proc.add_fe_to_prediction(other_q, GX, prediction);
            }
          }
          residual = variables(i) - prediction;
          fe_proc.compute_fe_coef_general(q, residual, GGX);
        }

        // Step 3: Try Irons-Tuck acceleration after warmup
        if (iter >= params.demean_warmup_iterations) {
          bool converged = irons_tuck(all_fe_coef, GX, GGX, delta_GX, delta2_X, params);
          if (converged) {
            break; // Convergence achieved through acceleration
          }
        } else {
          // Before warmup, use regular updates
          all_fe_coef = GGX;
        }
        
        // Recompute GX for next iteration (from updated all_fe_coef)
        for (size_t q = 0; q < n_fe_groups; ++q) {
          prediction.zeros();
          for (size_t other_q = 0; other_q < n_fe_groups; ++other_q) {
            if (other_q != q) {
              fe_proc.add_fe_to_prediction(other_q, all_fe_coef, prediction);
            }
          }
          residual = variables(i) - prediction;
          fe_proc.compute_fe_coef_general(q, residual, GX);
        }

        // Check convergence
        keepGoing = norm(all_fe_coef - old_coef, 2) > params.demean_2fe_tolerance;
      }

      fe_proc.compute_full_prediction(all_fe_coef, prediction);
      result.demeaned_vars(i) = variables(i) - prediction;

      // Save FE coefficients from first variable
      if (!first_var_processed) {
        result.fe_coefficients = all_fe_coef;
        first_var_processed = true;
      }

      if (save_fixed_effects) {
        all_fixed_effects = all_fe_coef;
      }
    }
  }

  if (save_fixed_effects) {
    result.fixed_effects.set_size(n_fe_groups);
    size_t offset = 0;
    for (size_t q = 0; q < n_fe_groups; ++q) {
      size_t n_coef = fe_proc.coefficient_counts()(q);
      result.fixed_effects(q) =
          all_fixed_effects.subvec(offset, offset + n_coef - 1);
      offset += n_coef;
    }
    result.has_fixed_effects = true;
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_DEMEAN_H
