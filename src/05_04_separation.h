// Combined separation detection

#ifndef CAPYBARA_SEPARATION_H
#define CAPYBARA_SEPARATION_H

namespace capybara {

inline SeparationResult check_separation(const vec &y, const mat &X,
                                         const vec &w, FlatFEMap &fe_map,
                                         const CapybaraParameters &params) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  const uvec boundary_sample = find(y == 0);
  const uvec interior_sample = find(y > 0);

  if (boundary_sample.n_elem == 0) {
    return result;
  }

  // Following ppmlhdfe: partial out X on the interior (y > 0) sample
  // For FE models: use full AP demeaning through FE structure
  // For non-FE models: just subtract weighted mean
  // Use pointer to avoid copying X when no centering is needed
  mat X_centered_storage;
  const mat *X_to_use = &X;
  const bool has_fe = (fe_map.K > 0 && fe_map.structure_built);

  if (X.n_cols > 0 && interior_sample.n_elem > 0) {
    // Create weight vector that zeros out boundary observations
    vec w_interior = w;
    w_interior.elem(boundary_sample).zeros();
    const double sum_w = accu(w_interior);

    if (sum_w > 0) {
      if (has_fe) {
        // FE case: use centering algorithm to partial out FE
        // This matches ppmlhdfe's HDFE._partial_out()
        X_centered_storage = X;

        // Update FE map with interior-only weights
        fe_map.update_weights(w_interior);

        // Center X through the FE structure
        center_variables(X_centered_storage, w_interior, fe_map,
                         params.center_tol, params.iter_center_max,
                         params.grand_acc_period);

        // Restore original weights
        fe_map.update_weights(w);
        X_to_use = &X_centered_storage;
      } else {
        // Non-FE case: simple mean centering
        const vec center_vec = (X.t() * w_interior) / sum_w;
        X_centered_storage = X;
        X_centered_storage.each_row() -= center_vec.t();
        X_to_use = &X_centered_storage;
      }
    }
    // else: X_to_use remains &X (no copy needed)
  }
  // else: X_to_use remains &X (no copy needed)

  const mat &X_centered = *X_to_use;

  // Simplex algorithm with collinearity-aware residual computation
  // (matches ppmlhdfe logic)
  if (params.sep_use_simplex && X.n_cols > 0) {
    SeparationResult simplex_result = detect_separation_simplex(
        X_centered, boundary_sample, interior_sample, w, params);

    if (simplex_result.num_separated > 0) {
      // Validate indices before conversion
      if (simplex_result.separated_obs.n_elem > 0) {
        const uword max_idx = simplex_result.separated_obs.max();
        if (max_idx >= boundary_sample.n_elem) {
          cpp4r::stop("Internal error in simplex separation: max index %u >= "
                      "boundary_sample size %u",
                      (unsigned)max_idx, (unsigned)boundary_sample.n_elem);
        }
      }
      // Convert boundary-relative indices to absolute indices
      result.separated_obs = boundary_sample.elem(simplex_result.separated_obs);
      result.num_separated = result.separated_obs.n_elem;
      result.converged = simplex_result.converged;
      // If simplex found separation, return immediately without running ReLU
      return result;
    }
  }

  // ReLU: only run if simplex didn't find separation (or was disabled)
  if (params.sep_use_relu) {
    SeparationResult relu_result =
        detect_separation_relu(y, X_centered, w, params);

    if (relu_result.num_separated > 0) {
      result.separated_obs = std::move(relu_result.separated_obs);
      result.num_separated = result.separated_obs.n_elem;
      result.support = std::move(relu_result.support);
      result.iterations = relu_result.iterations;
      result.converged = relu_result.converged;
    }
  }

  return result;
}

// Overload for non-FE case (backward compatibility)
inline SeparationResult check_separation(const vec &y, const mat &X,
                                         const vec &w,
                                         const CapybaraParameters &params) {
  FlatFEMap empty_fe_map;
  return check_separation(y, X, w, empty_fe_map, params);
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_H
