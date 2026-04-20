// Combined separation detection

#ifndef CAPYBARA_SEPARATION_H
#define CAPYBARA_SEPARATION_H

namespace capybara {

inline SeparationResult check_separation(const vec &y, const mat &X,
                                         const vec &w,
                                         const CapybaraParameters &params) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  const uvec boundary_sample = find(y == 0);
  const uvec interior_sample = find(y > 0);

  if (boundary_sample.n_elem == 0) {
    return result;
  }

  // Compute centering vector (weighted mean of X on interior samples)
  mat X_centered;
  bool needs_centering = false;
  if (X.n_cols > 0 && interior_sample.n_elem > 0) {
    vec w_interior = w;
    w_interior.elem(boundary_sample).zeros();
    const double sum_w = accu(w_interior);

    if (sum_w > 0) {
      const vec center_vec = (X.t() * w_interior) / sum_w;
      X_centered = X;
      X_centered.each_row() -= center_vec.t();
      needs_centering = true;
    }
  }

  const mat &X_for_sep = needs_centering ? X_centered : X;

  // Simplex algorithm with collinearity-aware residual computation
  // (matches ppmlhdfe logic)
  if (params.sep_use_simplex && X.n_cols > 0) {
    SeparationResult simplex_result = detect_separation_simplex(
        X_for_sep, boundary_sample, interior_sample, w, params);

    if (simplex_result.num_separated > 0) {
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
        detect_separation_relu(y, X_for_sep, w, params);

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

} // namespace capybara

#endif // CAPYBARA_SEPARATION_H
