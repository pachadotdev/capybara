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
  if (boundary_sample.n_elem == 0) {
    return result;
  }

  // Compute centering vector (weighted mean of X on interior samples)
  // Avoid full N*P copy when possible
  vec center_vec;
  bool needs_centering = false;
  if (X.n_cols > 0) {
    vec w_interior = w;
    w_interior.elem(boundary_sample).zeros();
    const double sum_w = accu(w_interior);

    if (sum_w > 0) {
      center_vec = (X.t() * w_interior) / sum_w;
      needs_centering = true;
    }
  }

  // Simplex: only create centered submatrix for boundary rows (much smaller)
  if (params.sep_use_simplex && X.n_cols > 0) {
    mat X_boundary = X.rows(boundary_sample);
    if (needs_centering) {
      X_boundary.each_row() -= center_vec.t();
    }

    SeparationResult simplex_result =
        detect_separation_simplex(X_boundary, params);

    if (simplex_result.num_separated > 0) {
      result.separated_obs = boundary_sample.elem(simplex_result.separated_obs);
      result.num_separated = result.separated_obs.n_elem;
    }
  }

  // ReLU: create full centered matrix only if ReLU is enabled
  if (params.sep_use_relu) {
    mat X_centered;
    if (needs_centering) {
      X_centered = X;
      X_centered.each_row() -= center_vec.t();
    }
    const mat &X_for_relu = needs_centering ? X_centered : X;

    SeparationResult relu_result =
        detect_separation_relu(y, X_for_relu, w, params);

    if (relu_result.num_separated > 0) {
      if (result.num_separated > 0) {
        result.separated_obs =
            unique(join_vert(result.separated_obs, relu_result.separated_obs));
        result.num_separated = result.separated_obs.n_elem;
      } else {
        result.separated_obs = relu_result.separated_obs;
        result.num_separated = relu_result.num_separated;
      }
      result.support = relu_result.support;
      result.iterations = relu_result.iterations;
    }
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_H
