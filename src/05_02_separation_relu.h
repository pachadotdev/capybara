// ReLU Separation Detection
// Algorithm: Iterative least squares with ReLU activation
// Reference: Section 3.2 of Correia, Guimaraes, Zylkin (2019)

#ifndef CAPYBARA_SEPARATION_RELU_H
#define CAPYBARA_SEPARATION_RELU_H

namespace capybara {

// Main ReLU separation detection algorithm (without FE)
inline SeparationResult
detect_separation_relu(const vec &y, const mat &X, const vec &w,
                       const CapybaraParameters &params) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;

  const uvec boundary_sample = find(y == 0);
  const uvec interior_sample = find(y > 0);
  const uword num_boundary = boundary_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  vec u = conv_to<vec>::from(y == 0);
  const double M = 1.0 / std::sqrt(datum::eps);

  vec xbd(n, fill::zeros);
  vec xbd_prev1(n, fill::zeros);
  vec xbd_prev2(n, fill::zeros);
  vec resid(n);
  double uu_old = dot(u, u);

  // Progress tracking for acceleration (from ppmlhdfe)
  double ee_cumulative = 0.0;
  const double ee_boundary = uu_old;
  double progress_ratio_prev1 = 0.0;
  double progress_ratio_prev2 = 0.0;
  uword num_candidates_prev1 = 0;
  uword num_candidates_prev2 = 0;
  bool convergence_is_stuck = false;
  double acceleration_value = 1.0;

  for (uword iter = 0; iter < params.sep_max_iter; ++iter) {
    if (iter % 100 == 0)
      check_user_interrupt();

    // Shift xbd history for acceleration detection
    std::swap(xbd_prev2, xbd_prev1);
    std::swap(xbd_prev1, xbd);

    // Build weights with potential acceleration
    vec weights(n, fill::ones);
    if (interior_sample.n_elem > 0) {
      weights.elem(interior_sample).fill(M);
    }

    // Apply acceleration to stuck negative boundary observations
    if (convergence_is_stuck && iter > 3) {
      const vec xbd_b = xbd_prev1.elem(boundary_sample);
      const vec xbd_b_p1 = xbd_prev2.elem(boundary_sample);
      // Find obs stuck at negative values
      for (uword i = 0; i < num_boundary; ++i) {
        if (xbd_b(i) < -0.1 * params.sep_tol && xbd_b_p1(i) < 1.01 * xbd_b(i)) {
          weights(boundary_sample(i)) = acceleration_value;
        }
      }
    }

    solve_wls(X, u, weights, resid);
    xbd = u - resid;

    const double ee = dot(resid, resid);
    const double epsilon = ee + params.sep_tol;
    const double delta = epsilon + params.sep_tol;

    // Track cumulative progress (from ppmlhdfe)
    ee_cumulative += ee;
    const double progress_ratio =
        ee_boundary > 0 ? 100.0 * ee_cumulative / ee_boundary : 100.0;

    // Count candidates for separation
    uword num_candidates = 0;
    {
      const vec boundary_xbd_tmp = xbd.elem(boundary_sample);
      for (uword i = 0; i < num_boundary; ++i) {
        if (boundary_xbd_tmp(i) > delta)
          num_candidates++;
      }
    }

    // Detect stuck convergence and enable acceleration (from ppmlhdfe)
    if (!convergence_is_stuck && iter > 3) {
      if ((progress_ratio - progress_ratio_prev2 < 1.0) &&
          (num_candidates == num_candidates_prev2)) {
        convergence_is_stuck = true;
        acceleration_value = 4.0;
      }
    } else if (convergence_is_stuck) {
      acceleration_value = std::min(256.0, 4.0 * acceleration_value);
    }

    // Update history
    progress_ratio_prev2 = progress_ratio_prev1;
    progress_ratio_prev1 = progress_ratio;
    num_candidates_prev2 = num_candidates_prev1;
    num_candidates_prev1 = num_candidates;

    // Enforce constraints on interior
    if (interior_sample.n_elem > 0) {
      xbd.elem(interior_sample).zeros();
    }

    // Zero out near-zero boundary values
    vec boundary_xbd = xbd.elem(boundary_sample);
    const uvec near_zero =
        find((boundary_xbd > -0.1 * delta) % (boundary_xbd < delta));
    if (near_zero.n_elem > 0) {
      xbd.elem(boundary_sample.elem(near_zero)).zeros();
      boundary_xbd = xbd.elem(boundary_sample);
    }

    // Check separation - all non-negative means we found it
    if (all(boundary_xbd >= 0)) {
      const uvec sep_ind_local = find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    resid.clean(params.sep_zero_tol);
    const vec boundary_resid = resid.elem(boundary_sample);

    if (boundary_resid.min() >= 0) {
      const uvec pos_resid_idx = find(boundary_resid > delta);
      if (pos_resid_idx.n_elem > 0) {
        xbd.elem(boundary_sample.elem(pos_resid_idx)).zeros();
      }
      boundary_xbd = xbd.elem(boundary_sample);
      const uvec sep_ind_local = find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // ReLU update: u = max(xbd, 0) on boundary
    u.zeros();
    u.elem(boundary_sample) = clamp(boundary_xbd, 0.0, datum::inf);

    const double uu = dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.sep_tol * 0.01) {
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  if (!result.converged) {
    result.iterations = params.sep_max_iter;
    const vec boundary_xbd = xbd.elem(boundary_sample);
    const uvec sep_ind_local = find(boundary_xbd > params.sep_tol);
    if (sep_ind_local.n_elem > 0) {
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
    }
  }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_RELU_H
