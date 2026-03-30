// ReLU Separation Detection
// Algorithm: Iterative least squares with ReLU activation
// Reference: Section 3.2 of Correia, Guimaraes, Zylkin (2019)

#ifndef CAPYBARA_SEPARATION_RELU_H
#define CAPYBARA_SEPARATION_RELU_H

namespace capybara {

// Workspace struct to avoid repeated allocations in ReLU separation
struct SeparationReluWorkspace {
  vec xbd;
  vec xbd_prev1;
  vec xbd_prev2;
  vec resid;
  vec u;
  vec weights;
  vec boundary_xbd;  // sized to num_boundary

  void ensure_size(uword n, uword num_boundary) {
    if (xbd.n_elem != n) {
      xbd.set_size(n);
      xbd_prev1.set_size(n);
      xbd_prev2.set_size(n);
      resid.set_size(n);
      u.set_size(n);
      weights.set_size(n);
    }
    if (boundary_xbd.n_elem != num_boundary) {
      boundary_xbd.set_size(num_boundary);
    }
  }
};

// Main ReLU separation detection algorithm (without FE)
inline SeparationResult
detect_separation_relu(const vec &y, const mat &X, const vec &w,
                       const CapybaraParameters &params,
                       SeparationReluWorkspace *ws = nullptr) {
  SeparationResult result;
  result.converged = false;
  result.num_separated = 0;

  const uword n = y.n_elem;

  const uvec boundary_sample = find(y == 0);
  const uvec interior_sample = find(y > 0);
  const uword num_boundary = boundary_sample.n_elem;
  const uword *bnd_ptr = boundary_sample.memptr();
  const uword *int_ptr = interior_sample.memptr();
  const uword num_interior = interior_sample.n_elem;

  if (num_boundary == 0) {
    result.converged = true;
    return result;
  }

  // Use workspace if provided, otherwise create local buffers
  SeparationReluWorkspace local_ws;
  SeparationReluWorkspace &work = ws ? *ws : local_ws;
  work.ensure_size(n, num_boundary);

  vec &xbd = work.xbd;
  vec &xbd_prev1 = work.xbd_prev1;
  vec &xbd_prev2 = work.xbd_prev2;
  vec &resid = work.resid;
  vec &u = work.u;
  vec &weights = work.weights;
  vec &boundary_xbd = work.boundary_xbd;

  xbd.zeros();
  xbd_prev1.zeros();
  xbd_prev2.zeros();

  // Initialize u = indicator(y == 0)
  u.zeros();
  for (uword i = 0; i < num_boundary; ++i) {
    u(bnd_ptr[i]) = 1.0;
  }

  const double M = 1.0 / std::sqrt(datum::eps);
  double uu_old = static_cast<double>(num_boundary);

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

    // Build weights with potential acceleration (reuse buffer)
    weights.ones();
    double *wgt_ptr = weights.memptr();
    for (uword i = 0; i < num_interior; ++i) {
      wgt_ptr[int_ptr[i]] = M;
    }

    // Apply acceleration to stuck negative boundary observations (direct access)
    if (convergence_is_stuck && iter > 3) {
      const double *xbd_p1_ptr = xbd_prev1.memptr();
      const double *xbd_p2_ptr = xbd_prev2.memptr();
      const double neg_tol = -0.1 * params.sep_tol;
      for (uword i = 0; i < num_boundary; ++i) {
        uword idx = bnd_ptr[i];
        double xb = xbd_p1_ptr[idx];
        double xb_p1 = xbd_p2_ptr[idx];
        if (xb < neg_tol && xb_p1 < 1.01 * xb) {
          wgt_ptr[idx] = acceleration_value;
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

    // Count candidates for separation (direct access, no temporary)
    uword num_candidates = 0;
    {
      const double *xbd_ptr = xbd.memptr();
      for (uword i = 0; i < num_boundary; ++i) {
        if (xbd_ptr[bnd_ptr[i]] > delta)
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

    // Enforce constraints on interior (direct access)
    double *xbd_ptr = xbd.memptr();
    for (uword i = 0; i < num_interior; ++i) {
      xbd_ptr[int_ptr[i]] = 0.0;
    }

    // Extract boundary_xbd and zero out near-zero values (direct access)
    const double neg_delta = -0.1 * delta;
    for (uword i = 0; i < num_boundary; ++i) {
      uword idx = bnd_ptr[i];
      double val = xbd_ptr[idx];
      if (val > neg_delta && val < delta) {
        xbd_ptr[idx] = 0.0;
        val = 0.0;
      }
      boundary_xbd(i) = xbd_ptr[idx];
    }

    // Check separation - all non-negative means we found it
    bool all_nonneg = true;
    for (uword i = 0; i < num_boundary && all_nonneg; ++i) {
      if (boundary_xbd(i) < 0) all_nonneg = false;
    }
    if (all_nonneg) {
      const uvec sep_ind_local = find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    resid.clean(params.sep_zero_tol);
    
    // Check boundary residuals (direct access)
    const double *resid_ptr = resid.memptr();
    double min_bnd_resid = datum::inf;
    for (uword i = 0; i < num_boundary; ++i) {
      double r = resid_ptr[bnd_ptr[i]];
      if (r < min_bnd_resid) min_bnd_resid = r;
    }

    if (min_bnd_resid >= 0) {
      // Zero out boundary obs with positive residuals
      for (uword i = 0; i < num_boundary; ++i) {
        uword idx = bnd_ptr[i];
        if (resid_ptr[idx] > delta) {
          xbd_ptr[idx] = 0.0;
        }
        boundary_xbd(i) = xbd_ptr[idx];
      }
      const uvec sep_ind_local = find(boundary_xbd > 0);
      result.separated_obs = boundary_sample.elem(sep_ind_local);
      result.num_separated = result.separated_obs.n_elem;
      result.support = xbd;
      result.converged = true;
      result.iterations = iter + 1;
      return result;
    }

    // ReLU update: u = max(xbd, 0) on boundary (direct access)
    u.zeros();
    double *u_ptr = u.memptr();
    for (uword i = 0; i < num_boundary; ++i) {
      u_ptr[bnd_ptr[i]] = std::max(boundary_xbd(i), 0.0);
    }

    const double uu = dot(u, u);
    if (std::abs(uu - uu_old) / (1.0 + uu_old) < params.sep_tol * 0.01) {
      result.iterations = iter + 1;
      break;
    }
    uu_old = uu;
  }

  if (!result.converged) {
    result.iterations = params.sep_max_iter;
    // Extract final boundary_xbd (direct access)
    const double *xbd_ptr = xbd.memptr();
    for (uword i = 0; i < num_boundary; ++i) {
      boundary_xbd(i) = xbd_ptr[bnd_ptr[i]];
    }
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
