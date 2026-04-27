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
  vec boundary_xbd; // sized to num_boundary

  // WLS solver buffers (avoid allocations in hot loop)
  mat Xw;         // n * p weighted design matrix
  mat XtWX;       // p * p normal equations matrix
  vec XtWy;       // p * 1 RHS of normal equations
  mat R;          // p * p Cholesky factor
  vec z_wls;      // p * 1 intermediate solve vector
  vec beta;       // p * 1 coefficients
  uvec excluded;  // p * 1 excluded columns mask
  uword cached_p; // cached number of columns

  SeparationReluWorkspace() : cached_p(0) {}

  void ensure_size(uword n, uword num_boundary, uword p = 0) {
    if (xbd.n_elem != n) {
      // Use zeros() for deterministic initialization
      xbd.zeros(n);
      xbd_prev1.zeros(n);
      xbd_prev2.zeros(n);
      resid.zeros(n);
      u.zeros(n);
      weights.zeros(n);
    }
    if (boundary_xbd.n_elem != num_boundary) {
      boundary_xbd.zeros(num_boundary);
    }
    // Allocate WLS buffers if needed
    if (p > 0 && cached_p != p) {
      XtWX.zeros(p, p);
      XtWy.zeros(p);
      R.zeros(p, p);
      z_wls.zeros(p);
      beta.zeros(p);
      excluded.zeros(p);
      cached_p = p;
    }
  }
};

// Inlined WLS solver using workspace buffers (avoids allocations)
inline void solve_wls_inplace(const mat &X, const vec &y, const vec &w,
                              vec &residuals, SeparationReluWorkspace &ws) {
  const uword p = X.n_cols;
  if (p == 0) {
    residuals = y;
    return;
  }

  // Reuse workspace buffers
  mat &Xw = ws.Xw;
  mat &XtWX = ws.XtWX;
  vec &XtWy = ws.XtWy;
  mat &R = ws.R;
  vec &z = ws.z_wls;
  vec &beta = ws.beta;
  uvec &excluded = ws.excluded;

  // Resize Xw only if dimensions changed (common case: same size)
  if (Xw.n_rows != X.n_rows || Xw.n_cols != p) {
    Xw.set_size(X.n_rows, p);
  }

  // Form weighted design matrix
  Xw = X.each_col() % w;

  // Form normal equations: X'WX and X'Wy
  XtWX = X.t() * Xw;
  XtWy = Xw.t() * y;

  // Rank-revealing Cholesky
  uword rank;
  chol_rank(R, excluded, rank, XtWX, "upper");

  beta.zeros();

  if (rank == p) {
    // Full rank: solve via back-substitution
    solve(z, trimatl(R.t()), XtWy, solve_opts::fast);
    solve(beta, trimatu(R), z, solve_opts::fast);
  } else if (rank > 0) {
    // Rank-deficient: solve on non-excluded columns
    const uvec included = find(excluded == 0);
    if (included.n_elem > 0) {
      const mat R_sub = R.submat(included, included);
      const vec XtWy_sub = XtWy.elem(included);
      vec z_sub;
      solve(z_sub, trimatl(R_sub.t()), XtWy_sub, solve_opts::fast);
      vec beta_sub;
      solve(beta_sub, trimatu(R_sub), z_sub, solve_opts::fast);
      beta.elem(included) = beta_sub;
    }
  }

  residuals = y - X * beta;
}

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

  const uword p = X.n_cols;

  // Use workspace if provided, otherwise create local buffers
  SeparationReluWorkspace local_ws;
  SeparationReluWorkspace &work = ws ? *ws : local_ws;
  work.ensure_size(n, num_boundary, p);

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

    // Apply acceleration to stuck negative boundary observations
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

    // Use inlined WLS solver with workspace buffers
    solve_wls_inplace(X, u, weights, resid, work);
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
      if (boundary_xbd(i) < 0)
        all_nonneg = false;
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
      if (r < min_bnd_resid)
        min_bnd_resid = r;
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
