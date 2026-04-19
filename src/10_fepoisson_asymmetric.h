// Asymmetric Poisson Pseudo-Maximum Likelihood (APPML) via expectile regression

#ifndef CAPYBARA_APPML_H
#define CAPYBARA_APPML_H

namespace capybara {

struct InferenceAPPML : public InferenceGLM {
  double expectile;
  uword iter_outer;
  bool conv_outer;
  double objective_function;
  double negative_residuals_share;
  vec residuals;
  vec appml_weights;

  InferenceAPPML(uword n, uword p)
      : InferenceGLM(n, p), expectile(0.5), iter_outer(0), conv_outer(false),
        objective_function(0.0), negative_residuals_share(0.0), residuals(n),
        appml_weights(n) {}
};

// APPML fit function: Iterative reweighted Poisson PML for expectile regression
// Based on Newey & Powell (1987) asymmetric least squares
InferenceAPPML fepoisson_asymmetric_fit(
    mat &X, const vec &y, const vec &w, const FlatFEMap &fe_map,
    const CapybaraParameters &params, const vec &offset = vec(),
    GlmWorkspace *workspace = nullptr, bool suppress_intercept = false,
    bool has_intercept_column = false) {

  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const bool has_offset = (offset.n_elem == n);
  const double tau = params.expectile;

  InferenceAPPML result(n, p);
  result.expectile = tau;

  // Workspace allocation - reuse if provided
  GlmWorkspace local_workspace;
  GlmWorkspace &ws = workspace ? *workspace : local_workspace;
  ws.ensure_size(n, p);

  // Initialize eta: use offset if provided, otherwise zeros
  vec eta = has_offset ? offset : vec(n, fill::zeros);
  vec beta_coef(p, fill::zeros);

  // Initial unweighted Poisson fit to get starting values
  InferenceGLM initial_fit =
      feglm_fit(beta_coef, eta, y, X, w, 0.0, POISSON, fe_map, params, &ws,
                nullptr, nullptr, false, nullptr, nullptr, true,
                suppress_intercept, has_intercept_column);

  if (!initial_fit.conv) {
    static_cast<InferenceGLM &>(result) = std::move(initial_fit);
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Extract coefficients and fitted values
  beta_coef = initial_fit.coef_table.col(0);
  eta = initial_fit.eta;
  vec mu = initial_fit.fitted_values;

  // Compute initial residuals and weights
  vec residuals = y - mu;
  vec appml_w(n);

  // Compute asymmetric weights: w_i = |tau - I(r_i < 0)|
  // For tau = 0.5, this is 0.5 for all observations (symmetric)
  // For tau < 0.5, negative residuals get weight tau, positive get (1-tau)
  // For tau > 0.5, negative residuals get weight tau, positive get (1-tau)
  for (uword i = 0; i < n; ++i) {
    appml_w(i) = std::abs(tau - static_cast<double>(residuals(i) < 0.0));
  }

  // Combine with original observation weights
  vec combined_w = w % appml_w;

  // Update FE map weights for combined weights
  FlatFEMap fe_map_copy = fe_map;
  if (fe_map_copy.structure_built) {
    fe_map_copy.update_weights(combined_w);
  }

  // Initialize convergence tracking
  double cv = std::numeric_limits<double>::infinity();
  vec beta_old; // Will be initialized after first iteration

  // Outer iteration: iterative reweighting
  const double tol = params.expectile_tol;
  const uword max_iter = params.expectile_iter_max;
  const bool trace = params.expectile_trace;

  for (uword iter = 0; iter < max_iter; ++iter) {
    result.iter_outer = iter + 1;

    // Fit weighted Poisson model with current APPML weights
    // Note: run_from_negbin=false to compute vcov (needed for convergence check)
    InferenceGLM glm_fit =
        feglm_fit(beta_coef, eta, y, X, combined_w, 0.0, POISSON, fe_map_copy,
                  params, &ws, nullptr, nullptr, false, nullptr, nullptr, false,
                  suppress_intercept, has_intercept_column);

    if (!glm_fit.conv) {
      if (trace) {
        Rprintf("APPML: Inner fit failed at iteration %lu\n",
                static_cast<unsigned long>(iter + 1));
      }
      // Return last successful result if available
      if (iter > 0) {
        result.conv_outer = false;
        return result;
      }
      static_cast<InferenceGLM &>(result) = std::move(glm_fit);
      result.conv = false;
      result.conv_outer = false;
      return result;
    }

    // Extract new coefficients
    vec beta_new = glm_fit.coef_table.col(0);

    // Initialize beta_old on first iteration
    if (iter == 0) {
      beta_old = zeros<vec>(beta_new.n_elem);
    }

    // Compute convergence criterion: (b - b_old)' * V^{-1} * (b - b_old)
    // Use vcov from the fit
    mat V = glm_fit.vcov;
    
    // Check dimensions match
    if (V.n_rows != beta_new.n_elem || V.n_cols != beta_new.n_elem) {
      if (trace) {
        Rprintf("APPML: vcov dimension mismatch at iteration %lu\n",
                static_cast<unsigned long>(iter + 1));
      }
      // Fallback: use simple coefficient norm change
      vec diff_b = beta_new - beta_old;
      cv = dot(diff_b, diff_b);
    } else {
      vec diff_b = beta_new - beta_old;

      // Try to invert V; if singular, use pseudo-inverse or fallback
      mat invV;
      bool inv_success = inv(invV, V);
      if (!inv_success) {
        // Fallback: use pinv for near-singular matrices
        invV = pinv(V);
      }

      cv = as_scalar(diff_b.t() * invV * diff_b);
    }

    if (trace) {
      Rprintf("APPML iteration %lu: objective function = %.6e\n",
              static_cast<unsigned long>(iter + 1), cv);
    }

    // Check convergence
    if (cv <= tol) {
      // Converged - glm_fit already has vcov computed (run_from_negbin=false)
      static_cast<InferenceGLM &>(result) = std::move(glm_fit);
      result.conv_outer = true;
      result.objective_function = cv;

      // Compute final residuals and statistics
      result.residuals = y - result.fitted_values;
      result.appml_weights = appml_w;

      // Compute share of negative residuals
      uword neg_count = 0;
      for (uword i = 0; i < n; ++i) {
        if (result.residuals(i) < 0.0) {
          neg_count++;
        }
      }
      result.negative_residuals_share =
          static_cast<double>(neg_count) / static_cast<double>(n);

      if (trace) {
        Rprintf("\nAPPML converged after %lu iterations\n",
                static_cast<unsigned long>(iter + 1));
        Rprintf("Tolerance = %.2e, Objective = %.6e\n", tol, cv);
        Rprintf("%% negative residuals = %.3f%%\n",
                100.0 * result.negative_residuals_share);
        Rprintf("Expectile = %.3f\n", tau);
      }

      return result;
    }

    // Update for next iteration
    beta_old = beta_new;
    beta_coef = beta_new;
    eta = glm_fit.eta;
    mu = glm_fit.fitted_values;

    // Update residuals and weights
    residuals = y - mu;
    for (uword i = 0; i < n; ++i) {
      appml_w(i) = std::abs(tau - static_cast<double>(residuals(i) < 0.0));
    }
    combined_w = w % appml_w;

    // Update FE map weights
    if (fe_map_copy.structure_built) {
      fe_map_copy.update_weights(combined_w);
    }
  }

  // Max iterations reached without convergence
  if (trace) {
    Rprintf("\nAPPML: Max iterations (%lu) reached without convergence\n",
            static_cast<unsigned long>(max_iter));
    Rprintf("Final objective function = %.6e\n", cv);
  }

  // Do final full fit to populate results
  InferenceGLM final_fit =
      feglm_fit(beta_coef, eta, y, X, combined_w, 0.0, POISSON, fe_map_copy,
                params, &ws, nullptr, nullptr, false, nullptr, nullptr, false,
                suppress_intercept, has_intercept_column);

  static_cast<InferenceGLM &>(result) = std::move(final_fit);
  result.conv_outer = false;
  result.objective_function = cv;

  // Compute final residuals and statistics
  result.residuals = y - result.fitted_values;
  result.appml_weights = appml_w;

  uword neg_count = 0;
  for (uword i = 0; i < n; ++i) {
    if (result.residuals(i) < 0.0) {
      neg_count++;
    }
  }
  result.negative_residuals_share =
      static_cast<double>(neg_count) / static_cast<double>(n);

  return result;
}

} // namespace capybara

#endif // CAPYBARA_APPML_H
