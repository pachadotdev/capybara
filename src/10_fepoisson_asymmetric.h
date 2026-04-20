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
  uvec working_obs_idx;  // Indices of observations used in working sample

  InferenceAPPML(uword n, uword p)
      : InferenceGLM(n, p), expectile(0.5), iter_outer(0), conv_outer(false),
        objective_function(0.0), negative_residuals_share(0.0), residuals(n),
        appml_weights(n) {}
};

// APPML fit function: Iterative reweighted Poisson PML for expectile regression
// Based on Newey & Powell (1987) asymmetric least squares
// Ported from Stata's appmlhdfe by Clance & Santos Silva
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

  // Workspace allocation for initial fit
  GlmWorkspace init_workspace;
  init_workspace.ensure_size(n, p);

  vec beta_coef(p, fill::zeros);

  const double tol = params.expectile_tol;
  const uword max_iter = params.expectile_iter_max;
  const bool trace = params.expectile_trace;

  // Following Stata appmlhdfe: if expectile == 0.5, just run standard Poisson
  // No iteration needed - it's equivalent to regular PPML
  if (std::abs(tau - 0.5) < 1e-10) {
    // Let feglm_fit initialize eta properly by passing empty vector
    // This matches how feglm_fit_() in capybara.cpp handles it
    vec eta_empty;  // empty - feglm_fit will initialize based on y
    const vec *offset_ptr = has_offset ? &offset : nullptr;

    InferenceGLM glm_fit =
        feglm_fit(beta_coef, eta_empty, y, X, w, 0.0, POISSON, fe_map, params,
                  &init_workspace, nullptr, offset_ptr, false, nullptr, nullptr,
                  false, suppress_intercept, has_intercept_column);

    static_cast<InferenceGLM &>(result) = std::move(glm_fit);
    result.conv_outer = result.conv;
    result.iter_outer = 1;
    result.objective_function = 0.0;

    // Compute residuals and statistics
    result.residuals = y - result.fitted_values;
    result.appml_weights = vec(n, fill::value(0.5));

    uword neg_count = 0;
    for (uword i = 0; i < n; ++i) {
      // Skip NaN (separated observations)
      if (std::isfinite(result.residuals(i)) && result.residuals(i) < 0.0) {
        neg_count++;
      }
    }
    // Count only non-separated observations
    uword valid_count = n;
    if (result.has_separation && result.num_separated > 0) {
      valid_count = n - result.num_separated;
    }
    result.negative_residuals_share =
        valid_count > 0
            ? static_cast<double>(neg_count) / static_cast<double>(valid_count)
            : 0.0;

    if (trace) {
      Rprintf(
          "APPML: expectile = 0.5, using standard Poisson (no iteration)\n");
      Rprintf("%% negative residuals = %.3f%%\n",
              100.0 * result.negative_residuals_share);
    }

    // For tau == 0.5, working sample is all observations
    result.working_obs_idx = regspace<uvec>(0, n - 1);

    return result;
  }

  // For tau != 0.5: Run iterative reweighted Poisson PML
  // Following the reference R implementation (appml_r2.R), we do NOT use
  // separation detection. The reference uses fixest without separation checking.
  // Separation detection causes numerical issues when we iterate with weights.
  
  const vec *offset_ptr = has_offset ? &offset : nullptr;

  // Create params copy with separation DISABLED for all APPML fits
  // This matches the reference implementation behavior
  CapybaraParameters appml_params = params;
  appml_params.check_separation = false;

  // Initial Poisson fit to get starting values (no separation detection)
  vec eta_init;  // empty - feglm_fit will initialize based on y
  mat X_init = X;

  InferenceGLM initial_fit =
      feglm_fit(beta_coef, eta_init, y, X_init, w, 0.0, POISSON, fe_map, appml_params,
                &init_workspace, nullptr, offset_ptr, false, nullptr, nullptr,
                false, suppress_intercept, has_intercept_column);

  if (!initial_fit.conv) {
    static_cast<InferenceGLM &>(result) = std::move(initial_fit);
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Extract coefficients and fitted values
  vec beta_old = initial_fit.coef_table.col(0);
  vec eta_work = initial_fit.eta;
  vec mu_work = initial_fit.fitted_values;

  // Replace any non-finite values with reasonable defaults
  double y_mean_safe = mean(y) + 0.1;
  double eta_default = std::log(y_mean_safe);

  for (uword i = 0; i < n; ++i) {
    if (!std::isfinite(eta_work(i))) {
      eta_work(i) = eta_default;
    }
    if (!std::isfinite(mu_work(i)) || mu_work(i) <= 0.0) {
      mu_work(i) = y_mean_safe;
    }
  }

  // Compute initial residuals
  vec residuals_all = y - mu_work;

  // Compute asymmetric weights: w_i = |tau - I(r_i < 0)|
  // Following reference: weights = abs(expectile - (residuals < 0))
  vec appml_w(n);
  for (uword i = 0; i < n; ++i) {
    appml_w(i) = std::abs(tau - static_cast<double>(residuals_all(i) < 0.0));
  }

  // Combined weights: original weights * APPML weights
  vec combined_w = w % appml_w;

  // Update FE map with combined weights
  FlatFEMap fe_map_iter = fe_map;
  fe_map_iter.update_weights(combined_w);

  // Initialize convergence tracking
  double cv = std::numeric_limits<double>::infinity();
  beta_coef = beta_old;

  if (trace) {
    Rprintf("\n");
  }

  // Outer iteration: iterative reweighting
  for (uword iter = 0; iter < max_iter; ++iter) {
    result.iter_outer = iter + 1;

    // Use fresh workspace for each iteration
    GlmWorkspace iter_ws;
    iter_ws.ensure_size(n, p);

    // Make a copy of X since feglm_fit may modify it
    mat X_iter = X;

    // Update FE map weights for current iteration
    fe_map_iter.update_weights(combined_w);

    // Fit weighted Poisson model with current APPML weights
    // run_from_negbin=true: Skip vcov computation in inner fit for efficiency
    InferenceGLM glm_fit =
        feglm_fit(beta_coef, eta_work, y, X_iter, combined_w, 0.0, POISSON,
                  fe_map_iter, appml_params, &iter_ws, nullptr, offset_ptr,
                  true, nullptr, nullptr, true, suppress_intercept,
                  has_intercept_column);

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

    // Compute convergence criterion using squared coefficient change
    vec diff_b = beta_new - beta_old;
    cv = dot(diff_b, diff_b);

    if (trace) {
      Rprintf("Iteration %lu: objective function = %.6e\n",
              static_cast<unsigned long>(iter + 1), cv);
    }

    // Check convergence
    if (cv <= tol) {
      // Converged - do final fit with vcov computation on full data
      GlmWorkspace final_ws;
      final_ws.ensure_size(n, p);
      mat X_final = X;
      vec eta_final = glm_fit.eta;

      // Update FE map for final fit
      fe_map_iter.update_weights(combined_w);

      InferenceGLM final_fit =
          feglm_fit(beta_new, eta_final, y, X_final, combined_w, 0.0,
                    POISSON, fe_map_iter, appml_params, &final_ws, nullptr,
                    offset_ptr, true, nullptr, nullptr, false,
                    suppress_intercept, has_intercept_column);

      static_cast<InferenceGLM &>(result) = std::move(final_fit);
      result.conv_outer = true;
      result.objective_function = cv;

      // Compute final residuals
      result.residuals = y - result.fitted_values;
      uword neg_count = 0;
      for (uword i = 0; i < n; ++i) {
        if (result.residuals(i) < 0.0) {
          neg_count++;
        }
      }
      result.appml_weights = appml_w;
      result.negative_residuals_share = static_cast<double>(neg_count) / static_cast<double>(n);

      if (trace) {
        Rprintf("\nAPPML converged after %lu iterations\n",
                static_cast<unsigned long>(iter + 1));
        Rprintf("Tolerance = %.2e, Objective = %.6e\n", tol, cv);
        Rprintf("%% negative residuals = %.3f%%\n",
                100.0 * result.negative_residuals_share);
        Rprintf("Expectile = %.3f\n", tau);
      }

      // All observations are in working sample (weights handle separation)
      result.working_obs_idx = regspace<uvec>(0, n - 1);

      return result;
    }

    // Update for next iteration
    beta_old = beta_new;
    beta_coef = beta_new;
    eta_work = glm_fit.eta;
    mu_work = glm_fit.fitted_values;

    // Update residuals and weights
    residuals_all = y - mu_work;
    for (uword i = 0; i < n; ++i) {
      appml_w(i) = std::abs(tau - static_cast<double>(residuals_all(i) < 0.0));
    }
    combined_w = w % appml_w;
  }

  // Max iterations reached without convergence
  if (trace) {
    Rprintf("\nAPPML: Max iterations (%lu) reached without convergence\n",
            static_cast<unsigned long>(max_iter));
    Rprintf("Final objective function = %.6e\n", cv);
  }

  // Do final full fit to populate results with vcov
  GlmWorkspace final_ws;
  final_ws.ensure_size(n, p);
  mat X_final = X;
  vec eta_final = eta_work;
  fe_map_iter.update_weights(combined_w);

  InferenceGLM final_fit =
      feglm_fit(beta_coef, eta_final, y, X_final, combined_w, 0.0, POISSON,
                fe_map_iter, appml_params, &final_ws, nullptr, offset_ptr,
                true, nullptr, nullptr, false, suppress_intercept,
                has_intercept_column);

  static_cast<InferenceGLM &>(result) = std::move(final_fit);
  result.conv_outer = false;
  result.objective_function = cv;

  // Compute final residuals
  result.residuals = y - result.fitted_values;
  uword neg_count = 0;
  for (uword i = 0; i < n; ++i) {
    if (result.residuals(i) < 0.0) {
      neg_count++;
    }
  }
  result.appml_weights = appml_w;
  result.negative_residuals_share = static_cast<double>(neg_count) / static_cast<double>(n);

  // All observations are in working sample
  result.working_obs_idx = regspace<uvec>(0, n - 1);

  return result;
}

} // namespace capybara

#endif // CAPYBARA_APPML_H
