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
  uvec working_obs_idx; // Indices of observations used in working sample
                        // (0-based into input data)

  InferenceAPPML(uword n, uword p)
      : InferenceGLM(n, p), expectile(0.5), iter_outer(0), conv_outer(false),
        objective_function(0.0), negative_residuals_share(0.0), residuals(n),
        appml_weights(n) {}
};

// Helper function: Run APPML iteration on given data
// Returns the result after convergence or max iterations
inline InferenceAPPML
appml_iterate(const mat &X, const vec &y, const vec &w, const FlatFEMap &fe_map,
              const CapybaraParameters &params, const vec &offset,
              const vec &beta_start, const vec &eta_start, const vec &mu_start,
              bool suppress_intercept, bool has_intercept_column) {

  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const bool has_offset = (offset.n_elem == n);
  const double tau = params.expectile;
  const double tol = params.expectile_tol;
  const uword max_iter = params.expectile_iter_max;
  const bool trace = params.expectile_trace;

  // Hybrid convergence criterion: relative tolerance scaled by coefficient
  // magnitude, with absolute floor to handle FMA rounding differences across
  // platforms (especially macOS)
  const double abs_tol_floor = 1e-14;

  InferenceAPPML result(n, p);
  result.expectile = tau;

  const vec *offset_ptr = has_offset ? &offset : nullptr;

  // Create params copy with separation DISABLED for iteration fits
  CapybaraParameters iter_params = params;
  iter_params.check_separation = false;

  // In single-step mode each outer iteration does exactly one Newton step, so
  // APPML asymmetric weights (which depend on current residuals) are updated
  // after every step rather than only after the inner GLM converges.
  const bool single_step = (params.expectile_glm_iter_max > 0);
  if (single_step) {
    iter_params.iter_max = params.expectile_glm_iter_max;
  }

  // Final fits (for vcov computation) always run to full convergence.
  CapybaraParameters final_params = iter_params;
  if (single_step) {
    final_params.iter_max = params.iter_max;
  }

  // Initialize from starting values
  vec beta_coef = beta_start;
  vec beta_old = beta_start;
  vec eta_work = eta_start;
  vec mu_work = mu_start;

  // Replace non-finite values with reasonable defaults
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

  // Compute initial residuals and asymmetric weights
  vec residuals_work = y - mu_work;
  vec appml_w(n);
  for (uword i = 0; i < n; ++i) {
    appml_w(i) = std::abs(tau - static_cast<double>(residuals_work(i) < 0.0));
  }

  // Combined weights
  vec combined_w = w % appml_w;

  // Copy FE map and update weights
  FlatFEMap fe_map_iter = fe_map;
  fe_map_iter.update_weights(combined_w);

  double cv = std::numeric_limits<double>::infinity();

  // Track cumulative Fisher Scoring iterations across all feglm_fit calls
  uword total_fs_iter = 0;

  if (trace) {
    cpp4r::message("\n");
  }

  // Outer iteration: iterative reweighting
  for (uword iter = 0; iter < max_iter; ++iter) {
    result.iter_outer = iter + 1;

    GlmWorkspace iter_ws;
    iter_ws.ensure_size(n, p);
    mat X_iter = X;
    fe_map_iter.update_weights(combined_w);

    // Fit weighted Poisson
    InferenceGLM glm_fit = feglm_fit(
        beta_coef, eta_work, y, X_iter, combined_w, 0.0, POISSON, fe_map_iter,
        iter_params, &iter_ws, nullptr, offset_ptr, true, nullptr, nullptr,
        true, suppress_intercept, has_intercept_column);

    total_fs_iter += glm_fit.iter;

    // In single-step mode conv=false is expected after each step (one Newton
    // step is never enough to satisfy the GLM stopping rule); only bail on
    // actual numerical failures (non-finite fitted values).
    const bool numeric_failure =
        !glm_fit.conv && !glm_fit.fitted_values.is_finite();
    if (!glm_fit.conv && (!single_step || numeric_failure)) {
      if (trace) {
        cpp4r::message("APPML: Inner fit failed at iteration %lu\n",
                       static_cast<unsigned long>(iter + 1));
      }
      if (iter > 0) {
        result.conv_outer = false;
        result.iter = total_fs_iter;
        return result;
      }
      static_cast<InferenceGLM &>(result) = std::move(glm_fit);
      result.conv = false;
      result.conv_outer = false;
      result.iter = total_fs_iter;
      return result;
    }

    vec beta_new = glm_fit.coef_table.col(0);
    vec diff_b = beta_new - beta_old;
    cv = dot(diff_b, diff_b);

    if (trace) {
      cpp4r::message("Iteration %lu: objective function = %.6e\n",
                     static_cast<unsigned long>(iter + 1), cv);
    }

    // Hybrid convergence criterion: relative to coefficient scale, with
    // absolute floor
    double beta_scale = std::max(norm(beta_coef, 2), 1.0);
    double cv_threshold = std::max(abs_tol_floor * abs_tol_floor,
                                   (tol * beta_scale) * (tol * beta_scale));

    if (cv <= cv_threshold) {
      // Converged - do final fit with vcov
      GlmWorkspace final_ws;
      final_ws.ensure_size(n, p);
      mat X_final = X;
      vec eta_final = glm_fit.eta;
      fe_map_iter.update_weights(combined_w);

      InferenceGLM final_fit = feglm_fit(
          beta_new, eta_final, y, X_final, combined_w, 0.0, POISSON,
          fe_map_iter, final_params, &final_ws, nullptr, offset_ptr, true,
          nullptr, nullptr, false, suppress_intercept, has_intercept_column);

      total_fs_iter += final_fit.iter;

      static_cast<InferenceGLM &>(result) = std::move(final_fit);
      result.iter = total_fs_iter; // Set cumulative iterations
      result.conv_outer = true;
      result.objective_function = cv;

      // Compute final statistics
      result.residuals = y - result.fitted_values;
      uword neg_count = 0;
      for (uword i = 0; i < n; ++i) {
        if (result.residuals(i) < 0.0)
          neg_count++;
      }
      result.appml_weights = appml_w;
      result.negative_residuals_share =
          static_cast<double>(neg_count) / static_cast<double>(n);

      if (trace) {
        cpp4r::message("\nAPPML converged after %lu iterations\n",
                       static_cast<unsigned long>(iter + 1));
        cpp4r::message("Tolerance = %.2e, Objective = %.6e\n", tol, cv);
        cpp4r::message("%% negative residuals = %.3f%%\n",
                       100.0 * result.negative_residuals_share);
        cpp4r::message("Expectile = %.3f\n", tau);
      }

      return result;
    }

    // Update for next iteration
    beta_old = beta_new;
    beta_coef = beta_new;
    eta_work = glm_fit.eta;
    mu_work = glm_fit.fitted_values;

    residuals_work = y - mu_work;
    for (uword i = 0; i < n; ++i) {
      appml_w(i) = std::abs(tau - static_cast<double>(residuals_work(i) < 0.0));
    }
    combined_w = w % appml_w;
  }

  // Max iterations reached
  if (trace) {
    cpp4r::message(
        "\nAPPML: Max iterations (%lu) reached without convergence\n",
        static_cast<unsigned long>(max_iter));
    cpp4r::message("Final objective function = %.6e\n", cv);
  }

  GlmWorkspace final_ws;
  final_ws.ensure_size(n, p);
  mat X_final = X;
  vec eta_final = eta_work;
  fe_map_iter.update_weights(combined_w);

  InferenceGLM final_fit = feglm_fit(
      beta_coef, eta_final, y, X_final, combined_w, 0.0, POISSON, fe_map_iter,
      final_params, &final_ws, nullptr, offset_ptr, true, nullptr, nullptr,
      false, suppress_intercept, has_intercept_column);

  total_fs_iter += final_fit.iter;

  static_cast<InferenceGLM &>(result) = std::move(final_fit);
  result.iter = total_fs_iter; // Set cumulative iterations
  result.conv_outer = false;
  result.objective_function = cv;

  result.residuals = y - result.fitted_values;
  uword neg_count = 0;
  for (uword i = 0; i < n; ++i) {
    if (result.residuals(i) < 0.0)
      neg_count++;
  }
  result.appml_weights = appml_w;
  result.negative_residuals_share =
      static_cast<double>(neg_count) / static_cast<double>(n);

  return result;
}

// Main APPML fit function
InferenceAPPML fepoisson_asymmetric_fit(mat &X, const vec &y, const vec &w,
                                        const FlatFEMap &fe_map,
                                        const CapybaraParameters &params,
                                        const vec &offset = vec(),
                                        GlmWorkspace *workspace = nullptr,
                                        bool suppress_intercept = false,
                                        bool has_intercept_column = false) {

  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const bool has_offset = (offset.n_elem == n);
  const double tau = params.expectile;
  const bool trace = params.expectile_trace;

  InferenceAPPML result(n, p);
  result.expectile = tau;

  // Workspace allocation for initial fit
  GlmWorkspace init_workspace;
  init_workspace.ensure_size(n, p);

  vec beta_coef(p, fill::zeros);
  const vec *offset_ptr = has_offset ? &offset : nullptr;

  // =========================================================================
  // Step 1: Initial Poisson fit (with or without separation based on params)
  // =========================================================================
  vec eta_empty;
  mat X_init = X;

  InferenceGLM initial_fit =
      feglm_fit(beta_coef, eta_empty, y, X_init, w, 0.0, POISSON, fe_map,
                params, &init_workspace, nullptr, offset_ptr, false, nullptr,
                nullptr, false, suppress_intercept, has_intercept_column);

  if (!initial_fit.conv) {
    static_cast<InferenceGLM &>(result) = std::move(initial_fit);
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // =========================================================================
  // Step 2: Check for separation and branch accordingly
  // =========================================================================
  const bool has_sep =
      initial_fit.has_separation && initial_fit.separated_obs.n_elem > 0;

  if (has_sep && trace) {
    cpp4r::message("Separation found in %lu observation(s)\n",
                   static_cast<unsigned long>(initial_fit.num_separated));
  }

  // For expectile == 0.5, just return standard Poisson result
  if (std::abs(tau - 0.5) < 1e-10) {
    static_cast<InferenceGLM &>(result) = std::move(initial_fit);
    result.conv_outer = result.conv;
    result.iter_outer = 1;
    result.objective_function = 0.0;
    result.residuals = y - result.fitted_values;
    result.appml_weights = vec(n, fill::value(0.5));

    uword neg_count = 0;
    uword valid_count = has_sep ? (n - initial_fit.num_separated) : n;
    for (uword i = 0; i < n; ++i) {
      if (std::isfinite(result.residuals(i)) && result.residuals(i) < 0.0) {
        neg_count++;
      }
    }
    result.negative_residuals_share =
        valid_count > 0
            ? static_cast<double>(neg_count) / static_cast<double>(valid_count)
            : 0.0;

    if (trace) {
      cpp4r::message(
          "APPML: expectile = 0.5, using standard Poisson (no iteration)\n");
      cpp4r::message("%% negative residuals = %.3f%%\n",
                     100.0 * result.negative_residuals_share);
    }

    // Working sample: all non-separated observations
    if (has_sep) {
      uvec keep_mask(n, fill::ones);
      keep_mask.elem(initial_fit.separated_obs).zeros();
      result.working_obs_idx = find(keep_mask == 1);
    } else {
      result.working_obs_idx = regspace<uvec>(0, n - 1);
    }

    return result;
  }

  // =========================================================================
  // Step 3: Run APPML iteration
  // =========================================================================

  if (has_sep) {
    // ==== PATH A: Separation detected - subset data and run APPML ====
    uvec keep_mask(n, fill::ones);
    keep_mask.elem(initial_fit.separated_obs).zeros();
    uvec keep_idx = find(keep_mask == 1);
    const uword n_work = keep_idx.n_elem;

    // Subset data
    vec y_work = y.elem(keep_idx);
    vec w_work = w.elem(keep_idx);
    mat X_work = X.rows(keep_idx);
    vec offset_work = has_offset ? offset.elem(keep_idx) : vec();

    // Rebuild FE map for subsetted data
    FlatFEMap fe_map_work;
    fe_map_work.K = fe_map.K;
    fe_map_work.n_obs = n_work;
    fe_map_work.n_groups.resize(fe_map.K);
    fe_map_work.fe_map.resize(fe_map.K);

    for (uword k = 0; k < fe_map.K; ++k) {
      std::vector<uword> new_map(n_work);
      for (uword i = 0; i < n_work; ++i) {
        new_map[i] = fe_map.fe_map[k][keep_idx(i)];
      }
      fe_map_work.fe_map[k] = std::move(new_map);
      fe_map_work.n_groups[k] = fe_map.n_groups[k];
    }
    fe_map_work.structure_built = true;
    fe_map_work.update_weights(w_work);

    // Extract starting values (subsetted)
    vec beta_start = initial_fit.coef_table.col(0);
    vec eta_start = initial_fit.eta.elem(keep_idx);
    vec mu_start = initial_fit.fitted_values.elem(keep_idx);

    // Run APPML on subsetted data
    InferenceAPPML iter_result = appml_iterate(
        X_work, y_work, w_work, fe_map_work, params, offset_work, beta_start,
        eta_start, mu_start, suppress_intercept, has_intercept_column);

    // Add initial fit iterations to total
    iter_result.iter += initial_fit.iter;

    // Copy separation info from initial fit
    iter_result.has_separation = true;
    iter_result.separated_obs = initial_fit.separated_obs;
    iter_result.num_separated = initial_fit.num_separated;
    iter_result.working_obs_idx = keep_idx;

    return iter_result;

  } else {
    // ==== PATH B: No separation - run APPML on full data ====
    vec beta_start = initial_fit.coef_table.col(0);
    vec eta_start = initial_fit.eta;
    vec mu_start = initial_fit.fitted_values;

    // Run APPML on full data with original FE map
    InferenceAPPML iter_result =
        appml_iterate(X, y, w, fe_map, params, offset, beta_start, eta_start,
                      mu_start, suppress_intercept, has_intercept_column);

    // Add initial fit iterations to total
    iter_result.iter += initial_fit.iter;

    iter_result.working_obs_idx = regspace<uvec>(0, n - 1);

    return iter_result;
  }
}

} // namespace capybara

#endif // CAPYBARA_APPML_H
