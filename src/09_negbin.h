// Negative binomial as a special case of Poisson GLM

#ifndef CAPYBARA_NEGBIN_H
#define CAPYBARA_NEGBIN_H

namespace capybara {

struct InferenceNegBin : public InferenceGLM {
  double theta;
  uword iter_outer;
  bool conv_outer;

  InferenceNegBin(uword n, uword p)
      : InferenceGLM(n, p), theta(1.0), iter_outer(0), conv_outer(false) {}
};

// Estimate theta using method of moments with Armadillo vectorized operations
// Uses fitted values (mu) from current iteration for stable estimation
// theta = mean(mu)^2 / (var(y) - mean(mu)), clamped to [theta_min, theta_max]
// When mu is not provided, falls back to using y as initial estimate
inline double estimate_theta(const vec &y, const vec &mu = vec(),
                             const double theta_min = 0.1,
                             const double theta_max = 1.0e6,
                             const double overdispersion_threshold = 0.01,
                             const double regularization = 1.0e-6) {
  const double n = static_cast<double>(y.n_elem);
  const double n_inv = 1.0 / n;

  // Use mu if provided, otherwise use y for initial estimate
  const bool has_mu = (mu.n_elem == y.n_elem);
  const double mu_mean = has_mu ? accu(mu) * n_inv : accu(y) * n_inv;

  // Variance of y
  const double y_mean = accu(y) * n_inv;
  const vec y_centered = y - y_mean;
  const double y_var = dot(y_centered, y_centered) / (n - 1.0);

  // Overdispersion: var(y) - mean(mu), with regularization for stability
  const double overdispersion = y_var - mu_mean + regularization;

  // Low overdispersion -> return very large theta (Poisson-like)
  if (overdispersion <=
      overdispersion_threshold * std::abs(mu_mean) + regularization) {
    return theta_max;
  }

  // Ensure mu_mean is positive for stability
  const double mu_mean_safe = std::max(mu_mean, regularization);

  // Method of moments: theta = mean(mu)^2 / (var(y) - mean(mu))
  const double theta_raw = mu_mean_safe * mu_mean_safe / overdispersion;

  return std::clamp(theta_raw, theta_min, theta_max);
}

InferenceNegBin fenegbin_fit(mat &X, const vec &y, const vec &w,
                             const FlatFEMap &fe_map,
                             const CapybaraParameters &params,
                             const vec &offset = vec(), double init_theta = 0.0,
                             GlmWorkspace *workspace = nullptr,
                             bool suppress_intercept = false,
                             bool has_intercept_column = false) {
  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const bool has_offset = (offset.n_elem == n);

  InferenceNegBin result(n, p);

  // Workspace allocation - reuse if provided
  GlmWorkspace local_workspace;
  GlmWorkspace &ws = workspace ? *workspace : local_workspace;
  ws.ensure_size(n, p);

  // Initialize eta: use offset if provided, otherwise zeros
  // Armadillo's conditional copy is efficient
  vec eta = has_offset ? offset : vec(n, fill::zeros);
  vec beta_coef(p, fill::zeros);

  // Initial Poisson fit to get good starting values
  InferenceGLM poisson_fit =
      feglm_fit(beta_coef, eta, y, X, w, 0.0, POISSON, fe_map, params, &ws,
                nullptr, nullptr, false, nullptr, nullptr, true,
                suppress_intercept, has_intercept_column);

  if (!poisson_fit.conv) {
    static_cast<InferenceGLM &>(result) = std::move(poisson_fit);
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Extract coefficients and linear predictor - use move where possible
  beta_coef = std::move(poisson_fit.coef_table.col(0));
  eta = std::move(poisson_fit.eta);

  // Compute initial mu from eta for theta estimation
  vec mu = exp(eta);

  // Estimate initial theta from y and mu statistics
  double theta = (init_theta > 0.0) ? init_theta : estimate_theta(y, mu);
  double theta_prev = theta;

  // Step dampening factor for theta updates (prevents oscillation on Mac BLAS)
  const double theta_dampening = 0.7;

  // Outer iteration: alternate GLM fit and theta update
  const double tol = params.dev_tol;
  const double tol_denom = 0.1; // Regularization for relative tolerance

  for (uword iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;
    theta_prev = theta;

    // Fit negative binomial GLM with current theta
    InferenceGLM glm_fit =
        feglm_fit(beta_coef, eta, y, X, w, theta, NEG_BIN, fe_map, params, &ws,
                  nullptr, nullptr, false, nullptr, nullptr, true,
                  suppress_intercept, has_intercept_column);

    if (!glm_fit.conv) {
      static_cast<InferenceGLM &>(result) = std::move(glm_fit);
      result.theta = theta;
      result.conv_outer = false;
      return result;
    }

    // Update mu from fitted eta for theta estimation
    mu = exp(glm_fit.eta);

    // Update theta estimate from current fit using both y and mu
    double theta_estimated = estimate_theta(y, mu);

    // Apply step dampening to prevent oscillation across platforms
    double theta_new = theta_dampening * theta_estimated +
                       (1.0 - theta_dampening) * theta_prev;

    // Validate theta estimate
    if (theta_new <= 0.0 || !std::isfinite(theta_new)) {
      theta_new = theta;
    }

    // Scale-invariant convergence: relative change in beta + theta
    // Green & Santos Silva 2025: deviance is scale-dependent for PPML.
    // Beta is fully scale-invariant, so we track beta instead of eta.
    const vec beta_new = glm_fit.coef_table.col(0);
    const double beta_norm_nb = std::sqrt(dot(beta_new, beta_new));
    const double beta_crit =
        std::sqrt(dot(beta_new - beta_coef, beta_new - beta_coef)) /
        std::max(beta_norm_nb, datum::eps);
    const double theta_crit =
        std::abs(theta_new - theta_prev) / (tol_denom + std::abs(theta_prev));

    // On ARM (Accelerate BLAS), exp() rounding via FMA can cause theta to
    // oscillate at machine-epsilon amplitude, never crossing tol.  If the
    // absolute change in theta is below the representable precision for this
    // magnitude, further iterations are numerically pointless.
    const bool theta_at_machine_eps =
        std::abs(theta_new - theta_prev) <=
        datum::eps * std::max(std::abs(theta_prev), 1.0);

    // Hybrid convergence criterion for cross-platform stability.
    // Add absolute floor (1e-13) to handle FMA rounding on Mac/ARM:
    // Platform buffer prevents false non-convergence when relative changes
    // oscillate just above/below the threshold due to floating-point
    // differences.
    const double abs_tol_floor_negbin = 1e-13;
    const double beta_threshold = std::max(abs_tol_floor_negbin, tol);
    const double theta_threshold = std::max(abs_tol_floor_negbin, tol);

    if (beta_crit <= beta_threshold &&
        (theta_crit <= theta_threshold || theta_at_machine_eps)) {
      // Converged - do one final full fit (run_from_negbin=false) to
      // compute Hessian, vcov, FE recovery, SE/z/p, R-sq, etc.
      beta_coef = glm_fit.coef_table.col(0);
      eta = glm_fit.eta;
      InferenceGLM final_fit =
          feglm_fit(beta_coef, eta, y, X, w, theta_new, NEG_BIN, fe_map, params,
                    &ws, nullptr, nullptr, false, nullptr, nullptr, false,
                    suppress_intercept, has_intercept_column);
      static_cast<InferenceGLM &>(result) = std::move(final_fit);
      result.theta = theta_new;
      result.conv_outer = true;
      return result;
    }

    // Update for next iteration
    theta = theta_new;

    // Move coefficients for warm start (avoid copy)
    beta_coef = std::move(glm_fit.coef_table.col(0));
    eta = std::move(glm_fit.eta);
  }

  // Max iterations reached without convergence
  // Do a final full fit to populate Hessian, vcov, FE, SE/z/p
  {
    InferenceGLM final_fit =
        feglm_fit(beta_coef, eta, y, X, w, theta, NEG_BIN, fe_map, params, &ws,
                  nullptr, nullptr, false, nullptr, nullptr, false,
                  suppress_intercept, has_intercept_column);
    static_cast<InferenceGLM &>(result) = std::move(final_fit);
  }
  result.theta = theta;
  result.conv_outer = false;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_NEGBIN_H
