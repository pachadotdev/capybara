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
// theta = mean^2 / (variance - mean), clamped to [theta_min, theta_max]
inline double estimate_theta(const vec &y, const double theta_min = 0.1,
                             const double theta_max = 1.0e6,
                             const double overdispersion_threshold = 0.01) {
  // Compute mean and variance in a single pass using Armadillo's optimized
  // routines

  const double n_inv = 1.0 / static_cast<double>(y.n_elem);
  const double y_mean = accu(y) * n_inv;

  // Variance: E[(y - mean)^2] using vectorized squared difference
  // dot(y - y_mean, y - y_mean) is faster than accu(square(y - y_mean))
  // for the variance computation
  const vec y_centered = y - y_mean;
  const double y_var = dot(y_centered, y_centered) / (y.n_elem - 1);

  const double overdispersion = y_var - y_mean;

  // Low overdispersion -> return very large theta (Poisson-like)
  if (overdispersion <= overdispersion_threshold * y_mean) {
    return theta_max;
  }

  // Method of moments: theta = mean^2 / (var - mean)
  return std::clamp(y_mean * y_mean / overdispersion, theta_min, theta_max);
}

InferenceNegBin fenegbin_fit(mat &X, const vec &y, const vec &w,
                             const FlatFEMap &fe_map,
                             const CapybaraParameters &params,
                             const vec &offset = vec(), double init_theta = 0.0,
                             GlmWorkspace *workspace = nullptr) {
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
      feglm_fit(beta_coef, eta, y, X, w, 0.0, POISSON, fe_map, params, &ws);

  if (!poisson_fit.conv) {
    static_cast<InferenceGLM &>(result) = std::move(poisson_fit);
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Extract coefficients and linear predictor - use move where possible
  beta_coef = std::move(poisson_fit.coef_table.col(0));
  eta = std::move(poisson_fit.eta);

  // Estimate initial theta from y statistics (no mu needed)
  double theta = (init_theta > 0.0) ? init_theta : estimate_theta(y);
  double theta_prev = theta;

  // Outer iteration: alternate GLM fit and theta update
  const double tol = params.dev_tol;
  const double tol_denom = 0.1; // Regularization for relative tolerance

  for (uword iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;
    theta_prev = theta;

    // Fit negative binomial GLM with current theta
    InferenceGLM glm_fit =
        feglm_fit(beta_coef, eta, y, X, w, theta, NEG_BIN, fe_map, params, &ws);

    if (!glm_fit.conv) {
      static_cast<InferenceGLM &>(result) = std::move(glm_fit);
      result.theta = theta;
      result.conv_outer = false;
      return result;
    }

    // Update theta estimate from current fit
    double theta_new = estimate_theta(y);

    // Validate theta estimate
    if (theta_new <= 0.0 || !std::isfinite(theta_new)) {
      theta_new = theta;
    }

    // Scale-invariant convergence: relative change in eta + theta
    // Green & Santos Silva 2025: deviance is scale-dependent for PPML
    const uword n_nb = glm_fit.eta.n_elem;
    const double eta_norm_nb = std::sqrt(dot(glm_fit.eta, glm_fit.eta) / n_nb);
    const double eta_crit =
        std::sqrt(dot(glm_fit.eta - eta, glm_fit.eta - eta) / n_nb) /
        std::max(eta_norm_nb, 1.0);
    const double theta_crit =
        std::abs(theta_new - theta_prev) / (tol_denom + std::abs(theta_prev));

    if (eta_crit <= tol && theta_crit <= tol) {
      // Converged - finalize result
      static_cast<InferenceGLM &>(result) = std::move(glm_fit);
      result.theta = theta_new;
      result.conv_outer = true;
      return result;
    }

    // Update for next iteration
    theta = theta_new;

    // Move coefficients for warm start (avoid copy)
    beta_coef = std::move(glm_fit.coef_table.col(0));
    eta = std::move(glm_fit.eta);

    // Store current state as best result (in case we hit max iterations)
    static_cast<InferenceGLM &>(result) = std::move(glm_fit);
  }

  // Max iterations reached without convergence
  result.theta = theta;
  result.conv_outer = false;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_NEGBIN_H
