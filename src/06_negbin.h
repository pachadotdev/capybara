// Computing generalized linear models with fixed effects
// eta = X beta + alpha + offset

#ifndef CAPYBARA_NEGBIN_H
#define CAPYBARA_NEGBIN_H

namespace capybara {

// Dedicated implementation for negative binomial models
struct InferenceNegBin : public InferenceGLM {
  double theta;
  size_t iter_outer;
  bool conv_outer;

  InferenceNegBin(size_t n, size_t p)
      : InferenceGLM(n, p), theta(1.0), iter_outer(0), conv_outer(false) {}
};

// Method of moments: theta = mu^2 / (var - mu)
inline double estimate_theta(const vec &y, const vec &mu,
                             double theta_min = 0.1, double theta_max = 1.0e6,
                             double overdispersion_threshold = 0.01) {
  const double y_mean = mean(y);
  const double y_var = var(y);
  const double overdispersion = y_var - y_mean;

  // Very little overdispersion => return very large theta (Poisson-like)
  if (overdispersion <= overdispersion_threshold * y_mean) {
    return theta_max;
  }

  // Estimate theta using method of moments
  double theta = y_mean * y_mean / overdispersion;

  // Ensure theta is within reasonable bounds
  return clamp(theta, theta_min, theta_max);
}

// Dedicated implementation for negative binomial models
InferenceNegBin fenegbin_fit(mat &X, const vec &y, const vec &w,
                             const field<field<uvec>> &fe_groups,
                             const CapybaraParameters &params,
                             double init_theta = 0.0) {
  const size_t n = y.n_elem;
  const size_t p = X.n_cols;

  InferenceNegBin result(n, p);

  // Initialize theta if not provided
  double theta = (init_theta > 0) ? init_theta : 1.0;

  // Start with Poisson for initialization
  Family poisson_family = POISSON;

  // Initialize beta and eta with Poisson fit
  vec beta(p, fill::zeros);
  vec eta(n, fill::zeros);

  // Get initial fit using Poisson
  InferenceGLM poisson_fit =
      feglm_fit(beta, eta, y, X, w, 0.0, poisson_family, fe_groups, params);

  if (!poisson_fit.conv) {
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Transfer initial values from Poisson fit
  beta = poisson_fit.coefficients;
  eta = poisson_fit.eta;
  vec mu = poisson_fit.fitted_values;

  // Estimate initial theta if not provided
  if (init_theta <= 0) {
    theta = estimate_theta(y, mu);
  }

  double dev0 = poisson_fit.deviance;
  double theta0 = theta;
  bool converged = false;

  // Alternate between fitting GLM and updating theta
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;

    // Save old theta
    theta0 = theta;

    // Fit GLM with current theta
    Family negbin_family = NEG_BIN;
    InferenceGLM glm_fit =
        feglm_fit(beta, eta, y, X, w, theta, negbin_family, fe_groups, params);

    if (!glm_fit.conv) {
      break;
    }

    // Update mu and compute new theta
    mu = glm_fit.fitted_values;
    double dev = glm_fit.deviance;

    // Estimate new theta based on current fit
    double theta_new = estimate_theta(y, mu);

    // Check validity
    if (!is_finite(theta_new) || theta_new <= 0) {
      theta_new = theta;
    }

    // Check convergence criteria
    double dev_crit = std::abs(dev - dev0) / (0.1 + std::abs(dev));
    double theta_crit = std::abs(theta_new - theta0) / (0.1 + std::abs(theta0));

    if (dev_crit <= params.dev_tol && theta_crit <= params.dev_tol) {
      converged = true;
      theta = theta_new;

      // Transfer results from glm_fit to result
      result.coefficients = beta;
      result.eta = eta;
      result.fitted_values = mu;
      result.weights = w;
      result.hessian = glm_fit.hessian;
      result.deviance = dev;
      result.null_deviance = glm_fit.null_deviance;
      result.conv = glm_fit.conv;
      result.iter = glm_fit.iter;
      result.coef_status = std::move(glm_fit.coef_status);
      result.fixed_effects = std::move(glm_fit.fixed_effects);
      result.has_fe = glm_fit.has_fe;
      result.TX = std::move(glm_fit.TX);
      result.has_tx = glm_fit.has_tx;
      result.theta = theta;
      result.conv_outer = true;

      break;
    }

    // Update values for next iteration
    theta = theta_new;
    dev0 = dev;

    // Save latest results for next iteration
    beta = glm_fit.coefficients;
    eta = glm_fit.eta;

    // Save results to return in case we hit max iterations
    result.coefficients = std::move(glm_fit.coefficients);
    result.eta = std::move(glm_fit.eta);
    result.fitted_values = std::move(glm_fit.fitted_values);
    result.weights = std::move(glm_fit.weights);
    result.hessian = std::move(glm_fit.hessian);
    result.deviance = glm_fit.deviance;
    result.null_deviance = glm_fit.null_deviance;
    result.conv = glm_fit.conv;
    result.iter = glm_fit.iter;
    result.coef_status = std::move(glm_fit.coef_status);
    result.fixed_effects = std::move(glm_fit.fixed_effects);
    result.has_fe = glm_fit.has_fe;
    result.TX = std::move(glm_fit.TX);
    result.has_tx = glm_fit.has_tx;
  }

  // Set final theta and convergence status
  result.theta = theta;
  result.conv_outer = converged;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_NEGBIN_H
