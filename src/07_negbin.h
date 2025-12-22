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

inline double estimate_theta(const vec &y, const vec &mu,
                             double theta_min = 0.1, double theta_max = 1.0e6,
                             double overdispersion_threshold = 0.01) {
  const double y_mean = mean(y);
  const double y_var = var(y);
  const double overdispersion = y_var - y_mean;

  // Low overdispersion -> return very large theta (Poisson-like)
  if (overdispersion <= overdispersion_threshold * y_mean) {
    return theta_max;
  }

  // Method of moments -> theta = mu^2 / (var - mu)
  double theta = y_mean * y_mean / overdispersion;

  return clamp(theta, theta_min, theta_max);
}

InferenceNegBin fenegbin_fit(mat &X, const vec &y, const vec &w,
                             const field<field<uvec>> &fe_groups,
                             const CapybaraParameters &params,
                             double init_theta = 0.0,
                             GlmWorkspace *workspace = nullptr) {
  const uword n = y.n_elem;
  const uword p = X.n_cols;

  InferenceNegBin result(n, p);

  double theta = (init_theta > 0) ? init_theta : 1.0;

  GlmWorkspace local_workspace;
  if (!workspace) {
    workspace = &local_workspace;
  }
  workspace->ensure_size(n, p);

  Family poisson_family = POISSON;

  // Initialize beta and eta with Poisson fit
  vec beta(p, fill::zeros);
  vec eta(n, fill::zeros);

  InferenceGLM poisson_fit = feglm_fit(beta, eta, y, X, w, 0.0, poisson_family,
                                       fe_groups, params, workspace);

  if (!poisson_fit.conv) {
    result.conv = false;
    result.conv_outer = false;
    // Copy coefficients even if convergence failed
    result.coefficients = poisson_fit.coefficients;
    result.eta = poisson_fit.eta;
    result.fitted_values = poisson_fit.fitted_values;
    result.weights = poisson_fit.weights;
    result.hessian = poisson_fit.hessian;
    result.deviance = poisson_fit.deviance;
    result.null_deviance = poisson_fit.null_deviance;
    result.iter = poisson_fit.iter;
    result.coef_status = poisson_fit.coef_status;
    result.fixed_effects = poisson_fit.fixed_effects;
    result.has_fe = poisson_fit.has_fe;
    result.TX = poisson_fit.TX;
    result.has_tx = poisson_fit.has_tx;
    return result;
  }

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
  for (uword iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;

    theta0 = theta;

    Family negbin_family = NEG_BIN;
    InferenceGLM glm_fit = feglm_fit(beta, eta, y, X, w, theta, negbin_family,
                                     fe_groups, params, workspace);

    if (!glm_fit.conv) {
      // Copy current results even if convergence failed
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
      break;
    }

    mu = glm_fit.fitted_values;
    double dev = glm_fit.deviance;

    double theta_new = estimate_theta(y, mu);

    if (!std::isfinite(theta_new) || theta_new <= 0) {
      theta_new = theta;
    }

    double dev_crit = std::abs(dev - dev0) / (0.1 + std::abs(dev));
    double theta_crit = std::abs(theta_new - theta0) / (0.1 + std::abs(theta0));

    if (dev_crit <= params.dev_tol && theta_crit <= params.dev_tol) {
      converged = true;
      theta = theta_new;

      result.coefficients = std::move(glm_fit.coefficients);
      result.eta = std::move(glm_fit.eta);
      result.fitted_values = std::move(glm_fit.fitted_values);
      result.weights = std::move(glm_fit.weights);
      result.hessian = std::move(glm_fit.hessian);
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

    theta = theta_new;
    dev0 = dev;

    beta = glm_fit.coefficients;
    eta = glm_fit.eta;

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
