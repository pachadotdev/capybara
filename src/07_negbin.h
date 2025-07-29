// Negative binomial GLM fitting with theta estimation
// Handles the alternating optimization between GLM coefficients and theta

#ifndef CAPYBARA_NEGBIN_H
#define CAPYBARA_NEGBIN_H

namespace capybara {

// Method of moments: theta = mu^2 / (var - mu)
// If variance is close to or less than mean, theta should be very large
// and approaching Poisson
inline double theta_ml(const vec &y, const vec &mu,
                       const CapybaraParameters &params) {

  const size_t n = y.n_elem;

  double y_mean = mean(y);
  double y_dev = var(y);

  double overdispersion = y_dev - y_mean;

  if (overdispersion <= params.nb_overdispersion_threshold * y_mean) {
    // Very little overdispersion => return very large theta (Poisson-like)
    return params.nb_theta_max;
  }

  double theta = y_mean * y_mean / overdispersion;

  // Ensure reasonable bounds
  theta = std::max(theta, params.nb_theta_min);
  theta = std::min(theta, params.nb_theta_max);

  // Newton-Raphson iterations for MLE
  for (size_t iter = 0; iter < params.iter_nb_theta; ++iter) {
    double theta_old = theta;

    // Compute score function: derivative of log-likelihood w.r.t. theta
    double score = 0.0;
    double info = 0.0; // Fisher information (negative second derivative)

    for (size_t i = 0; i < n; ++i) {
      double y_i = y(i);
      double mu_i = mu(i);

      // Score: d/d_theta log(L)
      // For negative binomial: sum over i of [digamma(y_i + theta) -
      // digamma(theta) + log(theta) - log(mu_i + theta)] Simplified
      // approximation using log differences
      score += log(theta) - log(mu_i + theta);
      if (y_i > 0) {
        // Approximate digamma differences for y_i + theta vs theta
        for (size_t k = 0; k < static_cast<size_t>(y_i); ++k) {
          score += 1.0 / (theta + k);
        }
      }

      info += 1.0 / theta - 1.0 / (mu_i + theta);
      if (y_i > 0) {
        for (size_t k = 0; k < static_cast<size_t>(y_i); ++k) {
          info += 1.0 / ((theta + k) * (theta + k));
        }
      }
    }

    // Newton-Raphson step with conservative updates
    if (std::abs(info) > params.nb_info_min) {
      double step = score / info;
      step = std::max(step, -params.nb_step_max_decrease *
                                theta); // Don't decrease by more than limit
      step = std::min(step, params.nb_step_max_increase *
                                theta); // Don't increase by more than limit
      theta = theta + step;
    }

    // Ensure theta stays positive and reasonable
    theta = std::max(theta, params.nb_theta_min);
    theta = std::min(theta, params.nb_theta_max);

    // Check convergence
    if (std::abs(theta - theta_old) <
        params.nb_theta_tol * (1 + std::abs(theta))) {
      break;
    }
  }

  return theta;
}

struct InferenceNegBin : public InferenceGLM {
  double theta;
  size_t iter_outer;
  bool conv_outer;

  InferenceNegBin(size_t n, size_t p)
      : InferenceGLM(n, p), theta(1.0), iter_outer(0), conv_outer(false) {}
};

inline InferenceNegBin
fenegbin_fit(const mat &X, const vec &y_orig, const vec &w,
             const field<uvec> &fe_indices, const uvec &nb_ids,
             const field<uvec> &fe_id_tables, const std::string &link_str,
             const CapybaraParameters &params, const double init_theta = 0.0) {

  const size_t n = y_orig.n_elem;
  const size_t p_orig = X.n_cols;

  InferenceNegBin result(n, p_orig);

  double theta = (init_theta > 0) ? init_theta : 1.0;

  std::string poisson_family = "poisson";
  InferenceGLM poisson_fit =
      feglm_fit(X, y_orig, w, fe_indices, nb_ids, fe_id_tables, poisson_family,
                params, 0.0);

  if (!poisson_fit.conv) {
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  vec beta = poisson_fit.coefficients;
  vec eta = poisson_fit.eta;
  vec mu = poisson_fit.fitted_values;

  if (init_theta <= 0) {

    theta = std::max(theta_ml(y_orig, mu, params), 10.0);
  }

  double dev_old = poisson_fit.deviance;
  double theta_old = theta;
  bool converged = false;

  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;

    theta_old = theta;

    std::string negbin_family = "negative_binomial";
    InferenceGLM glm_fit =
        feglm_fit(X, y_orig, w, fe_indices, nb_ids, fe_id_tables, negbin_family,
                  params, theta);

    if (!glm_fit.conv) {
      break;
    }

    result.coefficients = glm_fit.coefficients;
    result.eta = glm_fit.eta;
    result.fitted_values = glm_fit.fitted_values;
    result.weights = glm_fit.weights;
    result.hessian = glm_fit.hessian;
    result.deviance = glm_fit.deviance;
    result.null_deviance = glm_fit.null_deviance;
    result.conv = glm_fit.conv;
    result.iter = glm_fit.iter;
    result.coef_status = glm_fit.coef_status;
    result.residuals_working = glm_fit.residuals_working;
    result.residuals_response = glm_fit.residuals_response;
    result.fixed_effects = glm_fit.fixed_effects;
    result.nb_references = glm_fit.nb_references;
    result.is_regular = glm_fit.is_regular;
    result.has_fe = glm_fit.has_fe;
    result.X_dm = glm_fit.X_dm;
    result.has_mx = glm_fit.has_mx;

    mu = glm_fit.fitted_values;
    double dev = glm_fit.deviance;

    double theta_new = theta_ml(y_orig, mu, params);

    if (!is_finite(theta_new) || theta_new <= 0) {
      theta_new = theta;
    }

    double dev_crit =
        std::abs(dev - dev_old) / (params.rel_tol_denom + std::abs(dev));
    double theta_crit = std::abs(theta_new - theta_old) /
                        (params.rel_tol_denom + std::abs(theta_old));

    if (dev_crit <= params.dev_tol && theta_crit <= params.dev_tol) {
      converged = true;
      result.theta = theta_new;
      break;
    }

    theta = theta_new;
    dev_old = dev;
  }

  result.theta = theta;
  result.conv_outer = converged;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_NEGBIN_H
