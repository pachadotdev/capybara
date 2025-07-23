// Negative binomial GLM fitting with theta estimation
// Handles the alternating optimization between GLM coefficients and theta

#ifndef CAPYBARA_NEGBIN_H
#define CAPYBARA_NEGBIN_H

namespace capybara {
namespace negbin {

using convergence::Family;
using glm::dev_resids;
using glm::feglm_fit;
using glm::InferenceGLM;
using glm::link_inv;

//////////////////////////////////////////////////////////////////////////////
// THETA ML ESTIMATION
//////////////////////////////////////////////////////////////////////////////

inline double theta_ml(const vec &y, const vec &mu, size_t limit = 10) {
  // Maximum likelihood estimation of theta for negative binomial
  // Using moment-based estimation with proper bounds checking

  const size_t n = y.n_elem;

  // Calculate sample variance and mean
  double y_mean = mean(y);
  double y_dev = var(y);

  // Method of moments: theta = mu^2 / (var - mu)
  // If variance is close to or less than mean, theta should be very large
  // (approaching Poisson)
  double overdispersion = y_dev - y_mean;

  if (overdispersion <= 0.01 * y_mean) {
    // Very little overdispersion - return very large theta (Poisson-like)
    return 1e6;
  }

  double theta = y_mean * y_mean / overdispersion;

  // Ensure reasonable bounds
  theta = std::max(theta, 0.1);
  theta = std::min(theta, 1e6);

  // Newton-Raphson iterations for MLE
  for (size_t iter = 0; iter < limit; ++iter) {
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

      // Fisher information (negative second derivative)
      info += 1.0 / theta - 1.0 / (mu_i + theta);
      if (y_i > 0) {
        for (size_t k = 0; k < static_cast<size_t>(y_i); ++k) {
          info += 1.0 / ((theta + k) * (theta + k));
        }
      }
    }

    // Newton-Raphson step with conservative updates
    if (std::abs(info) > 1e-12) {
      double step = score / info;
      step = std::max(step, -0.1 * theta); // Don't decrease by more than 10%
      step = std::min(step, 0.5 * theta);  // Don't increase by more than 50%
      theta = theta + step;
    }

    // Ensure theta stays positive and reasonable
    theta = std::max(theta, 0.1);
    theta = std::min(theta, 1e6);

    // Check convergence
    if (std::abs(theta - theta_old) < 1e-6 * (1 + std::abs(theta))) {
      break;
    }
  }

  return theta;
}

//////////////////////////////////////////////////////////////////////////////
// NEGATIVE BINOMIAL RESULT STRUCTURE
//////////////////////////////////////////////////////////////////////////////

struct InferenceNegBin : public InferenceGLM {
  double theta;
  size_t iter_outer;
  bool conv_outer;

  InferenceNegBin(size_t n, size_t p)
      : InferenceGLM(n, p), theta(1.0), iter_outer(0), conv_outer(false) {}
};

//////////////////////////////////////////////////////////////////////////////
// NEGATIVE BINOMIAL FITTING
//////////////////////////////////////////////////////////////////////////////

inline InferenceNegBin
fenegbin_fit(const mat &X, const vec &y_orig, const vec &w,
             const field<uvec> &fe_indices, const uvec &nb_ids,
             const field<uvec> &fe_id_tables, const std::string &link_str,
             const CapybaraParameters &params, const double init_theta = 0.0) {

  const size_t n = y_orig.n_elem;
  const size_t p_orig = X.n_cols;

  InferenceNegBin result(n, p_orig);

  // Initialize theta
  double theta = (init_theta > 0) ? init_theta : 1.0;

  // First fit with Poisson to get starting values
  std::string poisson_family = "poisson";
  InferenceGLM poisson_fit =
      feglm_fit(X, y_orig, w, fe_indices, nb_ids, fe_id_tables, poisson_family,
                params, 0.0);

  if (!poisson_fit.conv) {
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Use Poisson results as starting values
  vec beta = poisson_fit.coefficients;
  vec eta = poisson_fit.eta;
  vec mu = poisson_fit.fitted_values;

  // Initial theta estimate if not provided - start with large value (closer to
  // Poisson)
  if (init_theta <= 0) {
    // Start with a large theta value to be closer to Poisson limit
    theta = std::max(theta_ml(y_orig, mu, params.iter_nb_theta), 10.0);
  }

  // Storage for convergence checking
  double dev_old = poisson_fit.deviance;
  double theta_old = theta;
  bool converged = false;

  // Alternate between fitting GLM and estimating theta
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;

    // Store old values
    theta_old = theta;

    // Fit GLM with current theta - using negative_binomial family
    std::string negbin_family = "negative_binomial";
    InferenceGLM glm_fit =
        feglm_fit(X, y_orig, w, fe_indices, nb_ids, fe_id_tables, negbin_family,
                  params, theta);

    if (!glm_fit.conv) {
      break;
    }

    // Update results - copy all fields from GLM result
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

    // Extract updated values
    mu = glm_fit.fitted_values;
    double dev = glm_fit.deviance;

    // Update theta
    double theta_new = theta_ml(y_orig, mu, params.iter_nb_theta);

    // Ensure theta is valid
    if (!is_finite(theta_new) || theta_new <= 0) {
      theta_new = theta; // Keep previous value
    }

    // Check convergence
    double dev_crit = std::abs(dev - dev_old) / (0.1 + std::abs(dev));
    double theta_crit =
        std::abs(theta_new - theta_old) / (0.1 + std::abs(theta_old));

    if (dev_crit <= params.dev_tol && theta_crit <= params.dev_tol) {
      converged = true;
      result.theta = theta_new;
      break;
    }

    // Update for next iteration
    theta = theta_new;
    dev_old = dev;
  }

  // Set final values
  result.theta = theta;
  result.conv_outer = converged;

  return result;
}

} // namespace negbin
} // namespace capybara

#endif // CAPYBARA_NEGBIN_H
