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

inline double theta_ml(const vec &y, const vec &mu, const CapybaraParameters &params) {
  const size_t n = y.n_elem;
  
  // Use params.iter_nb_theta if limit is 0 (default)
  const size_t max_iter = params.iter_nb_theta;

  // Calculate sample variance and mean using Armadillo functions
  double y_mean = mean(y);
  double y_dev = var(y);

  // Method of moments: theta = mu^2 / (var - mu)
  double overdispersion = y_dev - y_mean;

  if (overdispersion <= 0.01 * y_mean) {
    return 1e6;
  }

  double theta = std::max(0.1, std::min(y_mean * y_mean / overdispersion, 1e6));

  // Newton-Raphson iterations for MLE
  for (size_t iter = 0; iter < max_iter; ++iter) {
    double theta_old = theta;

    // Vectorized score and Fisher information computation
    vec log_theta_mu = log(theta + mu);
    double score = n * log(theta) - sum(log_theta_mu);
    double info = n / theta - sum(1.0 / (theta + mu));
    
    // Add contributions from positive y values
    uvec pos_y = find(y > 0);
    if (pos_y.n_elem > 0) {
      for (size_t idx = 0; idx < pos_y.n_elem; ++idx) {
        size_t i = pos_y(idx);
        double y_i = y(i);
        for (size_t k = 0; k < static_cast<size_t>(y_i); ++k) {
          score += 1.0 / (theta + k);
          info += 1.0 / ((theta + k) * (theta + k));
        }
      }
    }

    // Newton-Raphson step with conservative updates
    if (std::abs(info) > datum::eps) {
      double step = score / info;
      step = std::max(-0.1 * theta, std::min(step, 0.5 * theta));
      theta += step;
    }

    theta = std::max(0.1, std::min(theta, 1e6));

    // Check convergence
    if (std::abs(theta - theta_old) / (0.1 + std::abs(theta)) < params.dev_tol) {
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
             const CapybaraParameters &params, const double init_theta = 1.0) {

  const size_t n = y_orig.n_elem;
  const size_t p_orig = X.n_cols;

  InferenceNegBin result(n, p_orig);

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
  vec mu = poisson_fit.fitted_values;

  // Initialize theta
  double theta = (init_theta > 0) ? init_theta : std::max(theta_ml(y_orig, mu, params), 1.0);

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
    static_cast<InferenceGLM&>(result) = glm_fit;

    // Extract updated values
    mu = glm_fit.fitted_values;
    double dev = glm_fit.deviance;

    // Update theta
    double theta_new = theta_ml(y_orig, mu, params);

    // Ensure theta is valid
    theta_new = is_finite(theta_new) && theta_new > 0 ? theta_new : theta;

    // Check convergence
    double dev_crit = std::abs(dev - dev_old) / (0.1 + std::abs(dev));
    double theta_crit = std::abs(theta_new - theta_old) / (0.1 + std::abs(theta_old));

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
