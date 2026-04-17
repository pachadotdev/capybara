// Code used for GLM and NegBin model fitting
// Inverse link derivatives

#ifndef CAPYBARA_GLM_LINKS_H
#define CAPYBARA_GLM_LINKS_H

namespace capybara {

inline vec inverse_link_derivative(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
  case TOBIT:
    return vec(eta.n_elem, fill::ones);
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  case BINOMIAL: {
    // d/d(eta) [1/(1+exp(-eta))] = exp(eta)/(1+exp(eta))^2
    const vec exp_eta = exp(eta);
    return exp_eta / square(1.0 + exp_eta);
  }
  case PROBIT:
    // d/d(eta) [Phi(eta)] = phi(eta) = standard normal PDF
    return normpdf(eta);
  case GAMMA:
    return -1.0 / square(eta);
  case INV_GAUSSIAN:
    return -0.5 * pow(eta, -1.5);
  default:
    stop("Unknown family");
  }
  return vec();
}

inline vec variance(const vec &mu, const double &theta,
                    const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
  case TOBIT:
    return vec(mu.n_elem, fill::ones);
  case POISSON:
    return mu;
  case BINOMIAL:
  case PROBIT:
    return mu % (1.0 - mu);
  case GAMMA:
    return square(mu);
  case INV_GAUSSIAN:
    return pow(mu, 3.0);
  case NEG_BIN:
    return mu + square(mu) / theta;
  default:
    stop("Unknown family");
  }
  return vec();
}

} // namespace capybara

#endif // CAPYBARA_GLM_LINKS_H
