// Code used for GLM and NegBin model fitting
// Inverse link derivatives

#ifndef CAPYBARA_GLM_LINKS_H
#define CAPYBARA_GLM_LINKS_H

namespace capybara {

inline vec inverse_link_derivative(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return vec(eta.n_elem, fill::ones);
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  default:
    stop("Unknown family");
  }
  return vec();
}

inline vec variance(const vec &mu, const double &theta,
                    const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return vec(mu.n_elem, fill::ones);
  case POISSON:
    return mu;
  case NEG_BIN:
    return mu + square(mu) / theta;
  default:
    stop("Unknown family");
  }
  return vec();
}

} // namespace capybara

#endif // CAPYBARA_GLM_LINKS_H
