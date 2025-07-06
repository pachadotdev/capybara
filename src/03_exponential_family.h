#ifndef CAPYBARA_FAMILY_H
#define CAPYBARA_FAMILY_H

// Map string to family_type enum
inline family_type get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, family_type> family_map = {
      {"gaussian", GAUSSIAN},
      {"poisson", POISSON},
      {"binomial", BINOMIAL},
      {"gamma", GAMMA},
      {"inverse_gaussian", INV_GAUSSIAN},
      {"negative_binomial", NEG_BIN},
      {"negbinomial", NEG_BIN},
      {"negative binomial", NEG_BIN}};

  const auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

// Normalize and clean up family string
inline std::string tidy_family(const std::string &family) {
  std::string fam;
  fam.reserve(family.size());

  for (unsigned char c : family) {
    fam.push_back(std::tolower(c));
  }

  fam.erase(std::remove_if(fam.begin(), fam.end(),
                           [](char c) {
                             return std::isdigit(static_cast<unsigned char>(c));
                           }),
            fam.end());

  if (const auto pos = fam.find('('); pos != std::string::npos) {
    fam.resize(pos);
  }

  for (char &c : fam) {
    if (c == ' ' || c == '.') {
      c = '_';
    }
  }

  fam.erase(std::remove_if(fam.begin(), fam.end(),
                           [](char c) {
                             return std::isspace(static_cast<unsigned char>(c));
                           }),
            fam.end());

  return fam;
}

// Deviance residuals for Gaussian family
inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu));
}

// Deviance residuals for Poisson family
inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt,
                                 vec &dev_vec_work, vec &ratio_work) {
  ratio_work = y / mu;
  
  const uvec zero_indices = find(y <= datum::eps);
  if (!zero_indices.is_empty()) {
    ratio_work(zero_indices) = vec(zero_indices.n_elem, fill::value(datum::eps)) / mu(zero_indices);
  }
  
  const uvec positive_indices = find(y > 0);
  const uvec zero_y_indices = find(y == 0);
  
  if (!positive_indices.is_empty()) {
    dev_vec_work(positive_indices) = y(positive_indices) % log(ratio_work(positive_indices)) - 
                                     y(positive_indices) + mu(positive_indices);
  }
  
  if (!zero_y_indices.is_empty()) {
    dev_vec_work(zero_y_indices) = mu(zero_y_indices);
  }
  
  return 2.0 * dot(wt, dev_vec_work);
}

// Deviance residuals for Binomial family
inline double dev_resids_binomial(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  vec mu_safe(n, fill::none);
  vec y_safe(n, fill::none);
  mu_safe = clamp(mu, datum::eps, 1.0 - datum::eps);
  y_safe = clamp(y, datum::eps, 1.0 - datum::eps);

  vec dev_vec(n, fill::none);
  dev_vec = (y_safe % log(y_safe / mu_safe)) +
            ((1.0 - y_safe) % log((1.0 - y_safe) / (1.0 - mu_safe)));
  return 2.0 * dot(wt, dev_vec);
}

// Deviance residuals for Gamma family
inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  vec dev_vec(n, fill::none);
  dev_vec = -log(y / mu) + (y - mu) / mu;
  const uvec zero_idx = find(y == 0);
  if (!zero_idx.is_empty()) {
    dev_vec.elem(zero_idx).ones();
  }
  return -2.0 * dot(wt, dev_vec);
}

// Deviance residuals for Inverse Gaussian family
inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

// Deviance residuals for Negative Binomial family
inline double dev_resids_negbin(const vec &y, const vec &mu,
                                const double &theta, const vec &wt) {
  const uword n = y.n_elem;
  vec dev_vec(n, fill::none);
  vec y_theta(n, fill::none);
  vec mu_theta(n, fill::none);

  y_theta = y + theta;
  mu_theta = mu + theta;
  dev_vec = y % log(clamp(y, datum::eps, y.max()) / mu) -
            y_theta % log(clamp(y_theta, datum::eps, y_theta.max()) / mu_theta);

  const uvec idx = find(y < 1);
  if (!idx.is_empty()) {
    dev_vec.elem(idx) = log(1.0 + mu.elem(idx) / theta);
  }
  return 2.0 * dot(wt, dev_vec);
}

// Inverse link function for each family
inline void link_inv(vec &mu, const vec &eta, const family_type family) {
  const uword n = eta.n_elem;
  mu.set_size(n);
  mu.zeros();
  switch (family) {
  case GAUSSIAN:
    mu = eta;
    break;
  case POISSON:
  case NEG_BIN:
    mu = exp(eta);
    break;
  case BINOMIAL:
    mu = 1.0 / (1.0 + exp(-eta));
    break;
  case GAMMA:
    mu = 1.0 / eta;
    break;
  case INV_GAUSSIAN:
    mu = pow(eta, -0.5);
    break;
  default:
    stop("Unknown family");
  }
}

// Dispatch to correct deviance residuals for a family
inline double dev_resids(const vec &y, const vec &mu, const double &theta,
                         const vec &wt, const family_type family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return dev_resids_gaussian(y, mu, wt);
  case POISSON: {
    vec dev_vec_work(y.n_elem, fill::none);
    vec ratio_work(y.n_elem, fill::none);
    return dev_resids_poisson(y, mu, wt, dev_vec_work, ratio_work);
  }
  case BINOMIAL:
    return dev_resids_binomial(y, mu, wt);
  case GAMMA:
    return dev_resids_gamma(y, mu, wt);
  case INV_GAUSSIAN:
    return dev_resids_invgaussian(y, mu, wt);
  case NEG_BIN:
    return dev_resids_negbin(y, mu, theta, wt);
  default:
    stop("Unknown family");
  }
}

// Check if eta/mu are valid for a family
inline bool valid_eta_mu(const vec &eta, const vec &mu,
                         const family_type family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return true;
  case POISSON:
  case NEG_BIN:
    return is_finite(mu) && all(mu > 0);
  case BINOMIAL:
    return is_finite(mu) && all(mu > 0 && mu < 1);
  case GAMMA:
    return is_finite(eta) && all(eta != 0) && is_finite(mu) && all(mu > 0);
  case INV_GAUSSIAN:
    return is_finite(eta) && all(eta > 0);
  default:
    stop("Unknown family");
  }
}

// Derivative of inverse link (dmu/deta) for each family
inline void get_mu(vec &result, const vec &eta, const family_type family) {
  const uword n = eta.n_elem;
  result.set_size(n);
  switch (family) {
  case GAUSSIAN:
    result.ones();
    break;
  case POISSON:
  case NEG_BIN:
    result = exp(eta);
    break;
  case BINOMIAL: {
    const vec exp_eta = exp(eta);
    result = exp_eta / square(1.0 + exp_eta);
  } break;
  case GAMMA:
    result = -1.0 / square(eta);
    break;
  case INV_GAUSSIAN:
    result = -0.5 * pow(eta, -1.5);
    break;
  default:
    stop("Unknown family");
  }
}

// Variance function for each family
inline void variance(vec &result, const vec &mu, const double &theta,
                     const family_type family) {
  const uword n = mu.n_elem;
  result.set_size(n);
  switch (family) {
  case GAUSSIAN:
    result.ones();
    break;
  case POISSON:
    result = mu;
    break;
  case BINOMIAL: {
    const vec mu_safe = clamp(mu, datum::eps, 1.0 - datum::eps);
    result = mu_safe % (1.0 - mu_safe);
    break;
  }
  case GAMMA:
    result = square(mu);
    break;
  case INV_GAUSSIAN:
    result = pow(mu, 3.0);
    break;
  case NEG_BIN:
    result = mu + square(mu) / theta;
    break;
  default:
    stop("Unknown family");
  }
}

// Smart initialization of eta for each family
inline void smart_initialize_glm(vec &eta, const vec &y,
                                 const family_type family) {
  const uword n = y.n_elem;
  eta.set_size(n);
  const double small_const = 0.1;
  const double eps = 1e-4;
  vec y_clamped(n, fill::none);

  switch (family) {
  case GAUSSIAN:
    eta = y;
    break;
  case POISSON:
  case NEG_BIN:
    y_clamped = clamp(y, small_const, y.max() + small_const);
    eta = log(y_clamped);
    break;
  case BINOMIAL:
    y_clamped = clamp(y, eps, 1.0 - eps);
    eta = log(y_clamped / (1.0 - y_clamped));
    break;
  case GAMMA:
    y_clamped = clamp(y, small_const, y.max() + small_const);
    eta = 1.0 / y_clamped;
    break;
  case INV_GAUSSIAN:
    y_clamped = clamp(y, small_const, y.max() + small_const);
    eta = 1.0 / sqrt(y_clamped);
    break;
  default:
    eta.zeros();
  }
}

#endif // CAPYBARA_FAMILY_H
