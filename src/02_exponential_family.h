#ifndef CAPYBARA_FAMILY_H
#define CAPYBARA_FAMILY_H

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

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

inline std::string tidy_family(const std::string &family) {
  std::string fam;
  fam.reserve(family.size());
  // 1. Lowercase copy
  for (unsigned char c : family) {
    fam.push_back(std::tolower(c));
  }
  // 2. Remove digits
  fam.erase(std::remove_if(fam.begin(), fam.end(),
                           [](char c) {
                             return std::isdigit(static_cast<unsigned char>(c));
                           }),
            fam.end());
  // 3. Truncate at '('
  if (auto pos = fam.find('('); pos != std::string::npos) {
    fam.resize(pos);
  }
  // 4. Replace spaces and dots with underscores
  for (char &c : fam) {
    if (c == ' ' || c == '.') {
      c = '_';
    }
  }
  // 5. Remove any remaining whitespace
  fam.erase(std::remove_if(fam.begin(), fam.end(),
                           [](char c) {
                             return std::isspace(static_cast<unsigned char>(c));
                           }),
            fam.end());

  return fam;
}
inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu));
}

inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt,
                                 vec &dev_vec_work, vec &ratio_work) {
  ratio_work = clamp(y, datum::eps, y.max()) / mu;
  dev_vec_work = y % log(ratio_work) - y + mu;

  // y == 0 case
  uvec y0 = find(y == 0);
  if (!y0.is_empty()) {
    dev_vec_work.elem(y0) = mu.elem(y0);
  }

  return 2.0 * dot(wt, dev_vec_work);
}

// inline double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
//   const uword n = y.n_elem;
//   vec dev_vec(n, fill::zeros);

//   // Create binary mask (0 for y=0, 1 for y=1)
//   uvec mask(n, fill::zeros);
//   uvec idx1 = find(y == 1);

//   // y=1 cases: log(1.0/mu)
//   // y=0 cases: log(1.0/(1.0-mu))
//   if (!idx1.is_empty()) {
//     mask.elem(idx1).ones();
//     dev_vec = mask % log(1.0 / mu) + (1.0 - mask) % log(1.0 / (1.0 - mu));
//   } else {
//     dev_vec = log(1.0 / (1.0 - mu));
//   }

//   return 2.0 * dot(wt, dev_vec);
// }

inline double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
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

inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  vec dev_vec(n, fill::none);
  dev_vec = -log(y / mu) + (y - mu) / mu;
  uvec zero_idx = find(y == 0);
  if (!zero_idx.is_empty()) {
    dev_vec.elem(zero_idx).ones();
  }
  return -2.0 * dot(wt, dev_vec);
}

inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

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

  uvec idx = find(y < 1);
  if (!idx.is_empty()) {
    dev_vec.elem(idx) = log(1.0 + mu.elem(idx) / theta);
  }
  return 2.0 * dot(wt, dev_vec);
}

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

inline double dev_resids(const vec &y, const vec &mu, const double &theta,
                         const vec &wt, const family_type family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return dev_resids_gaussian(y, mu, wt);
  case POISSON: {
    const uword n = y.n_elem;
    vec dev_vec_work(n, fill::none);
    vec ratio_work(n, fill::none);
    return dev_resids_poisson(y, mu, wt, dev_vec_work, ratio_work);
  }
  case BINOMIAL:
    return dev_resids_logit(y, mu, wt);
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
    vec exp_eta(n, fill::none);
    exp_eta = exp(eta);
    result.set_size(n);
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
  case BINOMIAL:
    result = mu % (1.0 - mu);
    break;
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
