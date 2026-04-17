// Deviance residuals - templated for both scalar and vector mu

#ifndef CAPYBARA_GLM_DEVIANCE_H
#define CAPYBARA_GLM_DEVIANCE_H

namespace capybara {

// Helper to get mu[i] - works for both scalar (returns mu) and vector (returns
// mu[i])
template <typename MuType> inline double get_mu_i(const MuType &mu, uword i) {
  if constexpr (std::is_arithmetic_v<MuType>) {
    (void)i; // suppress unused warning
    return mu;
  } else {
    return mu(i);
  }
}

template <typename MuType>
inline double dev_resids_(const vec &y, const MuType &mu, const double &theta,
                          const vec &wt, const Family family_type) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();
  double sum = 0.0;

  switch (family_type) {
  case GAUSSIAN:
    for (uword i = 0; i < n; ++i) {
      double diff = y_ptr[i] - get_mu_i(mu, i);
      sum += wt_ptr[i] * diff * diff;
    }
    return sum;

  case POISSON:
    for (uword i = 0; i < n; ++i) {
      double yi = y_ptr[i];
      double mui = get_mu_i(mu, i);
      double y_clamped = (yi < 1.0) ? 1.0 : yi;
      sum += wt_ptr[i] * (yi * std::log(y_clamped / mui) - (yi - mui));
    }
    return 2.0 * sum;

  case BINOMIAL:
  case PROBIT:
    for (uword i = 0; i < n; ++i) {
      double yi = y_ptr[i];
      double mui = get_mu_i(mu, i);
      double y_safe = (yi < datum::eps)
                          ? datum::eps
                          : ((yi > 1.0 - datum::eps) ? 1.0 - datum::eps : yi);
      sum += wt_ptr[i] * (yi * std::log(y_safe / mui) +
                          (1.0 - yi) * std::log((1.0 - y_safe) / (1.0 - mui)));
    }
    return 2.0 * sum;

  case GAMMA:
    for (uword i = 0; i < n; ++i) {
      double yi = y_ptr[i];
      double mui = get_mu_i(mu, i);
      double ratio = yi / mui;
      double ratio_clamped = (ratio < datum::eps) ? datum::eps : ratio;
      sum += wt_ptr[i] * (std::log(ratio_clamped) - (yi - mui) / mui);
    }
    return -2.0 * sum;

  case INV_GAUSSIAN:
    for (uword i = 0; i < n; ++i) {
      double yi = y_ptr[i];
      double mui = get_mu_i(mu, i);
      double diff = yi - mui;
      sum += wt_ptr[i] * (diff * diff) / (yi * mui * mui);
    }
    return sum;

  case NEG_BIN:
    for (uword i = 0; i < n; ++i) {
      double yi = y_ptr[i];
      double mui = get_mu_i(mu, i);
      double y_clamped = (yi < 1.0) ? 1.0 : yi;
      double y_theta = yi + theta;
      sum += wt_ptr[i] * (yi * std::log(y_clamped / mui) -
                          y_theta * std::log(y_theta / (mui + theta)));
    }
    return 2.0 * sum;

  default:
    stop("Unknown family");
  }
  return 0.0;
}

// Convenience wrappers for the two common cases
inline double null_deviance(const vec &y, const double &theta, const vec &wt,
                            const Family family_type) {
  return dev_resids_(y, mean(y), theta, wt, family_type);
}

inline double dev_resids(const vec &y, const vec &mu, const double &theta,
                         const vec &wt, const Family family_type) {
  return dev_resids_(y, mu, theta, wt, family_type);
}

inline vec link_inv(const vec &eta, const Family family_type) {
  const uword n = eta.n_elem;
  vec mu(n);
  const double *eta_ptr = eta.memptr();
  double *mu_ptr = mu.memptr();

  switch (family_type) {
  case GAUSSIAN:
    std::memcpy(mu_ptr, eta_ptr, n * sizeof(double));
    break;
  case POISSON:
  case NEG_BIN:
    for (uword i = 0; i < n; ++i) {
      mu_ptr[i] = std::exp(eta_ptr[i]);
    }
    break;
  case BINOMIAL:
    for (uword i = 0; i < n; ++i) {
      mu_ptr[i] = 1.0 / (1.0 + std::exp(-eta_ptr[i]));
    }
    break;
  case PROBIT:
    for (uword i = 0; i < n; ++i) {
      // Phi(eta) using erfc for numerical stability
      mu_ptr[i] = 0.5 * std::erfc(-eta_ptr[i] * M_SQRT1_2);
    }
    break;
  case GAMMA:
    for (uword i = 0; i < n; ++i) {
      mu_ptr[i] = 1.0 / eta_ptr[i];
    }
    break;
  case INV_GAUSSIAN:
    for (uword i = 0; i < n; ++i) {
      mu_ptr[i] = 1.0 / std::sqrt(eta_ptr[i]);
    }
    break;
  default:
    stop("Unknown family");
  }
  return mu;
}

inline bool valid_eta(const vec &eta, const Family family_type) {
  if (!eta.is_finite())
    return false;

  switch (family_type) {
  case GAUSSIAN:
  case POISSON:
  case BINOMIAL:
  case PROBIT:
  case NEG_BIN:
    return true;
  case GAMMA:
    return !any(eta == 0.0);
  case INV_GAUSSIAN:
    return eta.min() > 0.0;
  default:
    stop("Unknown family");
  }
  return false;
}

inline bool valid_mu(const vec &mu, const Family family_type) {
  if (!mu.is_finite())
    return false;

  switch (family_type) {
  case GAUSSIAN:
  case INV_GAUSSIAN:
    return true;
  case POISSON:
  case NEG_BIN:
  case GAMMA:
    return mu.min() > 0.0;
  case BINOMIAL:
  case PROBIT:
    return mu.min() > 0.0 && mu.max() < 1.0;
  default:
    stop("Unknown family");
  }
  return false;
}

} // namespace capybara

#endif // CAPYBARA_GLM_DEVIANCE_H
