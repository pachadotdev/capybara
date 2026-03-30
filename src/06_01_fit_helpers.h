// Code used for LM, GLM and NegBin model fitting

#ifndef CAPYBARA_GLM_HELPERS_H
#define CAPYBARA_GLM_HELPERS_H

namespace capybara {

struct InferenceGLM {
  mat coef_table; // Coefficient table: [estimate, std.error, z, p-value]
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  mat vcov; // inverse Hessian or sandwich)
  double deviance;
  double null_deviance;
  bool conv;
  uword iter;
  uvec coef_status; // 1 = estimable, 0 = collinear

  field<vec> fixed_effects;
  double pseudo_rsq; // pseudo R-squared for Poisson
  bool has_fe = false;
  uvec iterations;

  mat TX;
  bool has_tx = false;

  vec means;

  // Separation detection fields
  bool has_separation = false;
  uword num_separated = 0;
  uvec separated_obs;
  vec separation_support;

  InferenceGLM(uword n, uword p)
      : coef_table(p, 4, fill::zeros), eta(n, fill::zeros),
        fitted_values(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), vcov(p, p, fill::zeros), deviance(0.0),
        null_deviance(0.0), conv(false), iter(0), coef_status(p, fill::ones),
        pseudo_rsq(0.0), has_fe(false), has_tx(false), has_separation(false),
        num_separated(0) {}
};

enum Family {
  UNKNOWN = 0,
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN
};

inline double predict_convergence(const vec &eps_history, double current_eps) {
  const uword n = eps_history.n_elem;
  if (n < 3 || !eps_history.is_finite()) {
    return current_eps;
  }

  const vec last3 = eps_history.tail(3);
  if (!last3.is_finite()) {
    return current_eps;
  }

  // Vectorized regression: log(eps) = a + b*x with x = {1,2,3}
  const vec log_eps = log(last3);
  // slope = (log_eps(2) - log_eps(0)) / 2, predict at x=4: mean + 2*slope
  return std::max(std::exp(mean(log_eps) + (log_eps(2) - log_eps(0))),
                  datum::eps);
}

inline std::string tidy_family(const std::string &family) {
  std::string fam = family;

  // Lowercase
  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Truncate at '(' if present
  const auto pos = fam.find('(');
  if (pos != std::string::npos) {
    fam.resize(pos);
  }

  // Remove digits and replace separators with underscore in one pass
  fam.erase(
      std::remove_if(fam.begin(), fam.end(),
                     [](char c) { return std::isdigit(c) || std::isspace(c); }),
      fam.end());

  std::replace(fam.begin(), fam.end(), '.', '_');

  return fam;
}

Family get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", GAUSSIAN},
      {"poisson", POISSON},
      {"binomial", BINOMIAL},
      {"gamma", GAMMA},
      {"inverse_gaussian", INV_GAUSSIAN},
      {"negative_binomial", NEG_BIN}};

  const auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

///////////////////////////////////////////////////////////////////////////
// Deviance residuals - templated for both scalar and vector mu
///////////////////////////////////////////////////////////////////////////

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
    return mu.min() > 0.0 && mu.max() < 1.0;
  default:
    stop("Unknown family");
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////
// Inverse link derivatives
///////////////////////////////////////////////////////////////////////////

inline vec inverse_link_derivative(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return vec(eta.n_elem, fill::ones);
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  case BINOMIAL: {
    // d/d(eta) [1/(1+exp(-eta))] = exp(eta)/(1+exp(eta))^2
    const vec exp_eta = exp(eta);
    return exp_eta / square(1.0 + exp_eta);
  }
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
    return vec(mu.n_elem, fill::ones);
  case POISSON:
    return mu;
  case BINOMIAL:
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

////////////////////////////////////////////////////////////////////////////////
// Group-level separation pre-filter
// Drops observations in FE groups where all y==0 (Poisson/NegBin)
// or all y==0 or all y==1 (Binomial). Iterates until a fixed point.
// Equivalent to the R-side drop_by_link_type_() but runs in C++.
////////////////////////////////////////////////////////////////////////////////

inline SeparationResult check_group_separation(const vec &y, const vec &w,
                                               const FlatFEMap &fe_map,
                                               Family family_type) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  // Only applicable for Poisson, NegBin, and Binomial with fixed effects
  if (fe_map.K == 0) {
    return result;
  }
  const bool is_binomial = (family_type == BINOMIAL);
  if (!is_binomial && family_type != POISSON && family_type != NEG_BIN) {
    return result;
  }

  const uword n = y.n_elem;
  uvec drop_mask(n, fill::zeros); // 1 = separated, 0 = keep

  // Iterate until no new observations are dropped
  // (dropping from one FE dimension can cause another group to become
  // degenerate in a different dimension)
  bool changed = true;
  while (changed) {
    changed = false;

    for (uword k = 0; k < fe_map.K; ++k) {
      const uword n_grp = fe_map.n_groups[k];
      const std::vector<uword> &map_k = fe_map.fe_map[k];

      // Compute weighted group sums and counts over kept observations
      vec grp_sum(n_grp, fill::zeros);
      vec grp_wt(n_grp, fill::zeros);

      for (uword i = 0; i < n; ++i) {
        if (drop_mask(i))
          continue;
        const uword g = map_k[i];
        const double wi = w(i);
        grp_sum(g) += wi * y(i);
        grp_wt(g) += wi;
      }

      // Identify degenerate groups
      for (uword i = 0; i < n; ++i) {
        if (drop_mask(i))
          continue;
        const uword g = map_k[i];
        if (grp_wt(g) <= 0.0)
          continue;

        const double grp_mean = grp_sum(g) / grp_wt(g);

        bool is_separated = false;
        if (is_binomial) {
          // Groups where mean(y) <= 0 or mean(y) >= 1 => perfect prediction
          is_separated = (grp_mean <= 0.0 || grp_mean >= 1.0);
        } else {
          // Poisson/NegBin: groups where mean(y) <= 0 => all zeros
          is_separated = (grp_mean <= 0.0);
        }

        if (is_separated) {
          drop_mask(i) = 1;
          changed = true;
        }
      }
    }
  }

  result.separated_obs = find(drop_mask);
  result.num_separated = result.separated_obs.n_elem;
  return result;
}

} // namespace capybara

#endif // CAPYBARA_GLM_HELPERS_H
