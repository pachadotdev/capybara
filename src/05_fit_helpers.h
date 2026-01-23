// Generalized linear model with fixed effects eta = alpha + X * beta
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
  if (eps_history.n_elem < 3 || !eps_history.is_finite()) {
    return current_eps;
  }

  const uvec finite_indices = find_finite(eps_history);
  if (finite_indices.n_elem < 3) {
    return current_eps;
  }

  // linear extrapolation based on last 3 values
  const vec log_eps = log(eps_history.elem(finite_indices.tail(3)));
  const vec x_vals = {1.0, 2.0, 3.0};

  // simple regression log(eps) = a + b*x
  // x_mean = 2.0, so x_centered = {-1, 0, 1}
  const vec x_centered = x_vals - 2.0;
  const double y_mean = accu(log_eps) / 3.0;
  const vec y_centered = log_eps - y_mean;

  // slope
  const double slope = dot(x_centered, y_centered) / 2.0;
  const double intercept = y_mean - slope * 2.0;

  // predict next value (x=4)
  return std::max(std::exp(intercept + slope * 4.0), datum::eps);
}

template <typename T>
inline T clamp(const T &value, const T &lower, const T &upper) {
  return std::clamp(value, lower, upper);
}

std::string tidy_family(const std::string &family) {
  std::string fam = family;

  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  const uword pos = fam.find("(");
  if (pos != std::string::npos) {
    fam.erase(pos, fam.size());
  }

  std::replace(fam.begin(), fam.end(), ' ', '_');
  std::replace(fam.begin(), fam.end(), '.', '_');

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isspace), fam.end());

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
// Link inverse functions
///////////////////////////////////////////////////////////////////////////

inline vec link_inv_gaussian(const vec &eta) { return eta; }

inline vec link_inv_poisson(const vec &eta) { return exp(eta); }

inline vec link_inv_logit(const vec &eta) { return 1.0 / (1.0 + exp(-eta)); }

inline vec link_inv_gamma(const vec &eta) { return 1.0 / eta; }

inline vec link_inv_invgaussian(const vec &eta) { return 1.0 / sqrt(eta); }

inline vec link_inv_negbin(const vec &eta) { return exp(eta); }

///////////////////////////////////////////////////////////////////////////
// Deviance residuals
///////////////////////////////////////////////////////////////////////////

inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  return accu(wt % square(y - mu));
}

inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt) {
  // Use clamp to avoid log(0): max(y, 1) ensures log argument >= 1 when y=0
  return 2.0 *
         accu(wt % (y % log(arma::max(y, ones(size(y))) / mu) - (y - mu)));
}

inline double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
  // Avoid log(0) using max()
  // For y=0: y*log(max(y,1)/mu) = 0*log(1/mu) = 0
  // For y=1: (1-y)*log(max(1-y,1)/(1-mu)) = 0*log(1/(1-mu)) = 0
  const vec y_safe = arma::max(y, ones(size(y)));
  const vec one_minus_y = 1.0 - y;
  const vec y_inv_safe = arma::max(one_minus_y, ones(size(y)));

  return 2.0 * accu(wt % (y % log(y_safe / mu) +
                          one_minus_y % log(y_inv_safe / (1.0 - mu))));
}

inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  // Use max(y/mu, 1) when y=0 to avoid log(0)
  const vec r_val = arma::max(y / mu, ones(size(y)));
  return -2.0 * accu(wt % (log(r_val) - (y - mu) / mu));
}

inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &wt) {
  return accu(wt % square(y - mu) / (y % square(mu)));
}

inline double dev_resids_negbin(const vec &y, const vec &mu,
                                const double &theta, const vec &wt) {
  // Use max(y, 1) to avoid log(0) when y < 1
  const vec y_safe = arma::max(y, ones(size(y)));
  const vec y_plus_theta = y + theta;

  return 2.0 * accu(wt % (y % log(y_safe / mu) -
                          y_plus_theta % log(y_plus_theta / (mu + theta))));
}

inline vec variance_gaussian(const vec &mu) { return ones(size(mu)); }

vec link_inv(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return link_inv_gaussian(eta);
  case POISSON:
    return link_inv_poisson(eta);
  case BINOMIAL:
    return link_inv_logit(eta);
  case GAMMA:
    return link_inv_gamma(eta);
  case INV_GAUSSIAN:
    return link_inv_invgaussian(eta);
  case NEG_BIN:
    return link_inv_negbin(eta);
  default:
    stop("Unknown family");
  }
  return eta; // Unreachable, but avoids compiler warning
}

double dev_resids(const vec &y, const vec &mu, const double &theta,
                  const vec &wt, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return dev_resids_gaussian(y, mu, wt);
  case POISSON:
    return dev_resids_poisson(y, mu, wt);
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

bool valid_eta(const vec &eta, const Family family_type) {
  if (!eta.is_finite())
    return false;

  switch (family_type) {
  case GAUSSIAN:
  case POISSON:
  case BINOMIAL:
  case NEG_BIN:
    return true;
  case GAMMA:
    return all(eta != 0.0);
  case INV_GAUSSIAN:
    return eta.min() > 0.0;
  default:
    stop("Unknown family");
  }
  return false; // Unreachable
}

bool valid_mu(const vec &mu, const Family family_type) {
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
  return false; // Unreachable
}

///////////////////////////////////////////////////////////////////////////
// Inverse link derivatives
///////////////////////////////////////////////////////////////////////////

vec inverse_link_derivative(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ones(size(eta));
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  case BINOMIAL: {
    // d/d(eta) [1/(1+exp(-eta))] = exp(eta)/(1+exp(eta))^2
    const vec exp_eta = exp(eta);
    const vec denom = 1.0 + exp_eta;
    return exp_eta / square(denom);
  }
  case GAMMA:
    // d/d(eta) [1/eta] = -1/eta^2
    return -1.0 / square(eta);
  case INV_GAUSSIAN:
    // d/d(eta) [1/sqrt(eta)] = -0.5 * eta^(-1.5)
    return -0.5 / pow(eta, 1.5);
  default:
    stop("Unknown family");
  }
  return vec(); // Unreachable
}

vec variance(const vec &mu, const double &theta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ones(size(mu));
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
  return vec(); // Unreachable
}

///////////////////////////////////////////////////////////////////////////
// Clustered covariance matrix (Sandwich estimator)
///////////////////////////////////////////////////////////////////////////

// MX: centered design matrix (n x p)
// y: response vector (n)
// mu: fitted values (n)
// H: Hessian matrix (p x p), i.e., MX' W MX
// cluster_groups: indices for each cluster
// Returns: sandwich covariance matrix (p x p)

mat compute_sandwich_vcov(const mat &MX, const vec &y, const vec &mu,
                          const mat &H, const field<uvec> &cluster_groups) {
  const uword p = MX.n_cols;
  const uword G = cluster_groups.n_elem;

  // Bread: H^{-1}

  mat H_inv;
  bool success = inv_sympd(H_inv, H);
  if (!success) {
    success = inv(H_inv, H);
    if (!success) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  const vec resid = y - mu;

  const double adj = (G > 1) ? static_cast<double>(G) / (G - 1.0) : 1.0;

  // Meat: B = adj * sum_g (sum_i in g s_i)' * (sum_i in g s_i)

  mat B(p, p, fill::zeros);

  for (uword g = 0; g < G; ++g) {
    const uvec &idx = cluster_groups(g);
    if (idx.n_elem == 0)
      continue;

    const vec cluster_score = MX.rows(idx).t() * resid.elem(idx);

    B += cluster_score * cluster_score.t();
  }

  B *= adj;

  // Sandwich: V = H^{-1} B H^{-1} or bread * meat * bread
  return H_inv * B * H_inv;
}

mat group_sums(const mat &M, const mat &w, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  vec b(P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    if (idx.n_elem == 0)
      continue;

    const double denom = accu(w.elem(idx));
    if (denom == 0.0)
      continue;

    b += sum(M.rows(idx), 0).t() / denom;
  }

  return b;
}

mat group_sums_spectral(const mat &M, const mat &v, const mat &w,
                        const size_t K, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword K1 = K;
  const uword P = M.n_cols;

  vec b(P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    const uword I = idx.n_elem;

    if (I <= 1)
      continue;

    const double denom = accu(w.elem(idx));
    if (denom == 0.0)
      continue;

    const vec v_group = v.elem(idx);

    vec v_shifted(I, fill::zeros);
    const uword max_k = std::min(K1, I - 1);
    for (uword k = 1; k <= max_k; ++k) {
      v_shifted.tail(I - k) += v_group.head(I - k);
    }

    const vec num =
        M.rows(idx).t() * (v_shifted * (static_cast<double>(I) / (I - 1.0)));
    b += num / denom;
  }

  return b;
}

mat group_sums_var(const mat &M, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat V(P, P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    if (idx.n_elem == 0)
      continue;

    const vec v = sum(M.rows(idx), 0).t();
    V += v * v.t();
  }

  return V;
}

mat group_sums_cov(const mat &M, const mat &N,
                   const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat V(P, P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);

    if (idx.n_elem < 2)
      continue;

    V += M.rows(idx).t() * N.rows(idx);
  }

  return V;
}

} // namespace capybara

#endif // CAPYBARA_GLM_HELPERS_H
