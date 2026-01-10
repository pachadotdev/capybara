// Generalized linear model with fixed effects eta = alpha + X * beta

#ifndef CAPYBARA_GLM_HELPERS_H
#define CAPYBARA_GLM_HELPERS_H

namespace capybara {

struct InferenceGLM {
  mat coef_table; // Coefficient table: [estimate, std.error, z, p-value]
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian; // Hessian matrix (needed for some internal computations)
  mat vcov;    // Covariance matrix (inverse Hessian or sandwich)
  double deviance;
  double null_deviance;
  bool conv;
  uword iter;
  uvec coef_status; // 1 = estimable, 0 = collinear

  field<vec> fixed_effects;
  double pseudo_rsq; // Pseudo R-squared (for Poisson only)
  bool has_fe = false;
  uvec iterations;

  mat TX;
  bool has_tx = false;

  vec means;

  InferenceGLM(uword n, uword p)
      : coef_table(p, 4, fill::zeros), eta(n, fill::zeros),
        fitted_values(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), vcov(p, p, fill::zeros), deviance(0.0),
        null_deviance(0.0), conv(false), iter(0), coef_status(p, fill::ones),
        pseudo_rsq(0.0), has_fe(false), has_tx(false) {}
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

  uvec finite_indices = find_finite(eps_history);
  if (finite_indices.n_elem < 3) {
    return current_eps;
  }

  // Linear extrapolation based on last 3 values
  vec log_eps = log(eps_history.elem(finite_indices.tail(3)));
  vec x_vals = linspace(1, 3, 3);

  // Simple regression log(eps) = a + b*x
  double x_mean = mean(x_vals);
  double y_mean = mean(log_eps);
  double slope = dot(x_vals - x_mean, log_eps - y_mean) /
                 dot(x_vals - x_mean, x_vals - x_mean);
  double intercept = y_mean - slope * x_mean;

  // Predict next value
  double hat_log_eps = intercept + slope * 4.0;
  return std::max(exp(hat_log_eps), datum::eps);
}

template <typename T>
inline T clamp(const T &value, const T &lower, const T &upper) {
  return (value < lower) ? lower : ((value > upper) ? upper : value);
}

std::string tidy_family(const std::string &family) {
  std::string fam = family;

  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  uword pos = fam.find("(");
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

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

vec link_inv_gaussian(const vec &eta) { return eta; }

vec link_inv_poisson(const vec &eta) { return exp(eta); }

vec link_inv_logit(const vec &eta) { return 1.0 / (1.0 + exp(-eta)); }

vec link_inv_gamma(const vec &eta) { return 1 / eta; }

vec link_inv_invgaussian(const vec &eta) { return 1 / sqrt(eta); }

vec link_inv_negbin(const vec &eta) { return exp(eta); }

double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu));
}

double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();
  
  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    if (y_ptr[i] > 0) {
      sum += wt_ptr[i] * (y_ptr[i] * std::log(y_ptr[i] / mu_ptr[i]) - (y_ptr[i] - mu_ptr[i]));
    } else {
      sum += mu_ptr[i] * wt_ptr[i];
    }
  }
  return 2.0 * sum;
}

// Adapted from binomial_dev_resids()
// in base R it can be found in src/library/stats/src/family.c
double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();
  
  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double contrib = 0.0;
    if (y_ptr[i] == 1.0) {
      contrib = y_ptr[i] * std::log(y_ptr[i] / mu_ptr[i]);
    } else if (y_ptr[i] == 0.0) {
      contrib = (1.0 - y_ptr[i]) * std::log((1.0 - y_ptr[i]) / (1.0 - mu_ptr[i]));
    }
    sum += wt_ptr[i] * contrib;
  }
  return 2.0 * sum;
}

double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  vec r = y / mu;

  uvec p = find(y == 0);
  r.elem(p).fill(1.0);
  r = wt % (log(r) - (y - mu) / mu);

  return -2 * accu(r);
}

double dev_resids_invgaussian(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

double dev_resids_negbin(const vec &y, const vec &mu, const double &theta,
                         const vec &wt) {
  vec r = y;

  uvec p = find(y < 1);
  r.elem(p).fill(1.0);
  r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2 * accu(r);
}

vec variance_gaussian(const vec &mu) { return ones<vec>(mu.n_elem); }

vec link_inv(const vec &eta, const Family family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case GAUSSIAN:
    result = link_inv_gaussian(eta);
    break;
  case POISSON:
    result = link_inv_poisson(eta);
    break;
  case BINOMIAL:
    result = link_inv_logit(eta);
    break;
  case GAMMA:
    result = link_inv_gamma(eta);
    break;
  case INV_GAUSSIAN:
    result = link_inv_invgaussian(eta);
    break;
  case NEG_BIN:
    result = link_inv_negbin(eta);
    break;
  default:
    stop("Unknown family");
  }

  return result;
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
  switch (family_type) {
  case GAUSSIAN:
  case POISSON:
  case BINOMIAL:
  case NEG_BIN:
    return true;
  case GAMMA:
    return eta.is_finite() && all(eta != 0.0);
  case INV_GAUSSIAN:
    return eta.is_finite() && all(eta > 0.0);
  default:
    stop("Unknown family");
  }
}

bool valid_mu(const vec &mu, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return true;
  case POISSON:
  case NEG_BIN:
    return mu.is_finite() && all(mu > 0);
  case BINOMIAL:
    return mu.is_finite() && all(mu > 0 && mu < 1);
  case GAMMA:
    return mu.is_finite() && all(mu > 0.0);
  case INV_GAUSSIAN:
    return true;
  default:
    stop("Unknown family");
  }
}

vec inverse_link_derivative(const vec &eta, const Family family_type) {
  const uword n = eta.n_elem;
  vec result(n);
  double *r = result.memptr();
  const double *e = eta.memptr();

  switch (family_type) {
  case GAUSSIAN:
    result.ones();
    break;
  case POISSON:
  case NEG_BIN:
    for (uword i = 0; i < n; ++i) {
      r[i] = std::exp(e[i]);
    }
    break;
  case BINOMIAL:
    for (uword i = 0; i < n; ++i) {
      double exp_eta = std::exp(e[i]);
      double denom = 1.0 + exp_eta;
      r[i] = exp_eta / (denom * denom);
    }
    break;
  case GAMMA:
    for (uword i = 0; i < n; ++i) {
      r[i] = -1.0 / (e[i] * e[i]);
    }
    break;
  case INV_GAUSSIAN:
    for (uword i = 0; i < n; ++i) {
      r[i] = -1.0 / (2.0 * std::pow(e[i], 1.5));
    }
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

vec variance(const vec &mu, const double &theta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ones<vec>(mu.n_elem);
  case POISSON:
    return mu;
  case BINOMIAL:
    return mu % (1 - mu);
  case GAMMA:
    return square(mu);
  case INV_GAUSSIAN:
    return pow(mu, 3.0);
  case NEG_BIN:
    return mu + square(mu) / theta;
  default:
    stop("Unknown family");
  }
}

// Compute clustered sandwich covariance matrix
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

  // Compute H^{-1}
  mat H_inv;
  bool success = inv_sympd(H_inv, H);
  if (!success) {
    // Fall back to regular inverse
    success = inv(H_inv, H);
    if (!success) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  // Compute residuals (scores for Poisson with log-link are X * (y - mu))
  vec resid = y - mu;

  // Compute score matrix: each row is MX_i * resid_i
  mat scores = MX.each_col() % resid; // element-wise: MX[i,j] * resid[i]

  // Cluster adjustment factor: G / (G - 1)
  double adj = (G > 1) ? static_cast<double>(G) / (G - 1.0) : 1.0;

  // Compute meat: B = adj * sum_g (sum_i in g s_i)' * (sum_i in g s_i)
  mat B(p, p, fill::zeros);

  for (uword g = 0; g < G; ++g) {
    const uvec &idx = cluster_groups(g);
    if (idx.n_elem == 0)
      continue;

    // Sum scores within cluster
    vec cluster_score = sum(scores.rows(idx), 0).t();

    // Outer product
    B += cluster_score * cluster_score.t();
  }

  B *= adj;

  // Sandwich: V = H^{-1} B H^{-1}
  mat V = H_inv * B * H_inv;

  return V;
}

mat group_sums(const mat &M, const mat &w, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem, P = M.n_cols;

  mat b(P, 1, fill::zeros);
  double *b_ptr = b.memptr();

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    const uword n_idx = indexes.n_elem;
    const uword *idx_ptr = indexes.memptr();
    
    // Compute denominator directly
    double denom = 0.0;
    for (uword t = 0; t < n_idx; ++t) {
      denom += w(idx_ptr[t]);
    }
    
    if (denom == 0.0) continue;
    double inv_denom = 1.0 / denom;
    
    // Accumulate group sums for each column
    for (uword p = 0; p < P; ++p) {
      const double *col_ptr = M.colptr(p);
      double col_sum = 0.0;
      for (uword t = 0; t < n_idx; ++t) {
        col_sum += col_ptr[idx_ptr[t]];
      }
      b_ptr[p] += col_sum * inv_denom;
    }
  }

  return b;
}

mat group_sums_spectral(const mat &M, const mat &v, const mat &w,
                        const size_t K, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem, K1 = K, P = M.n_cols;

  vec num(P, fill::none), v_shifted;
  mat b(P, 1, fill::zeros);
  double denom;

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    const uword I = indexes.n_elem;

    if (I <= 1)
      continue;

    num.fill(0.0);
    denom = accu(w.elem(indexes));

    v_shifted.zeros(I);
    for (uword k = 1; k <= K1 && k < I; ++k) {
      for (uword i = 0; i < I - k; ++i) {
        v_shifted(i + k) += v(indexes(i));
      }
    }

    num = M.rows(indexes).t() * (v_shifted * (I / (I - 1.0)));
    b += num / denom;
  }

  return b;
}

mat group_sums_var(const mat &M, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat v(P, 1, fill::none), V(P, P, fill::zeros);

  for (uword j = 0; j < J; ++j) {
    const uvec &indexes = group_indices(j);
    v = sum(M.rows(indexes), 0).t();
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
    const uvec &indexes = group_indices(j);

    if (indexes.n_elem < 2) {
      continue;
    }

    V += M.rows(indexes).t() * N.rows(indexes);
  }

  return V;
}

} // namespace capybara

#endif // CAPYBARA_GLM_HELPERS_H
