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
      : coef_table(p, 4), eta(n), fitted_values(n), weights(n, fill::ones),
        hessian(p, p), vcov(p, p), deviance(0.0), null_deviance(0.0),
        conv(false), iter(0), coef_status(p, fill::ones), pseudo_rsq(0.0),
        has_fe(false), has_tx(false), has_separation(false), num_separated(0) {
    coef_table.zeros();
    eta.zeros();
    fitted_values.zeros();
    hessian.zeros();
    vcov.zeros();
  }
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
// Deviance residuals - optimized with fused operations
///////////////////////////////////////////////////////////////////////////

inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double diff = y_ptr[i] - mu_ptr[i];
    sum += wt_ptr[i] * diff * diff;
  }
  return sum;
}

inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt) {
  // 2 * sum(wt * (y * log(max(y,1)/mu) - (y - mu)))
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = mu_ptr[i];
    double y_clamped = (yi < 1.0) ? 1.0 : yi;
    sum += wt_ptr[i] * (yi * std::log(y_clamped / mui) - (yi - mui));
  }
  return 2.0 * sum;
}

inline double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
  // 2 * sum(wt * (y*log(y/mu) + (1-y)*log((1-y)/(1-mu))))
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = mu_ptr[i];
    // Clamp y to avoid log(0)
    double y_safe = (yi < datum::eps)
                        ? datum::eps
                        : ((yi > 1.0 - datum::eps) ? 1.0 - datum::eps : yi);
    sum += wt_ptr[i] * (yi * std::log(y_safe / mui) +
                        (1.0 - yi) * std::log((1.0 - y_safe) / (1.0 - mui)));
  }
  return 2.0 * sum;
}

inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = mu_ptr[i];
    double ratio = yi / mui;
    double ratio_clamped = (ratio < datum::eps) ? datum::eps : ratio;
    sum += wt_ptr[i] * (std::log(ratio_clamped) - (yi - mui) / mui);
  }
  return -2.0 * sum;
}

inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = mu_ptr[i];
    double diff = yi - mui;
    sum += wt_ptr[i] * (diff * diff) / (yi * mui * mui);
  }
  return sum;
}

inline double dev_resids_negbin(const vec &y, const vec &mu,
                                const double &theta, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *mu_ptr = mu.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = mu_ptr[i];
    double y_clamped = (yi < 1.0) ? 1.0 : yi;
    double y_theta = yi + theta;
    sum += wt_ptr[i] * (yi * std::log(y_clamped / mui) -
                        y_theta * std::log(y_theta / (mui + theta)));
  }
  return 2.0 * sum;
}

inline vec link_inv(const vec &eta, const Family family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return eta;
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  case BINOMIAL:
    return 1.0 / (1.0 + exp(-eta));
  case GAMMA:
    return 1.0 / eta;
  case INV_GAUSSIAN:
    return 1.0 / sqrt(eta);
  default:
    stop("Unknown family");
  }
  return eta;
}

inline double dev_resids(const vec &y, const vec &mu, const double &theta,
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
  return 0.0;
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

///////////////////////////////////////////////////////////////////////////
// Clustered covariance matrix (Sandwich estimator)
///////////////////////////////////////////////////////////////////////////

// MX: centered design matrix (n x p)
// y: response vector (n)
// mu: fitted values (n)
// H: Hessian matrix (p x p), i.e., MX' W MX
// cluster_groups: indices for each cluster
// Returns: sandwich covariance matrix (p x p)

inline mat compute_sandwich_vcov(const mat &MX, const vec &y, const vec &mu,
                                 const mat &H,
                                 const field<uvec> &cluster_groups) {
  const uword p = MX.n_cols;
  const uword G = cluster_groups.n_elem;

  // Bread: H^{-1} - try symmetric positive definite inverse first
  mat H_inv;
  if (!inv_sympd(H_inv, H)) {
    if (!inv(H_inv, H)) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  // Precompute residuals
  vec resid = y - mu;
  const double *resid_ptr = resid.memptr();

  const double adj = (G > 1) ? static_cast<double>(G) / (G - 1.0) : 1.0;

  // Meat: B = sum_g (score_g * score_g')
  // Compute cluster scores directly without intermediate matrix
  mat B(p, p, fill::zeros);
  vec cluster_score(p);

  for (uword g = 0; g < G; ++g) {
    const uvec &idx = cluster_groups(g);
    const uword ng = idx.n_elem;
    if (ng == 0)
      continue;

    // Sum scores within cluster: score[j] = sum_i MX[i,j] * resid[i]
    cluster_score.zeros();
    double *cs_ptr = cluster_score.memptr();
    const uword *idx_ptr = idx.memptr();

    for (uword i = 0; i < ng; ++i) {
      const uword obs = idx_ptr[i];
      const double r = resid_ptr[obs];
      for (uword j = 0; j < p; ++j) {
        cs_ptr[j] += MX(obs, j) * r;
      }
    }

    // B += cluster_score * cluster_score'
    B += cluster_score * cluster_score.t();
  }

  // Sandwich: H^{-1} * (adj * B) * H^{-1}
  return (adj * H_inv) * B * H_inv;
}

///////////////////////////////////////////////////////////////////////////
// Dyadic for M-estimators and GMM
///////////////////////////////////////////////////////////////////////////

// This borrows from
// Dyad-Robust Inference for International Trade Data
// Colin Cameron (U.C. Davis) and Doug Miller (Cornell University) .
// Presented at IAAE session at ASSA Meetings
// January 5, 2024

// Consider dyads for countries g and h
// For simplicity consider cross-section data
// y_{gh} = x'_{gh} \beta + u_{gh}.
// Errors correlated between dyads (g,h) with at least one of g and h in common
// $E[u_{gh} u_{g' h'} | x_{gh} , x_{g' h'}] = 0
// unless $g = g'$ or $h = h'$ or $g = h'$ or $h = g'$
// Extra complication over two-way clustering is $g = h'$ or $h = g'$.
// Results generalize immediately to multiple observations per data such
// as panel data
// y_{ght} = x'_{ght} \beta + u_{ght}.

// Example: G=4 countries and bidirectional trade
// Six Pairs (1, 2), (1, 3), (1, 4), (2, 3), (2, 4) and (3, 4)
//  country-pair: only (g , h) = (g 0, h0) diagonal entries denoted CP
//  two-way: g = g 0 and/or h = h0 denoted CP and 2way.
//  dyadic: also g = h0 or h = g 0 denoted CP, 2way and DYAD.
// (g,h) / (g',h') | (1,2) | (1,3) | (1,4) | (2,3) | (2,4) | (3,4)
// ----------------|-------|-------|-------|-------|-------|-------
// (1,2)           | CP    | 2way  | 2way  | DYAD  | DYAD  |
// (1,3)           | 2way  | CP    | 2way  | 2way  |       | DYAD |
// (1,4)           | 2way  | 2way  | CP    |       | 2way  | 2way |
// (2,3)           | DYAD  | 2way  |       | CP    | 2way  | DYAD |
// (2,4)           | DYAD  |       | 2way  | 2way  | CP    | 2way |
// (3,4)           |       | DYAD  | 2way  | DYAD  | 2way  |   CP |
// For small G large fraction of correlation matrix is nonzero
//  G = 10 : 38% of error correlations are nonzero
//  G = 30 : 13% of error correlations are nonzero.
// For large G the fraction potentially correlated ! 4/(G  1).

// Extends to m-estimators (e.g. probit), IV, and GMM.
// M-estimator based on $E[m_{gh} (θ)] = 0$ solves $\sum_{g,h} m_{gh}
// (\hat{\theta}) = 0$.
// $\hat{\theta}$ is asymptotically normal with
// $\hat{V}[\hat{\theta}] = \hat{A}^{-1} \hat{B} \hat{A}^{-1}$
// $\hat{A} = \sum_{g, h} \left. \frac{\partial m_{gh}}{\partial \theta}
// \hat{\theta} \right|_{\hat{\theta}}$
// $\hat{B} = \sum_{g, h} 1[g = g' or h = h' or g = h' or h = g'] \times
// \hat{m}_{gh} \hat{m}_{g'h'}$ Straightforward generalization to GMM.
//  Santos and Silva (2006) gravity model has dependent variable in levels
//  (rather than logs) use an exponential mean model with multiplicative fixed
//  effects estimate by Poisson quasi-MLE Graham (2020ba) provides a dyadic
//  empirical application.

inline mat compute_sandwich_vcov_mestimator(const mat &A, const mat &scores,
                                            const field<uvec> &cluster_groups) {
  const uword p = A.n_cols;
  const uword G = cluster_groups.n_elem;

  // Bread: A^{-1} where A = sum_{g,h} d m_{gh} / d theta
  // (i.e. the Hessian / Jacobian of the moment conditions)
  mat A_inv;
  if (!inv_sympd(A_inv, A)) {
    if (!inv(A_inv, A)) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  // Small-sample degrees-of-freedom adjustment G / (G - 1)
  const double adj = (G > 1) ? static_cast<double>(G) / (G - 1.0) : 1.0;

  // Meat: B = sum_g s_g s_g'
  // where s_g = sum_{i in cluster g} scores_i  (cluster-level score)
  // scores is n x p, each row is the observation-level score m_{gh}(theta_hat)
  mat B(p, p, fill::zeros);
  vec cluster_score(p);

  for (uword g = 0; g < G; ++g) {
    const uvec &idx = cluster_groups(g);
    const uword ng = idx.n_elem;
    if (ng == 0)
      continue;

    // Sum observation-level scores within cluster g
    cluster_score.zeros();
    double *cs_ptr = cluster_score.memptr();
    const uword *idx_ptr = idx.memptr();

    for (uword i = 0; i < ng; ++i) {
      const uword obs = idx_ptr[i];
      for (uword j = 0; j < p; ++j) {
        cs_ptr[j] += scores(obs, j);
      }
    }

    // B += s_g * s_g'
    B += cluster_score * cluster_score.t();
  }

  // Sandwich: A^{-1} * (adj * B) * A^{-1}
  return (adj * A_inv) * B * A_inv;
}

///////////////////////////////////////////////////////////////////////////
// Group aggregation functions
///////////////////////////////////////////////////////////////////////////

inline vec group_sums(const mat &M, const vec &w,
                      const field<uvec> &group_indices) {
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

inline vec group_sums_spectral(const mat &M, const vec &v, const vec &w,
                               const uword K,
                               const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
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
    const vec v_cumsum = cumsum(v_group);
    const uword max_k = std::min(K, I - 1);

    // Compute shifted sum: v_shifted[i] = sum_{k=1}^{min(K,i)} v_group[i-k]
    vec v_shifted(I, fill::zeros);
    for (uword i = 1; i < I; ++i) {
      const uword start = (i > max_k) ? i - max_k : 0;
      v_shifted(i) = v_cumsum(i - 1) - (start > 0 ? v_cumsum(start - 1) : 0.0);
    }

    const double scale = static_cast<double>(I) / ((I - 1.0) * denom);
    b += M.rows(idx).t() * v_shifted * scale;
  }
  return b;
}

inline mat group_sums_var(const mat &M, const field<uvec> &group_indices) {
  const uword J = group_indices.n_elem;
  const uword P = M.n_cols;

  mat V(P, P, fill::zeros);
  for (uword j = 0; j < J; ++j) {
    const uvec &idx = group_indices(j);
    if (idx.n_elem == 0)
      continue;

    const rowvec row_sum = sum(M.rows(idx), 0);
    V += row_sum.t() * row_sum;
  }
  return V;
}

inline mat group_sums_cov(const mat &M, const mat &N,
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
