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
// Deviance residuals - templated for both scalar and vector mu
///////////////////////////////////////////////////////////////////////////

// Helper to get mu[i] - works for both scalar (returns mu) and vector (returns mu[i])
template <typename MuType>
inline double get_mu_i(const MuType &mu, uword i) {
  if constexpr (std::is_arithmetic_v<MuType>) {
    (void)i; // suppress unused warning
    return mu;
  } else {
    return mu(i);
  }
}

template <typename MuType>
inline double dev_resids_gaussian(const vec &y, const MuType &mu,
                                  const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double diff = y_ptr[i] - get_mu_i(mu, i);
    sum += wt_ptr[i] * diff * diff;
  }
  return sum;
}

template <typename MuType>
inline double dev_resids_poisson(const vec &y, const MuType &mu,
                                 const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = get_mu_i(mu, i);
    double y_clamped = (yi < 1.0) ? 1.0 : yi;
    sum += wt_ptr[i] * (yi * std::log(y_clamped / mui) - (yi - mui));
  }
  return 2.0 * sum;
}

template <typename MuType>
inline double dev_resids_logit(const vec &y, const MuType &mu, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
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
}

template <typename MuType>
inline double dev_resids_gamma(const vec &y, const MuType &mu, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = get_mu_i(mu, i);
    double ratio = yi / mui;
    double ratio_clamped = (ratio < datum::eps) ? datum::eps : ratio;
    sum += wt_ptr[i] * (std::log(ratio_clamped) - (yi - mui) / mui);
  }
  return -2.0 * sum;
}

template <typename MuType>
inline double dev_resids_invgaussian(const vec &y, const MuType &mu,
                                     const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = get_mu_i(mu, i);
    double diff = yi - mui;
    sum += wt_ptr[i] * (diff * diff) / (yi * mui * mui);
  }
  return sum;
}

template <typename MuType>
inline double dev_resids_negbin(const vec &y, const MuType &mu,
                                const double &theta, const vec &wt) {
  const uword n = y.n_elem;
  const double *y_ptr = y.memptr();
  const double *wt_ptr = wt.memptr();

  double sum = 0.0;
  for (uword i = 0; i < n; ++i) {
    double yi = y_ptr[i];
    double mui = get_mu_i(mu, i);
    double y_clamped = (yi < 1.0) ? 1.0 : yi;
    double y_theta = yi + theta;
    sum += wt_ptr[i] * (yi * std::log(y_clamped / mui) -
                        y_theta * std::log(y_theta / (mui + theta)));
  }
  return 2.0 * sum;
}

template <typename MuType>
inline double dev_resids_(const vec &y, const MuType &mu, const double &theta,
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

inline mat sandwich_vcov_(const mat &MX, const vec &y, const vec &mu,
                          const mat &H, const field<uvec> &cluster_groups) {
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
// Two-way cluster covariance matrix (Cameron-Gelbach-Miller 2011)
// V_{2way} = V_{c1} + V_{c2} - V_{c1 x c2}
///////////////////////////////////////////////////////////////////////////

inline mat sandwich_vcov_twoway_(const mat &MX, const vec &y, const vec &mu,
                                 const mat &H, const field<uvec> &cl1_groups,
                                 const field<uvec> &cl2_groups) {
  const uword n = MX.n_rows;
  const uword G1 = cl1_groups.n_elem;
  const uword G2 = cl2_groups.n_elem;

  // Build obs -> cl1 / cl2 reverse maps
  std::vector<uword> obs_to_cl1(n), obs_to_cl2(n);
  for (uword g = 0; g < G1; ++g) {
    const uvec &idx = cl1_groups(g);
    for (uword i = 0; i < idx.n_elem; ++i)
      obs_to_cl1[idx(i)] = g;
  }
  for (uword h = 0; h < G2; ++h) {
    const uvec &idx = cl2_groups(h);
    for (uword i = 0; i < idx.n_elem; ++i)
      obs_to_cl2[idx(i)] = h;
  }

  // Build interaction groups: (g, h) pair -> bucket of obs indices
  std::unordered_map<uword, uword> pair_map;
  pair_map.reserve(std::min(G1 * G2, n));
  std::vector<std::vector<uword>> pair_buckets;
  for (uword i = 0; i < n; ++i) {
    const uword key = obs_to_cl1[i] * G2 + obs_to_cl2[i];
    auto it = pair_map.find(key);
    if (it == pair_map.end()) {
      pair_map[key] = pair_buckets.size();
      pair_buckets.push_back({i});
    } else {
      pair_buckets[it->second].push_back(i);
    }
  }

  field<uvec> cl12_groups(pair_buckets.size());
  for (uword g = 0; g < pair_buckets.size(); ++g) {
    cl12_groups(g) = uvec(pair_buckets[g]);
  }

  const mat V1 = sandwich_vcov_(MX, y, mu, H, cl1_groups);
  const mat V2 = sandwich_vcov_(MX, y, mu, H, cl2_groups);
  const mat V12 = sandwich_vcov_(MX, y, mu, H, cl12_groups);

  return V1 + V2 - V12;
}

///////////////////////////////////////////////////////////////////////////
// Heteroskedastic-robust (HC0) covariance matrix
///////////////////////////////////////////////////////////////////////////

// HC0: V = H^{-1} * (sum_i e_i^2 * x_i x_i') * H^{-1}
// This is the sandwich with every observation as its own cluster.
// Works for both LM and GLM; just pass the demeaned design matrix (MX)
// and the fitted residuals e = y - mu (or y - fitted for LM).

inline mat sandwich_vcov_hetero_(const mat &MX, const vec &resid,
                                 const mat &H) {
  const uword p = MX.n_cols;
  const uword n = MX.n_rows;

  mat H_inv;
  if (!inv_sympd(H_inv, H)) {
    if (!inv(H_inv, H)) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  // Meat: sum_i e_i^2 * x_i x_i'
  // accumulated column-by-column to stay cache-friendly
  mat meat(p, p, fill::zeros);
  for (uword i = 0; i < n; ++i) {
    const double ei2 = resid(i) * resid(i);
    const rowvec xi = MX.row(i);
    meat += ei2 * xi.t() * xi;
  }

  return H_inv * meat * H_inv;
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

// Standard one-way clustering for M-estimators
inline mat sandwich_vcov_mestimator_(const mat &A, const mat &scores,
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

// Memory-efficient overload: computes scores on-the-fly from MX and resid
// Avoids N*P scores matrix allocation when score_i = resid_i * MX_i
inline mat sandwich_vcov_mestimator_(const mat &A, const mat &MX,
                                     const vec &resid,
                                     const field<uvec> &cluster_groups) {
  const uword p = A.n_cols;
  const uword G = cluster_groups.n_elem;

  mat A_inv;
  if (!inv_sympd(A_inv, A)) {
    if (!inv(A_inv, A)) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  const double adj = (G > 1) ? static_cast<double>(G) / (G - 1.0) : 1.0;
  const double *resid_ptr = resid.memptr();

  mat B(p, p, fill::zeros);
  vec cluster_score(p);

  for (uword g = 0; g < G; ++g) {
    const uvec &idx = cluster_groups(g);
    const uword ng = idx.n_elem;
    if (ng == 0)
      continue;

    cluster_score.zeros();
    double *cs_ptr = cluster_score.memptr();
    const uword *idx_ptr = idx.memptr();

    for (uword i = 0; i < ng; ++i) {
      const uword obs = idx_ptr[i];
      const double r = resid_ptr[obs];
      for (uword j = 0; j < p; ++j) {
        cs_ptr[j] += r * MX(obs, j);
      }
    }

    B += cluster_score * cluster_score.t();
  }

  return (adj * A_inv) * B * A_inv;
}

// Dyadic clustering for M-estimators
// For dyadic data, observations (g,h) and (g',h') are correlated if they share
// at least one entity: g==g', h==h', g==h', or h==g'
// This requires passing entity IDs separately from cluster IDs
inline mat sandwich_vcov_mestimator_dyadic_(const mat &A, const mat &scores,
                                            const field<uvec> &entity1_groups,
                                            const field<uvec> &entity2_groups) {
  const uword p = A.n_cols;
  const uword n_obs = scores.n_rows;
  const uword G1 =
      entity1_groups
          .n_elem; // Number of unique entity 1 values (e.g., exporters)
  const uword G2 =
      entity2_groups
          .n_elem; // Number of unique entity 2 values (e.g., importers)

  // Bread: A^{-1}
  mat A_inv;
  if (!inv_sympd(A_inv, A)) {
    if (!inv(A_inv, A)) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  // Build mapping from observation to its entities
  std::vector<uword> obs_to_entity1(n_obs);
  std::vector<uword> obs_to_entity2(n_obs);

  for (uword g = 0; g < G1; ++g) {
    const uvec &idx = entity1_groups(g);
    for (uword i = 0; i < idx.n_elem; ++i) {
      obs_to_entity1[idx(i)] = g;
    }
  }

  for (uword h = 0; h < G2; ++h) {
    const uvec &idx = entity2_groups(h);
    for (uword i = 0; i < idx.n_elem; ++i) {
      obs_to_entity2[idx(i)] = h;
    }
  }

  // Compute entity-level scores by summing observations within each entity
  std::vector<vec> entity1_scores(G1, vec(p, fill::zeros));
  std::vector<vec> entity2_scores(G2, vec(p, fill::zeros));

  for (uword i = 0; i < n_obs; ++i) {
    vec score_i = scores.row(i).t();
    entity1_scores[obs_to_entity1[i]] += score_i;
    entity2_scores[obs_to_entity2[i]] += score_i;
  }

  // Compute dyad-pair aggregate scores
  // T_{gh} = sum_{i: e1_i=g, e2_i=h} s_i
  // Key: g * G2 + h  (packed pair index using the aligned codebook)
  std::unordered_map<uword, vec> dyad_scores;
  dyad_scores.reserve(n_obs);
  for (uword i = 0; i < n_obs; ++i) {
    const uword key = obs_to_entity1[i] * G2 + obs_to_entity2[i];
    auto it = dyad_scores.find(key);
    if (it == dyad_scores.end()) {
      dyad_scores[key] = scores.row(i).t();
    } else {
      it->second += scores.row(i).t();
    }
  }

  // Dyadic meat — Cameron & Miller (2014) full decomposition:
  // B = B_11 + B_22 + B_12 + B_21 - B_same - B_rev
  //
  // where
  // S^1_g = sum_{i: e1_i=g} s_i,  S^2_h = sum_{i: e2_i=h} s_i,
  // T_{gh}  = sum_{i: e1_i=g, e2_i=h} s_i   (dyad aggregate score)
  //
  // B_11    = sum_g S^1_g (S^1_g)'  [same exporter]
  // B_22    = sum_h S^2_h (S^2_h)'  [same importer]
  // B_12    = sum_e S^1_e (S^2_e)'  [entity1_i = entity2_j]
  // B_21    = sum_e S^2_e (S^1_e)'  [entity2_i = entity1_j]
  // B_same  = sum_{g,h} T_{gh} T_{gh}'   [same dyad pair, corrects
  //           double-counting in B11+B22 for same-dyad pairs]
  // B_rev   = sum_{g,h} T_{gh} T_{hg}'   [reverse dyad (g,h)/(h,g),
  //           corrects double-counting in B12+B21 for reverse pairs]
  //
  // B_same + B_rev replaces the simpler -B_diag used in two-way clustering.
  // For cross-sectional data (one obs per dyad) B_same == B_diag, so the
  // reduction to B11+B22+B12+B21 - B_diag - B_rev is equivalent.

  // B_11
  mat B11(p, p, fill::zeros);
  for (uword g = 0; g < G1; ++g)
    B11 += entity1_scores[g] * entity1_scores[g].t();

  // B_22
  mat B22(p, p, fill::zeros);
  for (uword h = 0; h < G2; ++h)
    B22 += entity2_scores[h] * entity2_scores[h].t();

  // B_12 and B_21 (cross terms; G1==G2 with aligned codebook)
  const uword G_E = G1; // G1 == G2 after alignment
  mat B12(p, p, fill::zeros);
  mat B21(p, p, fill::zeros);
  for (uword e = 0; e < G_E; ++e) {
    B12 += entity1_scores[e] * entity2_scores[e].t();
    B21 += entity2_scores[e] * entity1_scores[e].t();
  }

  // B_same and B_rev from dyad aggregate scores
  mat B_same(p, p, fill::zeros);
  mat B_rev(p, p, fill::zeros);
  for (const auto &kv : dyad_scores) {
    const uword key_gh = kv.first;
    const vec &T_gh = kv.second;

    B_same += T_gh * T_gh.t();

    // Reverse key: h * G2 + g
    const uword g = key_gh / G2;
    const uword h = key_gh % G2;
    const uword key_hg = h * G2 + g;
    auto it_rev = dyad_scores.find(key_hg);
    if (it_rev != dyad_scores.end()) {
      B_rev += T_gh * it_rev->second.t();
    }
  }

  mat B = B11 + B22 + B12 + B21 - B_same - B_rev;

  // Degrees-of-freedom adjustment: G / (G - 1) where G = number of unique
  // entities (same set for both dimensions after aligned codebook).
  const double adj = (G_E > 1) ? static_cast<double>(G_E) / (G_E - 1.0) : 1.0;

  // Sandwich: A^{-1} * (adj * B) * A^{-1}
  return (adj * A_inv) * B * A_inv;
}

// Memory-efficient overload: computes scores on-the-fly from MX and resid
// Avoids N*P scores matrix allocation when score_i = resid_i * MX_i
inline mat sandwich_vcov_mestimator_dyadic_(const mat &A, const mat &MX,
                                            const vec &resid,
                                            const field<uvec> &entity1_groups,
                                            const field<uvec> &entity2_groups) {
  const uword p = A.n_cols;
  const uword n_obs = MX.n_rows;
  const uword G1 = entity1_groups.n_elem;
  const uword G2 = entity2_groups.n_elem;

  mat A_inv;
  if (!inv_sympd(A_inv, A)) {
    if (!inv(A_inv, A)) {
      return mat(p, p, fill::value(datum::inf));
    }
  }

  // Build mapping from observation to its entities
  std::vector<uword> obs_to_entity1(n_obs);
  std::vector<uword> obs_to_entity2(n_obs);

  for (uword g = 0; g < G1; ++g) {
    const uvec &idx = entity1_groups(g);
    for (uword i = 0; i < idx.n_elem; ++i)
      obs_to_entity1[idx(i)] = g;
  }

  for (uword h = 0; h < G2; ++h) {
    const uvec &idx = entity2_groups(h);
    for (uword i = 0; i < idx.n_elem; ++i)
      obs_to_entity2[idx(i)] = h;
  }

  // Compute entity-level scores on-the-fly
  const double *resid_ptr = resid.memptr();
  std::vector<vec> entity1_scores(G1, vec(p, fill::zeros));
  std::vector<vec> entity2_scores(G2, vec(p, fill::zeros));
  std::unordered_map<uword, vec> dyad_scores;
  dyad_scores.reserve(n_obs);

  for (uword i = 0; i < n_obs; ++i) {
    const double r = resid_ptr[i];
    const uword e1 = obs_to_entity1[i];
    const uword e2 = obs_to_entity2[i];
    const uword key = e1 * G2 + e2;

    // Compute score_i = r * MX.row(i) and accumulate
    for (uword j = 0; j < p; ++j) {
      const double s = r * MX(i, j);
      entity1_scores[e1](j) += s;
      entity2_scores[e2](j) += s;
    }

    // Accumulate dyad scores
    auto it = dyad_scores.find(key);
    if (it == dyad_scores.end()) {
      vec s(p);
      for (uword j = 0; j < p; ++j)
        s(j) = r * MX(i, j);
      dyad_scores[key] = std::move(s);
    } else {
      for (uword j = 0; j < p; ++j)
        it->second(j) += r * MX(i, j);
    }
  }

  // Compute dyadic meat components (same logic as above)
  mat B11(p, p, fill::zeros);
  for (uword g = 0; g < G1; ++g)
    B11 += entity1_scores[g] * entity1_scores[g].t();

  mat B22(p, p, fill::zeros);
  for (uword h = 0; h < G2; ++h)
    B22 += entity2_scores[h] * entity2_scores[h].t();

  const uword G_E = G1;
  mat B12(p, p, fill::zeros);
  mat B21(p, p, fill::zeros);
  for (uword e = 0; e < G_E; ++e) {
    B12 += entity1_scores[e] * entity2_scores[e].t();
    B21 += entity2_scores[e] * entity1_scores[e].t();
  }

  mat B_same(p, p, fill::zeros);
  mat B_rev(p, p, fill::zeros);
  for (const auto &kv : dyad_scores) {
    const uword key_gh = kv.first;
    const vec &T_gh = kv.second;
    B_same += T_gh * T_gh.t();

    const uword g = key_gh / G2;
    const uword h = key_gh % G2;
    const uword key_hg = h * G2 + g;
    auto it_rev = dyad_scores.find(key_hg);
    if (it_rev != dyad_scores.end()) {
      B_rev += T_gh * it_rev->second.t();
    }
  }

  mat B = B11 + B22 + B12 + B21 - B_same - B_rev;
  const double adj = (G_E > 1) ? static_cast<double>(G_E) / (G_E - 1.0) : 1.0;

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

// ============================================================================
// Group-level separation pre-filter
// Drops observations in FE groups where all y==0 (Poisson/NegBin)
// or all y==0 or all y==1 (Binomial). Iterates until a fixed point.
// Equivalent to the R-side drop_by_link_type_() but runs in C++.
// ============================================================================

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
