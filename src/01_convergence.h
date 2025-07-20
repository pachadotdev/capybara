// Computing optimal cluster coefficients (or fixed effects)

#ifndef CAPYBARA_CONVERGENCE_H
#define CAPYBARA_CONVERGENCE_H

namespace capybara {
namespace convergence {

// Family constants
enum class Family {
  POISSON,
  POISSON_LOG,
  NEGBIN,
  BINOMIAL,
  GAUSSIAN,
  INV_GAUSSIAN,
  GAMMA
};

// Utility functions
namespace utils {

// Safe division with configurable minimum threshold
inline vec safe_divide(const vec &numerator, const vec &denominator,
                       double min_val) {
  return numerator / max(denominator, min_val * ones<vec>(denominator.n_elem));
}

// Safe log with configurable minimum threshold
inline vec safe_log(const vec &x, double min_val) {
  return log(max(x, min_val * ones<vec>(x.n_elem)));
}

// Check if family is Poisson-type
inline bool is_poisson_family(Family family) {
  return (family == Family::POISSON || family == Family::POISSON_LOG);
}

// Check if family requires Newton-Raphson
inline bool requires_newton_raphson(Family family) {
  return (family == Family::NEGBIN || family == Family::BINOMIAL);
}
} // namespace utils

// Stopping criteria
inline bool continue_criterion(double a, double b, double diffMax,
                               const CapybaraParameters &params) {
  double diff = std::abs(a - b);
  double rel_diff = diff / (params.rel_tol_denom + std::abs(a));
  return (diff > diffMax) && (rel_diff > diffMax);
}

inline bool stopping_criterion(double a, double b, double diffMax,
                               const CapybaraParameters &params) {
  return !continue_criterion(a, b, diffMax, params);
}

//////////////////////////////////////////////////////////////////////////////
// CORE CLUSTER COEFFICIENT FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Irons-Tuck acceleration update
bool update_irons_tuck(vec &X, const vec &GX, const vec &GGX,
                       const CapybaraParameters &params) {
  vec delta_GX = GGX - GX;
  vec delta2_X = delta_GX - GX + X;

  double vprod = dot(delta_GX, delta2_X);
  double ssq = dot(delta2_X, delta2_X);

  if (ssq < params.irons_tuck_eps) {
    return true; // numerical convergence
  }

  double coef = vprod / ssq;
  X = GGX - coef * delta_GX;

  return false;
}

// Poisson cluster coefficients
void cluster_coef_poisson(const vec &exp_mu, const vec &sum_y, const uvec &dum,
                          vec &cluster_coef, const CapybaraParameters &params) {
  cluster_coef.zeros();

  // Accumulate exp_mu by cluster
  for (size_t i = 0; i < exp_mu.n_elem; ++i) {
    cluster_coef(dum(i)) += exp_mu(i);
  }

  // Compute coefficients with safe division
  cluster_coef =
      utils::safe_divide(sum_y, cluster_coef, params.safe_division_min) -
      exp_mu;
}

// Poisson log cluster coefficients
void cluster_coef_poisson_log(const vec &mu, const vec &sum_y, const uvec &dum,
                              vec &cluster_coef,
                              const CapybaraParameters &params) {
  size_t nb_cluster = cluster_coef.n_elem;
  size_t n_obs = mu.n_elem;

  // Find max mu for each cluster for numerical stability
  vec mu_max(nb_cluster);
  uvec initialized = zeros<uvec>(nb_cluster);

  for (size_t i = 0; i < n_obs; ++i) {
    uword d = dum(i);
    if (!initialized(d)) {
      mu_max(d) = mu(i);
      initialized(d) = 1;
    } else if (mu(i) > mu_max(d)) {
      mu_max(d) = mu(i);
    }
  }

  // Accumulate exp(mu - mu_max) by cluster
  cluster_coef.zeros();
  for (size_t i = 0; i < n_obs; ++i) {
    uword d = dum(i);
    cluster_coef(d) += std::exp(mu(i) - mu_max(d));
  }

  // Compute coefficients
  cluster_coef = utils::safe_log(sum_y, params.safe_log_min) -
                 utils::safe_log(cluster_coef, params.safe_log_min) - mu_max;
}

// Gaussian cluster coefficients
void cluster_coef_gaussian(const vec &mu, const vec &sum_y, const uvec &dum,
                           const uvec &table, vec &cluster_coef,
                           const CapybaraParameters &params) {
  cluster_coef.zeros();

  // Accumulate mu by cluster
  for (size_t i = 0; i < mu.n_elem; ++i) {
    cluster_coef(dum(i)) += mu(i);
  }

  // Compute coefficients
  vec table_dbl = conv_to<vec>::from(table);
  cluster_coef = utils::safe_divide(sum_y - cluster_coef, table_dbl,
                                    params.safe_division_min);
}

// Negative binomial cluster coefficients (Newton-Raphson + dichotomy)
void cluster_coef_negbin(const vec &mu, const vec &lhs, const vec &sum_y,
                         const uvec &dum, const uvec &obs_cluster,
                         const uvec &table, const uvec &cumtable, double theta,
                         double diffMax_NR, vec &cluster_coef,
                         const CapybaraParameters &params) {
  size_t nb_cluster = cluster_coef.n_elem;

  // Pre-compute bounds for each cluster
  vec lower_bound(nb_cluster);
  vec upper_bound(nb_cluster);

  for (size_t m = 0; m < nb_cluster; ++m) {
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    uvec cluster_indices = obs_cluster.subvec(u0, u_end - 1);
    vec cluster_mu = mu(cluster_indices);

    double mu_min = min(cluster_mu);
    double mu_max = max(cluster_mu);

    lower_bound(m) = std::log(sum_y(m)) - std::log(double(table(m))) - mu_max;
    upper_bound(m) = lower_bound(m) + (mu_max - mu_min);
  }

  // Solve for each cluster using Newton-Raphson + dichotomy
  cluster_coef.resize(nb_cluster);

  for (size_t m = 0; m < nb_cluster; ++m) {
    double x1 = 0.0;
    bool keepGoing = true;
    size_t iter = 0;

    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    double lb = lower_bound(m);
    double ub = upper_bound(m);

    // Ensure initial guess is within bounds
    if (x1 >= ub || x1 <= lb) {
      x1 = (lb + ub) / 2.0;
    }

    while (keepGoing && iter < params.iter_max_cluster) {
      ++iter;

      // Evaluate function f(x)
      double value = sum_y(m);
      for (size_t u = u0; u < u_end; ++u) {
        size_t i = obs_cluster(u);
        value -= (theta + lhs(i)) / (1.0 + theta * std::exp(-x1 - mu(i)));
      }

      // Update bounds
      if (value > 0) {
        lb = x1;
      } else {
        ub = x1;
      }

      double x0 = x1;

      if (std::abs(value) < 1e-12) {
        keepGoing = false;
      } else if (iter <= params.iter_full_dicho) {
        // Newton-Raphson step
        double derivative = 0.0;
        for (size_t u = u0; u < u_end; ++u) {
          size_t i = obs_cluster(u);
          double exp_mu = std::exp(x1 + mu(i));
          derivative -= theta * (theta + lhs(i)) /
                        ((theta / exp_mu + 1.0) * (theta + exp_mu));
        }

        if (std::abs(derivative) > 1e-12) {
          x1 = x0 - value / derivative;
        }

        // Fall back to dichotomy if out of bounds
        if (x1 >= ub || x1 <= lb) {
          x1 = (lb + ub) / 2.0;
        }
      } else {
        // Pure dichotomy
        x1 = (lb + ub) / 2.0;
      }

      if (stopping_criterion(x0, x1, diffMax_NR, params)) {
        keepGoing = false;
      }
    }

    cluster_coef(m) = x1;
  }
}

// Binomial cluster coefficients (similar to negbin)
void cluster_coef_binomial(const vec &mu, const vec &sum_y,
                           const uvec &obs_cluster, const uvec &table,
                           const uvec &cumtable, double diffMax_NR,
                           vec &cluster_coef,
                           const CapybaraParameters &params) {
  size_t nb_cluster = cluster_coef.n_elem;

  // Pre-compute bounds
  vec lower_bound(nb_cluster);
  vec upper_bound(nb_cluster);

  for (size_t m = 0; m < nb_cluster; ++m) {
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    uvec cluster_indices = obs_cluster.subvec(u0, u_end - 1);
    vec cluster_mu = mu(cluster_indices);

    double mu_min = min(cluster_mu);
    double mu_max = max(cluster_mu);

    lower_bound(m) =
        std::log(sum_y(m)) - std::log(table(m) - sum_y(m)) - mu_max;
    upper_bound(m) = lower_bound(m) + (mu_max - mu_min);
  }

  // Solve for each cluster
  cluster_coef.resize(nb_cluster);

  for (size_t m = 0; m < nb_cluster; ++m) {
    double x1 = 0.0;
    bool keepGoing = true;
    size_t iter = 0;

    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    double lb = lower_bound(m);
    double ub = upper_bound(m);

    if (x1 >= ub || x1 <= lb) {
      x1 = (lb + ub) / 2.0;
    }

    while (keepGoing && iter < params.iter_max) {
      ++iter;

      // Evaluate function
      double value = sum_y(m);
      for (size_t u = u0; u < u_end; ++u) {
        value -= 1.0 / (1.0 + std::exp(-x1 - mu(obs_cluster(u))));
      }

      // Update bounds
      if (value > 0) {
        lb = x1;
      } else {
        ub = x1;
      }

      double x0 = x1;

      if (std::abs(value) < 1e-12) {
        keepGoing = false;
      } else if (iter <= params.iter_full_dicho) {
        // Newton-Raphson step
        double derivative = 0.0;
        for (size_t u = u0; u < u_end; ++u) {
          double exp_mu = std::exp(x1 + mu(obs_cluster(u)));
          derivative -= 1.0 / ((1.0 / exp_mu + 1.0) * (1.0 + exp_mu));
        }

        if (std::abs(derivative) > 1e-12) {
          x1 = x0 - value / derivative;
        }

        if (x1 >= ub || x1 <= lb) {
          x1 = (lb + ub) / 2.0;
        }
      } else {
        x1 = (lb + ub) / 2.0;
      }

      if (stopping_criterion(x0, x1, diffMax_NR, params)) {
        keepGoing = false;
      }
    }

    cluster_coef(m) = x1;
  }
}

//////////////////////////////////////////////////////////////////////////////
// CLUSTER COEFFICIENT DISPATCHER
//////////////////////////////////////////////////////////////////////////////

// Unified cluster coefficient computation
void cluster_coefficients(Family family, const vec &mu, const vec &lhs,
                          const vec &sum_y, const uvec &dum,
                          const uvec &obs_cluster, const uvec &table,
                          const uvec &cumtable, double theta, double diffMax_NR,
                          vec &cluster_coef, const CapybaraParameters &params) {
  switch (family) {
  case Family::POISSON:
    cluster_coef_poisson(mu, sum_y, dum, cluster_coef, params);
    break;
  case Family::POISSON_LOG:
    cluster_coef_poisson_log(mu, sum_y, dum, cluster_coef, params);
    break;
  case Family::GAUSSIAN:
    cluster_coef_gaussian(mu, sum_y, dum, table, cluster_coef, params);
    break;
  case Family::NEGBIN:
    cluster_coef_negbin(mu, lhs, sum_y, dum, obs_cluster, table, cumtable,
                        theta, diffMax_NR, cluster_coef, params);
    break;
  case Family::BINOMIAL:
    cluster_coef_binomial(mu, sum_y, obs_cluster, table, cumtable, diffMax_NR,
                          cluster_coef, params);
    break;
  case Family::INV_GAUSSIAN:
    // For inverse Gaussian, use Gaussian approximation
    cluster_coef_gaussian(mu, sum_y, dum, table, cluster_coef, params);
    break;
  case Family::GAMMA:
    // For Gamma, use Gaussian approximation
    cluster_coef_gaussian(mu, sum_y, dum, table, cluster_coef, params);
    break;
  }
}

//////////////////////////////////////////////////////////////////////////////
// MAIN CONVERGENCE ALGORITHMS
//////////////////////////////////////////////////////////////////////////////

// Data structure for convergence algorithms - contains only data, not
// parameters
struct Convergence {
  Family family;
  size_t n_obs;
  size_t K;
  double theta;

  vec mu_init;
  vec lhs;
  uvec nb_cluster_all;

  field<uvec> dum_vector;
  field<uvec> table_vector;
  field<vec> sum_y_vector;
  field<uvec> cumtable_vector;
  field<uvec> obs_cluster_vector;
};

// Update mu with cluster coefficients
vec update_mu_with_coefficients(const vec &mu_base,
                                const field<vec> &cluster_coefs,
                                const field<uvec> &dum_vector, Family family) {
  vec mu_result = mu_base;
  size_t K = cluster_coefs.n_elem;

  for (size_t k = 0; k < K; ++k) {
    const uvec &my_dum = dum_vector(k);
    const vec &my_coef = cluster_coefs(k);

    if (utils::is_poisson_family(family)) {
      // Multiplicative updates for Poisson families
      mu_result %= my_coef(my_dum);
    } else {
      // Additive updates for other families
      mu_result += my_coef(my_dum);
    }
  }

  return mu_result;
}

// Compute all cluster coefficients for current mu
void all_cluster_coefficients(const Convergence &data,
                              const CapybaraParameters &params,
                              const vec &mu_with_coef,
                              field<vec> &cluster_coefs_dest,
                              const field<vec> &cluster_coefs_origin) {
  // Update mu starting from base
  vec mu_current = data.mu_init;

  // Add contributions from first K-1 cluster coefficients
  for (size_t k = 0; k < data.K - 1; ++k) {
    const uvec &my_dum = data.dum_vector(k);
    const vec &my_coef = cluster_coefs_origin(k);

    if (utils::is_poisson_family(data.family)) {
      mu_current %= my_coef(my_dum);
    } else {
      mu_current += my_coef(my_dum);
    }
  }

  // Compute optimal cluster coefficients from K down to 1
  for (int k = static_cast<int>(data.K) - 1; k >= 0; k--) {
    size_t uk = static_cast<size_t>(k);

    // Get current cluster data
    const uvec &my_table = data.table_vector(uk);
    const vec &my_sum_y = data.sum_y_vector(uk);
    const uvec &my_dum = data.dum_vector(uk);
    const uvec &my_cumtable = data.cumtable_vector(uk);
    const uvec &my_obs_cluster = data.obs_cluster_vector(uk);

    // Compute cluster coefficients for this level
    vec &my_cluster_coef = cluster_coefs_dest(uk);
    my_cluster_coef.resize(my_table.n_elem);

    cluster_coefficients(data.family, mu_current, data.lhs, my_sum_y, my_dum,
                         my_obs_cluster, my_table, my_cumtable, data.theta,
                         params.newton_raphson_tol, my_cluster_coef, params);

    // Update mu for next iteration if needed
    if (k > 0) {
      mu_current = data.mu_init;

      for (size_t h = 0; h < data.K; h++) {
        if (h == uk - 1)
          continue;

        const uvec &my_dum_h = data.dum_vector(h);
        const vec &my_coef_h =
            (h < uk - 1) ? cluster_coefs_origin(h) : cluster_coefs_dest(h);

        if (utils::is_poisson_family(data.family)) {
          mu_current %= my_coef_h(my_dum_h);
        } else {
          mu_current += my_coef_h(my_dum_h);
        }
      }
    }
  }
}

// Accelerated convergence algorithm
vec conv_accelerated(const Convergence &data, const CapybaraParameters &params,
                     size_t iterMax, double diffMax, size_t &final_iter,
                     bool &any_negative_poisson) {
  size_t K = data.K;

  // Initialize coefficient fields
  field<vec> X(K), GX(K), GGX(K);
  for (size_t k = 0; k < K; ++k) {
    size_t nk = data.nb_cluster_all(k);
    if (utils::is_poisson_family(data.family)) {
      X(k) = ones<vec>(nk);
    } else {
      X(k) = zeros<vec>(nk);
    }
    GX(k).resize(nk);
    GGX(k).resize(nk);
  }

  // First iteration
  all_cluster_coefficients(data, params, data.mu_init, GX, X);

  any_negative_poisson = false;

  // Check if iteration is needed
  bool keepGoing = false;
  for (size_t k = 0; k < K - 1; ++k) {
    const vec &X_k = X(k);
    const vec &GX_k = GX(k);
    for (size_t i = 0; i < X_k.n_elem; ++i) {
      if (continue_criterion(X_k(i), GX_k(i), diffMax, params)) {
        keepGoing = true;
        break;
      }
    }
    if (keepGoing)
      break;
  }

  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    // Compute GGX
    all_cluster_coefficients(data, params, data.mu_init, GGX, GX);

    // Apply Irons-Tuck acceleration to first K-1 coefficients
    for (size_t k = 0; k < K - 1; ++k) {
      bool numconv = update_irons_tuck(X(k), GX(k), GGX(k), params);
      if (numconv) {
        keepGoing = false;
        break;
      }
    }

    if (!keepGoing)
      break;

    // Check for negative Poisson coefficients
    if (utils::is_poisson_family(data.family)) {
      for (size_t k = 0; k < K - 1; ++k) {
        if (any(X(k) <= 0)) {
          any_negative_poisson = true;
          keepGoing = false;
          break;
        }
      }
    }

    if (!keepGoing)
      break;

    // Update GX
    all_cluster_coefficients(data, params, data.mu_init, GX, X);

    // Check convergence
    keepGoing = false;
    for (size_t k = 0; k < K - 1; ++k) {
      const vec &X_k = X(k);
      const vec &GX_k = GX(k);
      for (size_t i = 0; i < X_k.n_elem; ++i) {
        if (continue_criterion(X_k(i), GX_k(i), diffMax, params)) {
          keepGoing = true;
          break;
        }
      }
      if (keepGoing)
        break;
    }
  }

  // Final coefficient computation
  all_cluster_coefficients(data, params, data.mu_init, GGX, GX);

  // Compute final mu
  vec mu_result = update_mu_with_coefficients(data.mu_init, GGX,
                                              data.dum_vector, data.family);

  final_iter = iter;
  return mu_result;
}

// Sequential convergence algorithm
vec conv_sequential(const Convergence &data, const CapybaraParameters &params,
                    size_t iterMax, double diffMax, size_t &final_iter) {
  size_t K = data.K;

  // Initialize coefficient fields
  field<vec> X(K), X_new(K);
  for (size_t k = 0; k < K; ++k) {
    size_t nk = data.nb_cluster_all(k);
    if (utils::is_poisson_family(data.family)) {
      X(k) = ones<vec>(nk);
    } else {
      X(k) = zeros<vec>(nk);
    }
    X_new(k).resize(nk);
  }

  // First iteration
  all_cluster_coefficients(data, params, data.mu_init, X_new, X);

  bool keepGoing = true;
  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    // Alternate between X and X_new
    if (iter % 2 == 1) {
      all_cluster_coefficients(data, params, data.mu_init, X, X_new);
    } else {
      all_cluster_coefficients(data, params, data.mu_init, X_new, X);
    }

    // Check convergence for first K-1 coefficients
    keepGoing = false;
    const field<vec> &X_current = (iter % 2 == 1) ? X_new : X;
    const field<vec> &X_prev = (iter % 2 == 1) ? X : X_new;

    for (size_t k = 0; k < K - 1; ++k) {
      const vec &curr = X_current(k);
      const vec &prev = X_prev(k);
      for (size_t i = 0; i < curr.n_elem; ++i) {
        if (continue_criterion(prev(i), curr(i), diffMax, params)) {
          keepGoing = true;
          break;
        }
      }
      if (keepGoing)
        break;
    }
  }

  // Use the latest coefficients
  const field<vec> &final_coefs = (iter % 2 == 0) ? X_new : X;

  // Compute final mu
  vec mu_result = update_mu_with_coefficients(data.mu_init, final_coefs,
                                              data.dum_vector, data.family);

  final_iter = iter;
  return mu_result;
}

//////////////////////////////////////////////////////////////////////////////
// TWO-WAY FIXED EFFECTS FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// 2-way Poisson sequential convergence
vec conv_two_way_poisson_sequential(
    size_t n_i, size_t n_j, size_t n_cells, const uvec &index_i,
    const uvec &index_j, const field<uvec> &dum_vector, const vec &sum_y_vector,
    size_t iterMax, double diffMax, const vec &exp_mu_in, const uvec &order,
    const CapybaraParameters &params, size_t &final_iter) {
  // Build matrix representation
  uvec mat_row(n_cells);
  uvec mat_col(n_cells);
  vec mat_value(n_cells);

  size_t n_obs = exp_mu_in.n_elem;
  size_t index_current = 0;
  double value = exp_mu_in(order(0));

  for (size_t i = 1; i < n_obs; ++i) {
    if (index_j(i) != index_j(i - 1) || index_i(i) != index_i(i - 1)) {
      // Save current accumulated value
      mat_row(index_current) = index_i(i - 1);
      mat_col(index_current) = index_j(i - 1);
      mat_value(index_current) = value;
      index_current++;

      // Start new accumulation
      value = exp_mu_in(order(i));
    } else {
      value += exp_mu_in(order(i));
    }
  }

  // Save last value
  mat_row(index_current) = index_i(n_obs - 1);
  mat_col(index_current) = index_j(n_obs - 1);
  mat_value(index_current) = value;

  // Initialize coefficient vectors
  vec alpha = ones<vec>(n_i);
  vec beta = ones<vec>(n_j);
  vec alpha_new(n_i);
  vec beta_new(n_j);

  vec ca = sum_y_vector.head(n_i);
  vec cb = sum_y_vector.tail(n_j);

  // Helper function for 2-way Poisson computation
  auto two_way_poisson_coefs = [&](const vec &alpha_in, vec &alpha_out,
                                   vec &beta_out) {
    alpha_out.zeros();
    beta_out.zeros();

    // Compute beta
    for (size_t obs = 0; obs < n_cells; ++obs) {
      beta_out(mat_col(obs)) += mat_value(obs) * alpha_in(mat_row(obs));
    }
    beta_out = utils::safe_divide(cb, beta_out, params.safe_division_min);

    // Compute alpha
    for (size_t obs = 0; obs < n_cells; ++obs) {
      alpha_out(mat_row(obs)) += mat_value(obs) * beta_out(mat_col(obs));
    }
    alpha_out = utils::safe_divide(ca, alpha_out, params.safe_division_min);
  };

  bool keepGoing = true;
  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    if (iter % 2 == 1) {
      two_way_poisson_coefs(alpha, alpha_new, beta_new);
    } else {
      two_way_poisson_coefs(alpha_new, alpha, beta);
    }

    // Check convergence on alpha coefficients
    keepGoing = false;
    const vec &alpha_current = (iter % 2 == 1) ? alpha : alpha_new;
    const vec &alpha_prev = (iter % 2 == 1) ? alpha_new : alpha;

    for (size_t i = 0; i < n_i; ++i) {
      if (continue_criterion(alpha_current(i), alpha_prev(i), diffMax,
                             params)) {
        keepGoing = true;
        break;
      }
    }
  }

  // Use final coefficients
  const vec &alpha_final = (iter % 2 == 0) ? alpha_new : alpha;
  const vec &beta_final = (iter % 2 == 0) ? beta_new : beta;

  // Compute result mu
  vec result_mu(n_obs);
  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    result_mu(obs) =
        exp_mu_in(obs) * alpha_final(dum_i(obs)) * beta_final(dum_j(obs));
  }

  final_iter = iter;
  return result_mu;
}

// 2-way Gaussian accelerated convergence
vec conv_two_way_gaussian_accelerated(
    size_t n_i, size_t n_j, size_t n_cells, const uvec &mat_row,
    const uvec &mat_col, const vec &mat_value_Ab, const vec &mat_value_Ba,
    const field<uvec> &dum_vector, const vec &lhs,
    const vec &invTableCluster_vector, size_t iterMax, double diffMax,
    const vec &mu_in, size_t &final_iter, const CapybaraParameters &params) {
  size_t n_obs = mu_in.n_elem;
  vec resid = lhs - mu_in;

  // Compute constants
  vec const_a = zeros<vec>(n_i);
  vec const_b = zeros<vec>(n_j);

  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);
  const vec &invTable_i = invTableCluster_vector.head(n_i);
  const vec &invTable_j =
      invTableCluster_vector.tail(invTableCluster_vector.n_elem - n_i);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    double resid_val = resid(obs);
    const_a(dum_i(obs)) += resid_val * invTable_i(dum_i(obs));
    const_b(dum_j(obs)) += resid_val * invTable_j(dum_j(obs));
  }

  // Compute a_tilde = const_a - (Ab %*% const_b)
  vec a_tilde = const_a;
  for (size_t obs = 0; obs < n_cells; ++obs) {
    a_tilde(mat_row(obs)) -= mat_value_Ab(obs) * const_b(mat_col(obs));
  }

  // Helper function for 2-way Gaussian computation
  auto two_way_gaussian_coefs = [&](const vec &alpha_in, vec &alpha_out) {
    alpha_out = a_tilde;
    vec beta_temp = zeros<vec>(n_j);

    // Compute beta = Ba %*% alpha
    for (size_t obs = 0; obs < n_cells; ++obs) {
      beta_temp(mat_col(obs)) += mat_value_Ba(obs) * alpha_in(mat_row(obs));
    }

    // Update alpha = a_tilde + (Ab %*% beta)
    for (size_t obs = 0; obs < n_cells; ++obs) {
      alpha_out(mat_row(obs)) += mat_value_Ab(obs) * beta_temp(mat_col(obs));
    }
  };

  // Initialize with Irons-Tuck acceleration
  vec X = zeros<vec>(n_i);
  vec GX(n_i);
  vec GGX(n_i);

  two_way_gaussian_coefs(X, GX);

  bool keepGoing = true;
  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    two_way_gaussian_coefs(GX, GGX);

    bool numconv = update_irons_tuck(X, GX, GGX, params);
    if (numconv)
      break;

    two_way_gaussian_coefs(X, GX);

    // Check convergence
    keepGoing = false;
    for (size_t i = 0; i < n_i; ++i) {
      if (continue_criterion(X(i), GX(i), diffMax, params)) {
        keepGoing = true;
        break;
      }
    }
  }

  // Compute final beta and alpha
  vec beta_final = const_b;
  for (size_t obs = 0; obs < n_cells; ++obs) {
    beta_final(mat_col(obs)) -= mat_value_Ba(obs) * GX(mat_row(obs));
  }

  vec alpha_final = const_a;
  for (size_t obs = 0; obs < n_cells; ++obs) {
    alpha_final(mat_row(obs)) -= mat_value_Ab(obs) * beta_final(mat_col(obs));
  }

  // Compute result mu
  vec result_mu(n_obs);
  for (size_t obs = 0; obs < n_obs; ++obs) {
    result_mu(obs) =
        mu_in(obs) + alpha_final(dum_i(obs)) + beta_final(dum_j(obs));
  }

  final_iter = iter;
  return result_mu;
}

//////////////////////////////////////////////////////////////////////////////
// DERIVATIVE CONVERGENCE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Sequential derivative convergence
field<vec> conv_derivative_sequential(size_t iterMax, double diffMax,
                                      size_t n_vars, const uvec &nb_cluster_all,
                                      const vec &ll_d2,
                                      const field<vec> &jacob_matrix,
                                      const field<vec> &deriv_init_matrix,
                                      const field<uvec> &dum_matrix) {
  size_t n_obs = ll_d2.n_elem;
  size_t K = nb_cluster_all.n_elem;

  // Copy initial derivatives
  field<vec> deriv_matrix = deriv_init_matrix;

  // Compute sum_ll_d2 for each cluster
  field<vec> sum_ll_d2(K);
  for (size_t k = 0; k < K; ++k) {
    sum_ll_d2(k) = zeros<vec>(nb_cluster_all(k));
    const uvec &my_dum = dum_matrix(k);

    for (size_t i = 0; i < n_obs; ++i) {
      sum_ll_d2(k)(my_dum(i)) += ll_d2(i);
    }
  }

  // Process each variable
  for (size_t v = 0; v < n_vars; ++v) {
    vec &my_deriv = deriv_matrix(v);
    const vec &my_jac = jacob_matrix(v);

    bool keepGoing = true;
    size_t iter = 0;

    while (keepGoing && iter < iterMax) {
      ++iter;
      keepGoing = false;

      // Update clusters sequentially from K to 1
      for (int k = static_cast<int>(K) - 1; k >= 0; k--) {
        size_t uk = static_cast<size_t>(k);

        const uvec &my_dum = dum_matrix(uk);
        const vec &my_sum_ll_d2 = sum_ll_d2(uk);
        size_t nb_cluster = nb_cluster_all(uk);

        // Compute derivative coefficients
        vec my_deriv_coef = zeros<vec>(nb_cluster);
        for (size_t i = 0; i < n_obs; ++i) {
          my_deriv_coef(my_dum(i)) += (my_jac(i) + my_deriv(i)) * ll_d2(i);
        }
        my_deriv_coef /= -my_sum_ll_d2;

        // Update derivatives
        for (size_t i = 0; i < n_obs; ++i) {
          my_deriv(i) += my_deriv_coef(my_dum(i));
        }

        // Check stopping criterion
        if (!keepGoing && any(abs(my_deriv_coef) > diffMax)) {
          keepGoing = true;
        }
      }
    }
  }

  return deriv_matrix;
}

} // namespace convergence
} // namespace capybara

#endif // CAPYBARA_CONVERGENCE_H
