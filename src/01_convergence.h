// Computing optimal cluster coefficients (or fixed effects)

#ifndef CAPYBARA_CONVERGENCE_H
#define CAPYBARA_CONVERGENCE_H

namespace capybara {
namespace convergence {

enum class Family {
  POISSON,
  POISSON_LOG,
  NEGBIN,
  BINOMIAL,
  GAUSSIAN,
  INV_GAUSSIAN,
  GAMMA
};

namespace utils {

inline vec safe_divide(const vec &numerator, const vec &denominator,
                       double min_val) {
  return numerator / max(denominator, min_val);
}

inline vec safe_log(const vec &x, double min_val) {
  return log(max(x, min_val));
}

inline bool is_poisson_family(Family family) {
  return (family == Family::POISSON || family == Family::POISSON_LOG);
}

inline bool requires_newton_raphson(Family family) {
  return (family == Family::NEGBIN || family == Family::BINOMIAL);
}
} // namespace utils

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

inline bool vector_continue_criterion(const vec &a, const vec &b, double diffMax,
                                     const CapybaraParameters &params) {
  vec diff = abs(a - b);
  vec rel_diff = diff / (params.rel_tol_denom + abs(a));
  return any((diff > diffMax) % (rel_diff > diffMax));
}
inline bool vector_stopping_criterion(const vec &a, const vec &b, double diffMax,
                                     const CapybaraParameters &params) {
  return !vector_continue_criterion(a, b, diffMax, params);
}

struct ClusterWorkspace {
  vec accumulator;
  vec mu_max;
  vec exp_values;
  uvec counts;
  
  ClusterWorkspace(size_t max_cluster_size, size_t n_obs) {
    accumulator.set_size(max_cluster_size);
    mu_max.set_size(max_cluster_size);
    exp_values.set_size(n_obs);
    counts.set_size(max_cluster_size);
  }
};

inline void grouped_accu(const vec &values, const uvec &indices, 
                                  vec &result, ClusterWorkspace &ws) {
  result.zeros();
  
  for (size_t g = 0; g < result.n_elem; ++g) {
    result(g) = accu(values(find(indices == g)));
  }
}

bool irons_tuck(vec &X, const vec &GX, const vec &GGX,
                                  const CapybaraParameters &params) {
  vec delta_GX = GGX - GX;
  vec delta2_X = delta_GX - GX + X;

  double vprod = dot(delta_GX, delta2_X);
  double ssq = dot(delta2_X, delta2_X);

  if (ssq < params.irons_tuck_eps) {
    return true; 
  }

  X = GGX - (vprod / ssq) * delta_GX;
  return false;
}

void cluster_coef_poisson_vectorized(const vec &exp_mu, const vec &sum_y, const uvec &dum,
                                    vec &cluster_coef, ClusterWorkspace &ws) {
  size_t nb_cluster = cluster_coef.n_elem;
  
  vec accumulator(nb_cluster);
  accumulator.zeros();
  
  grouped_accu(exp_mu, dum, accumulator, ws);
  
  cluster_coef = sum_y / accumulator;
}

void cluster_coef_poisson_log_vectorized(const vec &mu, const vec &sum_y, const uvec &dum,
                                        vec &cluster_coef, ClusterWorkspace &ws) {
  size_t nb_cluster = cluster_coef.n_elem;
  size_t n_obs = mu.n_elem;

  cluster_coef.zeros();
  vec mu_max(nb_cluster);
  mu_max.fill(-datum::inf);

  for (size_t i = 0; i < n_obs; ++i) {
    uword cluster_id = dum(i);
    if (mu(i) > mu_max(cluster_id)) {
      mu_max(cluster_id) = mu(i);
    }
  }

  vec exp_diff(n_obs);
  for (size_t i = 0; i < n_obs; ++i) {
    exp_diff(i) = exp(mu(i) - mu_max(dum(i)));
  }
  
  grouped_accu(exp_diff, dum, cluster_coef, ws);
  
  cluster_coef = log(sum_y) - log(cluster_coef) - mu_max;
}

void cluster_coef_gaussian_vectorized(const vec &mu, const vec &sum_y, const uvec &dum,
                                     const uvec &table, vec &cluster_coef,
                                     ClusterWorkspace &ws, double safe_min) {
  grouped_accu(mu, dum, cluster_coef, ws);
  
  vec table_dbl = conv_to<vec>::from(table);
  cluster_coef = (sum_y - cluster_coef) / max(table_dbl, safe_min);
}

void cluster_coefficients_vectorized(Family family, const vec &mu, const vec &lhs,
                                    const vec &sum_y, const uvec &dum,
                                    const uvec &obs_cluster, const uvec &table,
                                    const uvec &cumtable, double theta, double diffMax_NR,
                                    vec &cluster_coef, ClusterWorkspace &ws,
                                    const CapybaraParameters &params) {
  switch (family) {
  case Family::POISSON:
    cluster_coef_poisson_vectorized(mu, sum_y, dum, cluster_coef, ws);
    break;
  case Family::POISSON_LOG:
    cluster_coef_poisson_log_vectorized(mu, sum_y, dum, cluster_coef, ws);
    break;
  case Family::GAUSSIAN:
  case Family::INV_GAUSSIAN:
  case Family::GAMMA:
    cluster_coef_gaussian_vectorized(mu, sum_y, dum, table, cluster_coef, ws,
                                    params.safe_division_min);
    break;
  case Family::NEGBIN:
    // Needs Newton-Raphson
    break;
  case Family::BINOMIAL:
    // Needs Newton-Raphson
    break;
  }
}

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
  
  ClusterWorkspace workspace;
  
  Convergence(Family fam, size_t n, size_t k, double th,
              const vec &mu_i, const vec &lhs_i, const uvec &nb_cl)
      : family(fam), n_obs(n), K(k), theta(th),
        mu_init(mu_i), lhs(lhs_i), nb_cluster_all(nb_cl),
        workspace(nb_cl.max(), n) {}
};

struct ConvergenceWorkspace {
  field<vec> X, GX, GGX, X_new;
  
  vec mu_current;
  vec mu_result;
  
  ConvergenceWorkspace(const Convergence &data) {
    size_t K = data.K;
    size_t n_obs = data.n_obs;
    
    X.set_size(K);
    GX.set_size(K);
    GGX.set_size(K);
    X_new.set_size(K);
    
    for (size_t k = 0; k < K; ++k) {
      size_t nk = data.nb_cluster_all(k);
      X(k).set_size(nk);
      GX(k).set_size(nk);
      GGX(k).set_size(nk);
      X_new(k).set_size(nk);
    }
    
    mu_current.set_size(n_obs);
    mu_result.set_size(n_obs);
  }
};

inline void update_mu_vectorized(vec &mu_result, const vec &mu_base,
                                const field<vec> &cluster_coefs,
                                const field<uvec> &dum_vector, Family family) {
  // TODO: avoid Initial copy
  mu_result = mu_base;
  size_t K = cluster_coefs.n_elem;

  if (utils::is_poisson_family(family)) {
    for (size_t k = 0; k < K; ++k) {
      mu_result %= cluster_coefs(k)(dum_vector(k));
    }
  } else {
    for (size_t k = 0; k < K; ++k) {
      mu_result += cluster_coefs(k)(dum_vector(k));
    }
  }
}

void all_cluster_coefficients_vectorized(const Convergence &data,
                                         const CapybaraParameters &params,
                                         field<vec> &cluster_coefs_dest,
                                         const field<vec> &cluster_coefs_origin,
                                         ConvergenceWorkspace &workspace) {
  vec &mu_current = workspace.mu_current;
  mu_current = data.mu_init;

  if (utils::is_poisson_family(data.family)) {
    for (size_t k = 0; k < data.K - 1; ++k) {
      mu_current %= cluster_coefs_origin(k)(data.dum_vector(k));
    }
  } else {
    for (size_t k = 0; k < data.K - 1; ++k) {
      mu_current += cluster_coefs_origin(k)(data.dum_vector(k));
    }
  }

  for (int k = static_cast<int>(data.K) - 1; k >= 0; k--) {
    size_t uk = static_cast<size_t>(k);

    cluster_coefficients_vectorized(
        data.family, mu_current, data.lhs, data.sum_y_vector(uk),
        data.dum_vector(uk), data.obs_cluster_vector(uk), 
        data.table_vector(uk), data.cumtable_vector(uk),
        data.theta, params.newton_raphson_tol, 
        cluster_coefs_dest(uk), const_cast<ClusterWorkspace&>(data.workspace), params);

    if (k > 0) {
      mu_current = data.mu_init;

      if (utils::is_poisson_family(data.family)) {
        for (size_t h = 0; h < data.K; h++) {
          if (h == uk - 1) continue;
          const vec &coef = (h < uk - 1) ? cluster_coefs_origin(h) : cluster_coefs_dest(h);
          mu_current %= coef(data.dum_vector(h));
        }
      } else {
        for (size_t h = 0; h < data.K; h++) {
          if (h == uk - 1) continue;
          const vec &coef = (h < uk - 1) ? cluster_coefs_origin(h) : cluster_coefs_dest(h);
          mu_current += coef(data.dum_vector(h));
        }
      }
    }
  }
}

vec conv_accelerated_vectorized(const Convergence &data, const CapybaraParameters &params,
                               size_t iterMax, double diffMax, size_t &final_iter,
                               bool &any_negative_poisson) {
  static thread_local std::unique_ptr<ConvergenceWorkspace> workspace_cache;
  if (!workspace_cache || workspace_cache->X.n_elem != data.K) {
    workspace_cache = std::make_unique<ConvergenceWorkspace>(data);
  }
  ConvergenceWorkspace &workspace = *workspace_cache;
  
  size_t K = data.K;
  field<vec> &X = workspace.X;
  field<vec> &GX = workspace.GX;
  field<vec> &GGX = workspace.GGX;

  for (size_t k = 0; k < K; ++k) {
    if (utils::is_poisson_family(data.family)) {
      X(k).ones();
    } else {
      X(k).zeros();
    }
  }

  all_cluster_coefficients_vectorized(data, params, GX, X, workspace);

  any_negative_poisson = false;

  bool keepGoing = false;
  for (size_t k = 0; k < K - 1; ++k) {
    if (vector_continue_criterion(X(k), GX(k), diffMax, params)) {
      keepGoing = true;
      break;
    }
  }

  size_t iter = 0;
  bool numconv = false;

  while (keepGoing && iter < iterMax) {
    ++iter;

    all_cluster_coefficients_vectorized(data, params, GGX, GX, workspace);

    numconv = true;
    for (size_t k = 0; k < K - 1; ++k) {
      bool k_converged = irons_tuck(X(k), GX(k), GGX(k), params);
      if (!k_converged) {
        numconv = false;
      }
    }
    
    if (numconv) break;

    if (utils::is_poisson_family(data.family)) {
      for (size_t k = 0; k < K - 1; ++k) {
        if (any(X(k) <= 0)) {
          any_negative_poisson = true;
          break;
        }
      }
      if (any_negative_poisson) break;
    }

    all_cluster_coefficients_vectorized(data, params, GX, X, workspace);

    keepGoing = false;
    for (size_t k = 0; k < K - 1; ++k) {
      if (vector_continue_criterion(X(k), GX(k), diffMax, params)) {
        keepGoing = true;
        break;
      }
    }
  }

  all_cluster_coefficients_vectorized(data, params, GGX, GX, workspace);

  update_mu_vectorized(workspace.mu_result, data.mu_init, GGX,
                      data.dum_vector, data.family);

  final_iter = iter;
  return workspace.mu_result;
}

} // namespace convergence
} // namespace capybara

#endif // CAPYBARA_CONVERGENCE_H
