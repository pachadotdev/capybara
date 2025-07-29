// Computing optimal cluster coefficients (or fixed effects)

#ifndef CAPYBARA_CONVERGENCE_H
#define CAPYBARA_CONVERGENCE_H

namespace capybara {

enum class Family {
  POISSON,
  POISSON_LOG,
  NEGBIN,
  BINOMIAL,
  GAUSSIAN,
  INV_GAUSSIAN,
  GAMMA
};

inline bool is_poisson_family(Family family) {
  return (family == Family::POISSON || family == Family::POISSON_LOG);
}

inline bool requires_newton_raphson(Family family) {
  return (family == Family::NEGBIN || family == Family::BINOMIAL);
}

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

inline bool vector_continue_criterion(const vec &a, const vec &b,
                                      double diffMax,
                                      const CapybaraParameters &params) {
  vec diff = abs(a - b);
  vec rel_diff = diff / (params.rel_tol_denom + abs(a));
  return any((diff > diffMax) % (rel_diff > diffMax));
}
inline bool vector_stopping_criterion(const vec &a, const vec &b,
                                      double diffMax,
                                      const CapybaraParameters &params) {
  return !vector_continue_criterion(a, b, diffMax, params);
}

struct ClusterWorkspace {
  vec accumulator;
  vec mu_max;
  vec exp_values;
  uvec counts;

  ClusterWorkspace(size_t max_cluster_size, size_t n_obs) {
    CAPYBARA_TIME_FUNCTION("ClusterWorkspace::ClusterWorkspace");

    accumulator.set_size(max_cluster_size);
    mu_max.set_size(max_cluster_size);
    exp_values.set_size(n_obs);
    counts.set_size(max_cluster_size);
  }
};

inline void grouped_accu(const vec &values, const uvec &indices, vec &result,
                         ClusterWorkspace &ws) {
  CAPYBARA_TIME_FUNCTION("grouped_accu");

  size_t n_obs = values.n_elem;
  result.zeros();

  const double *values_ptr = values.memptr();
  const uword *indices_ptr = indices.memptr();
  double *result_ptr = result.memptr();

  for (size_t i = 0; i < n_obs; ++i) {
    result_ptr[indices_ptr[i]] += values_ptr[i];
  }
}

bool irons_tuck(vec &X, const vec &GX, const vec &GGX,
                const CapybaraParameters &params) {
  CAPYBARA_TIME_FUNCTION("irons_tuck");

  size_t nb_coef = X.n_elem;
  const double *X_ptr = X.memptr();
  const double *GX_ptr = GX.memptr();
  const double *GGX_ptr = GGX.memptr();
  double *X_out_ptr = X.memptr();

  double vprod = 0.0;
  double ssq = 0.0;

  for (size_t i = 0; i < nb_coef; ++i) {
    double GX_tmp = GX_ptr[i];
    double delta_GX = GGX_ptr[i] - GX_tmp;
    double delta2_X = delta_GX - GX_tmp + X_ptr[i];

    vprod += delta_GX * delta2_X;
    ssq += delta2_X * delta2_X;
  }

  if (ssq < params.irons_tuck_eps) {
    return true; // Convergence reached
  }

  double coef = vprod / ssq;
  for (size_t i = 0; i < nb_coef; ++i) {
    double GX_tmp = GX_ptr[i];
    double delta_GX = GGX_ptr[i] - GX_tmp;
    X_out_ptr[i] = GGX_ptr[i] - coef * delta_GX;
  }

  return false;
}

void cluster_coef_poisson(const vec &exp_mu, const vec &sum_y, const uvec &dum,
                          vec &cluster_coef, ClusterWorkspace &ws) {
  CAPYBARA_TIME_FUNCTION("cluster_coef_poisson");

  size_t nb_cluster = cluster_coef.n_elem;
  size_t n_obs = exp_mu.n_elem;

  cluster_coef.zeros();

  const double *exp_mu_ptr = exp_mu.memptr();
  const uword *dum_ptr = dum.memptr();
  const double *sum_y_ptr = sum_y.memptr();
  double *coef_ptr = cluster_coef.memptr();

  for (size_t i = 0; i < n_obs; ++i) {
    coef_ptr[dum_ptr[i]] += exp_mu_ptr[i];
  }

  for (size_t m = 0; m < nb_cluster; ++m) {
    coef_ptr[m] = sum_y_ptr[m] / coef_ptr[m];
  }
}

void cluster_coef_poisson_log(const vec &mu, const vec &sum_y, const uvec &dum,
                              vec &cluster_coef, ClusterWorkspace &ws) {
  CAPYBARA_TIME_FUNCTION("cluster_coef_poisson_log");

  size_t nb_cluster = cluster_coef.n_elem;
  size_t n_obs = mu.n_elem;

  cluster_coef.zeros();

  vec &mu_max = ws.mu_max;
  mu_max.set_size(nb_cluster);
  mu_max.fill(-datum::inf);

  const double *mu_ptr = mu.memptr();
  const uword *dum_ptr = dum.memptr();
  const double *sum_y_ptr = sum_y.memptr();
  double *coef_ptr = cluster_coef.memptr();
  double *mu_max_ptr = mu_max.memptr();

  for (size_t i = 0; i < n_obs; ++i) {
    uword cluster_id = dum_ptr[i];
    if (mu_ptr[i] > mu_max_ptr[cluster_id]) {
      mu_max_ptr[cluster_id] = mu_ptr[i];
    }
  }

  for (size_t i = 0; i < n_obs; ++i) {
    uword cluster_id = dum_ptr[i];
    coef_ptr[cluster_id] += exp(mu_ptr[i] - mu_max_ptr[cluster_id]);
  }

  for (size_t m = 0; m < nb_cluster; ++m) {
    coef_ptr[m] = log(sum_y_ptr[m]) - log(coef_ptr[m]) - mu_max_ptr[m];
  }
}

void cluster_coef_gaussian(const vec &mu, const vec &sum_y, const uvec &dum,
                           const uvec &table, vec &cluster_coef,
                           ClusterWorkspace &ws, double safe_min) {
  CAPYBARA_TIME_FUNCTION("cluster_coef_gaussian");

  size_t nb_cluster = cluster_coef.n_elem;
  size_t n_obs = mu.n_elem;

  cluster_coef.zeros();

  const double *mu_ptr = mu.memptr();
  const uword *dum_ptr = dum.memptr();
  const double *sum_y_ptr = sum_y.memptr();
  const uword *table_ptr = table.memptr();
  double *coef_ptr = cluster_coef.memptr();

  for (size_t i = 0; i < n_obs; ++i) {
    coef_ptr[dum_ptr[i]] += mu_ptr[i];
  }

  for (size_t m = 0; m < nb_cluster; ++m) {
    double table_val = static_cast<double>(table_ptr[m]);
    coef_ptr[m] = (sum_y_ptr[m] - coef_ptr[m]) / std::max(table_val, safe_min);
  }
}

void cluster_coefficients(Family family, const vec &mu, const vec &lhs,
                          const vec &sum_y, const uvec &dum,
                          const uvec &obs_cluster, const uvec &table,
                          const uvec &cumtable, double theta, double diffMax_NR,
                          vec &cluster_coef, ClusterWorkspace &ws,
                          const CapybaraParameters &params) {
  CAPYBARA_TIME_FUNCTION("cluster_coefficients");

  switch (family) {
  case Family::POISSON:
    cluster_coef_poisson(mu, sum_y, dum, cluster_coef, ws);
    break;
  case Family::POISSON_LOG:
    cluster_coef_poisson_log(mu, sum_y, dum, cluster_coef, ws);
    break;
  case Family::GAUSSIAN:
  case Family::INV_GAUSSIAN:
  case Family::GAMMA:
    cluster_coef_gaussian(mu, sum_y, dum, table, cluster_coef, ws,
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

  Convergence(Family fam, size_t n, size_t k, double th, const vec &mu_i,
              const vec &lhs_i, const uvec &nb_cl)
      : family(fam), n_obs(n), K(k), theta(th), mu_init(mu_i), lhs(lhs_i),
        nb_cluster_all(nb_cl), workspace(nb_cl.max(), n) {
    CAPYBARA_TIME_FUNCTION("Convergence::Convergence");
  }
};

struct ConvergenceWorkspace {
  field<vec> X, GX, GGX, X_new;

  vec mu_current;
  vec mu_result;

  ConvergenceWorkspace(const Convergence &data) {
    CAPYBARA_TIME_FUNCTION("ConvergenceWorkspace::ConvergenceWorkspace");

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

inline void update_mu(vec &mu_result, const vec &mu_base,
                      const field<vec> &cluster_coefs,
                      const field<uvec> &dum_vector, Family family) {
  CAPYBARA_TIME_FUNCTION("update_mu");

  size_t K = cluster_coefs.n_elem;
  size_t n_obs = mu_base.n_elem;

  const double *mu_base_ptr = mu_base.memptr();
  double *mu_result_ptr = mu_result.memptr();

  if (mu_result.memptr() != mu_base.memptr()) {
    for (size_t i = 0; i < n_obs; ++i) {
      mu_result_ptr[i] = mu_base_ptr[i];
    }
  }

  if (is_poisson_family(family)) {
    // Multiplicative updates for Poisson
    for (size_t k = 0; k < K; ++k) {
      const double *coef_ptr = cluster_coefs(k).memptr();
      const uword *dum_ptr = dum_vector(k).memptr();

      for (size_t i = 0; i < n_obs; ++i) {
        mu_result_ptr[i] *= coef_ptr[dum_ptr[i]];
      }
    }
  } else {
    // Additive updates for Gaussian and others
    for (size_t k = 0; k < K; ++k) {
      const double *coef_ptr = cluster_coefs(k).memptr();
      const uword *dum_ptr = dum_vector(k).memptr();

      for (size_t i = 0; i < n_obs; ++i) {
        mu_result_ptr[i] += coef_ptr[dum_ptr[i]];
      }
    }
  }
}

void all_cluster_coefficients(const Convergence &data,
                              const CapybaraParameters &params,
                              field<vec> &cluster_coefs_dest,
                              const field<vec> &cluster_coefs_origin,
                              ConvergenceWorkspace &workspace) {
  CAPYBARA_TIME_FUNCTION("all_cluster_coefficients");

  vec &mu_current = workspace.mu_current;
  mu_current = data.mu_init;

  if (is_poisson_family(data.family)) {
    for (size_t k = 0; k < data.K - 1; ++k) {
      mu_current %= cluster_coefs_origin(k).elem(data.dum_vector(k));
    }
  } else {
    for (size_t k = 0; k < data.K - 1; ++k) {
      mu_current += cluster_coefs_origin(k).elem(data.dum_vector(k));
    }
  }

  for (int k = static_cast<int>(data.K) - 1; k >= 0; k--) {
    size_t uk = static_cast<size_t>(k);

    cluster_coefficients(
        data.family, mu_current, data.lhs, data.sum_y_vector(uk),
        data.dum_vector(uk), data.obs_cluster_vector(uk), data.table_vector(uk),
        data.cumtable_vector(uk), data.theta, params.newton_raphson_tol,
        cluster_coefs_dest(uk), const_cast<ClusterWorkspace &>(data.workspace),
        params);

    if (k > 0) {
      mu_current = data.mu_init;

      if (is_poisson_family(data.family)) {
        for (size_t h = 0; h < data.K; h++) {
          if (h == uk - 1)
            continue;
          const vec &coef =
              (h < uk - 1) ? cluster_coefs_origin(h) : cluster_coefs_dest(h);
          mu_current %= coef.elem(data.dum_vector(h));
        }
      } else {
        for (size_t h = 0; h < data.K; h++) {
          if (h == uk - 1)
            continue;
          const vec &coef =
              (h < uk - 1) ? cluster_coefs_origin(h) : cluster_coefs_dest(h);
          mu_current += coef.elem(data.dum_vector(h));
        }
      }
    }
  }
}

vec conv_accelerated(const Convergence &data, const CapybaraParameters &params,
                     size_t iterMax, double diffMax, size_t &final_iter,
                     bool &any_negative_poisson) {
  CAPYBARA_TIME_FUNCTION("conv_accelerated");

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
    if (is_poisson_family(data.family)) {
      X(k).ones();
    } else {
      X(k).zeros();
    }
  }

  all_cluster_coefficients(data, params, GX, X, workspace);

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

    all_cluster_coefficients(data, params, GGX, GX, workspace);

    numconv = true;
    for (size_t k = 0; k < K - 1; ++k) {
      bool k_converged = irons_tuck(X(k), GX(k), GGX(k), params);
      if (!k_converged) {
        numconv = false;
      }
    }

    if (numconv)
      break;

    if (is_poisson_family(data.family)) {
      for (size_t k = 0; k < K - 1; ++k) {
        const double *X_ptr = X(k).memptr();
        size_t n_coef = X(k).n_elem;
        for (size_t i = 0; i < n_coef; ++i) {
          if (X_ptr[i] <= 0) {
            any_negative_poisson = true;
            break;
          }
        }
        if (any_negative_poisson)
          break;
      }
      if (any_negative_poisson)
        break;
    }

    all_cluster_coefficients(data, params, GX, X, workspace);

    keepGoing = false;
    for (size_t k = 0; k < K - 1; ++k) {
      if (vector_continue_criterion(X(k), GX(k), diffMax, params)) {
        keepGoing = true;
        break;
      }
    }
  }

  all_cluster_coefficients(data, params, GGX, GX, workspace);

  update_mu(workspace.mu_result, data.mu_init, GGX, data.dum_vector,
            data.family);

  final_iter = iter;
  return workspace.mu_result;
}

} // namespace capybara

#endif // CAPYBARA_CONVERGENCE_H
