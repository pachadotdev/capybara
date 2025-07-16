#ifndef CAPYBARA_CONVERGENCE
#define CAPYBARA_CONVERGENCE

// Stopping / continuing criteria
inline bool continue_criterion(double a, double b, double diffMax) {
  double diff = std::abs(a - b);
  return ((diff > diffMax) && (diff / (0.1 + std::abs(a)) > diffMax));
}

inline bool stopping_criterion(double a, double b, double diffMax) {
  double diff = std::abs(a - b);
  return ((diff < diffMax) || (diff / (0.1 + std::abs(a)) < diffMax));
}

// Structs to lighten the writing of the functions
struct PARAM_CCC {
  int family;
  size_t n_obs;
  size_t K;
  double theta;
  double diffMax_NR;
  size_t nthreads;

  // vectors from input
  vec mu_init;
  field<uvec> pdum;
  vec lhs;

  // vectors of data
  field<uvec> ptable;
  field<vec> psum_y;
  field<uvec> pobsCluster;
  field<uvec> pcumtable;

  // value that will vary
  vec mu_with_coef;

  PARAM_CCC()
      : family(1), n_obs(0), K(0), theta(1.0), diffMax_NR(1e-8), nthreads(1) {}
};

// IT update + numerical convergence indicator
bool update_X_IronsTuck(size_t nb_coef_no_K, vec &X_flat, const vec &GX_flat,
                        const vec &GGX_flat, vec &delta_GX, vec &delta2_X) {
  // Ensure working vectors are properly sized
  if (delta_GX.n_elem != nb_coef_no_K) {
    delta_GX.set_size(nb_coef_no_K);
    delta2_X.set_size(nb_coef_no_K);
  }

  // Differences
  delta_GX = GGX_flat - GX_flat;
  delta2_X = delta_GX - GX_flat + X_flat;

  double vprod = dot(delta_GX, delta2_X);
  double ssq = dot(delta2_X, delta2_X);

  bool res = false;

  if (ssq == 0) {
    res = true;
  } else {
    double coef = vprod / ssq;
    X_flat = GGX_flat - coef * delta_GX;
  }

  return res;
}

void cluster_coefficients_poisson(size_t n_obs, size_t nb_cluster,
                                  vec &cluster_coef, const vec &exp_mu,
                                  const vec &sum_y, const uvec &dum) {
  // Initialize cluster coef
  cluster_coef.set_size(nb_cluster);
  cluster_coef.zeros();

  // Accumulate exp_mu values by cluster
  for (size_t i = 0; i < n_obs; ++i) {
    cluster_coef(dum(i)) += exp_mu(i);
  }

  // Cluster coefficients
  for (size_t m = 0; m < nb_cluster; ++m) {
    if (cluster_coef(m) > 0) {
      cluster_coef(m) = sum_y(m) / cluster_coef(m);
    }
  }
}

// Log Poisson is here because classic poisson does not always work with logs
// It can lead to very high values of the cluster coefs (in abs value)
// => We need to apply a trick of subtracting the max in the exp
void cluster_coefficients_poisson_log(size_t n_obs, size_t nb_cluster,
                                      vec &cluster_coef, const vec &mu,
                                      const vec &sum_y, const uvec &dum) {
  vec mu_max(nb_cluster, fill::none);
  uvec doInit(nb_cluster, fill::ones);

  // Initialize cluster coef
  cluster_coef.set_size(nb_cluster);
  cluster_coef.zeros();

  // Finding the max mu for each cluster
  for (size_t i = 0; i < n_obs; ++i) {
    uword d = dum(i);
    if (doInit(d)) {
      mu_max(d) = mu(i);
      doInit(d) = 0;
    } else if (mu(i) > mu_max(d)) {
      mu_max(d) = mu(i);
    }
  }

  // Accumulate exp(mu - mu_max) by cluster
  for (size_t i = 0; i < n_obs; ++i) {
    uword d = dum(i);
    cluster_coef(d) += std::exp(mu(i) - mu_max(d));
  }

  // Cluster coefficients
  for (size_t m = 0; m < nb_cluster; ++m) {
    if (cluster_coef(m) > 0 && sum_y(m) > 0) {
      cluster_coef(m) =
          std::log(sum_y(m)) - std::log(cluster_coef(m)) - mu_max(m);
    }
  }
}

void cluster_coefficients_gaussian(size_t n_obs, size_t nb_cluster,
                                   vec &cluster_coef, const vec &mu,
                                   const vec &sum_y, const uvec &dum,
                                   const uvec &table) {
  // Initialize cluster coef
  cluster_coef.set_size(nb_cluster);
  cluster_coef.zeros();

  // Accumulate mu values by cluster
  for (size_t i = 0; i < n_obs; ++i) {
    cluster_coef(dum(i)) += mu(i);
  }

  // Cluster coefficients
  for (size_t m = 0; m < nb_cluster; ++m) {
    if (table(m) > 0) {
      cluster_coef(m) =
          (sum_y(m) - cluster_coef(m)) / static_cast<double>(table(m));
    }
  }
}

// The negative binomial needs to apply dichotomy and Newton-Raphson algorithm
void cluster_coefficients_negbin(size_t nthreads, size_t nb_cluster,
                                 double theta, double diffMax_NR,
                                 vec &cluster_coef, const vec &mu,
                                 const vec &lhs, const vec &sum_y,
                                 const uvec &obsCluster, const uvec &table,
                                 const uvec &cumtable) {
  // first we find the min max for each cluster to get the bounds
  constexpr size_t iterMax = 100;
  constexpr size_t iterFullDicho = 10;

  // Pre-allocate bounds vectors
  vec lower_bound(nb_cluster, fill::none);
  vec upper_bound(nb_cluster, fill::none);
  cluster_coef.set_size(nb_cluster);

  // Finding the max/min values of mu for each cluster
  for (size_t m = 0; m < nb_cluster; ++m) {
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    double mu_min = mu(obsCluster(u0));
    double mu_max = mu(obsCluster(u0));

    for (size_t u = u0 + 1; u < u_end; ++u) {
      double value = mu(obsCluster(u));
      if (value < mu_min) {
        mu_min = value;
      } else if (value > mu_max) {
        mu_max = value;
      }
    }

    // Bounds
    lower_bound(m) =
        std::log(sum_y(m)) - std::log(static_cast<double>(table(m))) - mu_max;
    upper_bound(m) = lower_bound(m) + (mu_max - mu_min);
  }

  // Solve for each cluster
  // NOTE: This does not use OpenMP directly to avoid overhead with internal
  // Armadillo parallelization

  // Loop over each cluster
  for (size_t m = 0; m < nb_cluster; ++m) {
    // Initialize the cluster coefficient at 0
    double x1 = 0;
    bool keepGoing = true;
    size_t iter = 0;
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    double value, x0, derivative_val = 0, exp_mu;

    // Bounds
    double lb = lower_bound(m);
    double ub = upper_bound(m);

    // Update the value if it goes out of boundaries
    // We do not know ex-ante if 0 is within the bounds
    if (x1 >= ub || x1 <= lb) {
      x1 = (lb + ub) / 2;
    }

    while (keepGoing && iter < iterMax) {
      ++iter;

      // 1st step: Bounds initialization

      // Evaluate f(x)
      value = sum_y(m);
      for (size_t u = u0; u < u_end; ++u) {
        size_t i = obsCluster(u);
        value -= (theta + lhs(i)) / (1 + theta * std::exp(-x1 - mu(i)));
      }

      // Update the bounds
      if (value > 0) {
        lb = x1;
      } else {
        ub = x1;
      }

      // 2nd step: Newton-Raphson iteration or Dichotomy
      x0 = x1;
      if (std::abs(value) < 1e-12) {
        keepGoing = false;
      } else if (iter <= iterFullDicho) {
        // Derivative
        derivative_val = 0;
        for (size_t u = u0; u < u_end; ++u) {
          size_t i = obsCluster(u);
          exp_mu = std::exp(x1 + mu(i));
          derivative_val -= theta * (theta + lhs(i)) /
                            ((theta / exp_mu + 1) * (theta + exp_mu));
        }

        if (std::abs(derivative_val) > 1e-12) {
          x1 = x0 - value / derivative_val;
        }

        // 3rd step: dichotomy (if necessary)
        if (x1 >= ub || x1 <= lb) {
          x1 = (lb + ub) / 2;
        }
      } else {
        x1 = (lb + ub) / 2;
      }

      // Stopping criterion
      if (stopping_criterion(x0, x1, diffMax_NR)) {
        keepGoing = false;
      }
    }

    // After convergence: Update the cluster coefficient
    cluster_coef(m) = x1;
  }
}

// Similar to negative binomial
void cluster_coefficients_binomial(size_t nthreads, size_t nb_cluster,
                                   double diffMax_NR, vec &cluster_coef,
                                   const vec &mu, const vec &sum_y,
                                   const uvec &obsCluster, const uvec &table,
                                   const uvec &cumtable) {
  constexpr size_t iterMax = 100;
  constexpr size_t iterFullDicho = 10;

  vec lower_bound(nb_cluster, fill::none);
  vec upper_bound(nb_cluster, fill::none);
  cluster_coef.set_size(nb_cluster);

  for (size_t m = 0; m < nb_cluster; ++m) {
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    double mu_min = mu(obsCluster(u0));
    double mu_max = mu(obsCluster(u0));

    for (size_t u = u0 + 1; u < u_end; ++u) {
      double value = mu(obsCluster(u));
      if (value < mu_min) {
        mu_min = value;
      } else if (value > mu_max) {
        mu_max = value;
      }
    }

    lower_bound(m) =
        std::log(sum_y(m)) - std::log(table(m) - sum_y(m)) - mu_max;
    upper_bound(m) = lower_bound(m) + (mu_max - mu_min);
  }

  for (size_t m = 0; m < nb_cluster; ++m) {
    double x1 = 0;
    bool keepGoing = true;
    size_t iter = 0;
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);

    double value, x0, derivative_val = 0, exp_mu;

    double lb = lower_bound(m);
    double ub = upper_bound(m);

    if (x1 >= ub || x1 <= lb) {
      x1 = (lb + ub) / 2;
    }

    while (keepGoing && iter < iterMax) {
      ++iter;
      value = sum_y(m);
      for (size_t u = u0; u < u_end; ++u) {
        value -= 1 / (1 + std::exp(-x1 - mu(obsCluster(u))));
      }

      if (value > 0) {
        lb = x1;
      } else {
        ub = x1;
      }

      x0 = x1;
      if (std::abs(value) < 1e-12) {
        keepGoing = false;
      } else if (iter <= iterFullDicho) {
        derivative_val = 0;
        for (size_t u = u0; u < u_end; ++u) {
          exp_mu = std::exp(x1 + mu(obsCluster(u)));
          derivative_val -= 1 / ((1 / exp_mu + 1) * (1 + exp_mu));
        }

        if (std::abs(derivative_val) > 1e-12) {
          x1 = x0 - value / derivative_val;
        }

        if (x1 >= ub || x1 <= lb) {
          x1 = (lb + ub) / 2;
        }
      } else {
        x1 = (lb + ub) / 2;
      }

      if (stopping_criterion(x0, x1, diffMax_NR)) {
        keepGoing = false;
      }
    }

    cluster_coef(m) = x1;
  }
}

// Update the cluster
void cluster_coefficient_single(int family, size_t n_obs, size_t nb_cluster,
                                double theta, double diffMax_NR,
                                vec &cluster_coef, const vec &mu,
                                const vec &lhs, const vec &sum_y,
                                const uvec &dum, const uvec &obsCluster,
                                const uvec &table, const uvec &cumtable,
                                size_t nthreads) {
  switch (family) {
  case 1: // Poisson
    cluster_coefficients_poisson(n_obs, nb_cluster, cluster_coef, mu, sum_y,
                                 dum);
    break;
  case 2: // Negbin
    cluster_coefficients_negbin(nthreads, nb_cluster, theta, diffMax_NR,
                                cluster_coef, mu, lhs, sum_y, obsCluster, table,
                                cumtable);
    break;
  case 3: // Binomial
    cluster_coefficients_binomial(nthreads, nb_cluster, diffMax_NR,
                                  cluster_coef, mu, sum_y, obsCluster, table,
                                  cumtable);
    break;
  case 4: // Gaussian
    cluster_coefficients_gaussian(n_obs, nb_cluster, cluster_coef, mu, sum_y,
                                  dum, table);
    break;
  case 5: // Log-Poisson
    cluster_coefficients_poisson_log(n_obs, nb_cluster, cluster_coef, mu, sum_y,
                                     dum);
    break;
  }
}

// Update the cluster coefficients
void cluster_coefficient(field<vec> &pcluster_origin,
                         field<vec> &pcluster_destination, PARAM_CCC &args) {
  // First we update mu, then we update the cluster coefficients

  int family = args.family;
  size_t n_obs = args.n_obs;
  size_t K = args.K;
  size_t nthreads = args.nthreads;
  double theta = args.theta;
  double diffMax_NR = args.diffMax_NR;

  const vec &mu_init = args.mu_init;
  const vec &lhs = args.lhs;
  const field<uvec> &pdum = args.pdum;
  const field<uvec> &ptable = args.ptable;
  const field<vec> &psum_y = args.psum_y;
  const field<uvec> &pobsCluster = args.pobsCluster;
  const field<uvec> &pcumtable = args.pcumtable;

  // Value to modify
  vec &mu_with_coef = args.mu_with_coef;

  // Update each cluster coefficient, starting from K

  // Set the value of mu_with_coef
  mu_with_coef = mu_init;

  for (size_t k = 0; k < (K - 1); ++k) {
    const uvec &my_dum = pdum(k);
    const vec &my_cluster_coef = pcluster_origin(k);

    if (family == 1 || family == 5) { // Poisson or log Poisson
      for (size_t i = 0; i < n_obs; ++i) {
        mu_with_coef(i) *= my_cluster_coef(my_dum(i));
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        mu_with_coef(i) += my_cluster_coef(my_dum(i));
      }
    }
  }

  for (int k = static_cast<int>(K) - 1; k >= 0; k--) {
    size_t uk = static_cast<size_t>(k);

    // Optimal cluster coefficients
    vec &my_cluster_coef = pcluster_destination(uk);
    const uvec &my_table = ptable(uk);
    const vec &my_sum_y = psum_y(uk);
    const uvec &my_dum = pdum(uk);
    const uvec &my_cumtable = pcumtable(uk);
    const uvec &my_obsCluster = pobsCluster(uk);
    size_t nb_cluster = my_table.n_elem;

    // Update the cluster coefficients
    cluster_coefficient_single(family, n_obs, nb_cluster, theta, diffMax_NR,
                               my_cluster_coef, mu_with_coef, lhs, my_sum_y,
                               my_dum, my_obsCluster, my_table, my_cumtable,
                               nthreads);

    // Updating mu_with_coef if necessary
    if (k != 0) {
      mu_with_coef = mu_init;

      for (size_t h = 0; h < K; h++) {
        if (h == uk - 1)
          continue;

        const uvec &my_dum_h = pdum(h);
        const vec &my_cluster_coef_h =
            (h < uk - 1) ? pcluster_origin(h) : pcluster_destination(h);

        if (family == 1 || family == 5) { // Poisson or log Poisson
          for (size_t i = 0; i < n_obs; ++i) {
            mu_with_coef(i) *= my_cluster_coef_h(my_dum_h(i));
          }
        } else {
          for (size_t i = 0; i < n_obs; ++i) {
            mu_with_coef(i) += my_cluster_coef_h(my_dum_h(i));
          }
        }
      }
    }
  }

  // pcluster_destination is now fully updated, starting from K to 1
}

vec conv_acc_gnl(int family, size_t iterMax, double diffMax, double diffMax_NR,
                 double theta, const uvec &nb_cluster_all, const vec &lhs,
                 const vec &mu_init, const field<uvec> &dum_vector,
                 const field<uvec> &tableCluster_vector,
                 const field<vec> &sum_y_vector,
                 const field<uvec> &cumtable_vector,
                 const field<uvec> &obsCluster_vector, size_t nthreads,
                 size_t &final_iter, bool &any_negative_poisson) {
  size_t K = nb_cluster_all.n_elem;
  size_t n_obs = mu_init.n_elem;

  // Sending variables to struct

  PARAM_CCC args;
  args.family = family;
  args.n_obs = n_obs;
  args.K = K;
  args.nthreads = nthreads;
  args.theta = (family == 2) ? theta : 1.0;
  args.diffMax_NR = diffMax_NR;
  args.pdum = dum_vector;
  args.mu_init = mu_init;
  args.ptable = tableCluster_vector;
  args.psum_y = sum_y_vector;
  args.pcumtable = cumtable_vector;
  args.pobsCluster = obsCluster_vector;
  args.lhs = lhs;

  args.mu_with_coef.set_size(n_obs);

  // Preparation for Irons-Tuck iteration

  // Variables on 1:K
  field<vec> X(K), GX(K), GGX(K);
  for (size_t k = 0; k < K; ++k) {
    size_t nk = nb_cluster_all(k);
    X(k).set_size(nk);
    GX(k).set_size(nk);
    GGX(k).set_size(nk);
  }

  // Variables on 1:(K-1)
  size_t nb_coef_no_K = sum(nb_cluster_all.head(K - 1));
  vec delta_GX(nb_coef_no_K, fill::none);
  vec delta2_X(nb_coef_no_K, fill::none);

  // Main loop
  // Initialize X and GX
  if (family == 1 || family == 5) { // Poisson or log Poisson
    for (size_t k = 0; k < K; ++k) {
      X(k).ones();
    }
  } else {
    for (size_t k = 0; k < K; ++k) {
      X(k).zeros();
    }
  }

  // First iteration
  cluster_coefficient(X, GX, args);

  // Flag for problematic observations with poisson link
  any_negative_poisson = false;

  // Check if the loop should run
  bool keepGoing = false;
  for (size_t k = 0; k < K - 1; ++k) {
    const vec &X_k = X(k);
    const vec &GX_k = GX(k);
    for (size_t i = 0; i < X_k.n_elem; ++i) {
      if (continue_criterion(X_k(i), GX_k(i), diffMax)) {
        keepGoing = true;
        break;
      }
    }
    if (keepGoing)
      break;
  }

  size_t iter = 0;
  bool numconv = false;

  while (keepGoing && iter < iterMax) {
    ++iter;

    // GGX computation
    cluster_coefficient(GX, GGX, args);

    // Flatten first K-1 FE coefficients for Irons-Tuck
    vec X_flat(nb_coef_no_K, fill::none);
    vec GX_flat(nb_coef_no_K, fill::none);
    vec GGX_flat(nb_coef_no_K, fill::none);

    size_t coef_idx = 0;
    for (size_t k = 0; k < K - 1; ++k) {
      size_t nk = X(k).n_elem;
      X_flat.subvec(coef_idx, coef_idx + nk - 1) = X(k);
      GX_flat.subvec(coef_idx, coef_idx + nk - 1) = GX(k);
      GGX_flat.subvec(coef_idx, coef_idx + nk - 1) = GGX(k);
      coef_idx += nk;
    }

    // Irons-Tuck acceleration
    numconv = update_X_IronsTuck(nb_coef_no_K, X_flat, GX_flat, GGX_flat,
                                 delta_GX, delta2_X);
    if (numconv)
      break;

    // Unflatten back to field structure
    coef_idx = 0;
    for (size_t k = 0; k < K - 1; ++k) {
      size_t nk = X(k).n_elem;
      X(k) = X_flat.subvec(coef_idx, coef_idx + nk - 1);
      coef_idx += nk;
    }

    if (family == 1 || family == 5) { // Poisson or log Poisson
      // Control for possible problems with Poisson
      for (size_t k = 0; k < K - 1; ++k) {
        if (any(X(k) <= 0)) {
          any_negative_poisson = true;
          break;
        }
      }

      if (any_negative_poisson) {
        break;
      }
    }

    // Update GX
    cluster_coefficient(X, GX, args);

    keepGoing = false;
    for (size_t k = 0; k < K - 1; ++k) {
      const vec &X_k = X(k);
      const vec &GX_k = GX(k);
      for (size_t i = 0; i < X_k.n_elem; ++i) {
        if (continue_criterion(X_k(i), GX_k(i), diffMax)) {
          keepGoing = true;
          break;
        }
      }
      if (keepGoing)
        break;
    }
  }

  // Update resulting mu
  vec mu_result = mu_init;

  // Final cluster computation
  cluster_coefficient(GX, GGX, args);
  for (size_t k = 0; k < K; ++k) {
    const uvec &my_dum = dum_vector(k);
    const vec &my_cluster_coef = GGX(k);

    if (family == 1 || family == 5) { // Poisson or log Poisson
      for (size_t i = 0; i < n_obs; ++i) {
        mu_result(i) *= my_cluster_coef(my_dum(i));
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        mu_result(i) += my_cluster_coef(my_dum(i));
      }
    }
  }

  final_iter = iter;
  return mu_result;
}

vec conv_seq_gnl(int family, size_t iterMax, double diffMax, double diffMax_NR,
                 double theta, const uvec &nb_cluster_all, const vec &lhs,
                 const vec &mu_init, const field<uvec> &dum_vector,
                 const field<uvec> &tableCluster_vector,
                 const field<vec> &sum_y_vector,
                 const field<uvec> &cumtable_vector,
                 const field<uvec> &obsCluster_vector, size_t nthreads,
                 size_t &final_iter) {
  size_t K = nb_cluster_all.n_elem;
  size_t n_obs = mu_init.n_elem;

  // Initial mu
  vec mu_with_coef = mu_init;

  // Cluster coefficients
  field<vec> cluster_coef(K);
  for (size_t k = 0; k < K; ++k) {
    size_t nk = nb_cluster_all(k);
    cluster_coef(k).set_size(nk);
  }

  // Initialize cluster coefficients
  if (family == 1 || family == 5) { // Poisson or log Poisson
    for (size_t k = 0; k < K; ++k) {
      cluster_coef(k).ones();
    }
  } else {
    for (size_t k = 0; k < K; ++k) {
      cluster_coef(k).zeros();
    }
  }

  // Main loop

  bool keepGoing = true;
  size_t iter = 1;

  while (keepGoing && iter <= iterMax) {
    ++iter;
    keepGoing = false;

    /// Loop over all clusters => from K to 1
    for (int k = static_cast<int>(K) - 1; k >= 0; k--) {
      size_t uk = static_cast<size_t>(k);

      // 1) Compute the optimal cluster coefficient
      vec &my_cluster_coef = cluster_coef(uk);
      const uvec &my_table = tableCluster_vector(uk);
      const vec &my_sum_y = sum_y_vector(uk);
      const uvec &my_dum = dum_vector(uk);
      const uvec &my_cumtable = cumtable_vector(uk);
      const uvec &my_obsCluster = obsCluster_vector(uk);
      size_t nb_cluster = my_table.n_elem;
      // Update of the cluster coefficients
      cluster_coefficient_single(family, n_obs, nb_cluster, theta, diffMax_NR,
                                 my_cluster_coef, mu_with_coef, lhs, my_sum_y,
                                 my_dum, my_obsCluster, my_table, my_cumtable,
                                 nthreads);

      // 2) Update mu
      if (family == 1 || family == 5) { // Poisson or log Poisson
        for (size_t i = 0; i < n_obs; ++i) {
          mu_with_coef(i) *= my_cluster_coef(my_dum(i));
        }
      } else {
        for (size_t i = 0; i < n_obs; ++i) {
          mu_with_coef(i) += my_cluster_coef(my_dum(i));
        }
      }

      // Stopping criterion
      if (!keepGoing) {
        if (family == 1 || family == 5) { // Poisson or log Poisson
          vec diff_vec = abs(my_cluster_coef - 1.0);
          if (any(diff_vec > diffMax)) {
            keepGoing = true;
          }
        } else {
          vec abs_coef = abs(my_cluster_coef);
          if (any(abs_coef > diffMax)) {
            keepGoing = true;
          }
        }
      }
    }
  }

  final_iter = iter - 1;
  return mu_with_coef;
}

vec conv_seq_poi_2(size_t n_i, size_t n_j, size_t n_cells, const uvec &index_i,
                   const uvec &index_j, const field<uvec> &dum_vector,
                   const vec &sum_y_vector, size_t iterMax, double diffMax,
                   const vec &exp_mu_in, const uvec &order,
                   size_t &final_iter) {
  vec alpha(n_i, fill::none);

  // Compute Ab
  size_t index_current = 0;
  uvec mat_row(n_cells, fill::none);
  uvec mat_col(n_cells, fill::none);
  vec mat_value(n_cells, fill::none);

  size_t n_obs = exp_mu_in.n_elem;

  double value = exp_mu_in(order(0));

  for (size_t i = 1; i < n_obs; ++i) {
    if (index_j(i) != index_j(i - 1) || index_i(i) != index_i(i - 1)) {
      // Store value if we change index
      mat_row(index_current) = index_i(i - 1);
      mat_col(index_current) = index_j(i - 1);
      mat_value(index_current) = value;

      // New row
      index_current++;

      // New value
      value = exp_mu_in(order(i));
    } else {
      value += exp_mu_in(order(i));
    }
  }

  // Last save
  mat_row(index_current) = index_i(n_obs - 1);
  mat_col(index_current) = index_j(n_obs - 1);
  mat_value(index_current) = value;

  // coefficient vectors X and X_new
  vec X_new(n_i + n_j, fill::none);
  vec X(n_i + n_j, fill::none);

  // Initialize X
  X.head(n_i).ones();

  // ca and cb
  vec ca = sum_y_vector.head(n_i);
  vec cb = sum_y_vector.tail(n_j);

  // Helper for 2-FE Poisson computation (closed form)
  auto cluster_coefficients_poisson_2 = [&](const vec &origin,
                                            vec &destination) {
    vec beta = destination.tail(n_j);
    vec alpha_dest = destination.head(n_i);
    const vec &alpha_origin = origin.head(n_i);

    alpha_dest.zeros();
    beta.zeros();

    for (size_t obs = 0; obs < n_cells; ++obs) {
      beta(mat_col(obs)) += mat_value(obs) * alpha_origin(mat_row(obs));
    }

    for (size_t j = 0; j < n_j; ++j) {
      if (beta(j) > 0) {
        beta(j) = cb(j) / beta(j);
      }
    }

    for (size_t obs = 0; obs < n_cells; ++obs) {
      alpha_dest(mat_row(obs)) += mat_value(obs) * beta(mat_col(obs));
    }

    for (size_t i = 0; i < n_i; ++i) {
      if (alpha_dest(i) > 0) {
        alpha_dest(i) = ca(i) / alpha_dest(i);
      }
    }

    // Copy back to destination
    destination.head(n_i) = alpha_dest;
    destination.tail(n_j) = beta;
  };

  bool keepGoing = true;
  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    // No need to update the values of X with a loop
    if (iter % 2 == 1) {
      cluster_coefficients_poisson_2(X, X_new);
    } else {
      cluster_coefficients_poisson_2(X_new, X);
    }

    keepGoing = false;
    vec X_head = (iter % 2 == 1) ? X.head(n_i) : X_new.head(n_i);
    vec X_new_head = (iter % 2 == 1) ? X_new.head(n_i) : X.head(n_i);

    for (size_t i = 0; i < n_i; ++i) {
      if (continue_criterion(X_head(i), X_new_head(i), diffMax)) {
        keepGoing = true;
        break;
      }
    }
  }

  const vec &X_final = (iter % 2 == 1) ? X_new : X;

  vec result_mu(n_obs, fill::none);
  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);

  const vec &alpha_final = X_final.head(n_i);
  const vec &beta_final = X_final.tail(n_j);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    result_mu(obs) =
        exp_mu_in(obs) * alpha_final(dum_i(obs)) * beta_final(dum_j(obs));
  }

  final_iter = iter;
  return result_mu;
}

// Fixed cost computation for Gaussian 2-FE
struct FixedCostResult {
  uvec mat_row;
  uvec mat_col;
  vec mat_value_Ab;
  vec mat_value_Ba;
};

FixedCostResult fixed_cost_gaussian(size_t n_i, size_t n_cells,
                                    const uvec &index_i, const uvec &index_j,
                                    const uvec &order,
                                    const vec &invTableCluster_vector,
                                    const field<uvec> &dum_vector) {
  size_t n_obs = index_i.n_elem;
  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);
  const vec &invTable_i = invTableCluster_vector.head(n_i);
  const vec &invTable_j =
      invTableCluster_vector.tail(invTableCluster_vector.n_elem - n_i);

  // Compute Ab and Ba
  size_t index_current = 0;

  uvec mat_row(n_cells, fill::none);
  uvec mat_col(n_cells, fill::none);
  vec mat_value_Ab(n_cells, fill::none);
  vec mat_value_Ba(n_cells, fill::none);

  double value_Ab = invTable_i(dum_i(order(0)));
  double value_Ba = invTable_j(dum_j(order(0)));

  for (size_t obs = 1; obs < n_obs; ++obs) {
    if (index_j(obs) != index_j(obs - 1) || index_i(obs) != index_i(obs - 1)) {
      // Save the value if we change index
      mat_row(index_current) = index_i(obs - 1);
      mat_col(index_current) = index_j(obs - 1);
      mat_value_Ab(index_current) = value_Ab;
      mat_value_Ba(index_current) = value_Ba;

      // New row
      index_current++;

      // New value
      value_Ab = invTable_i(dum_i(order(obs)));
      value_Ba = invTable_j(dum_j(order(obs)));
    } else {
      value_Ab += invTable_i(dum_i(order(obs)));
      value_Ba += invTable_j(dum_j(order(obs)));
    }
  }

  // Last save
  mat_row(index_current) = index_i(n_obs - 1);
  mat_col(index_current) = index_j(n_obs - 1);
  mat_value_Ab(index_current) = value_Ab;
  mat_value_Ba(index_current) = value_Ba;

  return {std::move(mat_row), std::move(mat_col), std::move(mat_value_Ab),
          std::move(mat_value_Ba)};
}

void cluster_coefficients_gaussian_2(const vec &pcluster_origin,
                                     vec &pcluster_destination, size_t n_i,
                                     size_t n_j, size_t n_cells,
                                     const uvec &mat_row, const uvec &mat_col,
                                     const vec &mat_value_Ab,
                                     const vec &mat_value_Ba,
                                     const vec &a_tilde, vec &beta) {
  // alpha = a_tilde + (Ab %*% (Ba %*% alpha))
  pcluster_destination = a_tilde;
  beta.zeros();

  for (size_t obs = 0; obs < n_cells; ++obs) {
    beta(mat_col(obs)) += mat_value_Ba(obs) * pcluster_origin(mat_row(obs));
  }

  for (size_t obs = 0; obs < n_cells; ++obs) {
    pcluster_destination(mat_row(obs)) +=
        mat_value_Ab(obs) * beta(mat_col(obs));
  }
}

vec conv_acc_gau_2(size_t n_i, size_t n_j, size_t n_cells, const uvec &mat_row,
                   const uvec &mat_col, const vec &mat_value_Ab,
                   const vec &mat_value_Ba, const field<uvec> &dum_vector,
                   const vec &lhs, const vec &invTableCluster_vector,
                   size_t iterMax, double diffMax, const vec &mu_in,
                   size_t &final_iter) {
  size_t n_obs = mu_in.n_elem;

  vec resid = lhs - mu_in;

  // const_a and const_b
  vec const_a(n_i, fill::zeros);
  vec const_b(n_j, fill::zeros);
  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);
  const vec &invTable_i = invTableCluster_vector.head(n_i);
  const vec &invTable_j =
      invTableCluster_vector.tail(invTableCluster_vector.n_elem - n_i);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    double resid_tmp = resid(obs);
    uword d_i = dum_i(obs);
    uword d_j = dum_j(obs);

    const_a(d_i) += resid_tmp * invTable_i(d_i);
    const_b(d_j) += resid_tmp * invTable_j(d_j);
  }

  vec beta(n_j, fill::none);

  // a_tilde = const_a - (Ab %*% const_b)
  vec a_tilde = const_a;

  for (size_t obs = 0; obs < n_cells; ++obs) {
    a_tilde(mat_row(obs)) -= mat_value_Ab(obs) * const_b(mat_col(obs));
  }

  // Preparation for Irons-Tuck
  vec X(n_i, fill::none);
  vec GX(n_i, fill::none);
  vec GGX(n_i, fill::none);
  vec delta_GX(n_i, fill::none);
  vec delta2_X(n_i, fill::none);

  // Initialize X and GX
  X.zeros();

  // first iteration
  cluster_coefficients_gaussian_2(X, GX, n_i, n_j, n_cells, mat_row, mat_col,
                                  mat_value_Ab, mat_value_Ba, a_tilde, beta);

  bool numconv = false;
  bool keepGoing = true;
  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    // Origin: GX, destination: GGX
    cluster_coefficients_gaussian_2(GX, GGX, n_i, n_j, n_cells, mat_row,
                                    mat_col, mat_value_Ab, mat_value_Ba,
                                    a_tilde, beta);

    // Update the cluster coefficient
    numconv = update_X_IronsTuck(n_i, X, GX, GGX, delta_GX, delta2_X);
    if (numconv)
      break;

    // Origin: X, destination: GX
    cluster_coefficients_gaussian_2(X, GX, n_i, n_j, n_cells, mat_row, mat_col,
                                    mat_value_Ab, mat_value_Ba, a_tilde, beta);

    keepGoing = false;
    for (size_t i = 0; i < n_i; ++i) {
      if (continue_criterion(X(i), GX(i), diffMax)) {
        keepGoing = true;
        break;
      }
    }
  }

  // Compute beta and then alpha
  // beta = const_b - (Ba %*% alpha)
  vec beta_final = const_b;

  for (size_t obs = 0; obs < n_cells; ++obs) {
    beta_final(mat_col(obs)) -= mat_value_Ba(obs) * GX(mat_row(obs));
  }

  // alpha = const_a - (Ab %*% beta)
  vec alpha_final = const_a;

  for (size_t obs = 0; obs < n_cells; ++obs) {
    alpha_final(mat_row(obs)) -= mat_value_Ab(obs) * beta_final(mat_col(obs));
  }

  // Final mu
  vec result_mu(n_obs, fill::none);
  for (size_t obs = 0; obs < n_obs; ++obs) {
    result_mu(obs) =
        mu_in(obs) + alpha_final(dum_i(obs)) + beta_final(dum_j(obs));
  }

  final_iter = iter;
  return result_mu;
}

// analogous to conv_acc_gau_2
vec conv_seq_gau_2(size_t n_i, size_t n_j, size_t n_cells, const uvec &mat_row,
                   const uvec &mat_col, const vec &mat_value_Ab,
                   const vec &mat_value_Ba, const field<uvec> &dum_vector,
                   const vec &lhs, const vec &invTableCluster_vector,
                   size_t iterMax, double diffMax, const vec &mu_in,
                   size_t &final_iter) {
  size_t n_obs = mu_in.n_elem;

  vec resid = lhs - mu_in;

  vec const_a(n_i, fill::zeros);
  vec const_b(n_j, fill::zeros);
  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);
  const vec &invTable_i = invTableCluster_vector.head(n_i);
  const vec &invTable_j =
      invTableCluster_vector.tail(invTableCluster_vector.n_elem - n_i);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    double resid_tmp = resid(obs);
    uword d_i = dum_i(obs);
    uword d_j = dum_j(obs);

    const_a(d_i) += resid_tmp * invTable_i(d_i);
    const_b(d_j) += resid_tmp * invTable_j(d_j);
  }

  vec beta(n_j, fill::none);

  vec a_tilde = const_a;

  for (size_t obs = 0; obs < n_cells; ++obs) {
    a_tilde(mat_row(obs)) -= mat_value_Ab(obs) * const_b(mat_col(obs));
  }

  vec X(n_i, fill::none);
  vec X_new(n_i, fill::none);

  X = a_tilde;

  bool keepGoing = true;
  size_t iter = 0;

  while (keepGoing && iter < iterMax) {
    ++iter;

    if (iter % 2 == 1) {
      cluster_coefficients_gaussian_2(X, X_new, n_i, n_j, n_cells, mat_row,
                                      mat_col, mat_value_Ab, mat_value_Ba,
                                      a_tilde, beta);
    } else {
      cluster_coefficients_gaussian_2(X_new, X, n_i, n_j, n_cells, mat_row,
                                      mat_col, mat_value_Ab, mat_value_Ba,
                                      a_tilde, beta);
    }

    keepGoing = false;
    const vec &X_current = (iter % 2 == 1) ? X : X_new;
    const vec &X_new_current = (iter % 2 == 1) ? X_new : X;

    for (size_t i = 0; i < n_i; ++i) {
      if (continue_criterion(X_current(i), X_new_current(i), diffMax)) {
        keepGoing = true;
        break;
      }
    }
  }

  const vec &X_final = (iter % 2 == 1) ? X_new : X;

  vec beta_final = const_b;

  for (size_t obs = 0; obs < n_cells; ++obs) {
    beta_final(mat_col(obs)) -= mat_value_Ba(obs) * X_final(mat_row(obs));
  }

  vec alpha_final = const_a;

  for (size_t obs = 0; obs < n_cells; ++obs) {
    alpha_final(mat_row(obs)) -= mat_value_Ab(obs) * beta_final(mat_col(obs));
  }

  vec result_mu(n_obs, fill::none);
  for (size_t obs = 0; obs < n_obs; ++obs) {
    result_mu(obs) =
        mu_in(obs) + alpha_final(dum_i(obs)) + beta_final(dum_j(obs));
  }

  final_iter = iter;
  return result_mu;
}

// Derivative convergence functions

mat derivative_convergence_seq_gnl(size_t iterMax, double diffMax,
                                   size_t n_vars, const uvec &nb_cluster_all,
                                   const vec &ll_d2, const mat &jacob_matrix,
                                   const mat &deriv_init_matrix,
                                   const field<uvec> &dum_vector,
                                   size_t &final_iter) {
  size_t n_obs = ll_d2.n_elem;
  size_t K = nb_cluster_all.n_elem;

  size_t nb_coef = sum(nb_cluster_all);

  // Target: deriv
  mat deriv = deriv_init_matrix;

  // Derivative coefficients and sum of log-likelihood derivatives
  vec deriv_coef(nb_coef, fill::none);
  vec sum_ll_d2(nb_coef, fill::zeros);

  field<vec> pderiv_coef(K);
  field<vec> psum_ll_d2(K);

  size_t coef_idx = 0;
  for (size_t k = 0; k < K; ++k) {
    size_t nk = nb_cluster_all(k);
    pderiv_coef(k) = deriv_coef.subvec(coef_idx, coef_idx + nk - 1);
    psum_ll_d2(k) = sum_ll_d2.subvec(coef_idx, coef_idx + nk - 1);
    coef_idx += nk;
  }

  for (size_t k = 0; k < K; ++k) {
    vec &my_sum_ll_d2 = psum_ll_d2(k);
    const uvec &my_dum = dum_vector(k);
    for (size_t i = 0; i < n_obs; ++i) {
      my_sum_ll_d2(my_dum(i)) += ll_d2(i);
    }
  }

  size_t iter_all_max = 0;

  // Loop on each variable
  for (size_t v = 0; v < n_vars; ++v) {
    vec my_deriv = deriv.col(v);
    const vec &my_jac = jacob_matrix.col(v);

    bool keepGoing = true;
    size_t iter = 0;

    while (keepGoing && iter < iterMax) {
      ++iter;
      keepGoing = false;

      // Update the clusters sequentially
      // Loop over all clusters from K to 1
      for (int k = static_cast<int>(K) - 1; k >= 0; k--) {
        size_t uk = static_cast<size_t>(k);

        vec &my_deriv_coef = pderiv_coef(uk);
        const uvec &my_dum = dum_vector(uk);
        const vec &my_sum_ll_d2 = psum_ll_d2(uk);

        my_deriv_coef.zeros();

        // Sum the jacobian and derivatives
        for (size_t i = 0; i < n_obs; ++i) {
          my_deriv_coef(my_dum(i)) += (my_jac(i) + my_deriv(i)) * ll_d2(i);
        }

        // Divide by the log-likelihood sum
        my_deriv_coef /= -my_sum_ll_d2;

        // Update the derivative value
        for (size_t i = 0; i < n_obs; ++i) {
          my_deriv(i) += my_deriv_coef(my_dum(i));
        }

        // Stopping criterion
        if (!keepGoing) {
          if (any(abs(my_deriv_coef) > diffMax)) {
            keepGoing = true;
          }
        }
      }
    }

    if (iter > iter_all_max) {
      iter_all_max = iter;
    }

    // Save back to result matrix
    deriv.col(v) = my_deriv;
  }

  final_iter = iter_all_max;
  return deriv;
}

// Structure for derivative coefficient computation
struct PARAM_DERIV_COEF {
  size_t n_obs;
  size_t K;

  // Vectors of data
  field<uvec> pdum;
  field<vec> psum_ll_d2;
  field<vec> psum_jac_lld2;
  uvec pcluster;
  vec ll_d2;

  // Varying value
  vec deriv_with_coef;
};

// Compute the first two coefficients starting by the last item
void derivative_coefficients(field<vec> &pcoef_origin,
                             field<vec> &pcoef_destination,
                             const vec &my_deriv_init, PARAM_DERIV_COEF &args) {
  args.deriv_with_coef = my_deriv_init;

  for (size_t k = 0; k < (args.K - 1); ++k) {
    const uvec &my_dum = args.pdum(k);
    const vec &my_deriv_coef = pcoef_origin(k);

    for (size_t i = 0; i < args.n_obs; ++i) {
      args.deriv_with_coef(i) += my_deriv_coef(my_dum(i));
    }
  }

  for (int k = static_cast<int>(args.K) - 1; k >= 0; k--) {
    size_t uk = static_cast<size_t>(k);

    // Compute the optimal cluster coefficients
    vec &my_deriv_coef = pcoef_destination(uk);
    const vec &my_sum_ll_d2 = args.psum_ll_d2(uk);
    const vec &my_sum_jac_lld2 = args.psum_jac_lld2(uk);
    const uvec &my_dum = args.pdum(uk);

    // Update the derivative coefficients
    my_deriv_coef = my_sum_jac_lld2;

    // Sum the derivatives
    for (size_t i = 0; i < args.n_obs; ++i) {
      my_deriv_coef(my_dum(i)) += args.deriv_with_coef(i) * args.ll_d2(i);
    }

    // Divide by the log-likelihood sum
    my_deriv_coef /= -my_sum_ll_d2;

    // Update the value of deriv_with_coef (if necessary)
    if (k != 0) {
      args.deriv_with_coef = my_deriv_init;

      for (size_t h = 0; h < args.K; h++) {
        if (h == uk - 1)
          continue;

        const uvec &my_dum_h = args.pdum(h);
        const vec &my_deriv_coef_h =
            (h < uk - 1) ? pcoef_origin(h) : pcoef_destination(h);

        for (size_t i = 0; i < args.n_obs; ++i) {
          args.deriv_with_coef(i) += my_deriv_coef_h(my_dum_h(i));
        }
      }
    }
  }
}

mat derivative_convergence_acc_gnl(size_t iterMax, double diffMax,
                                   size_t n_vars, const uvec &nb_cluster_all,
                                   const vec &ll_d2, const mat &jacob_matrix,
                                   const mat &deriv_init_matrix,
                                   const field<uvec> &dum_vector,
                                   size_t &final_iter) {
  size_t n_obs = ll_d2.n_elem;
  size_t K = nb_cluster_all.n_elem;

  size_t nb_coef = sum(nb_cluster_all);

  vec deriv_coef(nb_coef, fill::none);
  vec sum_ll_d2(nb_coef, fill::zeros);
  vec sum_jac_lld2(nb_coef, fill::none);

  field<vec> pderiv_coef(K);
  field<vec> psum_ll_d2(K);
  field<vec> psum_jac_lld2(K);

  size_t coef_idx = 0;
  for (size_t k = 0; k < K; ++k) {
    size_t nk = nb_cluster_all(k);
    pderiv_coef(k) = deriv_coef.subvec(coef_idx, coef_idx + nk - 1);
    psum_ll_d2(k) = sum_ll_d2.subvec(coef_idx, coef_idx + nk - 1);
    psum_jac_lld2(k) = sum_jac_lld2.subvec(coef_idx, coef_idx + nk - 1);
    coef_idx += nk;
  }

  // Sum of log-likelihood derivatives
  for (size_t k = 0; k < K; ++k) {
    vec &my_sum_ll_d2 = psum_ll_d2(k);
    const uvec &my_dum = dum_vector(k);
    for (size_t i = 0; i < n_obs; ++i) {
      my_sum_ll_d2(my_dum(i)) += ll_d2(i);
    }
  }

  PARAM_DERIV_COEF args;
  args.n_obs = n_obs;
  args.K = K;
  args.pdum = dum_vector;
  args.psum_ll_d2 = psum_ll_d2;
  args.psum_jac_lld2 = psum_jac_lld2;
  args.pcluster = nb_cluster_all;
  args.ll_d2 = ll_d2;
  args.deriv_with_coef.set_size(n_obs);

  // Preparation for Irons-Tuck
  field<vec> X(K), GX(K), GGX(K);

  for (size_t k = 0; k < K; ++k) {
    size_t nk = nb_cluster_all(k);
    X(k).set_size(nk);
    GX(k).set_size(nk);
    GGX(k).set_size(nk);
  }

  // Variables on 1:(K-1)
  size_t nb_coef_no_K = sum(nb_cluster_all.head(K - 1));
  vec delta_GX(nb_coef_no_K, fill::none);
  vec delta2_X(nb_coef_no_K, fill::none);

  mat dxi_dbeta(n_obs, n_vars, fill::none);

  size_t iter_all_max = 0;

  // Loop on each variable
  for (size_t v = 0; v < n_vars; ++v) {
    const vec &my_deriv_init = deriv_init_matrix.col(v);
    const vec &my_jac = jacob_matrix.col(v);

    for (size_t k = 0; k < K; ++k) {
      const uvec &my_dum = dum_vector(k);
      vec &my_sum_jac_lld2 = psum_jac_lld2(k);

      my_sum_jac_lld2.zeros();

      for (size_t i = 0; i < n_obs; ++i) {
        my_sum_jac_lld2(my_dum(i)) += my_jac(i) * ll_d2(i);
      }
    }

    // Irons-Tuck loop

    for (size_t k = 0; k < K; ++k) {
      X(k).zeros();
    }

    derivative_coefficients(X, GX, my_deriv_init, args);

    bool keepGoing = true;
    size_t iter = 0;
    bool numconv = false;

    while (keepGoing && iter < iterMax) {
      ++iter;

      derivative_coefficients(GX, GGX, my_deriv_init, args);

      vec X_flat(nb_coef_no_K, fill::none);
      vec GX_flat(nb_coef_no_K, fill::none);
      vec GGX_flat(nb_coef_no_K, fill::none);

      coef_idx = 0;
      for (size_t k = 0; k < K - 1; ++k) {
        size_t nk = X(k).n_elem;
        X_flat.subvec(coef_idx, coef_idx + nk - 1) = X(k);
        GX_flat.subvec(coef_idx, coef_idx + nk - 1) = GX(k);
        GGX_flat.subvec(coef_idx, coef_idx + nk - 1) = GGX(k);
        coef_idx += nk;
      }

      numconv = update_X_IronsTuck(nb_coef_no_K, X_flat, GX_flat, GGX_flat,
                                   delta_GX, delta2_X);
      if (numconv)
        break;

      coef_idx = 0;
      for (size_t k = 0; k < K - 1; ++k) {
        size_t nk = X(k).n_elem;
        X(k) = X_flat.subvec(coef_idx, coef_idx + nk - 1);
        coef_idx += nk;
      }

      derivative_coefficients(X, GX, my_deriv_init, args);

      keepGoing = false;

      coef_idx = 0;
      for (size_t k = 0; k < K - 1; ++k) {
        const vec &X_k = X(k);
        const vec &GX_k = GX(k);
        for (size_t i = 0; i < X_k.n_elem; ++i) {
          if (continue_criterion(X_k(i), GX_k(i), diffMax)) {
            keepGoing = true;
            break;
          }
        }
        if (keepGoing)
          break;
      }
    }

    if (iter > iter_all_max) {
      iter_all_max = iter;
    }

    args.deriv_with_coef = my_deriv_init;

    for (size_t k = 0; k < K; ++k) {
      const uvec &my_dum = dum_vector(k);
      const vec &my_deriv_coef = GX(k);
      for (size_t i = 0; i < n_obs; ++i) {
        args.deriv_with_coef(i) += my_deriv_coef(my_dum(i));
      }
    }

    dxi_dbeta.col(v) = args.deriv_with_coef;
  }

  final_iter = iter_all_max;
  return dxi_dbeta;
}

void derivative_coefficients_2(const vec &alpha_origin, vec &alpha_destination,
                               size_t n_i, size_t n_j, size_t n_cells,
                               const vec &a_tilde, const uvec &mat_row,
                               const uvec &mat_col, const vec &mat_value_Ab,
                               const vec &mat_value_Ba, vec &beta) {
  // a_tilde + Ab * Ba * alpha
  alpha_destination = a_tilde;
  beta.zeros();

  for (size_t obs = 0; obs < n_cells; ++obs) {
    beta(mat_col(obs)) += mat_value_Ba(obs) * alpha_origin(mat_row(obs));
  }

  for (size_t obs = 0; obs < n_cells; ++obs) {
    alpha_destination(mat_row(obs)) += mat_value_Ab(obs) * beta(mat_col(obs));
  }
}

// Similar to derivative_convergence_acc_gnl
mat derivative_convergence_acc_2(
    size_t iterMax, double diffMax, size_t n_vars, const uvec &nb_cluster_all,
    size_t n_cells, const uvec &index_i, const uvec &index_j, const vec &ll_d2,
    const uvec &order, const mat &jacob_matrix, const mat &deriv_init_matrix,
    const field<uvec> &dum_vector, size_t &final_iter) {
  size_t n_obs = ll_d2.n_elem;
  size_t n_i = nb_cluster_all(0);
  size_t n_j = nb_cluster_all(1);

  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);

  vec sum_ll_d2_i(n_i, fill::zeros);
  vec sum_ll_d2_j(n_j, fill::zeros);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    sum_ll_d2_i(dum_i(obs)) += ll_d2(obs);
    sum_ll_d2_j(dum_j(obs)) += ll_d2(obs);
  }

  uvec mat_row(n_cells, fill::none);
  uvec mat_col(n_cells, fill::none);
  vec mat_value_Ab(n_cells, fill::none);
  vec mat_value_Ba(n_cells, fill::none);

  size_t index_current = 0;

  double value_Ab = ll_d2(order(0));
  double value_Ba = ll_d2(order(0));

  for (size_t obs = 1; obs < n_obs; ++obs) {
    if (index_j(obs) != index_j(obs - 1) || index_i(obs) != index_i(obs - 1)) {
      mat_row(index_current) = index_i(obs - 1);
      mat_col(index_current) = index_j(obs - 1);
      mat_value_Ab(index_current) = value_Ab / -sum_ll_d2_i(index_i(obs - 1));
      mat_value_Ba(index_current) = value_Ba / -sum_ll_d2_j(index_j(obs - 1));

      index_current++;

      value_Ab = ll_d2(order(obs));
      value_Ba = ll_d2(order(obs));
    } else {
      value_Ab += ll_d2(order(obs));
      value_Ba += ll_d2(order(obs));
    }
  }

  mat_row(index_current) = index_i(n_obs - 1);
  mat_col(index_current) = index_j(n_obs - 1);
  mat_value_Ab(index_current) = value_Ab / -sum_ll_d2_i(index_i(n_obs - 1));
  mat_value_Ba(index_current) = value_Ba / -sum_ll_d2_j(index_j(n_obs - 1));

  vec X(n_i, fill::none);
  vec GX(n_i, fill::none);
  vec GGX(n_i, fill::none);
  vec delta_GX(n_i, fill::none);
  vec delta2_X(n_i, fill::none);
  vec beta(n_j, fill::none);
  vec alpha_final(n_i, fill::none);
  vec beta_final(n_j, fill::none);

  mat dxi_dbeta(n_obs, n_vars, fill::none);

  size_t iter_all_max = 0;

  for (size_t v = 0; v < n_vars; ++v) {
    const vec &my_deriv_init = deriv_init_matrix.col(v);
    const vec &my_jac = jacob_matrix.col(v);

    vec a(n_i, fill::zeros);
    vec b(n_j, fill::zeros);

    for (size_t obs = 0; obs < n_obs; ++obs) {
      a(dum_i(obs)) += (my_jac(obs) + my_deriv_init(obs)) * ll_d2(obs);
      b(dum_j(obs)) += (my_jac(obs) + my_deriv_init(obs)) * ll_d2(obs);
    }

    a /= -sum_ll_d2_i;
    b /= -sum_ll_d2_j;

    vec a_tilde = a;
    for (size_t i = 0; i < n_cells; ++i) {
      a_tilde(mat_row(i)) += mat_value_Ab(i) * b(mat_col(i));
    }

    X.zeros();

    derivative_coefficients_2(X, GX, n_i, n_j, n_cells, a_tilde, mat_row,
                              mat_col, mat_value_Ab, mat_value_Ba, beta);

    bool keepGoing = true;
    size_t iter = 0;
    bool numconv = false;

    while (keepGoing && iter < iterMax) {
      ++iter;

      derivative_coefficients_2(GX, GGX, n_i, n_j, n_cells, a_tilde, mat_row,
                                mat_col, mat_value_Ab, mat_value_Ba, beta);

      numconv = update_X_IronsTuck(n_i, X, GX, GGX, delta_GX, delta2_X);
      if (numconv)
        break;

      derivative_coefficients_2(X, GX, n_i, n_j, n_cells, a_tilde, mat_row,
                                mat_col, mat_value_Ab, mat_value_Ba, beta);

      keepGoing = false;
      for (size_t m = 0; m < n_i; ++m) {
        if (continue_criterion(X(m), GX(m), diffMax)) {
          keepGoing = true;
          break;
        }
      }
    }

    if (iter > iter_all_max) {
      iter_all_max = iter;
    }

    alpha_final = a;
    beta_final = b;

    for (size_t obs = 0; obs < n_cells; ++obs) {
      beta_final(mat_col(obs)) += mat_value_Ba(obs) * GX(mat_row(obs));
    }

    for (size_t obs = 0; obs < n_cells; ++obs) {
      alpha_final(mat_row(obs)) += mat_value_Ab(obs) * beta_final(mat_col(obs));
    }

    for (size_t obs = 0; obs < n_obs; ++obs) {
      dxi_dbeta(obs, v) =
          my_deriv_init(obs) + alpha_final(dum_i(obs)) + beta_final(dum_j(obs));
    }
  }

  final_iter = iter_all_max;
  return dxi_dbeta;
}

// Similar to derivative_convergence_acc_gnl
mat derivative_convergence_seq_2(
    size_t iterMax, double diffMax, size_t n_vars, const uvec &nb_cluster_all,
    size_t n_cells, const uvec &index_i, const uvec &index_j, const uvec &order,
    const vec &ll_d2, const mat &jacob_matrix, const mat &deriv_init_matrix,
    const field<uvec> &dum_vector, size_t &final_iter) {
  size_t n_obs = ll_d2.n_elem;
  size_t n_i = nb_cluster_all(0);
  size_t n_j = nb_cluster_all(1);

  const uvec &dum_i = dum_vector(0);
  const uvec &dum_j = dum_vector(1);

  vec sum_ll_d2_i(n_i, fill::zeros);
  vec sum_ll_d2_j(n_j, fill::zeros);

  for (size_t obs = 0; obs < n_obs; ++obs) {
    sum_ll_d2_i(dum_i(obs)) += ll_d2(obs);
    sum_ll_d2_j(dum_j(obs)) += ll_d2(obs);
  }

  uvec mat_row(n_cells, fill::none);
  uvec mat_col(n_cells, fill::none);
  vec mat_value_Ab(n_cells, fill::none);
  vec mat_value_Ba(n_cells, fill::none);

  size_t index_current = 0;

  double value_current = ll_d2(order(0));

  for (size_t obs = 1; obs < n_obs; ++obs) {
    if (index_j(obs) != index_j(obs - 1) || index_i(obs) != index_i(obs - 1)) {
      mat_row(index_current) = index_i(obs - 1);
      mat_col(index_current) = index_j(obs - 1);
      mat_value_Ab(index_current) =
          value_current / -sum_ll_d2_i(index_i(obs - 1));
      mat_value_Ba(index_current) =
          value_current / -sum_ll_d2_j(index_j(obs - 1));

      index_current++;

      value_current = ll_d2(order(obs));
    } else {
      value_current += ll_d2(order(obs));
    }
  }

  mat_row(index_current) = index_i(n_obs - 1);
  mat_col(index_current) = index_j(n_obs - 1);
  mat_value_Ab(index_current) =
      value_current / -sum_ll_d2_i(index_i(n_obs - 1));
  mat_value_Ba(index_current) =
      value_current / -sum_ll_d2_j(index_j(n_obs - 1));

  vec X(n_i, fill::none);
  vec X_new(n_i, fill::none);
  vec beta(n_j, fill::none);
  vec alpha_final(n_i, fill::none);
  vec beta_final(n_j, fill::none);

  mat dxi_dbeta(n_obs, n_vars, fill::none);

  size_t iter_all_max = 0;

  for (size_t v = 0; v < n_vars; ++v) {
    const vec &my_deriv_init = deriv_init_matrix.col(v);
    const vec &my_jac = jacob_matrix.col(v);

    vec a(n_i, fill::zeros);
    vec b(n_j, fill::zeros);

    for (size_t obs = 0; obs < n_obs; ++obs) {
      a(dum_i(obs)) += (my_jac(obs) + my_deriv_init(obs)) * ll_d2(obs);
      b(dum_j(obs)) += (my_jac(obs) + my_deriv_init(obs)) * ll_d2(obs);
    }

    a /= -sum_ll_d2_i;
    b /= -sum_ll_d2_j;

    vec a_tilde = a;
    for (size_t i = 0; i < n_cells; ++i) {
      a_tilde(mat_row(i)) += mat_value_Ab(i) * b(mat_col(i));
    }

    X.zeros();

    bool keepGoing = true;
    size_t iter = 0;

    while (keepGoing && iter < iterMax) {
      ++iter;

      if (iter % 2 == 1) {
        derivative_coefficients_2(X, X_new, n_i, n_j, n_cells, a_tilde, mat_row,
                                  mat_col, mat_value_Ab, mat_value_Ba, beta);
      } else {
        derivative_coefficients_2(X_new, X, n_i, n_j, n_cells, a_tilde, mat_row,
                                  mat_col, mat_value_Ab, mat_value_Ba, beta);
      }

      keepGoing = false;
      const vec &X_current = (iter % 2 == 1) ? X : X_new;
      const vec &X_new_current = (iter % 2 == 1) ? X_new : X;

      for (size_t m = 0; m < n_i; ++m) {
        if (continue_criterion(X_current(m), X_new_current(m), diffMax)) {
          keepGoing = true;
          break;
        }
      }
    }

    if (iter > iter_all_max) {
      iter_all_max = iter;
    }

    const vec &X_final = (iter % 2 == 1) ? X_new : X;

    alpha_final = a;
    beta_final = b;

    for (size_t obs = 0; obs < n_cells; ++obs) {
      beta_final(mat_col(obs)) += mat_value_Ba(obs) * X_final(mat_row(obs));
    }

    for (size_t obs = 0; obs < n_cells; ++obs) {
      alpha_final(mat_row(obs)) += mat_value_Ab(obs) * beta_final(mat_col(obs));
    }

    for (size_t obs = 0; obs < n_obs; ++obs) {
      dxi_dbeta(obs, v) =
          my_deriv_init(obs) + alpha_final(dum_i(obs)) + beta_final(dum_j(obs));
    }
  }

  final_iter = iter_all_max;
  return dxi_dbeta;
}

mat update_derivative_single(size_t n_vars, size_t nb_coef, const vec &ll_d2,
                             const mat &jacob_matrix, const uvec &dum_vector) {
  size_t n_obs = ll_d2.n_elem;

  // Sum of log-likelihood derivative
  vec sum_ll_d2(nb_coef, fill::zeros);
  for (size_t obs = 0; obs < n_obs; ++obs) {
    sum_ll_d2(dum_vector(obs)) += ll_d2(obs);
  }

  vec coef_deriv(nb_coef, fill::none);

  mat result(n_obs, n_vars, fill::none);

  for (size_t v = 0; v < n_vars; ++v) {
    const vec &my_jac = jacob_matrix.col(v);

    // 1) Compute coefficients
    coef_deriv.zeros();

    for (size_t obs = 0; obs < n_obs; ++obs) {
      coef_deriv(dum_vector(obs)) += my_jac(obs) * ll_d2(obs);
    }

    coef_deriv /= -sum_ll_d2;

    // 2) Save the result
    for (size_t obs = 0; obs < n_obs; ++obs) {
      result(obs, v) = coef_deriv(dum_vector(obs));
    }
  }

  return result;
}

struct WeightedDemeanResult {
  mat demeaned_data;
  bool success;
};

inline WeightedDemeanResult
demean_variables(const mat &data, const umat &fe_matrix,
                 const vec &weights = vec(), double tol = 1e-8,
                 size_t max_iter = 10000,
                 const std::string &family = "gaussian") {
  WeightedDemeanResult result;
  result.success = false;

  if (fe_matrix.n_cols == 0) {
    // No fixed effects => return original data
    result.demeaned_data = data;
    result.success = true;
    return result;
  }

  const size_t n_obs = data.n_rows;
  const size_t n_vars = data.n_cols;
  const size_t Q = fe_matrix.n_cols;

  try {
    // Convert umat fe_matrix to field<ivec> format for FEClass
    field<ivec> fe_id_list(Q);
    ivec nb_id_Q(Q);
    ivec table_id_I;

    // Calculate total number of groups across all FE dimensions
    size_t total_groups = 0;
    for (size_t q = 0; q < Q; ++q) {
      uvec unique_ids = unique(fe_matrix.col(q));
      nb_id_Q(q) = unique_ids.n_elem;
      total_groups += unique_ids.n_elem;

      // TODO: maybe follow a 0 based index to save steps?
      // Convert to 1-based indexing for FEClass (fixest convention)
      fe_id_list(q) = conv_to<ivec>::from(fe_matrix.col(q)) + 1;
    }

    // Create table with the number of observations per group
    table_id_I.set_size(total_groups);
    size_t group_idx = 0;
    for (size_t q = 0; q < Q; ++q) {
      uvec unique_ids = unique(fe_matrix.col(q));
      for (size_t g = 0; g < unique_ids.n_elem; ++g) {
        uvec group_obs = find(fe_matrix.col(q) == unique_ids(g));
        table_id_I(group_idx++) = group_obs.n_elem;
      }
    }

    // Standard FE demeaning
    // TODO: No varying slopes for now
    ivec slope_flag_Q = zeros<ivec>(Q);
    field<mat> slope_vars_list(Q);
    for (size_t q = 0; q < Q; ++q) {
      slope_vars_list(q).set_size(0, 0);
    }

    // Setup weights vector following fixest convention
    vec weights_vec = weights;
    if (weights_vec.n_elem == 0) {
      weights_vec = ones<vec>(1); // Scalar 1 for no weights
    }

    // Create FEClass instance
    FEClass fe_info(n_obs, Q, weights_vec, fe_id_list, nb_id_Q, table_id_I,
                    slope_flag_Q, slope_vars_list);

    PARAM_DEMEAN demean_args;
    demean_args.n_obs = n_obs;
    demean_args.Q = Q;
    demean_args.nb_coef_T = fe_info.nb_coef_T;
    demean_args.iterMax = max_iter;
    demean_args.diffMax = tol;

    // TODO: these should be moved to R's fit_control.R to allow
    // user customization
    demean_args.algo_extraProj = 2;
    demean_args.algo_iter_warmup = 6;
    demean_args.algo_iter_projAfterAcc = 5;
    demean_args.algo_iter_grandAcc = 6;
    demean_args.save_fixef = false;

    demean_args.fixef_values = nullptr;
    demean_args.p_FE_info = &fe_info;

    // Control variables for the algorithm
    bool stopnow = false;
    demean_args.stopnow = &stopnow;

    std::vector<int> jobdone(n_vars, 0);
    std::vector<int> iterations_all(n_vars, 0);
    demean_args.jobdone = jobdone.data();
    demean_args.p_iterations_all = iterations_all.data();

    // Prepare input/output data
    result.demeaned_data.set_size(n_obs, n_vars);
    result.demeaned_data.zeros(); // Initialize to zeros

    demean_args.p_input.resize(n_vars);
    demean_args.p_output.resize(n_vars);

    for (size_t v = 0; v < n_vars; ++v) {
      demean_args.p_input[v] = data.col(v);
      demean_args.p_output[v] = result.demeaned_data.colptr(v);
    }

    // Demeaning for each variable
    for (size_t v = 0; v < n_vars; ++v) {
      if (Q == 1) {
        // Single FE case (closed form)
        demean_single_1(v, &demean_args);
      } else {
        // Multiple FE case (iterative algorithm)
        demean_single_gnl(v, &demean_args);
      }
    }

    // Demeaned data = original - fitted_fe_values
    for (size_t v = 0; v < n_vars; ++v) {
      vec original = demean_args.p_input[v];
      vec fitted_fe = vec(demean_args.p_output[v], n_obs);
      result.demeaned_data.col(v) = original - fitted_fe;
    }

    result.success = true;
  } catch (const std::exception &e) {
    // If demeaning fails, return original data
    result.demeaned_data = data;
    result.success = false;
  }

  return result;
}

#endif // CAPYBARA_CONVERGENCE
