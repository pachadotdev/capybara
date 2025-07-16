#ifndef CAPYBARA_CONVERGENCE
#define CAPYBARA_CONVERGENCE

// Forward declaration of FEClass from demeaning
class FEClass;

// Convergence result structures
struct ConvergenceResult {
  vec mu_new;
  bool success;
  size_t iterations;
  bool any_negative_poisson;

  ConvergenceResult() : success(false), iterations(0), any_negative_poisson(false) {}
};

// Family constants for convergence - using integer codes
// 1 = POISSON, 2 = NEGBIN, 3 = LOGIT, 4 = GAUSSIAN, 5 = LOG_POISSON

// Core cluster coefficient computation functions for each family

inline void ccc_poisson(size_t n_obs, size_t nb_cluster, vec &cluster_coef,
                       const vec &exp_mu, const vec &sum_y, const uvec &dum) {
  // Compute cluster coefficients for Poisson family
  cluster_coef.zeros();
  
  // Accumulate exp_mu values by cluster
  for (size_t i = 0; i < n_obs; ++i) {
    cluster_coef(dum(i)) += exp_mu(i);
  }
  
  // Calculate cluster coefficients
  for (size_t m = 0; m < nb_cluster; ++m) {
    if (cluster_coef(m) > 0) {
      cluster_coef(m) = sum_y(m) / cluster_coef(m);
    }
  }
}

inline void ccc_poisson_log(size_t n_obs, size_t nb_cluster, vec &cluster_coef,
                           const vec &mu, const vec &sum_y, const uvec &dum) {
  // Compute cluster coefficients for log Poisson (handles high values)
  vec mu_max(nb_cluster);
  uvec doInit(nb_cluster, fill::ones);
  
  cluster_coef.zeros();
  
  // Find max mu for each cluster
  for (size_t i = 0; i < n_obs; ++i) {
    size_t d = dum(i);
    if (doInit(d)) {
      mu_max(d) = mu(i);
      doInit(d) = 0;
    } else if (mu(i) > mu_max(d)) {
      mu_max(d) = mu(i);
    }
  }
  
  // Accumulate exp(mu - mu_max) by cluster
  for (size_t i = 0; i < n_obs; ++i) {
    size_t d = dum(i);
    cluster_coef(d) += std::exp(mu(i) - mu_max(d));
  }
  
  // Calculate cluster coefficients with log trick
  for (size_t m = 0; m < nb_cluster; ++m) {
    if (cluster_coef(m) > 0 && sum_y(m) > 0) {
      cluster_coef(m) = std::log(sum_y(m)) - std::log(cluster_coef(m)) - mu_max(m);
    }
  }
}

inline void ccc_gaussian(size_t n_obs, size_t nb_cluster, vec &cluster_coef,
                        const vec &mu, const vec &sum_y, const uvec &dum, 
                        const uvec &table) {
  // Compute cluster coefficients for Gaussian family
  cluster_coef.zeros();
  
  // Accumulate mu values by cluster
  for (size_t i = 0; i < n_obs; ++i) {
    cluster_coef(dum(i)) += mu(i);
  }
  
  // Calculate cluster coefficients
  for (size_t m = 0; m < nb_cluster; ++m) {
    if (table(m) > 0) {
      cluster_coef(m) = (sum_y(m) - cluster_coef(m)) / static_cast<double>(table(m));
    }
  }
}

inline void ccc_negbin(size_t nb_cluster, double theta, double diffMax_NR,
                      vec &cluster_coef, const vec &mu, const vec &lhs, 
                      const vec &sum_y, const uvec &obsCluster, 
                      const uvec &table, const uvec &cumtable) {
  // Compute cluster coefficients for negative binomial using Newton-Raphson + dichotomy
  size_t iterMax = 100;
  size_t iterFullDicho = 10;
  
  vec borne_inf(nb_cluster);
  vec borne_sup(nb_cluster);
  
  // Find bounds for each cluster
  for (size_t m = 0; m < nb_cluster; ++m) {
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);
    
    double mu_min = mu(obsCluster(u0));
    double mu_max = mu(obsCluster(u0));
    
    for (size_t u = u0 + 1; u < u_end; ++u) {
      double value = mu(obsCluster(u));
      if (value < mu_min) mu_min = value;
      if (value > mu_max) mu_max = value;
    }
    
    borne_inf(m) = std::log(sum_y(m)) - std::log(static_cast<double>(table(m))) - mu_max;
    borne_sup(m) = borne_inf(m) + (mu_max - mu_min);
  }
  
  // Solve for each cluster coefficient
  for (size_t m = 0; m < nb_cluster; ++m) {
    double x1 = 0.0;
    bool keepGoing = true;
    size_t iter = 0;
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);
    
    double lower_bound = borne_inf(m);
    double upper_bound = borne_sup(m);
    
    // Ensure x1 is within bounds
    if (x1 >= upper_bound || x1 <= lower_bound) {
      x1 = (lower_bound + upper_bound) / 2.0;
    }
    
    while (keepGoing && iter < iterMax) {
      ++iter;
      
      // Compute f(x)
      double value = sum_y(m);
      for (size_t u = u0; u < u_end; ++u) {
        size_t i = obsCluster(u);
        value -= (theta + lhs(i)) / (1.0 + theta * std::exp(-x1 - mu(i)));
      }
      
      // Update bounds
      if (value > 0) {
        lower_bound = x1;
      } else {
        upper_bound = x1;
      }
      
      double x0 = x1;
      if (std::abs(value) < 1e-12) {
        keepGoing = false;
      } else if (iter <= iterFullDicho) {
        // Newton-Raphson step
        double derivee = 0.0;
        for (size_t u = u0; u < u_end; ++u) {
          size_t i = obsCluster(u);
          double exp_mu = std::exp(x1 + mu(i));
          derivee -= theta * (theta + lhs(i)) / 
                    ((theta / exp_mu + 1.0) * (theta + exp_mu));
        }
        
        if (std::abs(derivee) > 1e-12) {
          x1 = x0 - value / derivee;
        }
        
        // Apply bounds
        if (x1 >= upper_bound || x1 <= lower_bound) {
          x1 = (lower_bound + upper_bound) / 2.0;
        }
      } else {
        // Pure dichotomy
        x1 = (lower_bound + upper_bound) / 2.0;
      }
      
      // Check convergence
      double diff = std::abs(x0 - x1);
      if (diff < diffMax_NR || diff / (0.1 + std::abs(x1)) < diffMax_NR) {
        keepGoing = false;
      }
    }
    
    cluster_coef(m) = x1;
  }
}

inline void ccc_logit(size_t nb_cluster, double diffMax_NR,
                     vec &cluster_coef, const vec &mu, const vec &sum_y,
                     const uvec &obsCluster, const uvec &table, 
                     const uvec &cumtable) {
  // Compute cluster coefficients for logit using Newton-Raphson + dichotomy
  size_t iterMax = 100;
  size_t iterFullDicho = 10;
  
  vec borne_inf(nb_cluster);
  vec borne_sup(nb_cluster);
  
  // Find bounds for each cluster
  for (size_t m = 0; m < nb_cluster; ++m) {
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);
    
    double mu_min = mu(obsCluster(u0));
    double mu_max = mu(obsCluster(u0));
    
    for (size_t u = u0 + 1; u < u_end; ++u) {
      double value = mu(obsCluster(u));
      if (value < mu_min) mu_min = value;
      if (value > mu_max) mu_max = value;
    }
    
    borne_inf(m) = std::log(sum_y(m)) - std::log(table(m) - sum_y(m)) - mu_max;
    borne_sup(m) = borne_inf(m) + (mu_max - mu_min);
  }
  
  // Solve for each cluster coefficient
  for (size_t m = 0; m < nb_cluster; ++m) {
    double x1 = 0.0;
    bool keepGoing = true;
    size_t iter = 0;
    size_t u0 = (m == 0) ? 0 : cumtable(m - 1);
    size_t u_end = cumtable(m);
    
    double lower_bound = borne_inf(m);
    double upper_bound = borne_sup(m);
    
    // Ensure x1 is within bounds
    if (x1 >= upper_bound || x1 <= lower_bound) {
      x1 = (lower_bound + upper_bound) / 2.0;
    }
    
    while (keepGoing && iter < iterMax) {
      ++iter;
      
      // Compute f(x)
      double value = sum_y(m);
      for (size_t u = u0; u < u_end; ++u) {
        size_t i = obsCluster(u);
        value -= 1.0 / (1.0 + std::exp(-x1 - mu(i)));
      }
      
      // Update bounds
      if (value > 0) {
        lower_bound = x1;
      } else {
        upper_bound = x1;
      }
      
      double x0 = x1;
      if (std::abs(value) < 1e-12) {
        keepGoing = false;
      } else if (iter <= iterFullDicho) {
        // Newton-Raphson step
        double derivee = 0.0;
        for (size_t u = u0; u < u_end; ++u) {
          size_t i = obsCluster(u);
          double exp_mu = std::exp(x1 + mu(i));
          derivee -= 1.0 / ((1.0 / exp_mu + 1.0) * (1.0 + exp_mu));
        }
        
        if (std::abs(derivee) > 1e-12) {
          x1 = x0 - value / derivee;
        }
        
        // Apply bounds
        if (x1 >= upper_bound || x1 <= lower_bound) {
          x1 = (lower_bound + upper_bound) / 2.0;
        }
      } else {
        // Pure dichotomy
        x1 = (lower_bound + upper_bound) / 2.0;
      }
      
      // Check convergence
      double diff = std::abs(x0 - x1);
      if (diff < diffMax_NR || diff / (0.1 + std::abs(x1)) < diffMax_NR) {
        keepGoing = false;
      }
    }
    
    cluster_coef(m) = x1;
  }
}

// Main cluster coefficient computation dispatcher
inline void compute_cluster_coef_single(int family, size_t n_obs, size_t nb_cluster,
                                       double theta, double diffMax_NR,
                                       vec &cluster_coef, const vec &mu, 
                                       const vec &lhs, const vec &sum_y,
                                       const uvec &dum, const uvec &obsCluster,
                                       const uvec &table, const uvec &cumtable) {
  
  switch (family) {
    case 1: // POISSON
      ccc_poisson(n_obs, nb_cluster, cluster_coef, mu, sum_y, dum);
      break;
    case 2: // NEGBIN
      ccc_negbin(nb_cluster, theta, diffMax_NR, cluster_coef, mu, lhs, 
                sum_y, obsCluster, table, cumtable);
      break;
    case 3: // LOGIT
      ccc_logit(nb_cluster, diffMax_NR, cluster_coef, mu, sum_y, 
               obsCluster, table, cumtable);
      break;
    case 4: // GAUSSIAN
      ccc_gaussian(n_obs, nb_cluster, cluster_coef, mu, sum_y, dum, table);
      break;
    case 5: // LOG_POISSON
      ccc_poisson_log(n_obs, nb_cluster, cluster_coef, mu, sum_y, dum);
      break;
  }
}

// Irons-Tuck acceleration update for convergence
inline bool update_conv_IronsTuck(size_t nb_coef_no_K, vec &X, const vec &GX, 
                                 const vec &GGX, vec &delta_GX, vec &delta2_X) {
  
  // Compute differences
  for (size_t i = 0; i < nb_coef_no_K; ++i) {
    double GX_tmp = GX(i);
    delta_GX(i) = GGX(i) - GX_tmp;
    delta2_X(i) = delta_GX(i) - GX_tmp + X(i);
  }
  
  // Compute dot products
  double vprod = dot(delta_GX, delta2_X);
  double ssq = dot(delta2_X, delta2_X);
  
  if (ssq == 0.0) {
    return true;  // Numerical convergence
  }
  
  double coef = vprod / ssq;
  
  // Update X with acceleration
  for (size_t i = 0; i < nb_coef_no_K; ++i) {
    X(i) = GGX(i) - coef * delta_GX(i);
  }
  
  return false;
}

// Structure to hold convergence parameters (similar to PARAM_CCC)
struct ConvParams {
  int family;
  size_t n_obs;
  size_t K;
  double theta;
  double diffMax_NR;
  
  // Input data
  vec mu_init;
  vec lhs;
  
  // FE structure data
  field<uvec> fe_indices;  // FE indices for each observation
  field<uvec> table;       // Number of obs per cluster for each FE
  field<vec> sum_y;        // Sum of y per cluster for each FE  
  field<uvec> obsCluster;  // For negbin/logit: observation indices by cluster
  field<uvec> cumtable;    // For negbin/logit: cumulative cluster sizes
  
  // Working vector
  vec mu_with_coef;
  
  ConvParams() : family(1), n_obs(0), K(0), theta(1.0), diffMax_NR(1e-8) {}
};

// Multi-FE cluster coefficient computation
inline void compute_cluster_coef_multi(const field<vec> &coef_origin,
                                      field<vec> &coef_destination,
                                      ConvParams &params) {
  
  // Initialize mu_with_coef
  params.mu_with_coef = params.mu_init;
  
  // Add all FE contributions except the last one
  for (size_t k = 0; k < params.K - 1; ++k) {
    const uvec &fe_idx = params.fe_indices(k);
    const vec &cluster_coef = coef_origin(k);
    
    if (params.family == 1 || params.family == 5) { // POISSON or LOG_POISSON
      for (size_t i = 0; i < params.n_obs; ++i) {
        params.mu_with_coef(i) *= cluster_coef(fe_idx(i));
      }
    } else {
      for (size_t i = 0; i < params.n_obs; ++i) {
        params.mu_with_coef(i) += cluster_coef(fe_idx(i));
      }
    }
  }
  
  // Compute coefficients for each FE in reverse order
  for (int k = static_cast<int>(params.K) - 1; k >= 0; --k) {
    size_t uk = static_cast<size_t>(k);
    
    const uvec &fe_idx = params.fe_indices(uk);
    vec &cluster_coef = coef_destination(uk);
    size_t nb_cluster = cluster_coef.n_elem;
    
    // Compute cluster coefficients for this FE
    compute_cluster_coef_single(params.family, params.n_obs, nb_cluster,
                               params.theta, params.diffMax_NR,
                               cluster_coef, params.mu_with_coef, params.lhs,
                               params.sum_y(uk), fe_idx,
                               params.obsCluster(uk), params.table(uk),
                               params.cumtable(uk));
    
    // Update mu_with_coef for next iteration (if not last)
    if (k != 0) {
      // Recompute from scratch
      params.mu_with_coef = params.mu_init;
      
      for (size_t h = 0; h < params.K; ++h) {
        if (h == uk - 1) continue;  // Skip the next one to be computed
        
        const uvec &h_fe_idx = params.fe_indices(h);
        const vec &h_cluster_coef = (h < uk - 1) ? coef_origin(h) : coef_destination(h);
        
        if (params.family == 1 || params.family == 5) { // POISSON or LOG_POISSON
          for (size_t i = 0; i < params.n_obs; ++i) {
            params.mu_with_coef(i) *= h_cluster_coef(h_fe_idx(i));
          }
        } else {
          for (size_t i = 0; i < params.n_obs; ++i) {
            params.mu_with_coef(i) += h_cluster_coef(h_fe_idx(i));
          }
        }
      }
    }
  }
}

// Main convergence function with Irons-Tuck acceleration
inline ConvergenceResult convergence_accelerated(const vec &mu_init, const vec &lhs,
                                                const field<uvec> &fe_indices,
                                                const field<uvec> &table,
                                                const field<vec> &sum_y,
                                                const field<uvec> &obsCluster,
                                                const field<uvec> &cumtable,
                                                int family, double theta = 1.0,
                                                size_t iterMax = 10000,
                                                double diffMax = 1e-8,
                                                double diffMax_NR = 1e-8) {
  
  ConvergenceResult result;
  size_t K = fe_indices.n_elem;
  size_t n_obs = mu_init.n_elem;
  
  // Set up parameters
  ConvParams params;
  params.family = family;
  params.n_obs = n_obs;
  params.K = K;
  params.theta = theta;
  params.diffMax_NR = diffMax_NR;
  params.mu_init = mu_init;
  params.lhs = lhs;
  params.fe_indices = fe_indices;
  params.table = table;
  params.sum_y = sum_y;
  params.obsCluster = obsCluster;
  params.cumtable = cumtable;
  params.mu_with_coef.set_size(n_obs);
  
  // Calculate total number of coefficients
  size_t nb_coef = 0;
  size_t nb_coef_no_K = 0;
  for (size_t k = 0; k < K; ++k) {
    size_t nk = table(k).n_elem;
    nb_coef += nk;
    if (k < K - 1) nb_coef_no_K += nk;
  }
  
  // Initialize coefficient vectors
  field<vec> X(K), GX(K), GGX(K);
  for (size_t k = 0; k < K; ++k) {
    size_t nk = table(k).n_elem;
    X(k).set_size(nk);
    GX(k).set_size(nk);
    GGX(k).set_size(nk);
    
    // Initialize coefficients
    if (family == 1 || family == 5) { // POISSON or LOG_POISSON
      X(k).ones();
    } else {
      X(k).zeros();
    }
  }
  
  // Acceleration vectors (only for first K-1 FE)
  vec delta_GX(nb_coef_no_K);
  vec delta2_X(nb_coef_no_K);
  
  // First iteration
  compute_cluster_coef_multi(X, GX, params);
  
  // Check if we need to iterate
  bool keepGoing = false;
  size_t coef_idx = 0;
  for (size_t k = 0; k < K - 1; ++k) {
    for (size_t i = 0; i < X(k).n_elem; ++i) {
      double diff = std::abs(X(k)(i) - GX(k)(i));
      if (diff > diffMax && diff / (0.1 + std::abs(GX(k)(i))) > diffMax) {
        keepGoing = true;
        break;
      }
    }
    if (keepGoing) break;
  }
  
  size_t iter = 0;
  bool numconv = false;
  bool any_negative_poisson = false;
  
  while (keepGoing && iter < iterMax) {
    ++iter;
    
    // GGX computation
    compute_cluster_coef_multi(GX, GGX, params);
    
    // Irons-Tuck acceleration (flatten first K-1 FE coefficients)
    vec X_flat(nb_coef_no_K);
    vec GX_flat(nb_coef_no_K);
    vec GGX_flat(nb_coef_no_K);
    
    coef_idx = 0;
    for (size_t k = 0; k < K - 1; ++k) {
      for (size_t i = 0; i < X(k).n_elem; ++i) {
        X_flat(coef_idx) = X(k)(i);
        GX_flat(coef_idx) = GX(k)(i);
        GGX_flat(coef_idx) = GGX(k)(i);
        coef_idx++;
      }
    }
    
    numconv = update_conv_IronsTuck(nb_coef_no_K, X_flat, GX_flat, GGX_flat, 
                                   delta_GX, delta2_X);
    if (numconv) break;
    
    // Unflatten back to field structure
    coef_idx = 0;
    for (size_t k = 0; k < K - 1; ++k) {
      for (size_t i = 0; i < X(k).n_elem; ++i) {
        X(k)(i) = X_flat(coef_idx);
        coef_idx++;
      }
    }
    
    // Check for negative Poisson coefficients
    if (family == 1 || family == 5) { // POISSON or LOG_POISSON
      for (size_t k = 0; k < K - 1; ++k) {
        if (any(X(k) <= 0)) {
          any_negative_poisson = true;
          break;
        }
      }
      if (any_negative_poisson) break;
    }
    
    // Update GX
    compute_cluster_coef_multi(X, GX, params);
    
    // Check convergence
    keepGoing = false;
    for (size_t k = 0; k < K - 1; ++k) {
      for (size_t i = 0; i < X(k).n_elem; ++i) {
        double diff = std::abs(X(k)(i) - GX(k)(i));
        if (diff > diffMax && diff / (0.1 + std::abs(GX(k)(i))) > diffMax) {
          keepGoing = true;
          break;
        }
      }
      if (keepGoing) break;
    }
  }
  
  // Final computation for result
  compute_cluster_coef_multi(GX, GGX, params);
  
  // Compute final mu
  vec mu_result = mu_init;
  for (size_t k = 0; k < K; ++k) {
    const uvec &fe_idx = fe_indices(k);
    const vec &cluster_coef = GGX(k);
    
    if (family == 1 || family == 5) { // POISSON or LOG_POISSON
      for (size_t i = 0; i < n_obs; ++i) {
        mu_result(i) *= cluster_coef(fe_idx(i));
      }
    } else {
      for (size_t i = 0; i < n_obs; ++i) {
        mu_result(i) += cluster_coef(fe_idx(i));
      }
    }
  }
  
  result.mu_new = mu_result;
  result.success = !any_negative_poisson && (numconv || iter < iterMax);
  result.iterations = iter;
  result.any_negative_poisson = any_negative_poisson;
  
  return result;
}

// Simple sequential convergence (no acceleration)
inline ConvergenceResult convergence_sequential(const vec &mu_init, const vec &lhs,
                                               const field<uvec> &fe_indices,
                                               const field<uvec> &table,
                                               const field<vec> &sum_y,
                                               const field<uvec> &obsCluster,
                                               const field<uvec> &cumtable,
                                               int family, double theta = 1.0,
                                               size_t iterMax = 10000,
                                               double diffMax = 1e-8,
                                               double diffMax_NR = 1e-8) {
  
  ConvergenceResult result;
  size_t K = fe_indices.n_elem;
  size_t n_obs = mu_init.n_elem;
  
  // Initialize cluster coefficients
  field<vec> cluster_coef(K);
  for (size_t k = 0; k < K; ++k) {
    size_t nk = table(k).n_elem;
    cluster_coef(k).set_size(nk);
    
    if (family == 1 || family == 5) { // POISSON or LOG_POISSON
      cluster_coef(k).ones();
    } else {
      cluster_coef(k).zeros();
    }
  }
  
  vec mu_with_coef = mu_init;
  bool keepGoing = true;
  size_t iter = 1;
  
  while (keepGoing && iter <= iterMax) {
    ++iter;
    keepGoing = false;
    
    // Loop over all FE dimensions from K to 1
    for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
      size_t uk = static_cast<size_t>(k);
      const uvec &fe_idx = fe_indices(uk);
      vec &my_cluster_coef = cluster_coef(uk);
      size_t nb_cluster = my_cluster_coef.n_elem;
      
      // Compute cluster coefficients
      compute_cluster_coef_single(family, n_obs, nb_cluster, theta, diffMax_NR,
                                 my_cluster_coef, mu_with_coef, lhs, sum_y(uk),
                                 fe_idx, obsCluster(uk), table(uk), cumtable(uk));
      
      // Update mu_with_coef
      if (family == 1 || family == 5) { // POISSON or LOG_POISSON
        for (size_t i = 0; i < n_obs; ++i) {
          mu_with_coef(i) *= my_cluster_coef(fe_idx(i));
        }
      } else {
        for (size_t i = 0; i < n_obs; ++i) {
          mu_with_coef(i) += my_cluster_coef(fe_idx(i));
        }
      }
      
      // Check stopping criterion
      if (!keepGoing) {
        if (family == 1 || family == 5) { // POISSON or LOG_POISSON
          for (size_t m = 0; m < nb_cluster; ++m) {
            if (std::abs(my_cluster_coef(m) - 1.0) > diffMax) {
              keepGoing = true;
              break;
            }
          }
        } else {
          for (size_t m = 0; m < nb_cluster; ++m) {
            if (std::abs(my_cluster_coef(m)) > diffMax) {
              keepGoing = true;
              break;
            }
          }
        }
      }
    }
  }
  
  result.mu_new = mu_with_coef;
  result.success = iter <= iterMax;
  result.iterations = iter - 1;
  result.any_negative_poisson = false;
  
  return result;
}

// Main convergence interface function
inline ConvergenceResult converge_fixed_effects(const vec &mu_init, const vec &lhs,
                                               const field<uvec> &fe_indices,
                                               const field<uvec> &table,
                                               const field<vec> &sum_y,
                                               const field<uvec> &obsCluster,
                                               const field<uvec> &cumtable,
                                               const std::string &family_str,
                                               bool use_acceleration = true,
                                               double theta = 1.0,
                                               size_t iterMax = 10000,
                                               double diffMax = 1e-8,
                                               double diffMax_NR = 1e-8) {
  
  // Convert family string to integer code
  int family;
  if (family_str == "poisson") {
    family = 1; // POISSON
  } else if (family_str == "negbin") {
    family = 2; // NEGBIN
  } else if (family_str == "logit") {
    family = 3; // LOGIT
  } else if (family_str == "gaussian") {
    family = 4; // GAUSSIAN
  } else if (family_str == "log_poisson") {
    family = 5; // LOG_POISSON
  } else {
    // Default to Poisson
    family = 1; // POISSON
  }
  
  if (use_acceleration && fe_indices.n_elem > 1) {
    return convergence_accelerated(mu_init, lhs, fe_indices, table, sum_y,
                                  obsCluster, cumtable, family, theta,
                                  iterMax, diffMax, diffMax_NR);
  } else {
    return convergence_sequential(mu_init, lhs, fe_indices, table, sum_y,
                                 obsCluster, cumtable, family, theta,
                                 iterMax, diffMax, diffMax_NR);
  }
}

#endif // CAPYBARA_CONVERGENCE
