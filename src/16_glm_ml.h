#ifndef CAPYBARA_GLM_ML_H
#define CAPYBARA_GLM_ML_H

// Maximum likelihood GLM implementation inspired by FENmlm
// Family-specific optimizations with pure Armadillo operations

// Forward declarations
struct glm_ml_workspace;

// Family-specific ML implementations
inline feglm_results feglm_ml_poisson(mat &MX, vec &beta, vec &eta, 
                                      const vec &y, const vec &wt,
                                      double center_tol, double dev_tol,
                                      size_t iter_max, size_t iter_center_max,
                                      const indices_info &indices,
                                      glm_ml_workspace &ws);

inline feglm_results feglm_ml_gaussian(mat &MX, vec &beta, vec &eta,
                                       const vec &y, const vec &wt,
                                       double center_tol, double dev_tol,
                                       size_t iter_max, size_t iter_center_max,
                                       const indices_info &indices,
                                       glm_ml_workspace &ws);

inline feglm_results feglm_ml_general(mat &MX, vec &beta, vec &eta,
                                      const vec &y, const vec &wt, double theta,
                                      family_type family, double center_tol, double dev_tol,
                                      size_t iter_max, size_t iter_center_max,
                                      const indices_info &indices,
                                      glm_ml_workspace &ws);

// Optimized ML workspace with minimal memory allocations
struct glm_ml_workspace {
  // Core matrices - pre-allocated to avoid reallocation
  mat MX_work;
  mat MX_orig;
  
  // ML iteration vectors
  vec mu, eta_work, beta_work;
  vec w, nu, nu_prev;
  vec eta_upd, beta_upd;
  
  // Centering workspace
  ml_center_workspace center_ws;
  
  // Temporary computation vectors
  vec temp_vec1, temp_vec2;
  mat temp_mat;
  
  // Convergence tracking
  std::vector<double> dev_history;
  
  // Beta solving workspace - initialized in constructor
  std::unique_ptr<beta_results> beta_ws;
  std::unique_ptr<crossproduct_results> cross_ws;
  
  glm_ml_workspace() = default;
  
  void reserve(size_t N, size_t P) {
    MX_work.set_size(N, P);
    MX_orig.set_size(N, P);
    
    mu.set_size(N);
    eta_work.set_size(N);
    beta_work.set_size(P);
    w.set_size(N);
    nu.set_size(N);
    nu_prev.set_size(N);
    eta_upd.set_size(N);
    beta_upd.set_size(P);
    
    temp_vec1.set_size(N);
    temp_vec2.set_size(N);
    temp_mat.set_size(N, P);
    
    dev_history.reserve(10);
    
    // Initialize workspaces
    beta_ws = std::make_unique<beta_results>(N, P);
    cross_ws = std::make_unique<crossproduct_results>(N, P);
  }
  
  void clear() {
    MX_work.reset();
    MX_orig.reset();
    mu.reset();
    eta_work.reset();
    beta_work.reset();
    w.reset();
    nu.reset();
    nu_prev.reset();
    eta_upd.reset();
    beta_upd.reset();
    center_ws.clear();
    temp_vec1.reset();
    temp_vec2.reset();
    temp_mat.reset();
    dev_history.clear();
    beta_ws.reset();
    cross_ws.reset();
  }
};

// Fast Poisson ML implementation following FENmlm approach
inline feglm_results feglm_ml_poisson(mat &MX, vec &beta, vec &eta, 
                                      const vec &y, const vec &wt,
                                      double center_tol, double dev_tol,
                                      size_t iter_max, size_t iter_center_max,
                                      const indices_info &indices,
                                      glm_ml_workspace &ws) {
  const size_t N = y.n_elem;
  const size_t P = MX.n_cols;
  const bool has_fe = (indices.fe_sizes.n_elem > 0);
  
  // Reserve workspace
  ws.reserve(N, P);
  
  // Store original matrix
  ws.MX_orig = MX;
  
  // Initialize eta if needed
  if (eta.is_empty() || all(eta == 0.0)) {
    eta.set_size(N);
    eta = log(y + 0.1); // Better initialization for Poisson
  }
  
  // Initial mu and deviance
  ws.mu = exp(eta);
  const vec ymean(N, fill::value(mean(y)));
  double dev = dev_resids(y, ws.mu, 0.0, wt, POISSON);
  const double null_dev = dev_resids(y, ymean, 0.0, wt, POISSON);
  
  bool conv = false;
  size_t actual_iters = 0;
  
  // Main ML iteration loop
  for (size_t it = 0; it < iter_max; ++it) {
    actual_iters = it + 1;
    const double dev_old = dev;
    const vec eta_old = eta;
    const vec beta_old = beta;
    
    // Poisson-specific weight computation (optimized)
    ws.w = wt % ws.mu; // For Poisson: w = wt * mu
    
    // Working response (optimized for Poisson)
    ws.nu = y - ws.mu;
    ws.nu /= ws.mu; // nu = (y - mu) / mu for Poisson
    
    // Center variables using ML approach
    if (has_fe) {
      ws.MX_work = ws.MX_orig;
      
      // Use unified centering for cache efficiency
      center_variables_ml(ws.MX_work, ws.nu, ws.w, indices, POISSON,
                          center_tol, iter_center_max, ws.center_ws);
    }
    
    // Solve for beta update
    solve_beta(ws.MX_work, ws.nu, ws.w, N, P, *ws.beta_ws, true);
    ws.beta_upd = ws.beta_ws->coefficients;
    
    // Compute eta update
    const uvec valid = find(ws.beta_ws->valid_coefficients);
    if (valid.n_elem < P) {
      ws.eta_upd = ws.MX_work.cols(valid) * ws.beta_upd.elem(valid);
    } else {
      ws.eta_upd = ws.MX_work * ws.beta_upd;
    }
    
    // Line search with adaptive damping
    double damping = 1.0;
    bool step_ok = false;
    
    for (size_t inner = 0; inner < 50; ++inner) {
      ws.eta_work = eta_old + damping * ws.eta_upd;
      ws.mu = exp(ws.eta_work);
      
      // Check for valid mu values
      if (!ws.mu.is_finite() || any(ws.mu <= 0)) {
        damping *= 0.5;
        continue;
      }
      
      const double dev_new = dev_resids(y, ws.mu, 0.0, wt, POISSON);
      if (!std::isfinite(dev_new)) {
        damping *= 0.5;
        continue;
      }
      
      // Accept step if deviance decreases
      const double dev_ratio = (dev_new - dev_old) / (0.1 + std::fabs(dev_old));
      if (dev_ratio <= -dev_tol) {
        eta = ws.eta_work;
        beta = beta_old + damping * ws.beta_upd;
        dev = dev_new;
        step_ok = true;
        break;
      }
      
      damping *= 0.5;
      if (damping < 1e-4) break;
    }
    
    if (!step_ok) {
      // Step failed, revert to previous values
      eta = eta_old;
      beta = beta_old;
      ws.mu = exp(eta);
      dev = dev_old;
    } else {
      // Track convergence history
      ws.dev_history.push_back(dev);
      if (ws.dev_history.size() > 5) {
        ws.dev_history.erase(ws.dev_history.begin());
      }
    }
    
    // Check convergence
    const double rel_change = std::fabs(dev - dev_old) / (0.1 + std::fabs(dev));
    if (rel_change < dev_tol) {
      conv = true;
      break;
    }
    
    if ((it % 100) == 0 && it > 0) {
      check_user_interrupt();
    }
  }
  
  if (!conv) {
    stop("Poisson ML GLM failed to converge");
  }
  
  // Compute Hessian
  mat H = crossproduct(ws.MX_work, ws.w, *ws.cross_ws, true);
  
  // Handle invalid coefficients
  for (size_t j = 0; j < P; ++j) {
    if (!ws.beta_ws->valid_coefficients(j)) {
      beta(j) = datum::nan;
    }
  }
  
  return feglm_results(std::move(beta),
                       std::move(ws.beta_ws->valid_coefficients),
                       std::move(eta), wt, std::move(H),
                       dev, null_dev, conv, actual_iters);
}

// Fast Gaussian ML implementation
inline feglm_results feglm_ml_gaussian(mat &MX, vec &beta, vec &eta,
                                       const vec &y, const vec &wt,
                                       double center_tol, double dev_tol,
                                       size_t iter_max, size_t iter_center_max,
                                       const indices_info &indices,
                                       glm_ml_workspace &ws) {
  const size_t N = y.n_elem;
  const size_t P = MX.n_cols;
  const bool has_fe = (indices.fe_sizes.n_elem > 0);
  const bool use_weights = !all(wt == 1.0);
  
  ws.reserve(N, P);
  
  // For Gaussian, this simplifies to linear regression
  if (has_fe) {
    // Center variables
    ws.MX_work = MX;
    vec y_centered = y;
    center_variables_ml(ws.MX_work, y_centered, wt, indices, GAUSSIAN,
                        center_tol, iter_center_max, ws.center_ws);
    
    // Solve normal equations
    solve_beta(ws.MX_work, y_centered, wt, N, P, *ws.beta_ws, use_weights);
    beta = ws.beta_ws->coefficients;
    
    // Compute fitted values
    if (use_weights) {
      eta = y - y_centered + ws.MX_work * beta;
    } else {
      eta = MX * beta;
    }
  } else {
    // No fixed effects - direct solution
    solve_beta(MX, y, wt, N, P, *ws.beta_ws, use_weights);
    beta = ws.beta_ws->coefficients;
    eta = MX * beta;
  }
  
  // Compute deviance
  ws.mu = eta; // For Gaussian, mu = eta
  const vec ymean(N, fill::value(mean(y)));
  const double dev = dev_resids(y, ws.mu, 0.0, wt, GAUSSIAN);
  const double null_dev = dev_resids(y, ymean, 0.0, wt, GAUSSIAN);
  
  // Compute Hessian
  mat H = crossproduct(ws.MX_work, wt, *ws.cross_ws, use_weights);
  
  return feglm_results(std::move(beta),
                       std::move(ws.beta_ws->valid_coefficients),
                       std::move(eta), wt, std::move(H),
                       dev, null_dev, true, 1);
}

// General ML implementation for other families
inline feglm_results feglm_ml_general(mat &MX, vec &beta, vec &eta,
                                      const vec &y, const vec &wt, double theta,
                                      family_type family, double center_tol, double dev_tol,
                                      size_t iter_max, size_t iter_center_max,
                                      const indices_info &indices,
                                      glm_ml_workspace &ws) {
  const size_t N = y.n_elem;
  const size_t P = MX.n_cols;
  const bool has_fe = (indices.fe_sizes.n_elem > 0);
  
  ws.reserve(N, P);
  ws.MX_orig = MX;
  
  // Initialize
  if (eta.is_empty() || all(eta == 0.0)) {
    smart_initialize_glm(eta, y, family);
  }
  link_inv(ws.mu, eta, family);
  
  const vec ymean(N, fill::value(mean(y)));
  double dev = dev_resids(y, ws.mu, theta, wt, family);
  const double null_dev = dev_resids(y, ymean, theta, wt, family);
  
  bool conv = false;
  size_t actual_iters = 0;
  
  for (size_t it = 0; it < iter_max; ++it) {
    actual_iters = it + 1;
    const double dev_old = dev;
    const vec eta_old = eta;
    const vec beta_old = beta;
    
    // Compute weights and working response
    get_mu(ws.temp_vec1, eta, family); // mu_eta
    variance(ws.temp_vec2, ws.mu, theta, family);
    
    ws.w = wt % square(ws.temp_vec1) / ws.temp_vec2;
    ws.nu = (y - ws.mu) / ws.temp_vec1;
    
    // Center variables
    if (has_fe) {
      ws.MX_work = ws.MX_orig;
      center_variables_ml(ws.MX_work, ws.nu, ws.w, indices, family,
                          center_tol, iter_center_max, ws.center_ws);
    }
    
    // Solve and update
    solve_beta(ws.MX_work, ws.nu, ws.w, N, P, *ws.beta_ws, true);
    ws.beta_upd = ws.beta_ws->coefficients;
    
    const uvec valid = find(ws.beta_ws->valid_coefficients);
    if (valid.n_elem < P) {
      ws.eta_upd = ws.MX_work.cols(valid) * ws.beta_upd.elem(valid);
    } else {
      ws.eta_upd = ws.MX_work * ws.beta_upd;
    }
    
    // Line search
    double damping = 1.0;
    bool step_ok = false;
    
    for (size_t inner = 0; inner < 50; ++inner) {
      ws.eta_work = eta_old + damping * ws.eta_upd;
      link_inv(ws.mu, ws.eta_work, family);
      
      if (!valid_eta_mu(ws.eta_work, ws.mu, family)) {
        damping *= 0.5;
        continue;
      }
      
      const double dev_new = dev_resids(y, ws.mu, theta, wt, family);
      if (!std::isfinite(dev_new)) {
        damping *= 0.5;
        continue;
      }
      
      const double dev_ratio = (dev_new - dev_old) / (0.1 + std::fabs(dev_old));
      if (dev_ratio <= -dev_tol) {
        eta = ws.eta_work;
        beta = beta_old + damping * ws.beta_upd;
        dev = dev_new;
        step_ok = true;
        break;
      }
      
      damping *= 0.5;
      if (damping < 1e-4) break;
    }
    
    if (!step_ok) {
      eta = eta_old;
      beta = beta_old;
      link_inv(ws.mu, eta, family);
      dev = dev_old;
    }
    
    // Check convergence
    const double rel_change = std::fabs(dev - dev_old) / (0.1 + std::fabs(dev));
    if (rel_change < dev_tol) {
      conv = true;
      break;
    }
    
    if ((it % 100) == 0 && it > 0) {
      check_user_interrupt();
    }
  }
  
  if (!conv) {
    stop("ML GLM failed to converge");
  }
  
  mat H = crossproduct(ws.MX_work, ws.w, *ws.cross_ws, true);
  
  for (size_t j = 0; j < P; ++j) {
    if (!ws.beta_ws->valid_coefficients(j)) {
      beta(j) = datum::nan;
    }
  }
  
  return feglm_results(std::move(beta),
                       std::move(ws.beta_ws->valid_coefficients),
                       std::move(eta), wt, std::move(H),
                       dev, null_dev, conv, actual_iters);
}

// Main entry point for ML GLM fitting
inline feglm_results feglm_ml(mat &MX, vec &beta, vec &eta, const vec &y, const vec &wt,
                              double theta, family_type family, double center_tol,
                              double dev_tol, size_t iter_max, size_t iter_center_max,
                              const indices_info &indices, glm_ml_workspace &ws) {
  
  // Dispatch to family-specific optimized implementations
  switch (family) {
    case POISSON:
      return feglm_ml_poisson(MX, beta, eta, y, wt, center_tol, dev_tol,
                              iter_max, iter_center_max, indices, ws);
    
    case GAUSSIAN:
      return feglm_ml_gaussian(MX, beta, eta, y, wt, center_tol, dev_tol,
                               iter_max, iter_center_max, indices, ws);
    
    default:
      return feglm_ml_general(MX, beta, eta, y, wt, theta, family, center_tol, dev_tol,
                              iter_max, iter_center_max, indices, ws);
  }
}

#endif // CAPYBARA_GLM_ML_H
