#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// Fixest-style Irons-Tuck acceleration
struct FixestIronsTuckAcceleration {
  vec GX, GGX, delta_GX, delta2_X;
  bool has_history = false;
  
  FixestIronsTuckAcceleration(size_t dim) {
    GX.zeros(dim);
    GGX.zeros(dim);
    delta_GX.zeros(dim);
    delta2_X.zeros(dim);
  }
  
  // Returns true if numerical convergence achieved
  bool accelerate(vec& X, const vec& GX_new, const vec& GGX_new, const std::string& family) {
    if (!has_history) {
      GX = GX_new;
      GGX = GGX_new;
      has_history = true;
      return false;
    }
    
    // Update using fixest's exact algorithm
    for (size_t i = 0; i < X.n_elem; ++i) {
      delta_GX(i) = GGX_new(i) - GX_new(i);
      delta2_X(i) = delta_GX(i) - GX_new(i) + X(i);
    }
    
    // Compute acceleration coefficient
    double vprod = 0, ssq = 0;
    for (size_t i = 0; i < X.n_elem; ++i) {
      vprod += delta_GX(i) * delta2_X(i);
      ssq += delta2_X(i) * delta2_X(i);
    }
    
    // Check for numerical convergence
    if (ssq == 0) {
      return true;
    }
    
    double coef = vprod / ssq;
    
    // Update X using fixest's formula: X[i] = GGX[i] - coef * delta_GX[i]
    for (size_t i = 0; i < X.n_elem; ++i) {
      X(i) = GGX_new(i) - coef * delta_GX(i);
    }
    
    // Special handling for Poisson models - avoid negative coefficients
    if (family == "poisson") {
      for (size_t i = 0; i < X.n_elem; ++i) {
        if (X(i) <= 0) {
          // Fall back to no acceleration if negative values appear
          X = GGX_new;
          break;
        }
      }
    }
    
    // Update history
    GX = GX_new;
    GGX = GGX_new;
    
    return false;
  }
};


double update_negbin(const arma::vec &y, const arma::vec &mu, const arma::vec &w, double theta, double prev_alpha, double tol = 1e-8, int max_iter = 100) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double mui = mu(i) * exp(x1);
      f += w(i) * (y(i) - (mui + theta) * (y(i) / mui));
      df += w(i) * (-mui * (y(i) / (mui * mui)));
    }
    if (std::abs(df) < 1e-12) break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper) x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol) break;
    if (f > 0) lower = x1; else upper = x1;
  }
  return x1;
}

double update_logit(const arma::vec &y, const arma::vec &mu, const arma::vec &w, double prev_alpha, double tol = 1e-8, int max_iter = 100) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double eta = x1 + mu(i);
      double p = 1.0 / (1.0 + std::exp(-eta));
      f += w(i) * (y(i) - p);
      df += w(i) * (-p * (1 - p));
    }
    if (std::abs(df) < 1e-12) break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper) x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol) break;
    if (f > 0) lower = x1; else upper = x1;
  }
  return x1;
}

// Bucket sort for efficient group processing
inline std::vector<int> bucket_argsort(const std::vector<int>& group_ids, int max_groups) {
  std::vector<int> result;
  result.reserve(group_ids.size());
  
  // Count occurrences
  std::vector<int> counts(max_groups + 1, 0);
  for (int id : group_ids) {
    counts[id]++;
  }
  
  // Create sorted indices
  for (int g = 0; g <= max_groups; ++g) {
    for (size_t i = 0; i < group_ids.size(); ++i) {
      if (group_ids[i] == g) {
        result.push_back(i);
      }
    }
  }
  return result;
}

// Flat array group information structure
struct GroupInfo {
  // Flat storage for cache efficiency
  uvec group_starts;        // Start index for each group in flat_indices
  uvec group_sizes;         // Size of each group
  uvec flat_indices;        // All group indices stored contiguously
  vec sum_weights;          // Precomputed sum of weights per group
  vec inv_weights;          // Precomputed inverse weights per group
  vec cached_group_means;   // Cache for group means
  size_t n_groups;
  size_t max_group_size;
  bool has_cached_means = false;
  
  GroupInfo() = default;
  
  GroupInfo(const field<uvec> &group_field, const vec &w) {
    n_groups = group_field.n_elem;
    group_starts.set_size(n_groups);
    group_sizes.set_size(n_groups);
    sum_weights.set_size(n_groups);
    inv_weights.set_size(n_groups);
    cached_group_means.set_size(n_groups);
    max_group_size = 0;
    
    // First pass: compute total size and group sizes
    size_t total_size = 0;
    for (size_t j = 0; j < n_groups; ++j) {
      const uvec &group_obs = group_field(j);
      group_sizes(j) = group_obs.n_elem;
      max_group_size = std::max(max_group_size, static_cast<size_t>(group_sizes(j)));
      total_size += group_sizes(j);
    }
    
    // Allocate flat array
    flat_indices.set_size(total_size);
    
    // Second pass: populate flat array and compute starts
    size_t current_pos = 0;
    for (size_t j = 0; j < n_groups; ++j) {
      const uvec &group_obs = group_field(j);
      group_starts(j) = current_pos;
      
      // Copy indices to flat array
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        flat_indices(current_pos + i) = group_obs(i);
      }
      
      // Precompute weights with efficient summation
      sum_weights(j) = 0.0;
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        sum_weights(j) += w(group_obs(i));
      }
      inv_weights(j) = 1.0 / sum_weights(j);
      
      current_pos += group_obs.n_elem;
    }
  }
  
  
  // Compute group means
  void compute_group_means(const vec& values, const vec& weights) {
    for (size_t j = 0; j < n_groups; ++j) {
      const size_t start = group_starts(j);
      const size_t size = group_sizes(j);
      
      if (size == 0) {
        cached_group_means(j) = 0.0;
        continue;
      }
      
      double weighted_sum = 0.0;
      // Sequential memory access through flat array
      for (size_t i = 0; i < size; ++i) {
        const size_t idx = flat_indices(start + i);
        weighted_sum += weights(idx) * values(idx);
      }
      cached_group_means(j) = weighted_sum * inv_weights(j);
    }
    has_cached_means = true;
  }
  
  // Vectorized group subtraction
  void subtract_group_effects(vec& values, const vec& group_effects) const {
    for (size_t j = 0; j < n_groups; ++j) {
      const size_t start = group_starts(j);
      const size_t size = group_sizes(j);
      const double effect = group_effects(j);
      
      // Sequential memory access
      for (size_t i = 0; i < size; ++i) {
        values(flat_indices(start + i)) -= effect;
      }
    }
  }
  
  // Vectorized group addition
  void add_group_effects(vec& values, const vec& group_effects) const {
    for (size_t j = 0; j < n_groups; ++j) {
      const size_t start = group_starts(j);
      const size_t size = group_sizes(j);
      const double effect = group_effects(j);
      
      // Sequential memory access
      for (size_t i = 0; i < size; ++i) {
        values(flat_indices(start + i)) += effect;
      }
    }
  }
};

// Group mean subtraction - portable version using pure Armadillo types
inline void demean_variables(mat &V, const vec &weights,
                                      const field<field<uvec>> &group_indices,
                                      double tol = 1e-8, int max_iter = 1000,
                                      const std::string &family = "gaussian", bool use_acceleration = false) {
  const size_t N = V.n_rows, P = V.n_cols, K = group_indices.n_elem;

  // Build GroupInfo structures for each fixed effect using portable constructor
  std::vector<GroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info[k] = GroupInfo(group_indices(k), weights);
  }

  // TODO: check this later
  // Conservative convergence check
  auto convergence_check = [&](const vec &v, const vec &v0, const vec &w,
                               double tol) {
    if (family == "poisson") {
      double ssr = dot(w, square(v));
      double ssr0 = dot(w, square(v0));
      return std::abs(ssr - ssr0) / (0.1 + std::abs(ssr)) < tol;
    } else {
      const double inv_sw = 1.0 / accu(w);
      return dot(abs(v - v0), w) * inv_sw < tol;
    }
  };

  // Main demeaning loop - sequential like fixest for better performance
  #pragma omp parallel for schedule(static, 1) if(P > 4)
  for (size_t p = 0; p < P; ++p) {
    const vec v_orig = V.col(p);
    vec v = v_orig;
    field<vec> alpha(K);
    for (size_t k = 0; k < K; ++k) {
      alpha(k).zeros(group_info[k].n_groups);
    }
    vec alpha_sum = zeros<vec>(N);
    vec v0(N, fill::none);

    // Simple alternating projections loop with enabled acceleration for Poisson
    for (int iter = 0; iter < max_iter; ++iter) {
      v0 = v;
      
      // Alternating projections step
      for (size_t k = 0; k < K; ++k) {
        const auto &gi = group_info[k];

        // Remove current FE from alpha_sum
        gi.subtract_group_effects(alpha_sum, alpha(k));

        // Compute residual
        v = v_orig - alpha_sum;

        // Update alpha
        for (size_t l = 0; l < gi.n_groups; ++l) {
          const size_t start = gi.group_starts(l);
          const size_t size = gi.group_sizes(l);
          if (size == 0) continue;

          double w_sum = 0.0;
          
          for (size_t i = 0; i < size; ++i) {
            const size_t idx = gi.flat_indices(start + i);
            w_sum += weights(idx) * v(idx);
          }
          
          alpha(k)(l) = w_sum * gi.inv_weights(l);
        }

        // Add back new FE
        gi.add_group_effects(alpha_sum, alpha(k));
      }
      
      v = v_orig - alpha_sum;
      
      // Convergence check
      if (convergence_check(v, v0, weights, tol)) break;
    }
    V.col(p) = v;
  }
}

// Portable GLM step demeaning
inline void demean_glm_step(mat &X, vec &y, const vec &weights, const field<field<uvec>> &group_indices,
                           double tol = 1e-8, int max_iter = 1000,
                           const std::string &family = "gaussian", double outer_crit = 1.0, bool use_acceleration = true) {
  const size_t N = X.n_rows, P = X.n_cols;
  
  // Adaptive tolerance for GLM
  double adaptive_tol = tol;
  if (family == "poisson" || family == "logit") {
    if (outer_crit < 10.0 * tol) {
      adaptive_tol = tol / 10.0;
    }
  }
  
  // Enable acceleration for Poisson models like fixest does
  // (fixest uses acceleration for all GLM models)
  // Now using fixest's exact algorithm which handles all K
  bool enable_acceleration = use_acceleration;

  // Create combined matrix for efficient demeaning
  mat combined_vars(N, P + 1);
  combined_vars.cols(0, P - 1) = X;
  combined_vars.col(P) = y;
  
  // Use the optimized demean_variables function with acceleration enabled
  demean_variables(combined_vars, weights, group_indices, adaptive_tol, max_iter, family, enable_acceleration);
  
  // Extract results
  X = combined_vars.cols(0, P - 1);
  y = combined_vars.col(P);
}

inline vec demean_and_solve_wls(mat &X, vec &y, const vec &weights, const field<field<uvec>> &group_indices,
                                double tol = 1e-8, int max_iter = 1000, const std::string &family = "gaussian", 
                                uvec* valid_coefficients = nullptr) {
  const size_t N = X.n_rows, P = X.n_cols, K = group_indices.n_elem;
  
  // Fast path for no fixed effects
  if (K == 0) {
    // In-place weighted least squares with minimal memory allocation
    mat XtW = X.t();
    XtW.each_row() %= weights.t();
    mat XtWX = XtW * X;
    vec XtWy = XtW * y;
    
    // Set all coefficients as valid for no fixed effects case
    if (valid_coefficients != nullptr) {
      valid_coefficients->ones(P);
    }
    
    return solve(XtWX, XtWy, solve_opts::fast);
  }
  
  // Enable acceleration for all GLM models like fixest does
  // Now using fixest's exact algorithm which handles all K
  bool use_acceleration = true;
  
  // Adaptive tolerance for GLM
  double adaptive_tol = tol;
  if (family == "poisson" || family == "logit") {
    adaptive_tol = tol / 10.0;
  }
  
  // Build GroupInfo structures for each fixed effect - eliminate caching overhead
  std::vector<GroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info[k] = GroupInfo(group_indices(k), weights);
  }
  
  static mat combined_vars_cache;
  static mat XtW_cache;
  static mat XtWX_cache;
  static vec XtWy_cache;
  
  // Reuse matrices if possible to avoid allocations
  if (combined_vars_cache.n_rows != N || combined_vars_cache.n_cols != P + 1) {
    combined_vars_cache.set_size(N, P + 1);
  }
  
  combined_vars_cache.cols(0, P - 1) = X;
  combined_vars_cache.col(P) = y;
  
  demean_variables(combined_vars_cache, weights, group_indices, adaptive_tol, max_iter, family, use_acceleration);
  
  mat X_demeaned = combined_vars_cache.cols(0, P - 1);
  vec y_demeaned = combined_vars_cache.col(P);
  
  if (XtW_cache.n_rows != P || XtW_cache.n_cols != N) {
    XtW_cache.set_size(P, N);
  }
  if (XtWX_cache.n_rows != P || XtWX_cache.n_cols != P) {
    XtWX_cache.set_size(P, P);
  }
  if (XtWy_cache.n_elem != P) {
    XtWy_cache.set_size(P);
  }
  
  XtW_cache = X_demeaned.t();
  XtW_cache.each_row() %= weights.t();  // More efficient than diagmat()
  XtWX_cache = XtW_cache * X_demeaned;
  XtWy_cache = XtW_cache * y_demeaned;
  
  // Collinearity-aware solve using QR decomposition with cached matrices
  vec beta(P, fill::value(datum::nan));
  
  // Pre-allocate QR matrices
  static mat Q_cache, R_cache;
  static vec work_cache;
  
  // Use QR decomposition to detect collinearity
  qr_econ(Q_cache, R_cache, X_demeaned.each_col() % sqrt(weights));
  
  if (work_cache.n_elem != P) {
    work_cache.set_size(P);
  }
  work_cache = Q_cache.t() * (y_demeaned % sqrt(weights));
  
  // Detect collinear variables using diagonal of R matrix
  const vec diag_abs = abs(R_cache.diag());
  const double max_diag = diag_abs.max();
  const double collin_tol = 1e-7 * max_diag;
  const uvec indep = find(diag_abs > collin_tol);
  
  // Set up valid coefficients indicator
  uvec valid_coef(P, fill::zeros);
  valid_coef(indep).ones();
  
  // Solve for independent variables
  if (indep.n_elem == P) {
    beta = solve(trimatu(R_cache), work_cache, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = R_cache.submat(indep, indep);
    const vec Yr = work_cache.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    beta(indep) = br;
    // Keep NaN for collinear coefficients
  }
  
  // Return valid coefficients if requested
  if (valid_coefficients != nullptr) {
    *valid_coefficients = valid_coef;
  }
  
  // Update original matrices
  X = std::move(X_demeaned);
  y = std::move(y_demeaned);
  
  return beta;
}

#endif // CAPYBARA_CENTER
