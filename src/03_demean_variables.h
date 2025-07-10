#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

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

// Optimized group information structure with flat memory layout
struct GroupInfo {
  // Flat storage for cache efficiency
  std::vector<std::vector<size_t>> indices;  // Use std::vector for better cache locality
  vec sum_weights;
  vec inv_weights;
  size_t n_groups;
  size_t max_group_size;
  
  GroupInfo() = default;
  GroupInfo(const list &jlist, const vec &w) {
    n_groups = jlist.size();
    indices.resize(n_groups);
    sum_weights.set_size(n_groups);
    inv_weights.set_size(n_groups);
    max_group_size = 0;
    
    for (size_t j = 0; j < n_groups; ++j) {
      uvec temp = as_uvec(as_cpp<integers>(jlist[j]));
      indices[j].resize(temp.n_elem);
      for (size_t i = 0; i < temp.n_elem; ++i) {
        indices[j][i] = temp(i);
      }
      max_group_size = std::max(max_group_size, indices[j].size());
      
      // Precompute weights
      sum_weights(j) = 0.0;
      for (size_t idx : indices[j]) {
        sum_weights(j) += w(idx);
      }
      inv_weights(j) = 1.0 / sum_weights(j);
    }
  }
};

// Group mean subtraction
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr,
                       const std::string &family = "gaussian") {
  const size_t N = V.n_rows, P = V.n_cols, K = klist.size();
  const double inv_sw = 1.0 / accu(w);

  // Precompute group information
  field<GroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info(k) = GroupInfo(klist[k], w);
  }

  auto convergence_check = [&](const vec &x, const vec &x0, const vec &w, double tol) {
    if (family == "poisson") {
      double ssr = dot(w, square(x));
      double ssr0 = dot(w, square(x0));
      return std::abs(ssr - ssr0) / (0.1 + std::abs(ssr)) < tol;
    } else if (family == "gaussian") {
      return dot(abs(x - x0), w) * inv_sw < tol;
    } else {
      return dot(abs(x - x0), w) * inv_sw < tol;
    }
  };

  for (size_t p = 0; p < P; ++p) {
    const vec x_orig = V.col(p);
    vec x = x_orig;
    field<vec> alpha(K);
    for (size_t k = 0; k < K; ++k) {
      alpha(k).zeros(group_info(k).n_groups);
    }
    vec alpha_sum = zeros<vec>(N);
    vec x0(N, fill::none), x1(N, fill::none), x2(N, fill::none);
    if (K == 2) {
      // K=2 specialization
      for (int iter = 0; iter < max_iter; ++iter) {
        x0 = x;
        for (size_t k = 0; k < 2; ++k) {
          const GroupInfo &gi = group_info(k);
          // Remove current FE from alpha_sum - optimized single pass
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;
            double old_alpha = alpha(k)(l);
            for (size_t idx : coords) {
              alpha_sum(idx) -= old_alpha;
            }
          }
          
          // Compute residual
          x = x_orig - alpha_sum;
          
          // Update alpha and add back - optimized computation
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;
            
            double new_alpha;
            if (family == "poisson" || family == "gaussian") {
              // Optimized weighted sum computation
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            } else if (family == "negbin") {
              // Extract subvectors for negbin update
              vec x_sub(coords.size()), w_sub(coords.size());
              for (size_t i = 0; i < coords.size(); ++i) {
                x_sub(i) = x(coords[i]);
                w_sub(i) = w(coords[i]);
              }
              new_alpha = update_negbin(x_sub, zeros<vec>(coords.size()), w_sub, /*theta*/1.0, alpha(k)(l));
            } else if (family == "logit" || family == "binomial") {
              // For logit, use the same weighted average approach as Gaussian
              // The specialized logit solver is for a different context
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            } else {
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            }
            
            alpha(k)(l) = new_alpha;
            for (size_t idx : coords) {
              alpha_sum(idx) += new_alpha;
            }
          }
        }
        x = x_orig - alpha_sum;
        if (convergence_check(x, x0, w, tol)) break;
      }
    } else {
      // K > 2 => use Irons-Tuck acceleration
      const int warmup = 15;
      const int grand_acc = 40;
      int acc_count = 0;
      int iter = 0;
      
      // Pre-compute sum of other FE contributions
      vec sum_other_coef(N, fill::zeros);
      
      // Warmup iterations
      for (; iter < std::min(max_iter, warmup); ++iter) {
        x0 = x;
        for (size_t k = 0; k < K; ++k) {
          const GroupInfo &gi = group_info(k);
          // Remove current FE from alpha_sum - optimized
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;
            double old_alpha = alpha(k)(l);
            for (size_t idx : coords) {
              alpha_sum(idx) -= old_alpha;
            }
          }
          
          x = x_orig - alpha_sum;  // Residual
          
          // Update alpha and add back - optimized computation
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;
            
            double new_alpha;
            if (family == "poisson" || family == "gaussian") {
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            } else if (family == "negbin") {
              vec x_sub(coords.size()), w_sub(coords.size());
              for (size_t i = 0; i < coords.size(); ++i) {
                x_sub(i) = x(coords[i]);
                w_sub(i) = w(coords[i]);
              }
              new_alpha = update_negbin(x_sub, zeros<vec>(coords.size()), w_sub, /*theta*/1.0, alpha(k)(l));
            } else if (family == "logit" || family == "binomial") {
              // For logit, use the same weighted average approach as Gaussian
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            } else {
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            }
            
            alpha(k)(l) = new_alpha;
            for (size_t idx : coords) {
              alpha_sum(idx) += new_alpha;
            }
          }
        }
        x = x_orig - alpha_sum;
        if (convergence_check(x, x0, w, tol)) break;
      }
      
      // Main iterations with acceleration
      x1 = x0; x2 = x0;
      for (; iter < max_iter; ++iter) {
        x2 = x1; x1 = x0; x0 = x;
        for (size_t k = 0; k < K; ++k) {
          const GroupInfo &gi = group_info(k);
          // Remove current FE from alpha_sum - optimized
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;
            double old_alpha = alpha(k)(l);
            for (size_t idx : coords) {
              alpha_sum(idx) -= old_alpha;
            }
          }
          
          x = x_orig - alpha_sum;  // Residual
          
          // Update alpha and add back - optimized computation
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const auto &coords = gi.indices[l];
            if (coords.empty()) continue;
            
            double new_alpha;
            if (family == "poisson" || family == "gaussian") {
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            } else if (family == "negbin") {
              vec x_sub(coords.size()), w_sub(coords.size());
              for (size_t i = 0; i < coords.size(); ++i) {
                x_sub(i) = x(coords[i]);
                w_sub(i) = w(coords[i]);
              }
              new_alpha = update_negbin(x_sub, zeros<vec>(coords.size()), w_sub, /*theta*/1.0, alpha(k)(l));
            } else if (family == "logit" || family == "binomial") {
              // For logit, use the same weighted average approach as Gaussian
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            } else {
              double sum_wx = 0.0;
              for (size_t idx : coords) {
                sum_wx += w(idx) * x(idx);
              }
              new_alpha = sum_wx * gi.inv_weights(l);
            }
            
            alpha(k)(l) = new_alpha;
            for (size_t idx : coords) {
              alpha_sum(idx) += new_alpha;
            }
          }
        }
        
        // Enhanced Irons-Tuck acceleration
        if (++acc_count == grand_acc) {
          acc_count = 0;
          vec delta1 = x0 - x1;
          vec delta2 = x1 - x2;
          vec delta_diff = delta1 - delta2;
          double vprod = dot(delta1, delta_diff);
          double ssq = dot(delta_diff, delta_diff);
          if (ssq > 1e-16) {
            double coef = vprod / ssq;
            // Apply acceleration with bounds checking
            vec x_new = x0 - coef * delta1;
            // Check for reasonable acceleration
            if (std::abs(coef) < 10.0) {  // Prevent excessive acceleration
              x = x_new;
            }
          }
        }
        
        x = x_orig - alpha_sum;
        if (convergence_check(x, x0, w, tol)) break;
      }
    }
    V.col(p) = x;
  }
}

// Wrapper functions to maintain compatibility
inline void demean_variables(mat &V, const vec &weights, const list &k_list,
                           double tol = 1e-8, int max_iter = 1000,
                           const std::string &family = "gaussian") {
  center_variables_(V, weights, k_list, tol, max_iter, 0, 0, family);
}

inline void demean_glm_step(mat &X, vec &y, const vec &weights, const list &k_list,
                           double tol = 1e-8, int max_iter = 1000,
                           const std::string &family = "gaussian") {
  // Combine X and y into a single matrix for joint demeaning
  mat V = join_rows(X, y);
  
  // Use the main demeaning function
  center_variables_(V, weights, k_list, tol, max_iter, 0, 0, family);
  
  // Extract demeaned X and y
  X = V.cols(0, X.n_cols - 1);
  y = V.col(V.n_cols - 1);
}

#endif // CAPYBARA_CENTER
