// Computing alpha a in a model with fixed effects Y = alpha + X beta given beta

#ifndef CAPYBARA_ALPHA_H
#define CAPYBARA_ALPHA_H

namespace capybara {

struct InferenceAlpha {
  field<vec> coefficients;
  uvec nb_references;
  bool is_regular;
  bool success;
  field<std::string> fe_names;         // Names for fixed effects categories
  field<field<std::string>> fe_levels; // Names for levels within each category

  InferenceAlpha() : is_regular(true), success(false) {}
};

struct AlphaGroupInfo {
  std::vector<size_t> group_start;  // Start index for each group
  std::vector<size_t> group_size;   // Size of each group
  std::vector<size_t> obs_to_group; // Map from observation to group
  size_t n_groups;
};

// Workspace structure to eliminate repeated allocations
struct AlphaWorkspace {
  vec y;                    // Residual vector
  vec alpha_old;            // Previous iteration values
  vec group_sums;           // Accumulator for group sums
  
  // Cache for reducing allocations
  size_t cached_N, cached_max_groups;
  bool is_initialized;

  // Default constructor
  AlphaWorkspace() : cached_N(0), cached_max_groups(0), is_initialized(false) {}

  AlphaWorkspace(size_t N, size_t max_groups) : cached_N(N), cached_max_groups(max_groups), is_initialized(true) {
    size_t safe_N = std::max(N, size_t(1));
    size_t safe_groups = std::max(max_groups, size_t(1));

    y.set_size(safe_N);
    alpha_old.set_size(safe_groups);
    group_sums.set_size(safe_groups);
  }
  
  // Efficient resize that avoids reallocation when possible
  void ensure_size(size_t N, size_t max_groups) {
    if (!is_initialized || N > cached_N || max_groups > cached_max_groups) {
      size_t new_N = std::max(N, cached_N);
      size_t new_groups = std::max(max_groups, cached_max_groups);
      
      if (y.n_elem < new_N) y.set_size(new_N);
      if (alpha_old.n_elem < new_groups) alpha_old.set_size(new_groups);
      if (group_sums.n_elem < new_groups) group_sums.set_size(new_groups);
      
      cached_N = new_N;
      cached_max_groups = new_groups;
      is_initialized = true;
    }
  }
  
  // Destructor to ensure proper cleanup
  ~AlphaWorkspace() {
    clear();
  }
  
  // Method to clear and release memory
  void clear() {
    y.reset();
    alpha_old.reset();
    group_sums.reset();
    cached_N = 0;
    cached_max_groups = 0;
    is_initialized = false;
  }
};

// Optimized get_alpha function with workspace and vectorization
inline field<vec> get_alpha(const vec &pi,
                            const field<field<uvec>> &group_indices,
                            double tol = 1e-8, size_t iter_max = 10000,
                            AlphaWorkspace *workspace = nullptr) {
  const size_t K = group_indices.n_elem;
  const size_t N = pi.n_elem;
  field<vec> coefficients(K);
  
  if (K == 0 || N == 0) {
    return coefficients;
  }
  
  // Pre-compute group information for faster access
  std::vector<AlphaGroupInfo> group_info(K);
  size_t max_groups = 0;
  
  for (size_t k = 0; k < K; ++k) {
    const size_t J = group_indices(k).n_elem;
    coefficients(k).set_size(J);
    coefficients(k).zeros();
    max_groups = std::max(max_groups, J);
    
    // Build mapping for fast access
    group_info[k].n_groups = J;
    group_info[k].group_size.resize(J);
    group_info[k].obs_to_group.resize(N, J); // J means "not in any group"
    
    for (size_t j = 0; j < J; ++j) {
      const uvec &indexes = group_indices(k)(j);
      group_info[k].group_size[j] = indexes.n_elem;
      
      for (size_t t = 0; t < indexes.n_elem; ++t) {
        group_info[k].obs_to_group[indexes(t)] = j;
      }
    }
  }
  
  // Create local workspace if none provided
  AlphaWorkspace local_workspace;
  if (!workspace) {
    workspace = &local_workspace;
  }
  
  // Ensure workspace has sufficient capacity
  workspace->ensure_size(N, max_groups);
  
  // Use workspace references to avoid pointer overhead
  vec &y = workspace->y;
  vec &alpha_old = workspace->alpha_old;
  vec &group_sums = workspace->group_sums;
  
  // Ensure workspace vectors are properly sized for this computation
  y.set_size(N);
  
  double crit = 1.0;
  size_t iter = 0;
  
  // Pre-allocate pointers for hot loop optimization
  const double *pi_ptr = pi.memptr();
  
  while (crit > tol && iter < iter_max) {
    double sum_sq_old = 0.0, sum_sq_diff = 0.0;
    
    // Update each fixed effect category
    for (size_t k = 0; k < K; ++k) {
      const AlphaGroupInfo &info = group_info[k];
      const size_t J = info.n_groups;
      
      // Ensure workspace vectors are sized correctly for this FE
      alpha_old.set_size(J);
      group_sums.set_size(J);
      
      // Store old values and compute sum of squares (avoid copy)
      double *alpha_old_ptr = alpha_old.memptr();
      const double *coef_k_ptr = coefficients(k).memptr();
      
      for (size_t j = 0; j < J; ++j) {
        alpha_old_ptr[j] = coef_k_ptr[j];
      }
      sum_sq_old += dot(alpha_old, alpha_old);
      
      // Compute residuals: y = pi - sum of other fixed effects
      // Use direct pointer access for optimal performance
      double *y_ptr = y.memptr();
      
      // Initialize y with pi (vectorized copy)
      std::memcpy(y_ptr, pi_ptr, N * sizeof(double));
      
      // Subtract contributions from other fixed effects
      for (size_t kk = 0; kk < K; ++kk) {
        if (kk != k) {
          const AlphaGroupInfo &info_kk = group_info[kk];
          const double *coef_kk_ptr = coefficients(kk).memptr();
          const std::vector<size_t> &obs_to_group_kk = info_kk.obs_to_group;
          
          // Vectorized subtraction with optimized memory access
          for (size_t i = 0; i < N; ++i) {
            size_t group = obs_to_group_kk[i];
            if (group < info_kk.n_groups) {
              y_ptr[i] -= coef_kk_ptr[group];
            }
          }
        }
      }
      
      // Compute group sums efficiently
      group_sums.zeros();
      double *group_sums_ptr = group_sums.memptr();
      const std::vector<size_t> &obs_to_group = info.obs_to_group;
      
      // Optimized accumulation with direct pointer access
      for (size_t i = 0; i < N; ++i) {
        size_t group = obs_to_group[i];
        if (group < J) {
          group_sums_ptr[group] += y_ptr[i];
        }
      }
      
      // Compute group means in-place (avoid intermediate allocations)
      double *coef_k_ptr_new = coefficients(k).memptr();
      const std::vector<size_t> &group_size = info.group_size;
      
      for (size_t j = 0; j < J; ++j) {
        if (group_size[j] > 0) {
          coef_k_ptr_new[j] = group_sums_ptr[j] / static_cast<double>(group_size[j]);
        } else {
          coef_k_ptr_new[j] = 0.0;
        }
      }
      
      // Compute squared differences efficiently (avoid temporary vector)
      double local_sum_sq_diff = 0.0;
      for (size_t j = 0; j < J; ++j) {
        double diff = coef_k_ptr_new[j] - alpha_old_ptr[j];
        local_sum_sq_diff += diff * diff;
      }
      sum_sq_diff += local_sum_sq_diff;
    }
    
    // Compute convergence criterion
    if (sum_sq_old > 0.0) {
      crit = std::sqrt(sum_sq_diff / sum_sq_old);
    } else if (sum_sq_diff > 0.0) {
      crit = std::sqrt(sum_sq_diff);
    } else {
      crit = 0.0;
    }
    
    ++iter;
  }
  
  return coefficients;
}

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
