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

inline field<vec> get_alpha(const vec &pi,
                            const field<field<uvec>> &group_indices,
                            double tol = 1e-8, size_t iter_max = 10000) {
  const size_t K = group_indices.n_elem;
  const size_t N = pi.n_elem;
  field<vec> coefficients(K);
  
  if (K == 0 || N == 0) {
    return coefficients;
  }
  
  // Pre-compute group information for faster access
  std::vector<AlphaGroupInfo> group_info(K);
  
  for (size_t k = 0; k < K; ++k) {
    const size_t J = group_indices(k).n_elem;
    coefficients(k).set_size(J);
    coefficients(k).zeros();
    
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
  
  // Pre-allocate working vectors
  vec y(N);
  
  double crit = 1.0;
  size_t iter = 0;
  
  while (crit > tol && iter < iter_max) {
    double sum_sq_old = 0.0, sum_sq_diff = 0.0;
    
    // Update each fixed effect category
    for (size_t k = 0; k < K; ++k) {
      const AlphaGroupInfo &info = group_info[k];
      
      // Store old values and compute sum of squares
      vec alpha_old = coefficients(k);
      sum_sq_old += dot(alpha_old, alpha_old);
      
      // Compute residuals: y = pi - sum of other fixed effects
      y = pi;
      
      for (size_t kk = 0; kk < K; ++kk) {
        if (kk != k) {
          const AlphaGroupInfo &info_kk = group_info[kk];
          
          for (size_t i = 0; i < N; ++i) {
            size_t group = info_kk.obs_to_group[i];
            if (group < info_kk.n_groups) {
              y(i) -= coefficients(kk)(group);
            }
          }
        }
      }
      
      // Compute group means
      coefficients(k).zeros();
      
      for (size_t i = 0; i < N; ++i) {
        size_t group = info.obs_to_group[i];
        if (group < info.n_groups) {
          coefficients(k)(group) += y(i);
        }
      }
      
      for (size_t j = 0; j < info.n_groups; ++j) {
        if (info.group_size[j] > 0) {
          coefficients(k)(j) /= static_cast<double>(info.group_size[j]);
        }
      }
      
      // Accumulate squared differences
      vec diff = coefficients(k) - alpha_old;
      sum_sq_diff += dot(diff, diff);
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
