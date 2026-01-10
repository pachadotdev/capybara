// Computing alpha a in a model with fixed effects Y = alpha + X beta given beta

#ifndef CAPYBARA_ALPHA_H
#define CAPYBARA_ALPHA_H

namespace capybara {

struct InferenceAlpha {
  field<vec> coefficients;
  uvec nb_references;
  bool is_regular;
  bool success;
  field<std::string> fe_names;
  field<field<std::string>> fe_levels;

  InferenceAlpha() : is_regular(true), success(false) {}
};

struct AlphaGroupInfo {
  uvec group_start;
  uvec group_size;
  uvec obs_to_group;
  uword n_groups;
};

// Precompute group sizes and observation-to-group maps (cached between calls)
inline field<AlphaGroupInfo>
precompute_alpha_group_info(const field<field<uvec>> &group_indices, uword N) {
  const uword K = group_indices.n_elem;
  field<AlphaGroupInfo> group_info(K);

  for (uword k = 0; k < K; ++k) {
    const uword J = group_indices(k).n_elem;
    AlphaGroupInfo &info = group_info(k);
    info.n_groups = J;
    info.group_size.set_size(J);
    info.obs_to_group.set_size(N);
    info.obs_to_group.fill(J); // Invalid index sentinel

    for (uword j = 0; j < J; ++j) {
      const uvec &indexes = group_indices(k)(j);
      info.group_size(j) = indexes.n_elem;

      for (uword t = 0; t < indexes.n_elem; ++t) {
        info.obs_to_group(indexes(t)) = j;
      }
    }
  }

  return group_info;
}

// Workspace to reuse allocations across iterations and calls
struct AlphaWorkspace {
  vec residual; // pi - sum(alpha_k)
  vec alpha0;   // Previous coefficients for one FE
  uword cached_N;
  uword cached_max_groups;
  bool is_initialized;

  AlphaWorkspace() : cached_N(0), cached_max_groups(0), is_initialized(false) {}

  void ensure_size(uword N, uword max_groups) {
    if (!is_initialized || N > cached_N || max_groups > cached_max_groups) {
      residual.set_size(N);
      alpha0.set_size(max_groups);
      cached_N = N;
      cached_max_groups = max_groups;
      is_initialized = true;
    }
  }

  void clear() {
    residual.reset();
    alpha0.reset();
    cached_N = 0;
    cached_max_groups = 0;
    is_initialized = false;
  }
};

inline field<vec>
get_alpha(const vec &pi, const field<field<uvec>> &group_indices,
          double tol = 1e-8, uword iter_max = 10000,
          field<AlphaGroupInfo> *precomputed_group_info = nullptr,
          AlphaWorkspace *workspace = nullptr) {
  const uword K = group_indices.n_elem;
  const uword N = pi.n_elem;
  field<vec> coefficients(K);

  if (K == 0 || N == 0) {
    return coefficients;
  }

  field<AlphaGroupInfo> local_group_info;
  const field<AlphaGroupInfo> *group_info_ptr = precomputed_group_info;

  if (!group_info_ptr) {
    local_group_info = precompute_alpha_group_info(group_indices, N);
    group_info_ptr = &local_group_info;
  }

  uword max_groups = 0;
  for (uword k = 0; k < K; ++k) {
    max_groups = std::max(max_groups, (*group_info_ptr)(k).n_groups);
    const uword J = (*group_info_ptr)(k).n_groups;
    coefficients(k).set_size(J);
    coefficients(k).zeros();
  }

  AlphaWorkspace local_workspace;
  AlphaWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, max_groups);

  vec &residual = ws->residual;
  std::memcpy(residual.memptr(), pi.memptr(), N * sizeof(double)); // initial residual = pi

  double crit = 1.0;
  uword iter = 0;
  
  // Get workspace pointers once outside main loop
  vec &alpha0 = ws->alpha0;
  double *alpha0_ptr = alpha0.memptr();
  double *residual_ptr = residual.memptr();

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (uword k = 0; k < K; ++k) {
      const AlphaGroupInfo &info = (*group_info_ptr)(k);
      const uword J = info.n_groups;
      const field<uvec> &fe_groups = group_indices(k);

      vec &coef_k = coefficients(k);
      double *coef_k_ptr = coef_k.memptr();
      const uword *group_size_ptr = info.group_size.memptr();

      // Save old coefficients and compute sum of squares in one pass
      for (uword j = 0; j < J; ++j) {
        const double val = coef_k_ptr[j];
        alpha0_ptr[j] = val;
        sum_sq0 += val * val;
      }

      // Accumulate group sums and update in single pass over groups
      for (uword j = 0; j < J; ++j) {
        const uvec &indexes = fe_groups(j);
        const uword *idx_ptr = indexes.memptr();
        const uword n_idx = indexes.n_elem;
        
        // Compute group sum
        const double a0j = alpha0_ptr[j];
        double s = a0j * static_cast<double>(n_idx);
        for (uword t = 0; t < n_idx; ++t) {
          s += residual_ptr[idx_ptr[t]];
        }
        
        // Compute new coefficient
        const double new_val = (group_size_ptr[j] > 0)
                                   ? s / static_cast<double>(group_size_ptr[j])
                                   : 0.0;
        const double diff = new_val - a0j;
        sum_sq_diff += diff * diff;
        coef_k_ptr[j] = new_val;

        // Update residual
        for (uword t = 0; t < n_idx; ++t) {
          residual_ptr[idx_ptr[t]] -= diff;
        }
      }
    }

    if (sum_sq0 > 0.0) {
      crit = std::sqrt(sum_sq_diff / sum_sq0);
    } else if (sum_sq_diff > 0.0) {
      crit = std::sqrt(sum_sq_diff);
    } else {
      crit = 0.0;
    }

    ++iter;
  }

  // Normalize fixed effects for identifiability to match fixest/Stata
  // convention
  // Set the first level of the last FE to zero, adjust first FE
  // accordingly
  // This ensures exp(FE) values are on the same scale as fixest for gravity
  // models
  if (K > 0) {
    uword last_k = K - 1;
    vec &last_coef = coefficients(last_k);

    if (last_coef.n_elem > 0) {
      // Get the value of the FIRST level of the last FE (fixest convention)
      const double first_fe_last_val = last_coef(0);

      // Shift last FE so its first level is zero (in-place loop)
      double *last_ptr = last_coef.memptr();
      const uword n_last = last_coef.n_elem;
      for (uword i = 0; i < n_last; ++i) {
        last_ptr[i] -= first_fe_last_val;
      }

      // Shift first FE in opposite direction to maintain sum constraint
      vec &first_coef = coefficients(0);
      double *first_ptr = first_coef.memptr();
      const uword n_first = first_coef.n_elem;
      for (uword i = 0; i < n_first; ++i) {
        first_ptr[i] += first_fe_last_val;
      }
    }
  }

  return coefficients;
}

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
