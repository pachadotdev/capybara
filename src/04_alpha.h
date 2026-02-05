// Computing alpha in a model with fixed effects Y = alpha + X beta given beta
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

// Group info using observation-to-group mapping for sparse ops
struct AlphaGroupInfo {
  uvec obs_to_group;
  vec group_sizes;
  uword n_groups;
};

// Precompute group info
inline field<AlphaGroupInfo>
precompute_alpha_group_info(const field<field<uvec>> &group_indices, uword N) {
  const uword K = group_indices.n_elem;
  field<AlphaGroupInfo> group_info(K);

  for (uword k = 0; k < K; ++k) {
    const uword J = group_indices(k).n_elem;
    AlphaGroupInfo &info = group_info(k);
    info.n_groups = J;
    info.obs_to_group.set_size(N);
    info.group_sizes.set_size(J);

    for (uword j = 0; j < J; ++j) {
      const uvec &indexes = group_indices(k)(j);
      info.group_sizes(j) = static_cast<double>(indexes.n_elem);
      info.obs_to_group.elem(indexes).fill(j);
    }
  }

  return group_info;
}

struct AlphaWorkspace {
  vec residual;       // pi - sum(alpha_k)
  vec group_sums;     // for accumulating group sums
  vec alpha_expanded; // Alpha values expanded to observation level
  uword cached_N;
  uword cached_max_groups;
  bool is_initialized;

  AlphaWorkspace() : cached_N(0), cached_max_groups(0), is_initialized(false) {}

  void ensure_size(uword N, uword max_groups) {
    if (!is_initialized || N > cached_N || max_groups > cached_max_groups) {
      residual.set_size(N);
      group_sums.set_size(max_groups);
      alpha_expanded.set_size(N);
      cached_N = N;
      cached_max_groups = max_groups;
      is_initialized = true;
    }
  }

  void clear() {
    residual.reset();
    group_sums.reset();
    alpha_expanded.reset();
    cached_N = 0;
    cached_max_groups = 0;
    is_initialized = false;
  }
};

// Accumulate values into groups
// group_sums[j] = sum of values[i] where obs_to_group[i] == j
inline void scatter_add(vec &group_sums, const vec &values,
                        const uvec &obs_to_group, uword n_groups,
                        const vec *w = nullptr) {
  group_sums.head(n_groups).zeros();
  const uword N = values.n_elem;
  const double *val_ptr = values.memptr();
  const uword *grp_ptr = obs_to_group.memptr();
  double *sum_ptr = group_sums.memptr();

  if (w) {
    const double *w_ptr = w->memptr();
    for (uword i = 0; i < N; ++i) {
      sum_ptr[grp_ptr[i]] += val_ptr[i] * w_ptr[i];
    }
  } else {
    for (uword i = 0; i < N; ++i) {
      sum_ptr[grp_ptr[i]] += val_ptr[i];
    }
  }
}

// Expand group values to observation level
// result[i] = group_values[obs_to_group[i]]
inline void gather(vec &result, const vec &group_values,
                   const uvec &obs_to_group) {
  result = group_values.elem(obs_to_group);
}

inline field<vec>
get_alpha(const vec &pi, const field<field<uvec>> &group_indices,
          double tol = 1e-8, uword iter_max = 10000,
          field<AlphaGroupInfo> *precomputed_group_info = nullptr,
          AlphaWorkspace *workspace = nullptr, const vec *weights = nullptr) {
  const uword K = group_indices.n_elem;
  const uword N = pi.n_elem;
  field<vec> coefficients(K);

  if (K == 0 || N == 0) {
    return coefficients;
  }

  // Group info
  field<AlphaGroupInfo> local_group_info;
  const field<AlphaGroupInfo> *group_info_ptr = precomputed_group_info;

  if (!group_info_ptr) {
    local_group_info = precompute_alpha_group_info(group_indices, N);
    group_info_ptr = &local_group_info;
  }

  // Initialize coefficients and find max groups
  uword max_groups = 0;
  for (uword k = 0; k < K; ++k) {
    const uword J = (*group_info_ptr)(k).n_groups;
    max_groups = std::max(max_groups, J);
    coefficients(k).zeros(J);
  }

  AlphaWorkspace local_workspace;
  AlphaWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, max_groups);

  ws->residual = pi;

  // Precompute denominators (sum of weights per group)
  field<vec> denominators(K);
  vec ones_vec;
  if (weights) {
    ones_vec.ones(N);
    for (uword k = 0; k < K; ++k) {
      const uword J = (*group_info_ptr)(k).n_groups;
      denominators(k).set_size(J);
      scatter_add(denominators(k), ones_vec, (*group_info_ptr)(k).obs_to_group,
                  J, weights);
      // Avoid division by zero
      denominators(k).replace(0.0, 1.0);
    }
  }

  double crit = 1.0;
  uword iter = 0;

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (uword k = 0; k < K; ++k) {
      const AlphaGroupInfo &info = (*group_info_ptr)(k);
      const uword J = info.n_groups;
      vec &coef_k = coefficients(k);

      sum_sq0 += dot(coef_k, coef_k);

      // Step 1: Expand current alpha to observation level
      gather(ws->alpha_expanded, coef_k, info.obs_to_group);

      // Step 2: Compute (residual + alpha_expanded)
      vec temp = ws->residual + ws->alpha_expanded;

      // Step 3: Scatter-add to get group sums
      scatter_add(ws->group_sums, temp, info.obs_to_group, J, weights);

      // Step 4: Compute new coefficients by dividing by group sizes/weights
      vec new_coef;
      if (weights) {
        new_coef = ws->group_sums.head(J) / denominators(k);
      } else {
        new_coef = ws->group_sums.head(J) / info.group_sizes;
      }

      // Step 5: Compute difference and update criterion
      vec diff = new_coef - coef_k;
      sum_sq_diff += dot(diff, diff);

      // Step 6: Update residual by subtracting the change in alpha
      gather(ws->alpha_expanded, diff, info.obs_to_group);
      ws->residual -= ws->alpha_expanded;

      // Step 7: Update coefficients
      coef_k = new_coef;
    }

    // Convergence criterion
    if (sum_sq0 > 0.0) {
      crit = std::sqrt(sum_sq_diff / sum_sq0);
    } else if (sum_sq_diff > 0.0) {
      crit = std::sqrt(sum_sq_diff);
    } else {
      crit = 0.0;
    }

    ++iter;
  }

  // Normalize fixed effects for identifiability (fixest/Stata convention)
  if (K > 0) {
    vec &last_coef = coefficients(K - 1);

    if (last_coef.n_elem > 0) {
      // Shift so first level of last FE is zero
      const double first_fe_last_val = last_coef(0);
      last_coef -= first_fe_last_val;
      coefficients(0) += first_fe_last_val;
    }
  }

  return coefficients;
}

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
