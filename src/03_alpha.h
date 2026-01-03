// Computing alpha a in a model with fixed effects Y = alpha + X beta given beta

#ifndef CAPYBARA_ALPHA_H
#define CAPYBARA_ALPHA_H

namespace capybara {

#ifdef _OPENMP
#include <omp.h>
#endif

// Get block size for cache-friendly indexed scatter operations
inline uword get_block_size(uword n, uword k) {
  constexpr uword L1_CACHE = 32768;
  constexpr uword element_size = sizeof(double) + sizeof(uword);
  return std::max(static_cast<uword>(1000),
                  std::min(n, L1_CACHE / (k * element_size)));
}

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
  vec residual;   // pi - sum(alpha_k)
  vec alpha0;     // Previous coefficients for one FE
  vec group_sums; // Accumulator for one FE
  uword cached_N;
  uword cached_max_groups;
  bool is_initialized;

  AlphaWorkspace() : cached_N(0), cached_max_groups(0), is_initialized(false) {}

  void ensure_size(uword N, uword max_groups) {
    if (!is_initialized || N > cached_N || max_groups > cached_max_groups) {
      residual.set_size(N);
      alpha0.set_size(max_groups);
      group_sums.set_size(max_groups);
      cached_N = N;
      cached_max_groups = max_groups;
      is_initialized = true;
    }
  }

  void clear() {
    residual.reset();
    alpha0.reset();
    group_sums.reset();
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
  residual = pi; // initial residual = pi - sum(alpha_k) with alpha_k = 0

  double crit = 1.0;
  uword iter = 0;

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (uword k = 0; k < K; ++k) {
      const AlphaGroupInfo &info = (*group_info_ptr)(k);
      const uword J = info.n_groups;

      double *alpha0_ptr = ws->alpha0.memptr();
      double *group_sums_ptr = ws->group_sums.memptr();
      const double *coef_k_ptr_old = coefficients(k).memptr();

      std::memcpy(alpha0_ptr, coef_k_ptr_old, J * sizeof(double));
      sum_sq0 += dot(ws->alpha0.head(J), ws->alpha0.head(J));

      ws->group_sums.head(J).zeros();
      const double *residual_ptr = residual.memptr();

      // Accumulate group sums by iterating groups (enables parallelization)
      // group_sums[j] = sum_{i in g_j} (residual[i] + alpha0[j])
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (uword j = 0; j < J; ++j) {
        const uvec &indexes = group_indices(k)(j);
        const uword *idx_ptr = indexes.memptr();
        const uword n_idx = indexes.n_elem;
        double s = 0.0;
        const double a0j = alpha0_ptr[j];
        for (uword t = 0; t < n_idx; ++t) {
          s += residual_ptr[idx_ptr[t]] + a0j;
        }
        group_sums_ptr[j] = s;
      }

      double *coef_k_ptr_new = coefficients(k).memptr();
      const uword *group_size = info.group_size.memptr();

      // Update alpha_k per group and accumulate convergence metric
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : sum_sq_diff)
#endif
      for (uword j = 0; j < J; ++j) {
        double new_val = 0.0;
        if (group_size[j] > 0) {
          new_val = group_sums_ptr[j] / static_cast<double>(group_size[j]);
        }
        coef_k_ptr_new[j] = new_val;
        double diff = new_val - alpha0_ptr[j];
        sum_sq_diff += diff * diff;
      }

      double *residual_ptr_mut = residual.memptr();

      // Update residual: residual = pi - sum(alpha_k)
      // residual[i] -= (alpha_k_new[group(i)] - alpha_k_old[group(i)])
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (uword j = 0; j < J; ++j) {
        const uvec &indexes = group_indices(k)(j);
        const uword *idx_ptr = indexes.memptr();
        const uword n_idx = indexes.n_elem;
        const double delta = coef_k_ptr_new[j] - alpha0_ptr[j];
        for (uword t = 0; t < n_idx; ++t) {
          residual_ptr_mut[idx_ptr[t]] -= delta;
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

  return coefficients;
}

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
