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

// Optimized alpha recovery using flat obs->group mapping (like centering)
struct AlphaFlatMap {
  std::vector<std::vector<uword>>
      fe_map;                   // K x N: fe_map[k][i] = group of obs i
  std::vector<vec> inv_weights; // K: precomputed 1/sum(w) per group
  std::vector<uword> n_groups;  // K: number of groups per FE
  uword n_obs;
  uword K;

  void build(const field<field<uvec>> &group_indices,
             const vec *weights = nullptr) {
    K = group_indices.n_elem;
    if (K == 0)
      return;

    n_groups.resize(K);
    n_obs = 0;

    // First pass: find n_obs and n_groups
    for (uword k = 0; k < K; ++k) {
      n_groups[k] = group_indices(k).n_elem;
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        if (idx.n_elem > 0) {
          n_obs = std::max(n_obs, idx.max() + 1);
        }
      }
    }

    // Allocate fe_map
    fe_map.resize(K);
    for (uword k = 0; k < K; ++k) {
      fe_map[k].assign(n_obs, 0);
    }

    // Fill fe_map
    for (uword k = 0; k < K; ++k) {
      uword *map_k = fe_map[k].data();
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        const uword *idx_ptr = idx.memptr();
        const uword cnt = idx.n_elem;
        for (uword j = 0; j < cnt; ++j) {
          map_k[idx_ptr[j]] = g;
        }
      }
    }

    // Compute inverse weights
    inv_weights.resize(K);
    const bool use_w = (weights != nullptr && weights->n_elem == n_obs);
    const double *w_ptr = use_w ? weights->memptr() : nullptr;

    for (uword k = 0; k < K; ++k) {
      inv_weights[k].zeros(n_groups[k]);
      double *inv_w_ptr = inv_weights[k].memptr();
      const uword *map_k = fe_map[k].data();

      if (use_w) {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += w_ptr[i];
        }
      } else {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += 1.0;
        }
      }

      // Invert
      for (uword g = 0; g < n_groups[k]; ++g) {
        inv_w_ptr[g] = (inv_w_ptr[g] > 1e-12) ? (1.0 / inv_w_ptr[g]) : 0.0;
      }
    }
  }
};

inline field<vec>
get_alpha(const vec &pi, const field<field<uvec>> &group_indices,
          double tol = 1e-8, uword iter_max = 10000,
          void *unused = nullptr, // kept for API compatibility
          const vec *weights = nullptr) {
  (void)unused; // suppress warning

  const uword K = group_indices.n_elem;
  const uword N = pi.n_elem;
  field<vec> coefficients(K);

  if (K == 0 || N == 0) {
    return coefficients;
  }

  // Build flat map (like centering does)
  AlphaFlatMap map;
  map.build(group_indices, weights);

  // Initialize coefficients to zero
  for (uword k = 0; k < K; ++k) {
    coefficients(k).zeros(map.n_groups[k]);
  }

  // Residual: r = pi - sum_k(alpha_k expanded)
  vec r = pi;
  const double *r_ptr = r.memptr();
  double *r_ptr_w = r.memptr();

  // Weight pointer (may be null)
  const bool use_w = (weights != nullptr && weights->n_elem == N);
  const double *w_ptr = use_w ? weights->memptr() : nullptr;

  double crit = 1.0;
  uword iter = 0;

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (uword k = 0; k < K; ++k) {
      const uword *gk = map.fe_map[k].data();
      const uword J = map.n_groups[k];
      const double *inv_wk = map.inv_weights[k].memptr();
      vec &alpha_k = coefficients(k);
      double *ak_ptr = alpha_k.memptr();

      sum_sq0 += dot(alpha_k, alpha_k);

      // Compute new alpha using scatter-gather with raw pointers
      vec new_alpha(J, fill::zeros);
      double *new_ak_ptr = new_alpha.memptr();

      // Accumulate: new_alpha[g] += w[i] * (r[i] + alpha_k[g[i]])
      if (use_w) {
        for (uword i = 0; i < N; ++i) {
          uword g = gk[i];
          new_ak_ptr[g] += w_ptr[i] * (r_ptr[i] + ak_ptr[g]);
        }
      } else {
        for (uword i = 0; i < N; ++i) {
          uword g = gk[i];
          new_ak_ptr[g] += r_ptr[i] + ak_ptr[g];
        }
      }

      // Apply inverse weights
      for (uword j = 0; j < J; ++j) {
        new_ak_ptr[j] *= inv_wk[j];
      }

      // Compute diff and update criterion
      double local_diff_sq = 0.0;
      for (uword j = 0; j < J; ++j) {
        double d = new_ak_ptr[j] - ak_ptr[j];
        local_diff_sq += d * d;
      }
      sum_sq_diff += local_diff_sq;

      // Update residual: r[i] -= (new_alpha[g[i]] - alpha_k[g[i]])
      for (uword i = 0; i < N; ++i) {
        uword g = gk[i];
        r_ptr_w[i] -= new_ak_ptr[g] - ak_ptr[g];
      }

      // Update coefficients
      alpha_k = std::move(new_alpha);
    }

    // Convergence criterion
    crit = (sum_sq0 > 0.0) ? std::sqrt(sum_sq_diff / sum_sq0)
                           : std::sqrt(sum_sq_diff);

    ++iter;
  }

  // Normalize: shift so first level of last FE is zero (fixest/Stata
  // convention)
  if (K > 0 && coefficients(K - 1).n_elem > 0) {
    const double shift = coefficients(K - 1)(0);
    coefficients(K - 1) -= shift;
    coefficients(0) += shift;
  }

  return coefficients;
}

// Legacy struct for API compatibility (not used internally anymore)
struct AlphaGroupInfo {
  uvec obs_to_group;
  vec group_sizes;
  uword n_groups;
};

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
