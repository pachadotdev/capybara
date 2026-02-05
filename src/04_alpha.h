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

// Group info using observation-to-group mapping
struct AlphaGroupInfo {
  uvec obs_to_group; // N x 1: maps observation i -> group j
  vec group_sizes;   // J x 1: count (or weighted sum) per group
  uword n_groups;
};

// Precompute group info
inline field<AlphaGroupInfo>
precompute_alpha_group_info(const field<field<uvec>> &group_indices, uword N,
                            const vec *weights = nullptr) {
  const uword K = group_indices.n_elem;
  field<AlphaGroupInfo> group_info(K);

  for (uword k = 0; k < K; ++k) {
    const uword J = group_indices(k).n_elem;
    AlphaGroupInfo &info = group_info(k);
    info.n_groups = J;
    info.obs_to_group.set_size(N);
    info.group_sizes.zeros(J);

    for (uword j = 0; j < J; ++j) {
      const uvec &idx = group_indices(k)(j);
      info.obs_to_group.elem(idx).fill(j);
      info.group_sizes(j) =
          weights ? accu(weights->elem(idx)) : static_cast<double>(idx.n_elem);
    }

    // Avoid division by zero (todo: remove this?)
    info.group_sizes.replace(0.0, 1.0);
  }

  return group_info;
}

inline field<vec>
get_alpha(const vec &pi, const field<field<uvec>> &group_indices,
          double tol = 1e-8, uword iter_max = 10000,
          field<AlphaGroupInfo> *precomputed_group_info = nullptr,
          const vec *weights = nullptr) {
  const uword K = group_indices.n_elem;
  const uword N = pi.n_elem;
  field<vec> coefficients(K);

  if (K == 0 || N == 0) {
    return coefficients;
  }

  // Group info (precompute if not provided)
  field<AlphaGroupInfo> local_group_info;
  const field<AlphaGroupInfo> *info = precomputed_group_info;

  if (!info) {
    local_group_info = precompute_alpha_group_info(group_indices, N, weights);
    info = &local_group_info;
  }

  // Initialize coefficients to zero
  for (uword k = 0; k < K; ++k) {
    coefficients(k).zeros((*info)(k).n_groups);
  }

  // Residual: r = pi - sum_k(alpha_k expanded)
  vec r = pi;

  double crit = 1.0;
  uword iter = 0;

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (uword k = 0; k < K; ++k) {
      const AlphaGroupInfo &ginfo = (*info)(k);
      const uvec &g = ginfo.obs_to_group;
      vec &alpha_k = coefficients(k);

      sum_sq0 += dot(alpha_k, alpha_k);

      // temp = r + alpha_k[g] (add back current FE contribution)
      vec temp = r + alpha_k.elem(g);

      // Compute new alpha: weighted or unweighted group means
      vec new_alpha(ginfo.n_groups, fill::zeros);
      for (uword j = 0; j < ginfo.n_groups; ++j) {
        const uvec &idx = group_indices(k)(j);
        new_alpha(j) = weights ? accu(temp.elem(idx) % weights->elem(idx)) /
                                     ginfo.group_sizes(j)
                               : accu(temp.elem(idx)) / ginfo.group_sizes(j);
      }

      // Update criterion
      vec diff = new_alpha - alpha_k;
      sum_sq_diff += dot(diff, diff);

      // Update residual: r -= diff[g]
      r -= diff.elem(g);

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

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
