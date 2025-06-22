#ifndef CAPYBARA_ALPHA_H
#define CAPYBARA_ALPHA_H

#include <armadillo>
using namespace arma;

inline void subtract_other_effects(vec &y, const vec &p,
                                   const field<vec> &alpha,
                                   const uvec &active_fes, size_t k_idx,
                                   const indices_info &indices) {
  // Initialize y once
  y = p; // uses pre-allocated memory
  size_t n_active = active_fes.n_elem;
  for (size_t l_idx = 0; l_idx < n_active; ++l_idx) {
    if (l_idx == k_idx)
      continue;
    size_t l = active_fes(l_idx);
    const uvec &groups_l = indices.nonempty_groups(l);
    const vec &alpha_l = alpha(l);
    for (uword j : groups_l) {
      uvec grp = indices.get_group(l, j);
      y.elem(grp) -= alpha_l(j);
    }
  }
}

inline void update_alpha_k(vec &alpha_k, const vec &y,
                           const indices_info &indices, size_t k) {
  // alpha_k pre-sized to fe_size
  alpha_k.zeros();
  const uvec &groups_k = indices.nonempty_groups(k);
  for (uword j : groups_k) {
    uvec grp = indices.get_group(k, j);
    alpha_k(j) = mean(y.elem(grp));
  }
}

inline solve_alpha_results solve_alpha(const vec &p,
                                       const indices_info &indices, double tol,
                                       size_t iter_max,
                                       size_t interrupt_iter0) {
  const uword K = indices.fe_sizes.n_elem;
  // Collect active fixed effects
  uvec active = find(indices.fe_sizes > 0);
  size_t n_active = active.n_elem;

  // Pre-allocate alpha field
  field<vec> alpha(K);
  for (uword k : active) {
    alpha(k).set_size(indices.fe_sizes(k));
    alpha(k).zeros();
  }

  if (n_active == 0) {
    return solve_alpha_results(std::move(alpha));
  }

  // Working vectors
  vec y(p.n_elem, fill::none);
  vec alpha_old(indices.fe_sizes.max(), fill::none);
  vec diff(alpha_old.n_elem, fill::none);

  size_t next_interrupt = interrupt_iter0;

  for (size_t it = 0; it < iter_max; ++it) {
    if (it == next_interrupt) {
      check_user_interrupt();
      next_interrupt += interrupt_iter0;
    }

    double change_sq = 0.0;
    double norm_sq = 0.0;

    // update each active FE
    for (size_t idx = 0; idx < n_active; ++idx) {
      uword k = active(idx);
      vec &ak = alpha(k);
      alpha_old.head(ak.n_elem) = ak;

      subtract_other_effects(y, p, alpha, active, idx, indices);
      update_alpha_k(ak, y, indices, k);

      // convergence metrics
      diff.head(ak.n_elem) = ak - alpha_old.head(ak.n_elem);
      change_sq += dot(diff.head(ak.n_elem), diff.head(ak.n_elem));
      norm_sq += dot(alpha_old.head(ak.n_elem), alpha_old.head(ak.n_elem));
    }

    double ratio = std::sqrt(change_sq / (norm_sq + 1e-10));
    if (ratio < tol)
      break;
  }

  return solve_alpha_results(std::move(alpha));
}

#endif // CAPYBARA_ALPHA_H
