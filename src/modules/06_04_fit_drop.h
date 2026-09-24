// Group-level separation pre-filter
// Drops observations in FE groups where all y==0 (Poisson/NegBin)
// or all y==0 or all y==1 (Binomial). Iterates until a fixed point.
// => drop observations that do not contribute to the likelihood

#ifndef CAPYBARA_GLM_DROP
#define CAPYBARA_GLM_DROP

namespace capybara {

inline SeparationResult check_group_separation(const vec &y, const vec &w,
                                               const FlatFEMap &fe_map,
                                               Family family_type) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  // Only applicable for Poisson/NegBin with fixed effects
  if (fe_map.K == 0) {
    return result;
  }

  if (family_type != POISSON && family_type != NEG_BIN) {
    return result;
  }

  const uword n = y.n_elem;
  const uword K = fe_map.K;

  // Validate input dimensions match FE map structure
  if (n != fe_map.n_obs || w.n_elem != fe_map.n_obs) {
    cpp4r::stop(
        "check_group_separation: dimension mismatch (y: %u, w: %u, fe_map: %u)",
        (unsigned)n, (unsigned)w.n_elem, (unsigned)fe_map.n_obs);
  }

  // Build inverse maps: group_members[k][g] = indices of observations in group
  // g This enables O(group_size) marking instead of O(n) scans
  std::vector<std::vector<std::vector<uword>>> group_members(K);
  for (uword k = 0; k < K; ++k) {
    const uword n_grp = fe_map.n_groups[k];
    const std::vector<uword> &map_k = fe_map.fe_map[k];

    if (map_k.size() != n) {
      cpp4r::stop("check_group_separation: fe_map[%u].size() != n (%u != %u)",
                  (unsigned)k, (unsigned)map_k.size(), (unsigned)n);
    }

    group_members[k].resize(n_grp);
    for (uword i = 0; i < n; ++i) {
      const uword g = map_k[i];
      if (g >= n_grp) {
        cpp4r::stop("check_group_separation: group index out of bounds (g=%u "
                    ">= n_groups[%u]=%u for obs %u)",
                    (unsigned)g, (unsigned)k, (unsigned)n_grp, (unsigned)i);
      }
      group_members[k][g].push_back(i);
    }
  }

  // Track dropped observations
  uvec drop_mask(n, fill::zeros); // 1 = separated, 0 = keep

  // Find max number of groups across all FE dimensions for buffer sizing
  uword max_grp = 0;
  for (uword k = 0; k < K; ++k) {
    max_grp = std::max(max_grp, fe_map.n_groups[k]);
  }

  // Preallocate buffers outside iteration loop
  vec grp_sum(max_grp);
  vec grp_wt(max_grp);

  // Iterate until no new observations are dropped
  // (dropping from one FE dimension can cause another group to become
  // degenerate in a different dimension)
  bool changed = true;
  while (changed) {
    changed = false;

    for (uword k = 0; k < K; ++k) {
      const uword n_grp = fe_map.n_groups[k];
      const std::vector<std::vector<uword>> &members_k = group_members[k];

      // Reset buffers (only the portion we use)
      std::fill(grp_sum.memptr(), grp_sum.memptr() + n_grp, 0.0);
      std::fill(grp_wt.memptr(), grp_wt.memptr() + n_grp, 0.0);

      // Compute weighted group sums using inverse map
      // This is still O(n) total but with better cache locality per group
      double *sum_ptr = grp_sum.memptr();
      double *wt_ptr = grp_wt.memptr();
      const double *y_ptr = y.memptr();
      const double *w_ptr = w.memptr();
      const uword *drop_ptr = drop_mask.memptr();

      for (uword g = 0; g < n_grp; ++g) {
        const std::vector<uword> &members = members_k[g];
        for (uword idx : members) {
          if (!drop_ptr[idx]) {
            const double wi = w_ptr[idx];
            sum_ptr[g] += wi * y_ptr[idx];
            wt_ptr[g] += wi;
          }
        }
      }

      // Identify degenerate groups and mark all their members
      // Loop over groups (not observations) - much faster when n >> n_grp
      uword *drop_ptr_mut = drop_mask.memptr();
      for (uword g = 0; g < n_grp; ++g) {
        if (wt_ptr[g] <= 0.0)
          continue;

        const double grp_mean = sum_ptr[g] / wt_ptr[g];

        bool is_separated = false;

        // Poisson/NegBin: groups where mean(y) <= 0 => all zeros
        is_separated = (grp_mean <= 0.0);

        if (is_separated) {
          // Mark all members of this degenerate group
          const std::vector<uword> &members = members_k[g];
          for (uword idx : members) {
            if (!drop_ptr_mut[idx]) {
              drop_ptr_mut[idx] = 1;
              changed = true;
            }
          }
        }
      }
    }
  }

  result.separated_obs = find(drop_mask);
  result.num_separated = result.separated_obs.n_elem;
  return result;
}

} // namespace capybara

#endif // CAPYBARA_GLM_DROP
