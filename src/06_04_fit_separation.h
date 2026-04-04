// Code used for GLM model fitting
// Group-level separation pre-filter
// Drops observations in FE groups where all y==0 (Poisson/NegBin)
// or all y==0 or all y==1 (Binomial). Iterates until a fixed point.

#ifndef CAPYBARA_GLM_DROP
#define CAPYBARA_GLM_DROP

namespace capybara {

inline SeparationResult check_group_separation(const vec &y, const vec &w,
                                               const FlatFEMap &fe_map,
                                               Family family_type) {
  SeparationResult result;
  result.num_separated = 0;
  result.converged = true;

  // Only applicable for Poisson, NegBin, and Binomial with fixed effects
  if (fe_map.K == 0) {
    return result;
  }
  const bool is_binomial = (family_type == BINOMIAL);
  if (!is_binomial && family_type != POISSON && family_type != NEG_BIN) {
    return result;
  }

  const uword n = y.n_elem;
  uvec drop_mask(n, fill::zeros); // 1 = separated, 0 = keep

  // Iterate until no new observations are dropped
  // (dropping from one FE dimension can cause another group to become
  // degenerate in a different dimension)
  bool changed = true;
  while (changed) {
    changed = false;

    for (uword k = 0; k < fe_map.K; ++k) {
      const uword n_grp = fe_map.n_groups[k];
      const std::vector<uword> &map_k = fe_map.fe_map[k];

      // Compute weighted group sums and counts over kept observations
      vec grp_sum(n_grp, fill::zeros);
      vec grp_wt(n_grp, fill::zeros);

      for (uword i = 0; i < n; ++i) {
        if (drop_mask(i))
          continue;
        const uword g = map_k[i];
        const double wi = w(i);
        grp_sum(g) += wi * y(i);
        grp_wt(g) += wi;
      }

      // Identify degenerate groups
      for (uword i = 0; i < n; ++i) {
        if (drop_mask(i))
          continue;
        const uword g = map_k[i];
        if (grp_wt(g) <= 0.0)
          continue;

        const double grp_mean = grp_sum(g) / grp_wt(g);

        bool is_separated = false;
        if (is_binomial) {
          // Groups where mean(y) <= 0 or mean(y) >= 1 => perfect prediction
          is_separated = (grp_mean <= 0.0 || grp_mean >= 1.0);
        } else {
          // Poisson/NegBin: groups where mean(y) <= 0 => all zeros
          is_separated = (grp_mean <= 0.0);
        }

        if (is_separated) {
          drop_mask(i) = 1;
          changed = true;
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
