#ifndef CAPYBARA_GLM_SUMS_H
#define CAPYBARA_GLM_SUMS_H

namespace capybara {

///////////////////////////////////////////////////////////////////////////
// Group aggregation functions
///////////////////////////////////////////////////////////////////////////

// Weighted group sums: sum_g (sum_i M_i / sum_i w_i) for each group g
inline vec group_sums(const mat &M, const vec &w, const FlatFEMap &fe_map,
                      uword k) {
  const uword n = M.n_rows;
  const uword P = M.n_cols;
  const uword G = fe_map.n_groups[k];
  const uword *gk = fe_map.fe_map[k].data();

  // Accumulate per group
  mat group_num(G, P, fill::zeros);
  vec group_denom(G, fill::zeros);

  for (uword i = 0; i < n; ++i) {
    const uword g = gk[i];
    group_num.row(g) += M.row(i);
    group_denom(g) += w(i);
  }

  // Sum across groups: sum_g (numerator_g / denominator_g)
  vec b(P, fill::zeros);
  for (uword g = 0; g < G; ++g) {
    if (group_denom(g) > 0.0) {
      b += group_num.row(g).t() / group_denom(g);
    }
  }
  return b;
}

// Group variance: V = sum_g (M_g * M_g') where M_g = sum_i M_i for obs in group g
inline mat group_sums_var(const mat &M, const FlatFEMap &fe_map, uword k) {
  const uword n = M.n_rows;
  const uword P = M.n_cols;
  const uword G = fe_map.n_groups[k];
  const uword *gk = fe_map.fe_map[k].data();

  // Accumulate group sums
  mat grp_sums(G, P, fill::zeros);
  for (uword i = 0; i < n; ++i) {
    grp_sums.row(gk[i]) += M.row(i);
  }

  // Compute V = sum_g (group_sum_g * group_sum_g')
  mat V(P, P, fill::zeros);
  for (uword g = 0; g < G; ++g) {
    V += grp_sums.row(g).t() * grp_sums.row(g);
  }
  return V;
}

// Group covariance: V = sum_g (M_g * N_g') where M_g, N_g are group sums
inline mat group_sums_cov(const mat &M, const mat &N, const FlatFEMap &fe_map,
                          uword k) {
  const uword n = M.n_rows;
  const uword P = M.n_cols;
  const uword G = fe_map.n_groups[k];
  const uword *gk = fe_map.fe_map[k].data();

  // Accumulate group sums for both matrices
  mat m_sums(G, P, fill::zeros);
  mat n_sums(G, P, fill::zeros);

  for (uword i = 0; i < n; ++i) {
    const uword g = gk[i];
    m_sums.row(g) += M.row(i);
    n_sums.row(g) += N.row(i);
  }

  // Compute V = sum_g m_g * n_g'
  mat V(P, P, fill::zeros);
  for (uword g = 0; g < G; ++g) {
    V += m_sums.row(g).t() * n_sums.row(g);
  }
  return V;
}

} // namespace capybara

#endif // CAPYBARA_GLM_SUMS_H
