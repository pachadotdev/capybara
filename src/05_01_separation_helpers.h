// Separation detection for Poisson models
// Based on Correia, Guimaraes, Zylkin (2019)
// "Verifying the existence of maximum likelihood estimates for generalized
// linear models"

#ifndef CAPYBARA_SEPARATION_HELPERS_H
#define CAPYBARA_SEPARATION_HELPERS_H

namespace capybara {

// Result structure for separation detection
struct SeparationResult {
  uvec separated_obs; // Indices of separated observations (0-based)
  vec support;        // z vector proving separation (supporting hyperplane)
  uword num_separated;
  bool converged;
  uword iterations;

  SeparationResult() : num_separated(0), converged(false), iterations(0) {}
};

// Solve weighted least squares: minimize ||sqrt(W)(y - X*beta)||^2
// Uses normal equations with rank-revealing Cholesky for robustness
inline vec solve_wls(const mat &X, const vec &y, const vec &w, vec &residuals) {
  if (X.n_cols == 0) {
    residuals = y;
    return vec();
  }

  const uword k = X.n_cols;

  // Form normal equations: X'WX * beta = X'Wy
  const mat Xw = X.each_col() % w;
  const mat XtWX = X.t() * Xw;
  const vec XtWy = Xw.t() * y;

  // Use rank-revealing Cholesky for robustness
  mat R;
  uvec excluded;
  uword rank;
  chol_rank(R, excluded, rank, XtWX, "upper");

  vec beta(k, fill::zeros);

  if (rank == k) {
    // Full rank: solve R'R * beta = XtWy via back-substitution
    vec z;
    solve(z, trimatl(R.t()), XtWy, solve_opts::fast);
    solve(beta, trimatu(R), z, solve_opts::fast);
  } else if (rank > 0) {
    // Rank-deficient: solve on non-excluded columns
    const uvec included = find(excluded == 0);
    if (included.n_elem > 0) {
      const mat R_sub = R.submat(included, included);
      const vec XtWy_sub = XtWy.elem(included);
      vec z;
      solve(z, trimatl(R_sub.t()), XtWy_sub, solve_opts::fast);
      vec beta_sub;
      solve(beta_sub, trimatu(R_sub), z, solve_opts::fast);
      beta.elem(included) = beta_sub;
    }
  }

  residuals = y - X * beta;
  return beta;
}

inline vec solve_lse_weighted(const mat &X, const vec &y,
                              const uvec &constrained_sample, double M_weight,
                              vec &residuals) {
  if (X.n_cols == 0) {
    residuals = y;
    return vec();
  }

  vec weights(y.n_elem, fill::ones);
  if (constrained_sample.n_elem > 0) {
    weights.elem(constrained_sample).fill(M_weight);
  }

  return solve_wls(X, y, weights, residuals);
}

} // namespace capybara

#endif // CAPYBARA_SEPARATION_HELPERS_H
