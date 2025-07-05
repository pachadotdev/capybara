#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

// #include "timing.h" // development only, for profiling

// Compute fitted values from coefficients, handling collinearity and centering
inline vec compute_fitted(const mat &X, const vec &orig_y,
                          const vec &centered_y, const beta_results &beta_ws) {
  vec fitted(X.n_rows, fill::none);

  const uvec valid = find(beta_ws.valid_coefficients);

  if (valid.n_elem < beta_ws.coefficients.n_elem) {
    fitted = X.cols(valid) * beta_ws.coefficients.elem(valid);
  } else {
    fitted = X * beta_ws.coefficients;
  }

  if (!centered_y.is_empty()) {
    fitted += orig_y - centered_y;
  }
  return fitted;
}

// Main linear model fitting routine (with or without fixed effects)
inline felm_results felm(mat &X, const vec &y, const vec &w, double center_tol,
                         size_t iter_center_max, size_t iter_interrupt,
                         const indices_info &indices,
                         const bool &use_acceleration) {
  // TIME_FUNCTION;

  const uword N = X.n_rows;
  const uword P = X.n_cols;

  const bool has_fe = (indices.fe_sizes.n_elem > 0);
  const bool use_w = !all(w == 1.0);

  beta_results beta_ws(N, P);
  vec fitted(N, fill::none);

  if (has_fe) {
    const mat X0 = X;
    vec yc = y;
    // Center variables for fixed effects
    center_variables(X, yc, w, X0, indices, center_tol, iter_center_max,
                     iter_interrupt);
    // Solve for beta coefficients
    solve_beta(X, yc, w, N, P, beta_ws, use_w);
    // Compute fitted values
    fitted = compute_fitted(X, y, yc, beta_ws);
  } else {
    // Solve for beta coefficients without fixed effects
    solve_beta(X, y, w, N, P, beta_ws, use_w);
    // Compute fitted values
    fitted = compute_fitted(X, y, vec(), beta_ws);
  }

  // Compute Hessian
  crossproduct_results cross_ws(N, P);
  crossproduct(X, w, cross_ws, use_w);
  mat &H = cross_ws.M;

  return felm_results(std::move(beta_ws.coefficients),
                      std::move(beta_ws.valid_coefficients), std::move(fitted),
                      w, std::move(H));
}

#endif // CAPYBARA_LM_H
