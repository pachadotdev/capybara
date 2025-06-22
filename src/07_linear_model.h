#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

inline vec compute_fitted(const mat &X, const vec &orig_y,
                          const vec &centered_y, const beta_results &beta_ws) {
  vec fitted(X.n_rows, fill::none);

  uvec valid = find(beta_ws.valid_coefficients);

  if (valid.n_elem < beta_ws.coefficients.n_elem) {
    fitted = X.cols(valid) * beta_ws.coefficients.elem(valid);
  } else {
    fitted = X * beta_ws.coefficients;
  }

  // Add back offset if centered_y provided
  if (!centered_y.is_empty()) {
    fitted += orig_y - centered_y;
  }
  return fitted;
}

inline felm_results felm(mat &X, const vec &y, const vec &w, double center_tol,
                         size_t iter_center_max, size_t iter_interrupt,
                         const indices_info &indices,
                         const bool &use_acceleration) {
  const uword N = X.n_rows;
  const uword P = X.n_cols;

  bool has_fe = (indices.fe_sizes.n_elem > 0);
  bool use_w = !all(w == 1.0);

  beta_results beta_ws(N, P);
  vec fitted(N, fill::none);

  if (has_fe) {
    mat X0 = X;
    vec yc = y; // Make a copy since it will be modified
    center_variables(X, yc, w, X0, indices, center_tol, iter_center_max,
                           iter_interrupt, use_w, use_acceleration);
    solve_beta(X, yc, w, N, P, beta_ws, use_w);
    fitted = compute_fitted(X, y, yc, beta_ws);
  } else {
    solve_beta(X, y, w, N, P, beta_ws, use_w);
    fitted = compute_fitted(X, y, vec(), beta_ws);
  }

  crossproduct_results cross_ws(N, P);
  mat H = crossproduct(X, w, cross_ws, use_w);

  return felm_results(std::move(beta_ws.coefficients),
                      std::move(beta_ws.valid_coefficients), std::move(fitted),
                      w, std::move(H));
}

#endif // CAPYBARA_LM_H
