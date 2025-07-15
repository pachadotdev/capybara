#ifndef CAPYBARA_LM
#define CAPYBARA_LM

struct LMResult {
  vec coefficients;
  vec fitted;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear

  cpp11::list to_list() const {
    return writable::list({"coefficients"_nm = as_doubles(coefficients),
                           "fitted.values"_nm = as_doubles(fitted),
                           "weights"_nm = as_doubles(weights),
                           "hessian"_nm = as_doubles_matrix(hessian),
                           "coef_status"_nm = as_integers(
                               arma::conv_to<ivec>::from(coef_status))});
  }
};

// Weighted cross-product computation
inline mat crossprod_(const mat &X, const vec &w) {
  if (all(w == 1.0)) {
    return X.t() * X;
  } else {
    mat wX = X.each_col() % w;
    return X.t() * wX;
  }
}

// Core function: pure Armadillo types
inline LMResult feols_fit(const mat &X, const vec &y, const vec &w,
                                const field<field<uvec>> &group_indices,
                                double center_tol, size_t iter_center_max,
                                size_t iter_interrupt, size_t iter_ssr) {
  // TIME_FUNCTION;
  LMResult res;
  mat Xc = X;
  vec MNU, beta, fitted;
  bool has_fixed_effects = group_indices.n_elem > 0;

  if (has_fixed_effects) {
    MNU = y;
    mat MNU_mat = MNU;
    demean_variables(MNU_mat, w, group_indices, center_tol, iter_center_max,
                     "gaussian");
    MNU = MNU_mat.col(0);
    demean_variables(Xc, w, group_indices, center_tol, iter_center_max,
                     "gaussian");
  } else {
    MNU = vec(y.n_elem, fill::zeros);
  }

  beta_results ws(Xc.n_rows, Xc.n_cols);
  beta = get_beta(Xc, MNU, w, Xc.n_rows, Xc.n_cols, ws, false);

  // Collinearity detection using QR decomposition
  uvec coef_status = ones<uvec>(beta.n_elem);
  mat Q, R;
  qr(Q, R, Xc);
  double tol = 1e-10;
  for (uword j = 0; j < R.n_cols && j < R.n_rows; ++j) {
    if (std::abs(R(j, j)) < tol) {
      beta(j) = 0.0;
      coef_status(j) = 0;
    }
  }

  if (has_fixed_effects) {
    fitted = y - MNU + Xc * beta;
  } else {
    fitted = Xc * beta;
  }

  res.coefficients = beta;
  res.fitted = fitted;
  res.weights = w;
  res.hessian = crossprod_(Xc, w);
  res.coef_status = coef_status;
  return res;
}

#endif // CAPYBARA_LM
