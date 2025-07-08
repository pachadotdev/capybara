#ifndef CAPYBARA_LM
#define CAPYBARA_LM

struct FelmFitResult {
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

mat crossprod_(const mat &X, const vec &w) {
  if (all(w == 1.0)) {
    return X.t() * X;
  } else {
    return X.t() * diagmat(w) * X;
  }
}

// Core function: pure Armadillo types
inline FelmFitResult felm_fit(const mat &X, const vec &y, const vec &w,
                              const list &k_list, double center_tol,
                              size_t iter_center_max, size_t iter_interrupt,
                              size_t iter_ssr) {
  FelmFitResult res;
  mat Xc = X;
  vec MNU, beta, fitted;
  bool has_fixed_effects = k_list.size() > 0;

  if (has_fixed_effects) {
    MNU = y;
    center_variables_(MNU, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);
    center_variables_(Xc, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);
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
