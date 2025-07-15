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
inline LMResult felm_fit(const mat &X, const vec &y, const vec &w,
                         const field<field<uvec>> &group_indices,
                         double center_tol, size_t iter_center_max,
                         size_t iter_interrupt, size_t iter_ssr,
                         double collin_tol) {
  // TIME_FUNCTION;
  LMResult res;
  mat Xc = X;
  vec MNU, beta, fitted;
  bool has_fixed_effects = group_indices.n_elem > 0;

  if (has_fixed_effects) {
    // Convert field<field<uvec>> to umat format for new demean_variables
    umat fe_matrix;
    if (group_indices.n_elem > 0) {
      size_t n_obs = y.n_elem;
      fe_matrix.set_size(n_obs, group_indices.n_elem);

      for (size_t k = 0; k < group_indices.n_elem; k++) {
        // Set FE levels based on group indices
        for (size_t g = 0; g < group_indices(k).n_elem; g++) {
          const uvec &group_obs = group_indices(k)(g);
          if (group_obs.n_elem > 0) {
            fe_matrix.submat(group_obs, uvec{k}).fill(g);
          }
        }
      }
    }

    MNU = y;
    mat MNU_mat = MNU;
    WeightedDemeanResult mnu_result = demean_variables(
        MNU_mat, fe_matrix, w, center_tol, iter_center_max, "gaussian");
    MNU = mnu_result.demeaned_data.col(0);

    WeightedDemeanResult xc_result = demean_variables(
        Xc, fe_matrix, w, center_tol, iter_center_max, "gaussian");
    Xc = xc_result.demeaned_data;
  } else {
    MNU = vec(y.n_elem, fill::zeros);
  }

  beta_results ws(Xc.n_rows, Xc.n_cols);
  beta = get_beta(Xc, MNU, w, Xc.n_rows, Xc.n_cols, ws, false, collin_tol);

  // Collinearity detection using QR decomposition
  uvec coef_status = ones<uvec>(beta.n_elem);
  mat Q, R;
  qr(Q, R, Xc);

  vec diag_R = R.diag();
  double max_diag = max(abs(diag_R));
  double tol_scaled = collin_tol * max_diag;
  uvec collinear_mask = (abs(diag_R) < tol_scaled);

  // Set collinear coefficients to zero
  if (any(collinear_mask)) {
    uvec collinear_indices = find(collinear_mask);
    beta.elem(collinear_indices).zeros();
    coef_status.elem(collinear_indices).zeros();
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
