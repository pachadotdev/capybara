#include "00_main.h"

[[cpp11::register]] list felm_fit_(const doubles &y_r,
                                   const doubles_matrix<> &x_r,
                                   const doubles &wt_r, const list &control,
                                   const list &k_list) {
  // Type conversion

  mat X = as_Mat(x_r);
  const vec y = as_Col(y_r);
  const vec w = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const double center_tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);

  // Auxiliary variables (storage)

  mat H(X.n_cols, X.n_cols, fill::none);
  vec MNU(y.n_elem, fill::none), beta(X.n_cols, fill::none),
      fitted(y.n_elem, fill::none);

  // Center variables

  const bool has_fixed_effects = k_list.size() > 0;

  if (has_fixed_effects) {
    // Initial response + centering for fixed effects
    MNU = y;
    center_variables_(MNU, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);
    center_variables_(X, w, k_list, center_tol, iter_center_max, iter_interrupt,
                      iter_ssr);
  } else {
    // No fixed effects
    MNU = vec(y.n_elem, fill::zeros);
  }

  // Solve the normal equations

  beta = solve_beta_(X, MNU, w);

  // Fitted values

  if (has_fixed_effects) {
    fitted = y - MNU + X * beta;
  } else {
    fitted = X * beta;
  }

  // Recompute Hessian

  H = crossprod_(X, w);

  // Generate result list

  return writable::list({"coefficients"_nm = as_doubles(beta),
                         "fitted.values"_nm = as_doubles(fitted),
                         "weights"_nm = as_doubles(w),
                         "hessian"_nm = as_doubles_matrix(H)});
}
