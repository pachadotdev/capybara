#include "00_main.h"

[[cpp11::register]] list felm_fit_(const doubles &y_r,
                                   const doubles_matrix<> &x_r,
                                   const doubles &wt_r, const list &control,
                                   const list &k_list) {
  // Type conversion

  vec y = as_Col(y_r);
  mat X = as_Mat(x_r);
  vec MNU = vec(y.n_elem, fill::zeros);
  vec w = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const double center_tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);

  // Auxiliary variables (storage)

  mat MX, H;

  // Center variables

  MNU += y;
  center_variables_(MNU, w, k_list, center_tol, iter_center_max,
                    iter_interrupt);
  center_variables_(X, w, k_list, center_tol, iter_center_max, iter_interrupt);

  // Solve the normal equations

  vec beta = solve_beta_(X, MNU, w);

  // Fitted values

  vec fitted = y - MNU + X * beta;

  // Recompute Hessian

  H = crossprod_(X, w);

  // Generate result list

  return writable::list({
    "coefficients"_nm = as_doubles(beta),
    "fitted.values"_nm = as_doubles(fitted),
    "weights"_nm = as_doubles(w),
    "hessian"_nm = as_doubles_matrix(H)
  });
}
