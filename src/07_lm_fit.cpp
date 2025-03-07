#include "00_main.h"

[[cpp11::register]] list felm_fit_(const doubles &y_r,
                                   const doubles_matrix<> &x_r,
                                   const doubles &wt_r, const list &control,
                                   const list &k_list) {
  // Type conversion

  Col<double> y = as_Col(y_r);
  Mat<double> X = as_Mat(x_r);
  Col<double> MNU = Col<double>(y.n_elem, fill::zeros);
  Col<double> w = as_Col(wt_r);

  // Auxiliary variables (fixed)

  double center_tol = as_cpp<double>(control["center_tol"]);
  int iter_center_max = 10000;

  // Auxiliary variables (storage)

  Mat<double> MX, H;

  // Center variables

  MNU += y;
  center_variables_(MNU, w, k_list, center_tol, iter_center_max);
  center_variables_(X, w, k_list, center_tol, iter_center_max);

  // Solve the normal equations

  Col<double> beta = solve_beta_(X, MNU, w);

  // Fitted values

  Col<double> fitted = y - MNU + X * beta;

  // Recompute Hessian

  H = crossprod_(X, w);

  // Generate result list

  writable::list out(4);
  out[0] = as_doubles(beta);
  out[1] = as_doubles(fitted);
  out[2] = as_doubles(w);
  out[3] = as_doubles_matrix(H);
  out.attr("names") = writable::strings(
      {"coefficients", "fitted.values", "weights", "hessian"});

  return out;
}
