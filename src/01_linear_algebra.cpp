#include "00_main.h"

mat crossprod_(const mat &X, const vec &w) {
  mat Y = X;
  Y.each_col() %= sqrt(w);
  return Y.t() * Y;
}

// vec solve_beta_(mat MX, const mat &MNU,
//                         const vec &w) {
//   const vec sqrt_w = sqrt(w);
//   MX.each_col() %= sqrt_w;

//   mat Q, R;
//   if (!qr_econ(Q, R, MX)) {
//     stop("QR decomposition failed");
//   }

//   return solve(trimatu(R), Q.t() * (MNU.each_col() % sqrt_w),
//   solve_opts::fast);
// }

// Cholesky decomposition
vec solve_beta_(mat MX, const mat &MNU, const vec &w) {
  const vec sqrt_w = sqrt(w);

  MX.each_col() %= sqrt_w;
  mat WMNU = MNU.each_col() % sqrt_w;

  mat XtX = MX.t() * MX;
  vec XtY = MX.t() * (MNU.each_col() % sqrt_w);

  // XtX = L * L.t()
  mat L;
  if (!chol(L, XtX, "lower")) {
    stop("Cholesky decomposition failed.");
  }

  // Solve L * z = Xty
  vec z = solve(trimatl(L), XtY, solve_opts::fast);

  // Solve Lt * beta = z
  return solve(trimatu(L.t()), z, solve_opts::fast);
}
