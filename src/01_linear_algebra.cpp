#include "00_main.h"

// Check if the rank of R is less than p
// Demmel Ch. 3: If m >> n, QR and SVD have similar cost. Otherwise, QR is a bit
// cheaper.
[[cpp11::register]] int check_linear_dependence_svd_(const doubles &y,
                                                     const doubles_matrix<> &x,
                                                     const int &p) {
  mat Y = as_mat(y);
  mat X = as_mat(x);
  X = join_rows(Y, X); // paste y and x together
  int r = rank(X);
  if (r < p) {
    return 1;
  }
  return 0;
}

mat crossprod_(const mat &X, const vec &w) {
  mat Y = X;
  Y.each_col() %= sqrt(w);
  return Y.t() * Y;
}

// replacement: QR to Cholesky (see below)
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
