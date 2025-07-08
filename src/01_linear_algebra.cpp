#include "00_main.h"

// Check if the rank of R is less than p
// Demmel Ch. 3: If m >> n, QR and SVD have similar cost. Otherwise, QR is a bit
// cheaper.
// Armadillo's rank() uses SVD, here we count non-zero pivots with an econ-QR
[[cpp11::register]] int check_linear_dependence_qr_(const doubles &y,
                                                    const doubles_matrix<> &x,
                                                    const int &p) {
  mat X = as_mat(x);
  X = join_rows(X, as_mat(y));

  mat Q, R;
  if (!qr_econ(Q, R, X)) {
    stop("QR decomposition failed");
  }

  double tol_qr = std::numeric_limits<double>::epsilon() *
                  std::max(X.n_rows, X.n_cols) * norm(R, "inf");
  int r = accu(arma::abs(diagvec(R)) > tol_qr);

  return (r < p) ? 1 : 0;
}

mat crossprod_(const mat &X, const vec &w) { return X.t() * diagmat(w) * X; }

// Cholesky decomposition
vec solve_beta_(mat &MX, const mat &MNU, const vec &w) {
  mat MXW = MX.t() * diagmat(w);
  mat XtX = MXW * MX;
  vec XtY = MXW * MNU;

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
