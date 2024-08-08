#include "00_main.h"

// WinvJ < -solve(object[["Hessian"]] / nt.full, J)
// Gamma < -(MX %*% WinvJ - PPsi) * v / nt.full

// [[cpp11::register]] doubles_matrix<>
// gamma_(const doubles_matrix<> &mx, const doubles_matrix<> &hessian,
//        const doubles_matrix<> &j, const doubles_matrix<> &ppsi,
//        const doubles &v, const SEXP &nt_full) {
//   double inv_N = 1.0 / as_cpp<double>(nt_full);

//   Mat<double> res =
//       (as_Mat(mx) * solve(as_Mat(hessian) * inv_N, as_Mat(j))) -
//       as_Mat(ppsi);
//   res = (res.each_col() % as_Mat(v)) * inv_N;

//   return as_doubles_matrix(res);
// }

// solve(H)

// [[cpp11::register]] doubles_matrix<> inv_(const doubles_matrix<> &h) {
//   Mat<double> H = inv(as_Mat(h));
//   return as_doubles_matrix(H);
// }

// qr(X)$rank

// [[cpp11::register]] int rank_(const doubles_matrix<> &x) {
//   return arma::rank(as_Mat(x));  // SVD
// }

// Y <- crossprod(X)
// Y <- t(X) %*% X

Mat<double> crossprod_(const Mat<double> &X, const Col<double> &w, const int &n,
                       const int &p, const bool &weighted,
                       const bool &root_weights) {
  Mat<double> res(p, p);

  if (weighted == false) {
    res = X.t() * X;
  } else {
    Mat<double> Y(n, p);
    if (root_weights == false) {
      Y = X.each_col() % w;
    } else {
      Y = X.each_col() % sqrt(w);
    }
    res = Y.t() * Y;
  }

  return res;
}

Col<double> solve_beta_(const Mat<double> &MX, const Mat<double> &MNU,
                        const Col<double> &w) {
  Col<double> wtilde = sqrt(w);

  Mat<double> Q, R;

  bool computable = qr_econ(Q, R, MX.each_col() % wtilde);

  if (!computable) {
    stop("QR decomposition failed");
  }

  return solve(R, Q.t() * (MNU.each_col() % wtilde));
}
