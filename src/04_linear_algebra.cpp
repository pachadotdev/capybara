#include "00_main.h"

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
