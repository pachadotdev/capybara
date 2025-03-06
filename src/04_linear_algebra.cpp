#include "00_main.h"

Mat<double> crossprod_(const Mat<double> &X, const Col<double> &w) {
  Mat<double> Y = X;
  Y.each_col() %= sqrt(w);
  return Y.t() * Y;
}

Col<double> solve_beta_(Mat<double> MX, const Mat<double> &MNU,
                        const Col<double> &w) {
  MX.each_col() %= sqrt(w);

  Mat<double> Q, R;
  if (!qr_econ(Q, R, MX)) {
    stop("QR decomposition failed");
  }

  return solve(R, Q.t() * (MNU.each_col() % sqrt(w)));
}
