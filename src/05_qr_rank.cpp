#include "00_main.h"

[[cpp11::register]] int qr_rank_(const doubles_matrix<> &x) {
  Mat<double> X = as_Mat(x);

  Mat<double> Q;
  Mat<double> R;

  bool computable = qr_econ(Q, R, X);

  if (!computable) {
    stop("QR decomposition failed");
  } else {
    // rank = non-zero diagonal elements
    int rank = sum(R.diag() != 0.0);
    return rank;
  }
}
