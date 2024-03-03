#include "00_main.h"

[[cpp11::register]] doubles solve_beta_(const doubles_matrix<> &mx,
                                        const doubles_matrix<> &mnu,
                                        const doubles wtilde, double epsilon,
                                        bool weighted) {
  // Types conversion
  Mat<double> X = as_Mat(mx);
  Mat<double> Y = as_Mat(mnu);

  // Weight the X and Y matrices
  if (weighted) {
    // Additional type conversion
    Mat<double> W = as_Mat(wtilde);

    // Multiply each column of X by W pair-wise
    X = X.each_col() % W;

    // Multiply each column of Y by W pair-wise
    Y = Y.each_col() % W;
  }

  // Now we need to solve the system X * beta = Y
  // We proceed with the Economic QR

  Mat<double> Q;
  Mat<double> R;

  bool computable = qr_econ(Q, R, X);

  if (!computable) {
    stop("QR decomposition failed");
  } else {
    // backsolve
    Mat<double> beta = solve(R, Q.t() * Y);
    return as_doubles(beta);
  }
}
