#include "00_main.h"

[[cpp11::register]] doubles_matrix<>
chol_crossprod_(const doubles_matrix<> &x) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  // Crossprod
  Mat<double> Y = X.t() * X;

  // Cholesky decomposition
  Mat<double> res = chol(Y);

  return as_doubles_matrix(res);
}

// chol2inv(X)

// r comes from a Cholesky decomposition in the R code
// no need to check upper triangularity
[[cpp11::register]] doubles_matrix<> chol2inv_(const doubles_matrix<> &r) {
  // Types conversion
  Mat<double> R = as_Mat(r);

  // (X'X)^(-1) from the R part of the Cholesky decomposition
  Mat<double> res = inv(R.t() * R);

  return as_doubles_matrix(res);
}

// chol(X)

[[cpp11::register]] doubles_matrix<> chol_(const doubles_matrix<> &x) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  // Cholesky decomposition
  Mat<double> res = chol(X);

  return as_doubles_matrix(res);
}
