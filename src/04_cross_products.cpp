// Y <- crossprod(X)
// Y <- t(X) %*% X

#include "00_main.h"

[[cpp11::register]] doubles_matrix<> crossprod_(const doubles_matrix<>& x,
                                                const doubles& w, bool weighted,
                                                bool root_weights) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  if (weighted) {
    // Additional type conversion
    Col<double> W = as_Col(w);

    if (root_weights) {
      W = sqrt(W);
    }

    // Multiply each column of X by W pair-wise
    X = X.each_col() % W;
  }

  Mat<double> Y = X.t() * X;

  return as_doubles_matrix(Y);
}

// chol(crossprod(X))

[[cpp11::register]] doubles_matrix<> chol_crossprod_(
    const doubles_matrix<>& x) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  // Crossprod
  Mat<double> Y = X.t() * X;

  // Cholesky decomposition
  Mat<double> res;
  res = chol(Y);

  return as_doubles_matrix(res);
}
