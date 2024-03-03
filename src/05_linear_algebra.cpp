// Y <- crossprod(X)
// Y <- t(X) %*% X

#include "00_main.h"

[[cpp11::register]] doubles_matrix<> crossprod_(const doubles_matrix<> &x,
                                                const doubles &w, bool weighted,
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

// WinvJ <- solve_(object[["Hessian"]] / nt.full, J)
// Gamma <- (MX %*% WinvJ - PPsi) * v / nt.full
// V <- crossprod((MX %*% WinvJ - PPsi) * v / nt.full)

[[cpp11::register]] doubles_matrix<> gamma_(const doubles_matrix<> &mx,
                                            const doubles_matrix<> &hessian,
                                            const doubles_matrix<> j,
                                            const doubles_matrix<> &ppsi,
                                            const doubles &v,
                                            const int &nt_full) {
  // Types conversion
  Mat<double> MX = as_Mat(mx);
  Mat<double> H = as_Mat(hessian);
  Mat<double> J = as_Mat(j);
  Mat<double> PPsi = as_Mat(ppsi);
  Mat<double> V = as_Col(v);

  double N = static_cast<double>(nt_full);
  Mat<double> res = ((MX * solve(H / N, J) - PPsi) % V) / N;

  return as_doubles_matrix(res);
}

[[cpp11::register]] doubles_matrix<> chol_crossprod_(
    const doubles_matrix<> &x) {
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

[[cpp11::register]] doubles solve_bias_(const doubles &beta_uncorr,
                                        const doubles_matrix<> &hessian,
                                        const double &nt, const doubles &b) {
  // Types conversion
  Col<double> Beta_uncorr = as_Col(beta_uncorr);
  Mat<double> H = as_Mat(hessian);
  Col<double> B = as_Col(b);

  // Solve
  Col<double> res = Beta_uncorr - solve(H / nt, B);

  return as_doubles(res);
}

[[cpp11::register]] doubles solve_y_(const doubles_matrix<> &a,
                                     const doubles &x) {
  // Types conversion
  Mat<double> A = as_Mat(a);
  Col<double> X = as_Col(x);

  // Solve
  Col<double> res = A * X;

  return as_doubles(res);
}

[[cpp11::register]] doubles_matrix<> sandwich_(const doubles_matrix<> &a,
                                               const doubles_matrix<> &b) {
  // Types conversion
  Mat<double> A = as_Mat(a);
  Mat<double> B = as_Mat(b);

  // Sandwich
  Mat<double> res = (A * B) * A;

  return as_doubles_matrix(res);
}
