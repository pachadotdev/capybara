#include "00_main.h"

// Y <- crossprod(X)
// Y <- t(X) %*% X

[[cpp11::register]] doubles_matrix<> crossprod_(const doubles_matrix<> &x,
                                                const doubles &w,
                                                const bool &weighted,
                                                const bool &root_weights) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  if (weighted) {
    // Additional type conversion
    Mat<double> W = as_Mat(w);

    if (root_weights) {
      W = sqrt(W);
    }

    // Multiply each column of X by W pair-wise
    X = X.each_col() % W;
  }

  Mat<double> res = X.t() * X;
  return as_doubles_matrix(res);
}

// WinvJ < -solve(object[["Hessian"]] / nt.full, J)
// Gamma < -(MX %*% WinvJ - PPsi) * v / nt.full
// V < -crossprod(Gamma)

[[cpp11::register]] doubles_matrix<>
gamma_(const doubles_matrix<> &mx, const doubles_matrix<> &hessian,
       const doubles_matrix<> &j, const doubles_matrix<> &ppsi,
       const doubles &v, const SEXP &nt_full) {
  // Types conversion
  Mat<double> MX = as_Mat(mx);
  Mat<double> H = as_Mat(hessian);
  Mat<double> J = as_Mat(j);
  Mat<double> PPsi = as_Mat(ppsi);
  Mat<double> V = as_Mat(v);

  double N = as_cpp<double>(nt_full);

  Mat<double> res = (MX * solve(H / N, J)) - PPsi;
  res = (res.each_col() % V) / N;
  return as_doubles_matrix(res);
}

// chol(crossprod(X))

[[cpp11::register]] doubles_matrix<>
chol_crossprod_(const doubles_matrix<> &x) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  // Crossprod + Cholesky decomposition
  Mat<double> res = chol(X.t() * X);
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

// qr(X)$rank

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

// Beta_uncorr - solve(H / nt, B)

[[cpp11::register]] doubles solve_bias_(const doubles &beta_uncorr,
                                        const doubles_matrix<> &hessian,
                                        const double &nt, const doubles &b) {
  // Types conversion
  Mat<double> Beta_uncorr = as_Mat(beta_uncorr);
  Mat<double> H = as_Mat(hessian);
  Mat<double> B = as_Mat(b);

  // Solve
  return as_doubles(Beta_uncorr - solve(H / nt, B));
}

// A %*% x

[[cpp11::register]] doubles solve_y_(const doubles_matrix<> &a,
                                     const doubles &x) {
  // Types conversion
  Mat<double> A = as_Mat(a);
  Mat<double> X = as_Mat(x);

  // Solve
  return as_doubles(A * X);
}

// A %*% B %*% A

[[cpp11::register]] doubles_matrix<> sandwich_(const doubles_matrix<> &a,
                                               const doubles_matrix<> &b) {
  // Types conversion
  Mat<double> A = as_Mat(a);
  Mat<double> B = as_Mat(b);

  // Sandwich

  Mat<double> res = A * B * A;
  return as_doubles_matrix(res);
}

// eta <- eta.old + rho * eta.upd

[[cpp11::register]] doubles
update_beta_eta_(const doubles &old, const doubles &upd, const double &param) {
  int n = old.size();
  writable::doubles res(n);

  double *old_data = REAL(old);
  double *upd_data = REAL(upd);

  for (int i = 0; i < n; ++i) {
    res[i] = old_data[i] + (upd_data[i] * param);
  }

  return res;
}

// nu <- (y - mu) / mu.eta

[[cpp11::register]] doubles update_nu_(const SEXP &y, const SEXP &mu,
                                       const SEXP &mu_eta) {
  int n = Rf_length(y);
  writable::doubles res(n);

  double *y_data = REAL(y);
  double *mu_data = REAL(mu);
  double *mu_eta_data = REAL(mu_eta);

  for (int i = 0; i < n; ++i) {
    res[i] = (y_data[i] - mu_data[i]) / mu_eta_data[i];
  }

  return res;
}

[[cpp11::register]] doubles solve_beta_(const doubles_matrix<> &mx,
                                        const doubles_matrix<> &mnu,
                                        const doubles &wtilde,
                                        const double &epsilon,
                                        const bool &weighted) {
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
    return as_doubles(solve(R, Q.t() * Y));
  }
}

// eta.upd <- nu - as.vector(Mnu - MX %*% beta.upd)

[[cpp11::register]] doubles solve_eta_(const doubles_matrix<> &mx,
                                       const doubles_matrix<> &mnu,
                                       const doubles &nu, const doubles &beta) {
  // Types conversion
  Mat<double> MX = as_Mat(mx);
  Mat<double> Mnu = as_Mat(mnu);
  Mat<double> Nu = as_Mat(nu);
  Mat<double> Beta = as_Mat(beta);

  return as_doubles(Nu - (Mnu - (MX * Beta)));
}

// eta.upd <- yadj - as.vector(Myadj) + offset - eta

[[cpp11::register]] doubles solve_eta2_(const doubles &yadj,
                                        const doubles_matrix<> &myadj,
                                        const doubles &offset,
                                        const doubles &eta) {
  // Types conversion
  int n = yadj.size();
  writable::doubles res(n);
  for (int i = 0; i < n; ++i) {
    res[i] = yadj[i] - myadj(i, 0) + offset[i] - eta[i];
  }
  return res;
}

// w <- sqrt(w)

[[cpp11::register]] doubles sqrt_(const doubles &w) {
  // Types conversion
  int n = w.size();
  writable::doubles res(n);
  for (int i = 0; i < n; ++i) {
    res[i] = sqrt(w[i]);
  }
  return res;
}
