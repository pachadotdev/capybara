#include "00_main.h"

// Y <- crossprod(X)
// Y <- t(X) %*% X

[[cpp11::register]] doubles_matrix<> crossprod_(const doubles_matrix<> &x,
                                                const doubles &w,
                                                const bool &weighted,
                                                const bool &root_weights) {
  Mat<double> X = as_Mat(x);
  int P = X.n_cols;

  Mat<double> res(P, P);

  if (!weighted) {
    res = X.t() * X;
  } else {
    Mat<double> W = as_Mat(w);

    if (root_weights) {
      W = sqrt(W);
    }

    X = X.each_col() % W;

    res = X.t() * X;
  }

  return as_doubles_matrix(res);
}

// WinvJ < -solve(object[["Hessian"]] / nt.full, J)
// Gamma < -(MX %*% WinvJ - PPsi) * v / nt.full
// V < -crossprod(Gamma)

[[cpp11::register]] doubles_matrix<>
gamma_(const doubles_matrix<> &mx, const doubles_matrix<> &hessian,
       const doubles_matrix<> &j, const doubles_matrix<> &ppsi,
       const doubles &v, const SEXP &nt_full) {
  Mat<double> MX = as_Mat(mx);
  Mat<double> H = as_Mat(hessian);
  Mat<double> J = as_Mat(j);
  Mat<double> PPsi = as_Mat(ppsi);
  Mat<double> V = as_Mat(v);

  double inv_N = 1.0 / as_cpp<double>(nt_full);
  
  Mat<double> res = (MX * solve(H * inv_N, J)) - PPsi;
  res = (res.each_col() % V) * inv_N;

  return as_doubles_matrix(res);
}

// solve(H)

[[cpp11::register]] doubles_matrix<> inv_(const doubles_matrix<> &h) {
  Mat<double> H = inv(as_Mat(h));
  return as_doubles_matrix(H);
}

// qr(X)$rank

[[cpp11::register]] int rank_(const doubles_matrix<> &x) {
  Mat<double> X = as_Mat(x);
  return arma::rank(X); // SVD
}

// Beta_uncorr - solve(H / nt, B)

[[cpp11::register]] doubles solve_bias_(const doubles &beta_uncorr,
                                        const doubles_matrix<> &hessian,
                                        const double &nt, const doubles &b) {
  Mat<double> Beta_uncorr = as_Mat(beta_uncorr);
  Mat<double> H = as_Mat(hessian);
  Mat<double> B = as_Mat(b);

  double inv_nt = 1.0 / nt;

  return as_doubles(Beta_uncorr - solve(H * inv_nt, B));
}

// A %*% x

[[cpp11::register]] doubles solve_y_(const doubles_matrix<> &a,
                                     const doubles &x) {
  Mat<double> A = as_Mat(a);
  Mat<double> X = as_Mat(x);

  return as_doubles(A * X);
}

// A %*% B %*% A

[[cpp11::register]] doubles_matrix<> sandwich_(const doubles_matrix<> &a,
                                               const doubles_matrix<> &b) {
  Mat<double> A = as_Mat(a);
  Mat<double> B = as_Mat(b);

  Mat<double> res = A * (B * A);
  return as_doubles_matrix(res);
}

// eta <- eta.old + rho * eta.upd

[[cpp11::register]] doubles
update_beta_eta_(const doubles &old, const doubles &upd, const double &param) {
  int N = old.size();
  writable::doubles res(N);

  double *old_data = REAL(old);
  double *upd_data = REAL(upd);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int n = 0; n < N; ++n) {
    res[n] = old_data[n] + (upd_data[n] * param);
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
                                        const bool &weighted) {
  Mat<double> X = as_Mat(mx);
  Mat<double> Y = as_Mat(mnu);

  // Weight the X and Y matrices
  if (weighted) {
    Mat<double> w = as_Mat(wtilde);
    X = X.each_col() % w; // element-wise multiplication
    Y = Y.each_col() % w;
  }

  // Solve the system X * beta = Y

  // QR decomposition

  Mat<double> Q, R;

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
  Mat<double> MX = as_Mat(mx);
  Mat<double> MNU = as_Mat(mnu);
  Mat<double> Nu = as_Mat(nu);
  Mat<double> Beta = as_Mat(beta);

  return as_doubles(Nu - (MNU - MX * Beta));
}

// eta.upd <- yadj - as.vector(Myadj) + offset - eta

[[cpp11::register]] doubles solve_eta2_(const SEXP &yadj, const SEXP &myadj,
                                        const SEXP &offset, const SEXP &eta) {
  int N = Rf_length(yadj);
  writable::doubles res(N);

  double *Yadj_data = REAL(yadj);
  double *Myadj_data = REAL(myadj);
  double *Offset_data = REAL(offset);
  double *Eta_data = REAL(eta);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int n = 0; n < N; ++n) {
    res[n] = Yadj_data[n] - Myadj_data[n] + Offset_data[n] - Eta_data[n];
  }

  return res;
}

// w <- sqrt(w)

[[cpp11::register]] doubles sqrt_(const SEXP &w) {
  int n = Rf_length(w);
  writable::doubles res(n);

  double *w_data = REAL(w);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    res[i] = sqrt(w_data[i]);
  }

  return res;
}
