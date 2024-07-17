#include "00_main.h"

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

Col<double> solve_beta_(const Mat<double> &MX, const Mat<double> &MNU,
                                 const Col<double> &w) {
  Col<double> wtilde = sqrt(w);

  Mat<double> Q, R;

  bool computable = qr_econ(Q, R, MX.each_col() % wtilde);

  if (!computable) {
    stop("QR decomposition failed");
  } else {
    // backsolve
    return solve(R, Q.t() * (MNU.each_col() % wtilde));
  }
}

// eta.upd <- nu - as.vector(Mnu - MX %*% beta.upd)

Col<double> solve_eta_(const Mat<double> &MX, const Mat<double> &MNU,
                        const Col<double> &nu, const Col<double> &beta) {
  return nu - MNU + MX * beta;
}

// eta.upd <- yadj - as.vector(Myadj) + offset - eta

[[cpp11::register]] doubles solve_eta2_(const doubles &yadj, const doubles_matrix<> &myadj,
                                        const doubles &offset, const doubles &eta) {
  Mat<double> Yadj = as_Mat(yadj);
  Mat<double> Myadj = as_Mat(myadj);
  Mat<double> Offset = as_Mat(offset);
  Mat<double> Eta = as_Mat(eta);

  return as_doubles(Yadj - Myadj + Offset - Eta);
}
