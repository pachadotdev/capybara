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
    w = sqrt(w);
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

[[cpp11::register]] doubles solve_eta2_(const doubles &yadj, const doubles_matrix<> &myadj,
                                        const doubles &offset, const doubles &eta) {
  Mat<double> Yadj = as_Mat(yadj);
  Mat<double> Myadj = as_Mat(myadj);
  Mat<double> Offset = as_Mat(offset);
  Mat<double> Eta = as_Mat(eta);

  return as_doubles(Yadj - Myadj + Offset - Eta);
}

std::string tidy_family(const std::string &family) {
  // tidy family param
  std::string fam = family;

  // 1. put all in lowercase
  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // 2. remove numbers
  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  // 3. remove parentheses and everything inside
  size_t pos = fam.find("(");
  if (pos != std::string::npos) {
    fam.erase(pos, fam.size());
  }

  // 4. replace spaces and dots
  std::replace(fam.begin(), fam.end(), ' ', '_');
  std::replace(fam.begin(), fam.end(), '.', '_');

  // 5. trim
  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isspace), fam.end());

  return fam;
}

[[cpp11::register]] doubles linkinv_(const doubles &eta_r,
                                     const std::string &family) {
  Col<double> eta = as_Col(eta_r);
  Col<double> res(eta.n_elem);
  
  std::string fam = tidy_family(family);

  if (fam == "gaussian") {
    res = eta;
  } else if (fam == "poisson") {
    res = exp(eta);
  } else if (fam == "binomial") {
    // res = exp(eta) / (1.0 + exp(eta));
    res = 1.0 / (1.0 + exp(-eta));
  } else if (fam == "gamma") {
    res = 1.0 / eta;
  } else if (fam == "inverse_gaussian") {
    res = 1.0 / sqrt(eta);
  } else if (fam == "negative_binomial") {
    res = exp(eta);
  } else {
    stop("Unknown family");
  }

  return as_doubles(res);
}

[[cpp11::register]] double dev_resids_(const doubles &y_r, const doubles &mu_r,
                                       const double &theta, const doubles &wt_r,
                                       const std::string &family) {
  Col<double> y = as_Col(y_r);
  Col<double> mu = as_Col(mu_r);
  Col<double> wt = as_Col(wt_r);
  double res;

  std::string fam = tidy_family(family);

  if (fam == "gaussian") {
    res = accu(wt % square(y - mu));
  } else if (fam == "poisson") {
    uvec p = find(y > 0.0);
    Col<double> r = mu % wt;
    r(p) = y(p) % log(y(p) / mu(p)) - (y(p) - mu(p));
    res = 2.0 * accu(r);
  } else if (fam == "binomial") {
    uvec p = find(y != 0.0);
    uvec q = find(y != 1.0);
    Col<double> r = y / mu;
    Col<double> s = (1.0 - y) / (1.0 - mu);
    r(p) = log(r(p));
    s(q) = log(s(q));
    res = 2.0 * accu(wt % (y % r + (1.0 - y) % s));
  } else if (fam == "gamma") {
    uvec p = find(y == 0.0);
    Col<double> r = y / mu;
    r.elem(p).fill(1.0);
    res = -2.0 * accu(wt % (log(r) - (y - mu) / mu));
  } else if (fam == "inverse_gaussian") {
    res = accu(wt % square(y - mu) / (y % square(mu)));
  } else if (fam == "negative_binomial") {
    uvec p = find(y < 1.0);
    Col<double> r = y;
    r.elem(p).fill(1.0);
    res = 2.0 * accu(
        wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta))));
  } else {
    stop("Unknown family");
  }

  return res;
}
