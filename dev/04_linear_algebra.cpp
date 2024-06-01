#include "00_main.h"

// helper to convert std::vector<double> to doubles_matrix<>
writable::doubles_matrix<> stdvec_to_dblmatrix_(const std::vector<double> &vec,
                                                int rows, int cols) {
  writable::doubles_matrix<> mat(rows, cols);

#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat(i, j) = vec[i * cols + j];
    }
  }
  return mat;
}

// Y <- crossprod(X)
// Y <- t(X) %*% X

[[cpp11::register]] doubles_matrix<> crossprod_(const doubles_matrix<> &x,
                                                const doubles &w,
                                                const bool &weighted,
                                                const bool &root_weights) {
  int N = x.nrow();
  int P = x.ncol();

  const double *x_data = REAL(x.data());
  const double *w_data = REAL(w);

  std::vector<double> res_vec(P * P, 0.0);

  if (!weighted) {
    // Multiply X.t() by X
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      for (int p = 0; p < P; p++) {
        for (int q = 0; q < P; q++) {
          double temp = x_data[n + N * p] * x_data[n + N * q];
#pragma omp atomic
          res_vec[p * P + q] += temp;
        }
      }
    }
  } else {
    std::vector<double> weights(N);

    if (!root_weights) {
#pragma omp parallel for
      for (int n = 0; n < N; n++) {
        weights[n] = w_data[n];
      }
    } else {
#pragma omp parallel for
      for (int n = 0; n < N; n++) {
        weights[n] = sqrt(w_data[n]);
      }
    }

    // Multiply weighted X.t() by weighted X
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      for (int p = 0; p < P; p++) {
        for (int q = 0; q < P; q++) {
          double temp =
              x_data[n + N * p] * x_data[n + N * q] * pow(weights[n], 2);
#pragma omp atomic
          res_vec[p * P + q] += temp;
        }
      }
    }
  }

  // writable::doubles_matrix<> res(P, P);

  // #pragma omp parallel for
  // for (int p = 0; p < P; p++) {
  //   for (int q = 0; q < P; q++) {
  //     res(p, q) = res_vec[p * P + q];
  //   }
  // }

  // return res;

  return stdvec_to_dblmatrix_(res_vec, P, P);
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

  double inv_N = 1.0 / as_cpp<double>(nt_full);

  Mat<double> res = (MX * solve(H * inv_N, J)) - PPsi;
  res = res.each_col() % V;
  res *= inv_N;
  return as_doubles_matrix(res);
}

// chol(crossprod(X))

[[cpp11::register]] doubles_matrix<>
chol_crossprod_(const doubles_matrix<> &x) {
  // Types conversion
  Mat<double> X = as_Mat(x);

  // Cholesky decomposition of X'X
  Mat<double> res = chol(X, "upper");
  return as_doubles_matrix(res);
}

// chol2inv(X)
// r comes from a Cholesky decomposition in the R code
// no need to check upper triangularity

[[cpp11::register]] doubles_matrix<> chol2inv_(const doubles_matrix<> &r) {
  // Types conversion
  Mat<double> R = as_Mat(r);

  // (X'X)^(-1) from the R part of the Cholesky decomposition
  Mat<double> res = inv_sympd(R.t() * R);
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

  Mat<double> Q, R;

  bool computable = qr_econ(Q, R, X);

  if (!computable) {
    stop("QR decomposition failed");
  } else {
    // rank = non-zero diagonal elements
    return sum(R.diag() != 0.0);
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

  double inv_nt = 1.0 / nt;

  // Solve
  return as_doubles(Beta_uncorr - solve(H * inv_nt, B));
}

// A %*% x

[[cpp11::register]] doubles solve_y_(const doubles_matrix<> &a,
                                     const doubles &x) {
  int N = a.nrow();
  int P = a.ncol();

  writable::doubles res(N);

  const double *a_data = REAL(a.data());
  const double *x_data = REAL(x);

// Perform matrix multiplication
#pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    double sum = 0.0;
    for (int p = 0; p < P; ++p) {
      sum += a_data[n + N * p] * x_data[p];
    }
    res[n] = sum;
  }

  return res;
}

// A %*% B %*% A

[[cpp11::register]] doubles_matrix<> sandwich_(const doubles_matrix<> &a,
                                               const doubles_matrix<> &b) {
  int N = a.nrow();
  int P = a.ncol();

  writable::doubles_matrix<> res(N, N);

  const double *a_data = REAL(a.data());
  const double *b_data = REAL(b.data());

  // Compute the matrix product (A %*% B) %*% A in a single set of nested loops
#pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < N; ++m) {
      double sum = 0.0;
      for (int p = 0; p < P; ++p) {
        for (int q = 0; q < P; ++q) {
          sum += a_data[n + N * p] * b_data[p + P * q] * a_data[q + N * m];
        }
      }
      res(n, m) = sum;
    }
  }

  return res;
}

// eta <- eta.old + rho * eta.upd

[[cpp11::register]] doubles
update_beta_eta_(const doubles &old, const doubles &upd, const double &param) {
  int N = old.size();
  writable::doubles res(N);

  double *old_data = REAL(old);
  double *upd_data = REAL(upd);

#pragma omp parallel for
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
  int N = mx.nrow();
  int P = mx.ncol();
  int n, p;
  double sum;

  writable::doubles res(N);

  const double *mx_data = REAL(mx.data());
  const double *mnu_data = REAL(mnu.data());
  const double *nu_data = REAL(nu);
  const double *beta_data = REAL(beta);

  // Matrix multiplication and subtraction
  for (n = 0; n < N; ++n) {
    sum = 0.0;
    for (p = 0; p < P; ++p) {
      sum += mx_data[n + N * p] * beta_data[p];
    }
    res[n] = nu_data[n] - (mnu_data[n] - sum);
  }

  return res;
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

#pragma omp parallel for
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

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    res[i] = sqrt(w_data[i]);
  }

  return res;
}
