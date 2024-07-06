// Compute QR decomposition

list qr_econ_(const doubles_matrix<> &x, bool rank_only) {
  int N = x.nrow();

  writable::list res;

  const double *x_data = REAL(x.data());

  // X = QR
  // Q is orthogonal, R is upper triangular

  // 1: Compute Q

  // Use Gram–Schmidt

  // projection of x onto u = <x, u> / <u, u> * u

  // u1 = x1
  // u2 = x2 - <x2, u1> / <u1, u1> * u1
  // u3 = x3 - <x3, u1> / <u1, u1> * u1 - <x3, u2> / <u2, u2> * u2
  // ...
  // un = xn - <xn, u1> / <u1, u1> * u1 - <xn, u2> / <u2, u2> * u2 - ... - <xn,
  // un-1> / <un-1, un-1> * un-1

  // e1 = u1 / ||u1||
  // e2 = u2 / ||u2||
  // e3 = u3 / ||u3||
  // ...
  // en = un / ||un||

  // x1 = <e1, x1> * e1
  // x2 = <e1, x2> * e1 + <e2, x2> * e2
  // x3 = <e1, x3> * e1 + <e2, x3> * e2 + <e3, x3> * e3
  // ...
  // xn = <e1, xn> * e1 + <e2, xn> * e2 + <e3, xn> * e3 + ... + <en, xn> * en

  // => X = QR
  // with Q = [e1, e2, e3, ...]

  std::vector<double> vec_q(N * N);

  // q = [u1/||u1| u2||u2| u3||u3| ...]

  // q1
  for (int i = 0; i < N; i++) {
    vec_u[i] = x_data[i];
  }

  // q2, q3, ..., qn
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < N; j++) {
      double dot = 0.0;
      for (int k = 0; k < j; k++) {
        dot += vec_u[k * N + i] * vec_u[k * N + i];
      }
      for (int k = 0; k < j; k++) {
        dot += vec_u[k * N + i] * vec_u[k * N + i];
      }
      vec_u[j * N + i] = x_data[j * N + i] - dot;
    }
  }

  // normalize q1, q2, q3, ..., qn
  for (int i = 0; i < N; i++) {
    double norm = 0.0;
    for (int j = 0; j < N; j++) {
      norm += vec_u[j * N + i] * vec_u[j * N + i];
    }
    norm = sqrt(norm);
    if (norm == 0.0) {
      stop("QR decomposition failed");
    }
    for (int j = 0; j < N; j++) {
      vec_q[j * N + i] = vec_u[j * N + i] / norm;
    }
  }

  // 2: Compute R

  // R = Q^T X

  std::vector<double> vec_r(N * N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double dot = 0.0;
      for (int k = 0; k < N; k++) {
        dot += vec_q[k * N + i] * x_data[k * N + j];
      }
      vec_r[i * N + j] = dot;
    }
  }

  if (rank_only) {
    return sum(R.diag() != 0.0);
  }
}
