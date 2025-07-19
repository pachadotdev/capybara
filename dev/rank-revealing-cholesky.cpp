//////////////////////////////////////////////////////////////////////////////
// MATRIX SOLVERS
//////////////////////////////////////////////////////////////////////////////

// Rank-revealing Cholesky decomposition
inline bool cholesky_decomp(mat &R, uvec &excluded, const mat &XtX,
                            double tol) {
  const size_t p = XtX.n_cols;
  R.zeros(p, p);
  excluded.zeros(p);

  size_t n_excluded = 0;

  for (size_t j = 0; j < p; ++j) {
    // Diagonal elements
    double R_jj = XtX(j, j);

    // Sum of squares
    if (j > 0) {
      const rowvec R_row = R.submat(0, j, j - 1, j);
      R_jj -= dot(R_row, R_row);
    }

    // Check for collinearity
    if (R_jj < tol) {
      excluded(j) = 1;
      ++n_excluded;
      if (n_excluded == p) {
        return false; // All variables excluded
      }
      continue;
    }

    R_jj = std::sqrt(R_jj);
    R(j, j) = R_jj;

    // Off-diagonal elements
    if (j < p - 1) {
      vec values = XtX.submat(j + 1, j, p - 1, j);

      if (j > 0) {
        const mat R_prev = R.submat(0, j + 1, j - 1, p - 1);
        const vec R_j_prev = R.submat(0, j, j - 1, j);
        values -= R_prev.t() * R_j_prev;
      }

      values /= R_jj;
      R.submat(j, j + 1, j, p - 1) = values.t();
    }
  }

  return true;
}

// Forward/backward substitution solver
inline vec solve_triangular(const mat &R, const vec &b, bool upper = true) {
  const size_t n = R.n_cols;
  vec x(n, fill::none);

  if (upper) {
    // Backward substitution
    for (size_t i = n; i-- > 0;) {
      double sum = b(i);
      for (size_t j = i + 1; j < n; ++j) {
        sum -= R(i, j) * x(j);
      }
      x(i) = sum / R(i, i);
    }
  } else {
    // Forward substitution
    for (size_t i = 0; i < n; ++i) {
      double sum = b(i);
      for (size_t j = 0; j < i; ++j) {
        sum -= R(i, j) * x(j);
      }
      x(i) = sum / R(i, i);
    }
  }

  return x;
}
