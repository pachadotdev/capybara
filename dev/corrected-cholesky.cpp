// Corrected rank-revealing Cholesky implementation
#include <armadillo>
using namespace arma;

//////////////////////////////////////////////////////////////////////////////
// CORRECTED MATRIX SOLVERS
//////////////////////////////////////////////////////////////////////////////

// Pivoted Cholesky decomposition for rank detection
inline bool pivoted_cholesky(mat &L, uvec &perm, uvec &excluded, const mat &XtX,
                             double tol) {
  const size_t p = XtX.n_cols;
  L.zeros(p, p);
  perm = regspace<uvec>(0, p - 1);
  excluded.zeros(p);

  mat A = XtX; // Working copy
  vec diag_vals = A.diag();

  size_t rank = 0;

  for (size_t k = 0; k < p; ++k) {
    // Find pivot (largest remaining diagonal element)
    uvec remaining = regspace<uvec>(k, p - 1);
    vec remaining_diag = diag_vals.elem(remaining);

    uword max_idx;
    double max_val = remaining_diag.max(max_idx);
    max_idx += k; // Adjust for offset

    // Check for rank deficiency
    if (max_val < tol) {
      // Mark remaining variables as excluded
      for (size_t j = k; j < p; ++j) {
        excluded(perm(j)) = 1;
      }
      break;
    }

    // Swap columns/rows if needed
    if (max_idx != k) {
      A.swap_cols(k, max_idx);
      A.swap_rows(k, max_idx);
      std::swap(perm(k), perm(max_idx));
      std::swap(diag_vals(k), diag_vals(max_idx));
    }

    // Compute L(k,k)
    L(k, k) = std::sqrt(A(k, k));
    rank++;

    // Update column k of L
    if (k < p - 1) {
      vec L_col = A.submat(k + 1, k, p - 1, k) / L(k, k);
      L.submat(k + 1, k, p - 1, k) = L_col;

      // Update remaining submatrix
      for (size_t i = k + 1; i < p; ++i) {
        for (size_t j = k + 1; j <= i; ++j) {
          A(i, j) -= L(i, k) * L(j, k);
        }
        diag_vals(i) = A(i, i);
      }
    }
  }

  return rank > 0;
}

// Forward substitution for lower triangular system Lx = b
inline vec solve_lower_triangular(const mat &L, const vec &b,
                                  const uvec &valid_cols) {
  const size_t p = L.n_cols;
  vec x(p, fill::zeros);

  for (size_t i = 0; i < valid_cols.n_elem; ++i) {
    size_t idx = valid_cols(i);
    if (idx >= p)
      continue;

    double sum = b(idx);
    for (size_t j = 0; j < i; ++j) {
      size_t prev_idx = valid_cols(j);
      if (prev_idx < p) {
        sum -= L(idx, prev_idx) * x(prev_idx);
      }
    }

    if (std::abs(L(idx, idx)) > 1e-14) {
      x(idx) = sum / L(idx, idx);
    }
  }

  return x;
}

// Alternative: Use modified Gram-Schmidt for better numerical stability
inline bool modified_gram_schmidt(mat &Q, mat &R, uvec &excluded, const mat &X,
                                  double tol) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;

  Q = X; // Start with copy of X
  R.zeros(p, p);
  excluded.zeros(p);

  for (size_t j = 0; j < p; ++j) {
    // Compute R(j,j) = ||Q(:,j)||
    R(j, j) = norm(Q.col(j));

    // Check for linear dependence
    if (R(j, j) < tol) {
      excluded(j) = 1;
      continue;
    }

    // Normalize Q(:,j)
    Q.col(j) /= R(j, j);

    // Orthogonalize remaining columns
    for (size_t k = j + 1; k < p; ++k) {
      if (excluded(k))
        continue;

      R(j, k) = dot(Q.col(j), Q.col(k));
      Q.col(k) -= R(j, k) * Q.col(j);
    }
  }

  return true;
}

// Robust solver using the above methods
inline vec robust_solve(const mat &XtX, const vec &XtY, double tol,
                        uvec &excluded) {
  const size_t p = XtX.n_cols;
  vec coefficients(p, fill::none);
  coefficients.fill(datum::nan);

  // Try pivoted Cholesky first
  mat L;
  uvec perm;

  if (pivoted_cholesky(L, perm, excluded, XtX, tol)) {
    // Permute XtY according to pivoting
    vec b_perm = XtY.elem(perm);

    // Find valid (non-excluded) variables
    uvec valid = find(excluded == 0);

    if (valid.n_elem > 0) {
      // Solve LL^T x = b using forward and backward substitution
      vec y = solve_lower_triangular(L, b_perm, valid);
      vec x_perm = solve_lower_triangular(L.t(), y, valid);

      // Unpermute the solution
      coefficients.elem(perm.elem(valid)) = x_perm.elem(valid);
    }
  }

  return coefficients;
}

//////////////////////////////////////////////////////////////////////////////
// COMPARISON WITH WORKING APPROACH
//////////////////////////////////////////////////////////////////////////////

// Your working QR approach (for reference)
inline vec qr_solve(const mat &X, const vec &y, double tol, uvec &excluded) {
  const size_t p = X.n_cols;

  mat Q, R;
  qr_econ(Q, R, X);

  vec QTy = Q.t() * y;

  const vec diag_abs = abs(R.diag());
  const double max_diag = diag_abs.max();
  const double effective_tol = tol * max_diag;
  const uvec indep = find(diag_abs > effective_tol);

  excluded.ones(p);
  excluded.elem(indep).zeros();

  vec coefficients(p, fill::none);
  coefficients.fill(datum::nan);

  if (indep.n_elem == p) {
    coefficients = solve(trimatu(R), QTy, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = R.submat(indep, indep);
    const vec Yr = QTy.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    coefficients.elem(indep) = br;
  }

  return coefficients;
}
