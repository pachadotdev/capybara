#include "00_main.hpp"

Mat<double> doubles_matrix_to_Mat_(const doubles_matrix<>& A) {
  int nrows = A.nrow();
  int ncols = A.ncol();

  Mat<double> B(nrows, ncols);

  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      double a_ij = A(i, j);
      B(i, j) = a_ij;  // Assuming you can access elements of A in this way
    }
  }

  return B;
}

doubles_matrix<> Mat_to_doubles_matrix(const Mat<double>& A) {
  int nrows = A.n_rows;
  int ncols = A.n_cols;

  writable::doubles_matrix<> B(nrows, ncols);

  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      double a_ij = A(i, j);
      B(i, j) = a_ij;
    }
  }

  return B;
}

Col<double> doubles_to_Vec_(const doubles& x) {
  int n = x.size();
  Col<double> y(n);

  for (int i = 0; i < n; ++i) {
    // double x_i = x[i];
    y(i) = x[i];
  }

  return y;
}

doubles Vec_to_doubles_(const Col<double>& x) {
  int n = x.n_rows;
  writable::doubles y(n);

  for (int i = 0; i < n; ++i) {
    double x_i = x[i];
    y[i] = x_i;
  }

  return y;
}

doubles_matrix<> Vec_to_doubles_matrix_(const Col<double>& x) {
  int n = x.n_rows;
  // confusing: the dimension is counted as (n, 1) instead of (n, 0) for a
  // column vector
  writable::doubles_matrix<> Y(n, 1);

  for (int i = 0; i < n; ++i) {
    double x_i = x[i];
    Y(i, 0) = x_i;
  }

  return Y;
}

// SpMat<double> doubles_matrix_to_SpMat_(const doubles_matrix<>& A) {
//   int nrows = A.nrow();
//   int ncols = A.ncol();

//   SpMat<double> B(nrows, ncols);

//   for (int i = 0; i < nrows; ++i) {
//     for (int j = 0; j < ncols; ++j) {
//       double a_ij = A(i, j);
//       B(i, j) = a_ij;  // Assuming you can access elements of A in this
//       way
//     }
//   }

//   return B;
// }

// doubles_matrix<> SpMat_to_doubles_matrix(const SpMat<double>& A) {
//   int nrows = A.n_rows;
//   int ncols = A.n_cols;

//   writable::doubles_matrix<> B(nrows, ncols);

//   for (int i = 0; i < nrows; ++i) {
//     for (int j = 0; j < ncols; ++j) {
//       double a_ij = A(i, j);
//       B(i, j) = a_ij;
//     }
//   }

//   return B;
// }
