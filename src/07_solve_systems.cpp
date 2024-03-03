#include "00_main.h"

[[cpp11::register]] doubles solve_(const doubles_matrix<> &a,
                                   const doubles &b) {
  // Types conversion
  Mat<double> A = as_Mat(a);
  Col<double> B = as_Col(b);

  // Solve the system
  Col<double> res = solve(A, B);

  return as_doubles(res);
}
