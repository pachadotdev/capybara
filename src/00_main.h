#include <algorithm>
#include <armadillo.hpp>
#include <cmath>
#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
#include <numeric>
#include <vector>
#include "Rmath.h"

// #include <iostream>

using namespace arma;
using namespace cpp11;

// used across the scripts

Mat<double> center_variables_(const Mat<double> &V, const Col<double> &w,
                              const list &klist, const double &tol,
                              const int &maxiter);

Col<double> solve_beta_(const Mat<double> &MX, const Mat<double> &MNU,
                        const Col<double> &w);

Col<double> solve_eta_(const Mat<double> &MX, const Mat<double> &MNU,
                       const Col<double> &nu, const Col<double> &beta);

Mat<double> crossprod_(const Mat<double> &X, const Col<double> &w, const int &n,
                       const int &p, const bool &weighted,
                       const bool &root_weights);
