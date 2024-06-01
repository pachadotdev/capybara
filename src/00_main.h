// #include <omp.h>

#include <algorithm>
#include <armadillo.hpp>
#include <cmath>
#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
// #include <iostream>
#include <vector>

using namespace arma;
using namespace cpp11;

// helpers used across scripts

#ifndef HELPERS_H
#define HELPERS_H

uvec as_uvec(const cpp11::integers &x);

#endif  // HELPERS_H
