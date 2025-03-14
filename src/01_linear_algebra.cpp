#include "00_main.h"

mat crossprod_(const mat &X, const vec &w) {
  mat Y = X;
  Y.each_col() %= sqrt(w);
  return Y.t() * Y;
}

vec solve_beta_(mat MX, const mat &MNU, const vec &w) {
  const vec sqrt_w = sqrt(w);
  MX.each_col() %= sqrt_w;

  mat Q, R;
  if (!qr_econ(Q, R, MX)) {
    stop("QR decomposition failed");
  }

  return solve(trimatu(R), Q.t() * (MNU.each_col() % sqrt_w), solve_opts::fast);
}
