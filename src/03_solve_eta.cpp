// eta.upd <- nu - as.vector(Mnu - MX %*% beta.upd)

#include "00_main.h"

[[cpp11::register]] doubles solve_eta_(const doubles_matrix<> &mx,
                                       const doubles_matrix<> &mnu,
                                       const doubles &nu, const doubles &beta) {
  // Types conversion
  Mat<double> MX = as_Mat(mx);
  Mat<double> Mnu = as_Mat(mnu);
  Mat<double> Nu = as_Mat(nu);
  Mat<double> Beta = as_Mat(beta);

  Mat<double> res = Nu - (Mnu - (MX * Beta));

  return as_doubles(res);
}

// eta.upd <- yadj - as.vector(Myadj) + offset - eta

[[cpp11::register]] doubles solve_eta2_(const doubles &yadj,
                                        const doubles_matrix<> &myadj,
                                        const doubles &offset,
                                        const doubles &eta) {
  // Types conversion
  Mat<double> Yadj = as_Mat(yadj);
  Mat<double> Myadj = as_Mat(myadj);
  Mat<double> Offset = as_Mat(offset);
  Mat<double> Eta = as_Mat(eta);

  Mat<double> res = Yadj - Myadj + Offset - Eta;

  return as_doubles(res);
}
