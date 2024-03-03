// eta.upd <- nu - as.vector(Mnu - MX %*% beta.upd)

#include "00_main.h"

[[cpp11::register]] doubles solve_eta_(const doubles_matrix<>& mx,
                                       const doubles_matrix<>& mnu,
                                       const doubles& nu, const doubles& beta) {
  // Types conversion
  Mat<double> MX = as_Mat(mx);
  Mat<double> Mnu = as_Mat(mnu);
  Col<double> Nu = as_Col(nu);
  Col<double> Beta = as_Col(beta);

  Col<double> Eta = Nu - (Mnu - (MX * Beta));

  return as_doubles(Eta);
}

// eta.upd <- yadj - as.vector(Myadj) + offset - eta

// [[cpp11::register]] doubles solve_eta2_(const doubles& yadj,
//                                         const doubles_matrix<>& myadj,
//                                         const doubles& offset,
//                                         const doubles& eta) {
//   // Types conversion
//   Col<double> Yadj = as_Col(yadj);
//   Mat<double> Myadj = as_Mat(myadj);
//   Col<double> Offset = as_Col(offset);
//   Col<double> Eta = as_Col(eta);

//   Col<double> res = Yadj - Myadj + Offset - Eta;

//   return as_doubles(res);
// }
