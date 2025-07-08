#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) || defined(__ARM_FEATURE_FMA)
  #ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
    #define ARMA_DONT_PRINT_FAST_MATH_WARNING
  #endif
#endif

#include <cmath>
#include <cpp11armadillo.hpp>
#include <limits>
#include <regex>
#include <unordered_map>

using arma::field;
using arma::mat;
using arma::uvec;
using arma::uword;
using arma::vec;

using cpp11::doubles;
using cpp11::doubles_matrix;
using cpp11::integers;
using cpp11::list;

#include "01_get_beta.h"
#include "02_center_variables.h"
#include "03_lm_fit.h"
#include "04_glm_fit.h"
#include "05_glm_offset_fit.h"
#include "06_get_alpha.h"
#include "07_group_sums.h"

[[cpp11::register]] doubles_matrix<> center_variables_r_(
    const doubles_matrix<> &V_r, const doubles &w_r, const list &klist,
    const double &tol, const int &max_iter, const int &iter_interrupt,
    const int &iter_ssr) {
  mat V = as_mat(V_r);
  center_variables_(V, as_col(w_r), klist, tol, max_iter, iter_interrupt,
                    iter_ssr);
  return as_doubles_matrix(V);
}

[[cpp11::register]] list felm_fit_(const doubles &y_r,
                                   const doubles_matrix<> &x_r,
                                   const doubles &wt_r, const list &control,
                                   const list &k_list) {
  mat X = as_Mat(x_r);
  const vec y = as_Col(y_r);
  const vec w = as_Col(wt_r);
  const double center_tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  FelmFitResult res = felm_fit(X, y, w, k_list, center_tol,
                                    iter_center_max, iter_interrupt, iter_ssr);
  // Replace collinear coefficients with R's NA_REAL
  for (arma::uword i = 0; i < res.coefficients.n_elem; ++i) {
    if (res.coef_status(i) == 0) {
      res.coefficients(i) = NA_REAL;
    }
  }
  return res.to_list();
}

[[cpp11::register]] list feglm_fit_(const doubles &beta_r, const doubles &eta_r,
                                    const doubles &y_r,
                                    const doubles_matrix<> &x_r,
                                    const doubles &wt_r, const double &theta,
                                    const std::string &family,
                                    const list &control, const list &k_list) {
  mat MX = as_Mat(x_r);
  vec beta = as_Col(beta_r);
  vec eta = as_Col(eta_r);
  const vec y = as_Col(y_r);
  const vec wt = as_Col(wt_r);
  const std::string fam = tidy_family_(family);
  const FamilyType family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  FeglmFitResult res =
      feglm_fit(MX, beta, eta, y, wt, theta, k_list, center_tol, dev_tol,
                     keep_mx, iter_max, iter_center_max, iter_inner_max,
                     iter_interrupt, iter_ssr, fam, family_type);
  // Replace collinear coefficients with R's NA_REAL
  for (arma::uword i = 0; i < res.coefficients.n_elem; ++i) {
    if (res.coef_status(i) == 0) {
      res.coefficients(i) = NA_REAL;
    }
  }
  return res.to_list(keep_mx);
}

[[cpp11::register]] doubles feglm_offset_fit_(
    const doubles &eta_r, const doubles &y_r, const doubles &offset_r,
    const doubles &wt_r, const std::string &family, const list &control,
    const list &k_list) {
  vec eta = as_Col(eta_r);
  vec y = as_Col(y_r);
  const vec offset = as_Col(offset_r);
  const vec wt = as_Col(wt_r);
  const std::string fam = tidy_family_(family);
  const FamilyType family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const size_t iter_max = as_cpp<int>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  FeglmOffsetFitResult res =
      feglm_offset_fit(eta, y, offset, wt, k_list, center_tol, dev_tol,
                            iter_max, iter_center_max, iter_inner_max,
                            iter_interrupt, iter_ssr, fam, family_type);
  return res.to_doubles();
}

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const list &control) {
  const vec p = as_Mat(p_r);
  const double tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_max = as_cpp<int>(control["iter_max"]);
  GetAlphaResult res = get_alpha(p, klist, tol, iter_max);
  return res.to_list();
}

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  const mat M = as_Mat(M_r);
  const mat w = as_Mat(w_r);
  GroupSumsResult res = group_sums(M, w, jlist);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<> group_sums_spectral_(
    const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
    const doubles_matrix<> &w_r, const int K, const list &jlist) {
  const mat M = as_Mat(M_r);
  const mat v = as_Mat(v_r);
  const mat w = as_Mat(w_r);
  GroupSumsResult res = group_sums_spectral(M, v, w, K, jlist);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<> group_sums_var_(
    const doubles_matrix<> &M_r, const list &jlist) {
  const mat M = as_Mat(M_r);
  GroupSumsResult res = group_sums_var(M, jlist);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<> group_sums_cov_(
    const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
    const list &jlist) {
  const mat M = as_Mat(M_r);
  const mat N = as_Mat(N_r);
  GroupSumsResult res = group_sums_cov(M, N, jlist);
  return res.to_matrix();
}
