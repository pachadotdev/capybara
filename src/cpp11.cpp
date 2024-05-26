// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"
#include <R_ext/Visibility.h>

// 01_center_variables.cpp
doubles_matrix<> center_variables_(const doubles_matrix<> & V_r, const doubles & v_sum_r, const doubles & w_r, const list & klist, const double & tol, const int & maxiter, const bool & sum_v);
extern "C" SEXP _capybara_center_variables_(SEXP V_r, SEXP v_sum_r, SEXP w_r, SEXP klist, SEXP tol, SEXP maxiter, SEXP sum_v) {
  BEGIN_CPP11
    return cpp11::as_sexp(center_variables_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(V_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(v_sum_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(w_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(klist), cpp11::as_cpp<cpp11::decay_t<const double &>>(tol), cpp11::as_cpp<cpp11::decay_t<const int &>>(maxiter), cpp11::as_cpp<cpp11::decay_t<const bool &>>(sum_v)));
  END_CPP11
}
// 02_get_alpha.cpp
list get_alpha_(const doubles_matrix<> & p_r, const list & klist, const double & tol);
extern "C" SEXP _capybara_get_alpha_(SEXP p_r, SEXP klist, SEXP tol) {
  BEGIN_CPP11
    return cpp11::as_sexp(get_alpha_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(p_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(klist), cpp11::as_cpp<cpp11::decay_t<const double &>>(tol)));
  END_CPP11
}
// 03_group_sums.cpp
doubles_matrix<> group_sums_(const doubles_matrix<> & M_r, const doubles_matrix<> & w_r, const list & jlist);
extern "C" SEXP _capybara_group_sums_(SEXP M_r, SEXP w_r, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(w_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 03_group_sums.cpp
doubles_matrix<> group_sums_spectral_(const doubles_matrix<> & M_r, const doubles_matrix<> & v_r, const doubles_matrix<> & w_r, const int K, const list & jlist);
extern "C" SEXP _capybara_group_sums_spectral_(SEXP M_r, SEXP v_r, SEXP w_r, SEXP K, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_spectral_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(v_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(w_r), cpp11::as_cpp<cpp11::decay_t<const int>>(K), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 03_group_sums.cpp
doubles_matrix<> group_sums_var_(const doubles_matrix<> & M_r, const list & jlist);
extern "C" SEXP _capybara_group_sums_var_(SEXP M_r, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_var_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 03_group_sums.cpp
doubles_matrix<> group_sums_cov_(const doubles_matrix<> & M_r, const doubles_matrix<> & N_r, const list & jlist);
extern "C" SEXP _capybara_group_sums_cov_(SEXP M_r, SEXP N_r, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_cov_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(N_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> crossprod_(const doubles_matrix<> & x, const doubles & w, const bool & weighted, const bool & root_weights);
extern "C" SEXP _capybara_crossprod_(SEXP x, SEXP w, SEXP weighted, SEXP root_weights) {
  BEGIN_CPP11
    return cpp11::as_sexp(crossprod_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(w), cpp11::as_cpp<cpp11::decay_t<const bool &>>(weighted), cpp11::as_cpp<cpp11::decay_t<const bool &>>(root_weights)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> gamma_(const doubles_matrix<> & mx, const doubles_matrix<> & hessian, const doubles_matrix<> & j, const doubles_matrix<> & ppsi, const doubles & v, const SEXP & nt_full);
extern "C" SEXP _capybara_gamma_(SEXP mx, SEXP hessian, SEXP j, SEXP ppsi, SEXP v, SEXP nt_full) {
  BEGIN_CPP11
    return cpp11::as_sexp(gamma_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(mx), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(hessian), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(j), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(ppsi), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(v), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(nt_full)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> chol_crossprod_(const doubles_matrix<> & x);
extern "C" SEXP _capybara_chol_crossprod_(SEXP x) {
  BEGIN_CPP11
    return cpp11::as_sexp(chol_crossprod_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> chol2inv_(const doubles_matrix<> & r);
extern "C" SEXP _capybara_chol2inv_(SEXP r) {
  BEGIN_CPP11
    return cpp11::as_sexp(chol2inv_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(r)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> chol_(const doubles_matrix<> & x);
extern "C" SEXP _capybara_chol_(SEXP x) {
  BEGIN_CPP11
    return cpp11::as_sexp(chol_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x)));
  END_CPP11
}
// 04_linear_algebra.cpp
int qr_rank_(const doubles_matrix<> & x);
extern "C" SEXP _capybara_qr_rank_(SEXP x) {
  BEGIN_CPP11
    return cpp11::as_sexp(qr_rank_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles solve_bias_(const doubles & beta_uncorr, const doubles_matrix<> & hessian, const double & nt, const doubles & b);
extern "C" SEXP _capybara_solve_bias_(SEXP beta_uncorr, SEXP hessian, SEXP nt, SEXP b) {
  BEGIN_CPP11
    return cpp11::as_sexp(solve_bias_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(beta_uncorr), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(hessian), cpp11::as_cpp<cpp11::decay_t<const double &>>(nt), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(b)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles solve_y_(const doubles_matrix<> & a, const doubles & x);
extern "C" SEXP _capybara_solve_y_(SEXP a, SEXP x) {
  BEGIN_CPP11
    return cpp11::as_sexp(solve_y_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(a), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(x)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> sandwich_(const doubles_matrix<> & a, const doubles_matrix<> & b);
extern "C" SEXP _capybara_sandwich_(SEXP a, SEXP b) {
  BEGIN_CPP11
    return cpp11::as_sexp(sandwich_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(a), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(b)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles update_beta_eta_(const doubles & old, const doubles & upd, const double & param);
extern "C" SEXP _capybara_update_beta_eta_(SEXP old, SEXP upd, SEXP param) {
  BEGIN_CPP11
    return cpp11::as_sexp(update_beta_eta_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(old), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(upd), cpp11::as_cpp<cpp11::decay_t<const double &>>(param)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles update_nu_(const SEXP & y, const SEXP & mu, const SEXP & mu_eta);
extern "C" SEXP _capybara_update_nu_(SEXP y, SEXP mu, SEXP mu_eta) {
  BEGIN_CPP11
    return cpp11::as_sexp(update_nu_(cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(y), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(mu), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(mu_eta)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles solve_beta_(const doubles_matrix<> & mx, const doubles_matrix<> & mnu, const doubles & wtilde, const double & epsilon, const bool & weighted);
extern "C" SEXP _capybara_solve_beta_(SEXP mx, SEXP mnu, SEXP wtilde, SEXP epsilon, SEXP weighted) {
  BEGIN_CPP11
    return cpp11::as_sexp(solve_beta_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(mx), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(mnu), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(wtilde), cpp11::as_cpp<cpp11::decay_t<const double &>>(epsilon), cpp11::as_cpp<cpp11::decay_t<const bool &>>(weighted)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles solve_eta_(const doubles_matrix<> & mx, const doubles_matrix<> & mnu, const doubles & nu, const doubles & beta);
extern "C" SEXP _capybara_solve_eta_(SEXP mx, SEXP mnu, SEXP nu, SEXP beta) {
  BEGIN_CPP11
    return cpp11::as_sexp(solve_eta_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(mx), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(mnu), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(nu), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(beta)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles solve_eta2_(const SEXP & yadj, const SEXP & myadj, const SEXP & offset, const SEXP & eta);
extern "C" SEXP _capybara_solve_eta2_(SEXP yadj, SEXP myadj, SEXP offset, SEXP eta) {
  BEGIN_CPP11
    return cpp11::as_sexp(solve_eta2_(cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(yadj), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(myadj), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(offset), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(eta)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles sqrt_(const SEXP & w);
extern "C" SEXP _capybara_sqrt_(SEXP w) {
  BEGIN_CPP11
    return cpp11::as_sexp(sqrt_(cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(w)));
  END_CPP11
}
// 05_pairwise_correlation.cpp
double pairwise_cor_(const doubles & y, const doubles & yhat);
extern "C" SEXP _capybara_pairwise_cor_(SEXP y, SEXP yhat) {
  BEGIN_CPP11
    return cpp11::as_sexp(pairwise_cor_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(y), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(yhat)));
  END_CPP11
}

extern "C" {
static const R_CallMethodDef CallEntries[] = {
    {"_capybara_center_variables_",    (DL_FUNC) &_capybara_center_variables_,    7},
    {"_capybara_chol2inv_",            (DL_FUNC) &_capybara_chol2inv_,            1},
    {"_capybara_chol_",                (DL_FUNC) &_capybara_chol_,                1},
    {"_capybara_chol_crossprod_",      (DL_FUNC) &_capybara_chol_crossprod_,      1},
    {"_capybara_crossprod_",           (DL_FUNC) &_capybara_crossprod_,           4},
    {"_capybara_gamma_",               (DL_FUNC) &_capybara_gamma_,               6},
    {"_capybara_get_alpha_",           (DL_FUNC) &_capybara_get_alpha_,           3},
    {"_capybara_group_sums_",          (DL_FUNC) &_capybara_group_sums_,          3},
    {"_capybara_group_sums_cov_",      (DL_FUNC) &_capybara_group_sums_cov_,      3},
    {"_capybara_group_sums_spectral_", (DL_FUNC) &_capybara_group_sums_spectral_, 5},
    {"_capybara_group_sums_var_",      (DL_FUNC) &_capybara_group_sums_var_,      2},
    {"_capybara_pairwise_cor_",        (DL_FUNC) &_capybara_pairwise_cor_,        2},
    {"_capybara_qr_rank_",             (DL_FUNC) &_capybara_qr_rank_,             1},
    {"_capybara_sandwich_",            (DL_FUNC) &_capybara_sandwich_,            2},
    {"_capybara_solve_beta_",          (DL_FUNC) &_capybara_solve_beta_,          5},
    {"_capybara_solve_bias_",          (DL_FUNC) &_capybara_solve_bias_,          4},
    {"_capybara_solve_eta2_",          (DL_FUNC) &_capybara_solve_eta2_,          4},
    {"_capybara_solve_eta_",           (DL_FUNC) &_capybara_solve_eta_,           4},
    {"_capybara_solve_y_",             (DL_FUNC) &_capybara_solve_y_,             2},
    {"_capybara_sqrt_",                (DL_FUNC) &_capybara_sqrt_,                1},
    {"_capybara_update_beta_eta_",     (DL_FUNC) &_capybara_update_beta_eta_,     3},
    {"_capybara_update_nu_",           (DL_FUNC) &_capybara_update_nu_,           3},
    {NULL, NULL, 0}
};
}

extern "C" attribute_visible void R_init_capybara(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
