// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"
#include <R_ext/Visibility.h>

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
doubles_matrix<> gamma_(const doubles_matrix<> & mx, const doubles_matrix<> & hessian, const doubles_matrix<> & j, const doubles_matrix<> & ppsi, const doubles & v, const SEXP & nt_full);
extern "C" SEXP _capybara_gamma_(SEXP mx, SEXP hessian, SEXP j, SEXP ppsi, SEXP v, SEXP nt_full) {
  BEGIN_CPP11
    return cpp11::as_sexp(gamma_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(mx), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(hessian), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(j), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(ppsi), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(v), cpp11::as_cpp<cpp11::decay_t<const SEXP &>>(nt_full)));
  END_CPP11
}
// 04_linear_algebra.cpp
doubles_matrix<> inv_(const doubles_matrix<> & h);
extern "C" SEXP _capybara_inv_(SEXP h) {
  BEGIN_CPP11
    return cpp11::as_sexp(inv_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(h)));
  END_CPP11
}
// 04_linear_algebra.cpp
int rank_(const doubles_matrix<> & x);
extern "C" SEXP _capybara_rank_(SEXP x) {
  BEGIN_CPP11
    return cpp11::as_sexp(rank_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x)));
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
doubles solve_eta2_(const doubles & yadj, const doubles_matrix<> & myadj, const doubles & offset, const doubles & eta);
extern "C" SEXP _capybara_solve_eta2_(SEXP yadj, SEXP myadj, SEXP offset, SEXP eta) {
  BEGIN_CPP11
    return cpp11::as_sexp(solve_eta2_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(yadj), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(myadj), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(offset), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(eta)));
  END_CPP11
}
// 05_glm_fit.cpp
list feglm_fit_(const doubles & beta_r, const doubles & eta_r, const doubles & y_r, const doubles_matrix<> & x_r, const double & nt, const doubles & wt_r, const double & theta, const std::string & family, const list & control, const list & k_list);
extern "C" SEXP _capybara_feglm_fit_(SEXP beta_r, SEXP eta_r, SEXP y_r, SEXP x_r, SEXP nt, SEXP wt_r, SEXP theta, SEXP family, SEXP control, SEXP k_list) {
  BEGIN_CPP11
    return cpp11::as_sexp(feglm_fit_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(beta_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(eta_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(y_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x_r), cpp11::as_cpp<cpp11::decay_t<const double &>>(nt), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(wt_r), cpp11::as_cpp<cpp11::decay_t<const double &>>(theta), cpp11::as_cpp<cpp11::decay_t<const std::string &>>(family), cpp11::as_cpp<cpp11::decay_t<const list &>>(control), cpp11::as_cpp<cpp11::decay_t<const list &>>(k_list)));
  END_CPP11
}
// 06_kendall_correlation.cpp
double kendall_cor_(const doubles_matrix<> & m);
extern "C" SEXP _capybara_kendall_cor_(SEXP m) {
  BEGIN_CPP11
    return cpp11::as_sexp(kendall_cor_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(m)));
  END_CPP11
}
// 06_kendall_correlation.cpp
doubles pkendall_(doubles Q, int n);
extern "C" SEXP _capybara_pkendall_(SEXP Q, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(pkendall_(cpp11::as_cpp<cpp11::decay_t<doubles>>(Q), cpp11::as_cpp<cpp11::decay_t<int>>(n)));
  END_CPP11
}

extern "C" {
static const R_CallMethodDef CallEntries[] = {
    {"_capybara_feglm_fit_",           (DL_FUNC) &_capybara_feglm_fit_,           10},
    {"_capybara_gamma_",               (DL_FUNC) &_capybara_gamma_,                6},
    {"_capybara_get_alpha_",           (DL_FUNC) &_capybara_get_alpha_,            3},
    {"_capybara_group_sums_",          (DL_FUNC) &_capybara_group_sums_,           3},
    {"_capybara_group_sums_cov_",      (DL_FUNC) &_capybara_group_sums_cov_,       3},
    {"_capybara_group_sums_spectral_", (DL_FUNC) &_capybara_group_sums_spectral_,  5},
    {"_capybara_group_sums_var_",      (DL_FUNC) &_capybara_group_sums_var_,       2},
    {"_capybara_inv_",                 (DL_FUNC) &_capybara_inv_,                  1},
    {"_capybara_kendall_cor_",         (DL_FUNC) &_capybara_kendall_cor_,          1},
    {"_capybara_pkendall_",            (DL_FUNC) &_capybara_pkendall_,             2},
    {"_capybara_rank_",                (DL_FUNC) &_capybara_rank_,                 1},
    {"_capybara_sandwich_",            (DL_FUNC) &_capybara_sandwich_,             2},
    {"_capybara_solve_bias_",          (DL_FUNC) &_capybara_solve_bias_,           4},
    {"_capybara_solve_eta2_",          (DL_FUNC) &_capybara_solve_eta2_,           4},
    {"_capybara_solve_y_",             (DL_FUNC) &_capybara_solve_y_,              2},
    {"_capybara_update_beta_eta_",     (DL_FUNC) &_capybara_update_beta_eta_,      3},
    {"_capybara_update_nu_",           (DL_FUNC) &_capybara_update_nu_,            3},
    {NULL, NULL, 0}
};
}

extern "C" attribute_visible void R_init_capybara(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
