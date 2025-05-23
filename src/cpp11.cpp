// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"
#include <R_ext/Visibility.h>

// 01_linear_algebra.cpp
int check_linear_dependence_qr_(const doubles & y, const doubles_matrix<> & x, const int & p);
extern "C" SEXP _capybara_check_linear_dependence_qr_(SEXP y, SEXP x, SEXP p) {
  BEGIN_CPP11
    return cpp11::as_sexp(check_linear_dependence_qr_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(y), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x), cpp11::as_cpp<cpp11::decay_t<const int &>>(p)));
  END_CPP11
}
// 02_center_variables.cpp
doubles_matrix<> center_variables_r_(const doubles_matrix<> & V_r, const doubles & w_r, const list & klist, const double & tol, const int & max_iter, const int & iter_interrupt, const int & iter_ssr);
extern "C" SEXP _capybara_center_variables_r_(SEXP V_r, SEXP w_r, SEXP klist, SEXP tol, SEXP max_iter, SEXP iter_interrupt, SEXP iter_ssr) {
  BEGIN_CPP11
    return cpp11::as_sexp(center_variables_r_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(V_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(w_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(klist), cpp11::as_cpp<cpp11::decay_t<const double &>>(tol), cpp11::as_cpp<cpp11::decay_t<const int &>>(max_iter), cpp11::as_cpp<cpp11::decay_t<const int &>>(iter_interrupt), cpp11::as_cpp<cpp11::decay_t<const int &>>(iter_ssr)));
  END_CPP11
}
// 03_lm_fit.cpp
list felm_fit_(const doubles & y_r, const doubles_matrix<> & x_r, const doubles & wt_r, const list & control, const list & k_list);
extern "C" SEXP _capybara_felm_fit_(SEXP y_r, SEXP x_r, SEXP wt_r, SEXP control, SEXP k_list) {
  BEGIN_CPP11
    return cpp11::as_sexp(felm_fit_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(y_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(wt_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(control), cpp11::as_cpp<cpp11::decay_t<const list &>>(k_list)));
  END_CPP11
}
// 04_glm_fit.cpp
list feglm_fit_(const doubles & beta_r, const doubles & eta_r, const doubles & y_r, const doubles_matrix<> & x_r, const doubles & wt_r, const double & theta, const std::string & family, const list & control, const list & k_list);
extern "C" SEXP _capybara_feglm_fit_(SEXP beta_r, SEXP eta_r, SEXP y_r, SEXP x_r, SEXP wt_r, SEXP theta, SEXP family, SEXP control, SEXP k_list) {
  BEGIN_CPP11
    return cpp11::as_sexp(feglm_fit_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(beta_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(eta_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(y_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(x_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(wt_r), cpp11::as_cpp<cpp11::decay_t<const double &>>(theta), cpp11::as_cpp<cpp11::decay_t<const std::string &>>(family), cpp11::as_cpp<cpp11::decay_t<const list &>>(control), cpp11::as_cpp<cpp11::decay_t<const list &>>(k_list)));
  END_CPP11
}
// 05_glm_offset_fit.cpp
doubles feglm_offset_fit_(const doubles & eta_r, const doubles & y_r, const doubles & offset_r, const doubles & wt_r, const std::string & family, const list & control, const list & k_list);
extern "C" SEXP _capybara_feglm_offset_fit_(SEXP eta_r, SEXP y_r, SEXP offset_r, SEXP wt_r, SEXP family, SEXP control, SEXP k_list) {
  BEGIN_CPP11
    return cpp11::as_sexp(feglm_offset_fit_(cpp11::as_cpp<cpp11::decay_t<const doubles &>>(eta_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(y_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(offset_r), cpp11::as_cpp<cpp11::decay_t<const doubles &>>(wt_r), cpp11::as_cpp<cpp11::decay_t<const std::string &>>(family), cpp11::as_cpp<cpp11::decay_t<const list &>>(control), cpp11::as_cpp<cpp11::decay_t<const list &>>(k_list)));
  END_CPP11
}
// 06_get_alpha.cpp
list get_alpha_(const doubles_matrix<> & p_r, const list & klist, const list & control);
extern "C" SEXP _capybara_get_alpha_(SEXP p_r, SEXP klist, SEXP control) {
  BEGIN_CPP11
    return cpp11::as_sexp(get_alpha_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(p_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(klist), cpp11::as_cpp<cpp11::decay_t<const list &>>(control)));
  END_CPP11
}
// 07_group_sums.cpp
doubles_matrix<> group_sums_(const doubles_matrix<> & M_r, const doubles_matrix<> & w_r, const list & jlist);
extern "C" SEXP _capybara_group_sums_(SEXP M_r, SEXP w_r, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(w_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 07_group_sums.cpp
doubles_matrix<> group_sums_spectral_(const doubles_matrix<> & M_r, const doubles_matrix<> & v_r, const doubles_matrix<> & w_r, const int K, const list & jlist);
extern "C" SEXP _capybara_group_sums_spectral_(SEXP M_r, SEXP v_r, SEXP w_r, SEXP K, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_spectral_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(v_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(w_r), cpp11::as_cpp<cpp11::decay_t<const int>>(K), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 07_group_sums.cpp
doubles_matrix<> group_sums_var_(const doubles_matrix<> & M_r, const list & jlist);
extern "C" SEXP _capybara_group_sums_var_(SEXP M_r, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_var_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}
// 07_group_sums.cpp
doubles_matrix<> group_sums_cov_(const doubles_matrix<> & M_r, const doubles_matrix<> & N_r, const list & jlist);
extern "C" SEXP _capybara_group_sums_cov_(SEXP M_r, SEXP N_r, SEXP jlist) {
  BEGIN_CPP11
    return cpp11::as_sexp(group_sums_cov_(cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(M_r), cpp11::as_cpp<cpp11::decay_t<const doubles_matrix<> &>>(N_r), cpp11::as_cpp<cpp11::decay_t<const list &>>(jlist)));
  END_CPP11
}

extern "C" {
static const R_CallMethodDef CallEntries[] = {
    {"_capybara_center_variables_r_",         (DL_FUNC) &_capybara_center_variables_r_,         7},
    {"_capybara_check_linear_dependence_qr_", (DL_FUNC) &_capybara_check_linear_dependence_qr_, 3},
    {"_capybara_feglm_fit_",                  (DL_FUNC) &_capybara_feglm_fit_,                  9},
    {"_capybara_feglm_offset_fit_",           (DL_FUNC) &_capybara_feglm_offset_fit_,           7},
    {"_capybara_felm_fit_",                   (DL_FUNC) &_capybara_felm_fit_,                   5},
    {"_capybara_get_alpha_",                  (DL_FUNC) &_capybara_get_alpha_,                  3},
    {"_capybara_group_sums_",                 (DL_FUNC) &_capybara_group_sums_,                 3},
    {"_capybara_group_sums_cov_",             (DL_FUNC) &_capybara_group_sums_cov_,             3},
    {"_capybara_group_sums_spectral_",        (DL_FUNC) &_capybara_group_sums_spectral_,        5},
    {"_capybara_group_sums_var_",             (DL_FUNC) &_capybara_group_sums_var_,             2},
    {NULL, NULL, 0}
};
}

extern "C" attribute_visible void R_init_capybara(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
