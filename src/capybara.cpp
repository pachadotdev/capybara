#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cpp11armadillo.hpp>
#include <cstdint>
#include <cstring>
#include <limits>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using arma::field;
using arma::mat;
using arma::uvec;
using arma::uword;
using arma::vec;

using cpp11::doubles;
using cpp11::doubles_matrix;
using cpp11::integers;
using cpp11::list;
using cpp11::strings;

// #include "timing.h"

#include "01_get_beta.h"
#include "02_group_ops.h"
#include "03_demean_variables.h"
#include "04_lm_fit.h"
#include "05_glm_fit.h"
#include "06_glm_offset_fit.h"
#include "07_get_alpha.h"
#include "08_group_sums.h"

// Convert R k_list to portable Armadillo field structure
// Each element in k_list is a list of groups, each group contains observation indices
inline field<field<uvec>> convert_klist_to_field(const list &k_list) {
  const size_t K = k_list.size();
  field<field<uvec>> group_indices(K);
  
  for (size_t k = 0; k < K; ++k) {
    const list group_list = as_cpp<list>(k_list[k]);
    const size_t n_groups = group_list.size();
    
    group_indices(k).set_size(n_groups);
    
    for (size_t g = 0; g < n_groups; ++g) {
      const integers group_obs = as_cpp<integers>(group_list[g]);
      uvec obs_indices(group_obs.size());
      
      for (size_t i = 0; i < group_obs.size(); ++i) {
        obs_indices(i) = static_cast<uword>(group_obs[i]);
      }
      
      group_indices(k)(g) = obs_indices;
    }
  }
  
  return group_indices;
}

[[cpp11::register]] doubles_matrix<>
demean_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &klist, const double &tol, const int &max_iter,
                  const int &iter_interrupt, const int &iter_ssr,
                  const std::string &family) {
  mat V = as_mat(V_r);
  
  field<field<uvec>> group_indices = convert_klist_to_field(klist);
  
  demean_variables(V, as_col(w_r), group_indices, tol, max_iter, family);
  
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
  
  // Convert R list to portable Armadillo structure
  field<field<uvec>> group_indices = convert_klist_to_field(k_list);
  
  FelmFitResult res = felm_fit(X, y, w, group_indices, center_tol, iter_center_max,
                               iter_interrupt, iter_ssr);
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
  
  // Convert R list to portable Armadillo structure
  field<field<uvec>> group_indices = convert_klist_to_field(k_list);
  
  FeglmFitResult res = feglm_fit(MX, beta, eta, y, wt, theta, group_indices,
                                         center_tol, dev_tol, keep_mx, iter_max, 
                                         iter_center_max, iter_inner_max,
                                         iter_interrupt, iter_ssr, fam, family_type);
  
  // Replace collinear coefficients with R's NA_REAL
  for (arma::uword i = 0; i < res.coefficients.n_elem; ++i) {
    if (res.coef_status(i) == 0) {
      res.coefficients(i) = NA_REAL;
    }
  }
  return res.to_list(keep_mx);
}

[[cpp11::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
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
  // Convert R list to portable Armadillo structure
  field<field<uvec>> group_indices = convert_klist_to_field(k_list);
  
  FeglmOffsetFitResult res =
      feglm_offset_fit(eta, y, offset, wt, group_indices, center_tol, dev_tol,
                       iter_max, iter_center_max, iter_inner_max,
                       iter_interrupt, iter_ssr, fam, family_type);
  return res.to_doubles();
}

[[cpp11::register]] list get_alpha_(const doubles_matrix<> &p_r,
                                    const list &klist, const list &control) {
  const vec p = as_Mat(p_r);
  const double tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_max = as_cpp<int>(control["iter_max"]);
  
  // Convert R list to portable Armadillo structure
  field<field<uvec>> group_indices = convert_klist_to_field(klist);
  
  GetAlphaResult res = get_alpha(p, group_indices, tol, iter_max);
  return res.to_list();
}

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  const mat M = as_Mat(M_r);
  const mat w = as_Mat(w_r);
  
  // Convert R list to single-level Armadillo field
  const size_t J = jlist.size();
  field<uvec> group_indices(J);
  for (size_t j = 0; j < J; ++j) {
    group_indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
  }
  
  GroupSumsResult res = group_sums(M, w, group_indices);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const int K,
                     const list &jlist) {
  const mat M = as_Mat(M_r);
  const mat v = as_Mat(v_r);
  const mat w = as_Mat(w_r);
  
  // Convert R list to single-level Armadillo field
  const size_t J = jlist.size();
  field<uvec> group_indices(J);
  for (size_t j = 0; j < J; ++j) {
    group_indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
  }
  
  GroupSumsResult res = group_sums_spectral(M, v, w, K, group_indices);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  const mat M = as_Mat(M_r);
  
  // Convert R list to single-level Armadillo field
  const size_t J = jlist.size();
  field<uvec> group_indices(J);
  for (size_t j = 0; j < J; ++j) {
    group_indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
  }
  
  GroupSumsResult res = group_sums_var(M, group_indices);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  const mat M = as_Mat(M_r);
  const mat N = as_Mat(N_r);
  
  // Convert R list to single-level Armadillo field
  const size_t J = jlist.size();
  field<uvec> group_indices(J);
  for (size_t j = 0; j < J; ++j) {
    group_indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
  }
  
  GroupSumsResult res = group_sums_cov(M, N, group_indices);
  return res.to_matrix();
}
