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
#include <set> // for Poisson separation check
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
#include "02_demean_variables.h"
// #include "03_convergence.h"
#include "04_get_alpha.h"
#include "05_lm_fit.h"
#include "06_exponential_family.h"
#include "07_glm_fit.h"
#include "08_glm_offset_fit.h"
#include "09_group_sums.h"

// Helper function to convert R indices to C++ uvec (R uses 1-based, C++ uses 0-based)
inline uvec r_to_cpp_indices(const integers& r_indices) {
  uvec cpp_indices(r_indices.size());
  
  // Use std::transform for efficient vectorized conversion
  std::transform(r_indices.begin(), r_indices.end(), cpp_indices.begin(),
    [](int r_val) -> uword {
      // if (r_val < 1) {
      //   cpp11::stop("Invalid R index: %d (R indices must be >= 1)", r_val);
      // }
      return static_cast<uword>(r_val - 1);  // Convert to 0-based
    });
  
  return cpp_indices;
}

// Convert R k_list to field<field<uvec>> format using 0-based indexing
inline field<field<uvec>> convert_klist_to_field(const list &k_list) {
  const size_t K = k_list.size();
  field<field<uvec>> group_indices(K);
  
  for (size_t k = 0; k < K; ++k) {
    const list group_list = as_cpp<list>(k_list[k]);
    const size_t n_groups = group_list.size();
    
    group_indices(k).set_size(n_groups);
    for (size_t g = 0; g < n_groups; ++g) {
      const integers group_obs = as_cpp<integers>(group_list[g]);
      group_indices(k)(g) = r_to_cpp_indices(group_obs);  // Convert to 0-based
    }
  }
  
  return group_indices;
}

// Convert field<field<uvec>> to umat format for modern functions
inline umat convert_field_to_umat(const field<field<uvec>>& group_indices, size_t n_obs) {
  const size_t K = group_indices.n_elem;
  
  if (K == 0) {
    return umat(n_obs, 0);
  }
  
  umat fe_matrix(n_obs, K);
  fe_matrix.zeros();

  for (size_t k = 0; k < K; ++k) {
    const field<uvec>& groups = group_indices(k);
    const size_t n_groups = groups.n_elem;

    for (size_t g = 0; g < n_groups; ++g) {
      const uvec& group_obs = groups(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_matrix(group_obs(i), k) = g;  // 0-based group IDs
      }
    }
  }

  return fe_matrix;
}

[[cpp11::register]] doubles_matrix<>
demean_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &klist, const double &tol, const int &max_iter,
                  const int &iter_interrupt, const int &iter_ssr,
                  const std::string &family) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  // Convert R list to field<field<uvec>> format
  field<field<uvec>> group_indices = convert_klist_to_field(klist);

  // Convert to the format expected by joint_demean
  const size_t K = group_indices.n_elem;
  field<uvec> fe_ids(K);
  
  for (size_t k = 0; k < K; ++k) {
    const field<uvec>& groups = group_indices(k);
    const size_t n_groups = groups.n_elem;
    
    // Find the maximum observation index to size the fe_id vector
    size_t max_obs = 0;
    for (size_t g = 0; g < n_groups; ++g) {
      if (groups(g).n_elem > 0) {
        max_obs = std::max(max_obs, static_cast<size_t>(groups(g).max()));
      }
    }
    
    fe_ids(k).set_size(max_obs + 1);  // +1 because indices are 0-based
    
    // Assign group IDs
    for (size_t g = 0; g < n_groups; ++g) {
      const uvec& group_obs = groups(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_ids(k)(group_obs(i)) = g;  // Use 0-based group IDs
      }
    }
  }

  // Use the traditional demean function
  DemeanResult result = joint_demean(V, fe_ids, w, tol, max_iter);

  return as_doubles_matrix(result.demeaned_data);
}

[[cpp11::register]] list felm_fit_(const doubles &y_r,
                                   const doubles_matrix<> &x_r,
                                   const doubles &wt_r, const list &control,
                                   const list &k_list) {
  mat X = as_Mat(x_r);
  const vec y = as_Col(y_r);
  const vec w = as_Col(wt_r);
  const double center_tol = as_cpp<double>(control["center_tol"]);
  const double collin_tol = as_cpp<double>(control["collin_tol"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);

  // Convert R list to efficient umat format
  field<field<uvec>> group_indices = convert_klist_to_field(k_list);
  umat fe_matrix = convert_field_to_umat(group_indices, y.n_elem);

  LMResult res = felm_fit(X, y, w, fe_matrix, center_tol, iter_center_max,
                          iter_interrupt, iter_ssr, collin_tol);
  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (res.coef_status == 0);
  if (any(collinear_mask)) {
    res.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
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
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const double collin_tol = as_cpp<double>(control["collin_tol"]);
  const bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);

  // Convert R list to efficient umat format
  field<field<uvec>> group_indices = convert_klist_to_field(k_list);
  umat fe_matrix = convert_field_to_umat(group_indices, y.n_elem);

  GLMResult res =
      feglm_fit(MX, y, wt, fe_matrix, center_tol, dev_tol, keep_mx,
                iter_max, iter_center_max, iter_inner_max, fam, collin_tol);

  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (res.coef_status == 0);
  if (any(collinear_mask)) {
    res.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
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
  const double collin_tol = as_cpp<double>(control["collin_tol"]);
  const size_t iter_max = as_cpp<int>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  // Convert R list to efficient umat structure
  field<field<uvec>> group_indices = convert_klist_to_field(k_list);
  umat fe_matrix = convert_field_to_umat(group_indices, y.n_elem);

  FeglmOffsetFitResult res =
      feglm_offset_fit(eta, y, offset, wt, fe_matrix, center_tol, dev_tol,
                       iter_max, iter_center_max, iter_inner_max,
                       iter_interrupt, iter_ssr, fam, family_type, collin_tol);
  return res.to_doubles();
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
    group_indices(j) = r_to_cpp_indices(as_cpp<integers>(jlist[j]));
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
    group_indices(j) = r_to_cpp_indices(as_cpp<integers>(jlist[j]));
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
    group_indices(j) = r_to_cpp_indices(as_cpp<integers>(jlist[j]));
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
    group_indices(j) = r_to_cpp_indices(as_cpp<integers>(jlist[j]));
  }

  GroupSumsResult res = group_sums_cov(M, N, group_indices);
  return res.to_matrix();
}
