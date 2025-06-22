#include <cpp11armadillo.hpp>

using namespace arma;
using namespace cpp11;

// Include all header files
#include "01_types.h"
#include "02_exponential_family.h"
#include "03_indices.h"
#include "04_linear_algebra.h"
#include "05_center.h"
#include "06_beta.h"
#include "07_linear_model.h"
#include "08_generalized_linear_model.h"
#include "09_alpha.h"
#include "10_groups.h"

////////////////////////////////////////////////////////////////////////////////
// R wrappers
////////////////////////////////////////////////////////////////////////////////

[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &k_list, const double &tol, const size_t &max_iter,
                  const size_t &iter_interrupt) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);
  indices_info indices = list_to_indices_info(k_list);

  const bool use_weights = !all(w == 1.0);
  center_variables(V, w, indices, tol, max_iter, iter_interrupt, use_weights);

  return as_doubles_matrix(V);
}

[[cpp11::register]] list felm_(const doubles &y_r, const doubles_matrix<> &x_r,
                               const doubles &wt_r, const list &control,
                               const list &k_list) {
  mat X = as_mat(x_r);
  const vec y = as_col(y_r);
  const vec w = as_col(wt_r);

  const double center_tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  const size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  const bool use_acceleration = as_cpp<bool>(control["use_acceleration"]);

  indices_info indices = list_to_indices_info(k_list);

  felm_results results = felm(X, y, w, center_tol, iter_center_max,
                              iter_interrupt, indices, use_acceleration);

  if (!any(results.valid_coefficients == 0)) {
    return results.to_list();
  } else {
    writable::list res = results.to_list();

    writable::doubles coefs = as_cpp<doubles>(res["coefficients"]);
    writable::integers invalid_positions =
        as_integers(find(results.valid_coefficients == 0));
    for (int i = 0; i < invalid_positions.size(); ++i) {
      coefs[invalid_positions[i]] = datum::nan;
    }

    res["coefficients"] = coefs;
    return res;
  }
}

[[cpp11::register]] list feglm_(const doubles &beta_r, const doubles &eta_r,
                                const doubles &y_r, const doubles_matrix<> &x_r,
                                const doubles &wt_r, const double &theta,
                                const std::string &family, const list &control,
                                const list &k_list) {
  mat MX = as_mat(x_r);
  vec beta = as_col(beta_r);
  vec eta = as_col(eta_r);
  const vec y = as_col(y_r);
  const vec wt = as_col(wt_r);

  const std::string fam = tidy_family(family);
  const family_type family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]);
  const double dev_tol = as_cpp<double>(control["dev_tol"]);
  const bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  const size_t iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
  const size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  const bool use_acceleration = as_cpp<bool>(control["use_acceleration"]);

  indices_info indices = list_to_indices_info(k_list);

  // Create workspace
  const size_t n = y.n_elem;
  const size_t p = MX.n_cols;
  glm_workspace ws(n, p);

  feglm_results results = feglm(MX, beta, eta, y, wt, theta, family_type,
                                center_tol, dev_tol, iter_max, iter_center_max,
                                iter_inner_max, iter_interrupt, indices, ws,
                                use_acceleration);

  if (keep_mx) {
    results.centered_matrix = std::move(MX);
  }

  // Check if we have invalid coefficients
  if (!any(results.valid_coefficients == 0)) {
    return results.to_list(keep_mx);
  } else {
    writable::list res = results.to_list(keep_mx);

    // Create coefficients with NA values for invalid ones
    writable::doubles coefs(results.coefficients.n_elem);
    const size_t I = results.coefficients.n_elem;
    for (size_t i = 0; i < I; ++i) {
      if (results.valid_coefficients[i] == 0) {
        coefs[i] = NA_REAL;
      } else {
        coefs[i] = results.coefficients[i];
      }
    }

    res["coefficients"] = coefs;
    return res;
  }
}

[[cpp11::register]] doubles
feglm_offset_(const doubles &eta_r, const doubles &y_r, const doubles &offset_r,
              const doubles &wt_r, const std::string &family,
              const list &control, const list &k_list) {
  vec eta = as_col(eta_r);
  const vec y = as_col(y_r);
  const vec offset = as_col(offset_r);
  const vec wt = as_col(wt_r);

  const std::string fam = tidy_family(family);
  const family_type family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]);
  const double dev_tol = as_cpp<double>(control["dev_tol"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  const size_t iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
  const size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  const bool use_acceleration = as_cpp<bool>(control["use_acceleration"]);

  indices_info indices = list_to_indices_info(k_list);

  // Create workspace
  const size_t n = y.n_elem;
  const size_t p = 1; // Single column for offset GLM
  glm_workspace ws(n, p);

  feglm_offset_results result = feglm_offset(
      eta, y, offset, wt, family_type, center_tol, dev_tol, iter_max,
      iter_center_max, iter_inner_max, iter_interrupt, indices, ws,
      use_acceleration);

  if (!any(result.valid_coefficients == 0)) {
    return as_doubles(result.coefficients);
  } else {
    writable::doubles eta_out(result.coefficients.n_elem);
    for (size_t i = 0; i < result.coefficients.n_elem; ++i) {
      if (result.valid_coefficients[i] == 0) {
        eta_out[i] = NA_REAL;
      } else {
        eta_out[i] = result.coefficients[i];
      }
    }
    return eta_out;
  }
}

[[cpp11::register]] list solve_alpha_(const doubles_matrix<> &p_r,
                                      const list &k_list, const list &control) {
  const vec p = as_col(p_r);
  const double tol = as_cpp<double>(control["center_tol"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  const size_t interrupt_iter0 = as_cpp<size_t>(control["iter_interrupt"]);

  indices_info indices = list_to_indices_info(k_list);

  solve_alpha_results results =
      solve_alpha(p, indices, tol, iter_max, interrupt_iter0);

  return results.to_list();
}

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  const mat M = as_mat(M_r);
  const vec w = as_col(w_r);
  single_fe_indices indices = list_to_single_fe_indices(jlist);
  mat result = group_sums(M, w, indices);
  return as_doubles_matrix(result);
}

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const size_t K,
                     const list &jlist) {
  const mat M = as_mat(M_r);
  const vec v = as_col(v_r);
  const vec w = as_col(w_r);
  single_fe_indices indices = list_to_single_fe_indices(jlist);
  mat result = group_sums_spectral(M, v, w, K, indices);
  return as_doubles_matrix(result);
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  const mat M = as_mat(M_r);
  single_fe_indices indices = list_to_single_fe_indices(jlist);
  mat result = group_sums_var(M, indices);
  return as_doubles_matrix(result);
}

[[cpp11::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  const mat M = as_mat(M_r);
  const mat N = as_mat(N_r);
  single_fe_indices indices = list_to_single_fe_indices(jlist);
  mat result = group_sums_cov(M, N, indices);
  return as_doubles_matrix(result);
}
