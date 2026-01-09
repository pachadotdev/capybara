#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <armadillo4r.hpp>

using arma::field;
using arma::mat;
using arma::uvec;
using arma::uword;
using arma::vec;

using cpp4r::doubles;
using cpp4r::doubles_matrix;
using cpp4r::integers;
using cpp4r::list;
using cpp4r::strings;

// Passing parameters from R to C++ functions
// see R/fit_control.R
struct CapybaraParameters {
  double dev_tol;
  double center_tol;
  double collin_tol;
  double step_halving_factor;
  double alpha_tol;
  size_t iter_max;
  size_t iter_center_max;
  size_t iter_inner_max;
  size_t iter_alpha_max;
  size_t iter_interrupt;
  bool return_fe;
  bool keep_tx;

  // Step-halving parameters
  double step_halving_memory;
  size_t max_step_halving;
  double start_inner_tol;

  CapybaraParameters()
      : dev_tol(1.0e-08), center_tol(1.0e-08), collin_tol(1.0e-10),
        step_halving_factor(0.5), alpha_tol(1.0e-08), iter_max(25),
        iter_center_max(10000), iter_inner_max(50), iter_alpha_max(10000),
        iter_interrupt(1000), return_fe(true), keep_tx(false),
        step_halving_memory(0.9), max_step_halving(2), start_inner_tol(1e-06) {}

  explicit CapybaraParameters(const cpp4r::list &control) {
    dev_tol = as_cpp<double>(control["dev_tol"]);
    center_tol = as_cpp<double>(control["center_tol"]);
    collin_tol = as_cpp<double>(control["collin_tol"]);
    step_halving_factor = as_cpp<double>(control["step_halving_factor"]);
    alpha_tol = as_cpp<double>(control["alpha_tol"]);
    iter_max = as_cpp<size_t>(control["iter_max"]);
    iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
    iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
    iter_alpha_max = as_cpp<size_t>(control["iter_alpha_max"]);
    iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
    return_fe = as_cpp<bool>(control["return_fe"]);
    keep_tx = as_cpp<bool>(control["keep_tx"]);
    step_halving_memory = as_cpp<double>(control["step_halving_memory"]);
    max_step_halving = as_cpp<size_t>(control["max_step_halving"]);
    start_inner_tol = as_cpp<double>(control["start_inner_tol"]);
  }
};

#include "01_center.h"
#include "02_beta.h"
#include "03_alpha.h"
#include "04_fit_helpers.h"
#include "05_lm.h"
#include "06_glm.h"
#include "07_negbin.h"
#include "08_separation.h"

using LMResult = capybara::InferenceLM;
using GLMResult = capybara::InferenceGLM;
using NegBinResult = capybara::InferenceNegBin;

// Convert R indexing to C++ indexing
inline uvec R_1based_to_Cpp_0based_indices(const integers &r_indices) {
  uvec cpp_indices(r_indices.size());

  std::transform(
      r_indices.begin(), r_indices.end(), cpp_indices.begin(),
      [](size_t r_val) -> uword { return static_cast<uword>(r_val - 1); });

  return cpp_indices;
}

// Convert R FEs from list to Armadillo field<field<uvec>>
inline field<field<uvec>> R_list_to_Armadillo_field(const list &FEs) {
  const size_t K = FEs.size();
  field<field<uvec>> group_indices(K);

  for (size_t k = 0; k < K; ++k) {
    const list group_list = as_cpp<list>(FEs[k]);
    const size_t n_groups = group_list.size();

    group_indices(k).set_size(n_groups);
    for (size_t g = 0; g < n_groups; ++g) {
      const integers group_obs = as_cpp<integers>(group_list[g]);

      uvec indices(group_obs.size());
      size_t I = group_obs.size();
      for (size_t i = 0; i < I; ++i) {
        size_t r_idx = group_obs[i];
        // if (r_idx < 1) {
        //   r_idx = 1; // Set to first element if invalid
        // }
        indices[i] = static_cast<uword>(r_idx - 1);
      }

      group_indices(k)(g) = indices;
    }
  }

  return group_indices;
}

// this function is not visible by the end-user, so we use multiple parameters
// instead of a CapybaraParameters object
[[cpp4r::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &klist, const double &tol, const size_t &max_iter,
                  const size_t &iter_interrupt) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  field<field<uvec>> group_indices = R_list_to_Armadillo_field(klist);

  capybara::center_variables(V, w, group_indices, tol, max_iter,
                             iter_interrupt);

  return as_doubles_matrix(V);
}

[[cpp4r::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                   const doubles &y_r, const doubles &w_r,
                                   const list &FEs, const list &control,
                                   const list &cl_list) {
  CapybaraParameters params(control);

  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  // Convert cluster list to Armadillo field<uvec>
  field<uvec> cluster_groups;
  bool has_clusters = cl_list.size() > 0;
  if (has_clusters) {
    cluster_groups.set_size(cl_list.size());
    for (R_xlen_t g = 0; g < cl_list.size(); ++g) {
      const integers group_obs = as_cpp<integers>(cl_list[g]);
      uvec indices(group_obs.size());
      for (size_t i = 0; i < static_cast<size_t>(group_obs.size()); ++i) {
        indices[i] = static_cast<uword>(group_obs[i] - 1);
      }
      cluster_groups(g) = indices;
    }
  }
  const field<uvec> *cluster_ptr = has_clusters ? &cluster_groups : nullptr;

  capybara::InferenceLM result =
      capybara::felm_fit(X, y, w, fe_groups, params, nullptr, cluster_ptr);

  field<std::string> fe_names(FEs.size());
  field<field<std::string>> fe_levels(FEs.size());

  if (!FEs.names().empty()) {
    cpp4r::strings fe_names_r = FEs.names();
    for (R_xlen_t i = 0; i < fe_names_r.size(); i++) {
      fe_names(i) = std::string(fe_names_r[i]);
    }
  }

  for (R_xlen_t k = 0; k < FEs.size(); k++) {
    const list &group_list = as_cpp<list>(FEs[k]);
    fe_levels(k).set_size(group_list.size());

    if (!group_list.names().empty()) {
      cpp4r::strings level_names = group_list.names();
      for (R_xlen_t j = 0; j < level_names.size(); j++) {
        fe_levels(k)(j) = std::string(level_names[j]);
      }
    }
  }

  // Replace collinear coefficients (NaN) with R's NA_REAL in coef_table
  vec coefficients = result.coef_table.col(0);
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    coefficients.elem(find(collinear_mask)).fill(NA_REAL);
    result.coef_table.col(0) = coefficients;
  }

  auto ret = writable::list(
      {"fitted_values"_nm = as_doubles(result.fitted_values),
       "residuals"_nm = as_doubles(result.residuals),
       "weights"_nm = as_doubles(result.weights),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "r_squared"_nm = result.r_squared,
       "adj_r_squared"_nm = result.adj_r_squared,
       "coef_status"_nm = as_integers(result.coef_status),
       "success"_nm = result.success, "has_fe"_nm = result.has_fe});

  // Add fixed effects information if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);

    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < fe_levels.n_elem && fe_levels(k).n_elem > 0) {
        writable::strings level_names(fe_levels(k).n_elem);
        for (size_t j = 0; j < fe_levels(k).n_elem; j++) {
          if (!fe_levels(k)(j).empty()) {
            level_names[j] = fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1); // fallback to numeric names
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (!FEs.names().empty() && k < static_cast<size_t>(FEs.names().size())) {
        fe_list_names[k] = FEs.names()[k];
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;

    ret.push_back({"fixed_effects"_nm = fe_list});

    ret.push_back({"has_fe"_nm = result.has_fe});
  }

  if (!result.iterations.is_empty()) {
    ret.push_back({"iterations"_nm = as_integers(result.iterations)});
  }

  if (params.keep_tx && result.has_tx) {
    ret.push_back({"TX"_nm = as_doubles_matrix(result.TX)});
  }

  return ret;
}

[[cpp4r::register]] list
feglm_fit_(const doubles &beta_r, const doubles &eta_r, const doubles &y_r,
           const doubles_matrix<> &x_r, const doubles &wt_r,
           const doubles &offset_r, const double &theta,
           const std::string &family, const list &control, const list &k_list,
           const list &cl_list) {
  mat X = as_mat(x_r);
  vec beta = as_col(beta_r);
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec w = as_col(wt_r);
  vec offset = as_col(offset_r);

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  CapybaraParameters params(control);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);

  // Convert cluster list to Armadillo field<uvec>
  field<uvec> cluster_groups;
  bool has_clusters = cl_list.size() > 0;
  if (has_clusters) {
    cluster_groups.set_size(cl_list.size());
    for (R_xlen_t g = 0; g < cl_list.size(); ++g) {
      const integers group_obs = as_cpp<integers>(cl_list[g]);
      uvec indices(group_obs.size());
      for (size_t i = 0; i < static_cast<size_t>(group_obs.size()); ++i) {
        indices[i] = static_cast<uword>(group_obs[i] - 1);
      }
      cluster_groups(g) = indices;
    }
  }

  // Add offset to eta (the linear predictor is eta = X*beta + alpha + offset)
  eta += offset;

  // Pass offset pointer so fixed effects can be computed correctly
  const vec *offset_ptr = (any(offset != 0.0)) ? &offset : nullptr;

  capybara::InferenceGLM result = capybara::feglm_fit(
      beta, eta, y, X, w, theta, family_type, fe_groups, params, nullptr,
      has_clusters ? &cluster_groups : nullptr, offset_ptr);

  field<std::string> fe_names(k_list.size());
  field<field<std::string>> fe_levels(k_list.size());

  if (!k_list.names().empty()) {
    cpp4r::strings fe_names_r = k_list.names();
    for (R_xlen_t i = 0; i < fe_names_r.size(); i++) {
      fe_names(i) = std::string(fe_names_r[i]);
    }
  }

  for (R_xlen_t k = 0; k < k_list.size(); k++) {
    const list &group_list = as_cpp<list>(k_list[k]);
    fe_levels(k).set_size(group_list.size());

    if (!group_list.names().empty()) {
      cpp4r::strings level_names = group_list.names();
      for (R_xlen_t j = 0; j < level_names.size(); j++) {
        fe_levels(k)(j) = std::string(level_names[j]);
      }
    }
  }

  vec coefficients = result.coef_table.col(0);
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    coefficients.elem(find(collinear_mask)).fill(NA_REAL);
    result.coef_table.col(0) = coefficients;
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)})});

  // Add pseudo R-squared for Poisson models
  if (family_type == capybara::POISSON && result.pseudo_rsq > 0.0) {
    out.push_back({"pseudo.rsq"_nm = result.pseudo_rsq});
  }

  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);

    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < fe_levels.n_elem && fe_levels(k).n_elem > 0) {
        writable::strings level_names(fe_levels(k).n_elem);
        for (size_t j = 0; j < fe_levels(k).n_elem; j++) {
          if (!fe_levels(k)(j).empty()) {
            level_names[j] = fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (!k_list.names().empty() &&
          k < static_cast<size_t>(k_list.names().size())) {
        fe_list_names[k] = k_list.names()[k];
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;

    out.push_back({"fixed_effects"_nm = fe_list});
  }

  if (params.keep_tx && result.has_tx) {
    out.push_back({"TX"_nm = as_doubles_matrix(result.TX)});
  }

  return out;
}

[[cpp4r::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
                  const list &k_list) {
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec offset = as_col(offset_r);
  vec w = as_col(wt_r);

  CapybaraParameters params(control);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  vec result = capybara::feglm_offset_fit(eta, y, offset, w, family_type,
                                          fe_groups, params);

  return as_doubles(result);
}

[[cpp4r::register]] list
fenegbin_fit_(const doubles_matrix<> &X_r, const doubles &y_r,
              const doubles &w_r, const list &FEs, const std::string &link,
              const doubles &beta_r, const doubles &eta_r,
              const double &init_theta, const doubles &offset_r,
              const list &control) {
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);
  vec offset_vec = as_col(offset_r);

  CapybaraParameters params(control);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  capybara::InferenceNegBin result = capybara::fenegbin_fit(
      X, y, w, fe_groups, params, offset_vec, init_theta);

  vec coefficients = result.coef_table.col(0);
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    coefficients.elem(find(collinear_mask)).fill(NA_REAL);
    result.coef_table.col(0) = coefficients;
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)}),
       "theta"_nm = writable::doubles({result.theta}),
       "iter.outer"_nm =
           writable::integers({static_cast<int>(result.iter_outer)}),
       "conv_outer"_nm = writable::logicals({result.conv_outer})});

  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      fe_list[k] = as_doubles(result.fixed_effects(k));
    }
    out.push_back({"fixed_effects"_nm = fe_list});

    out.push_back({"has_fe"_nm = result.has_fe});
  }

  if (result.has_tx) {
    out.push_back({"TX"_nm = as_doubles_matrix(result.TX)});
  }

  return out;
}

[[cpp4r::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  const mat M = as_mat(M_r);
  const mat w = as_mat(w_r);

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  mat result = capybara::group_sums(M, w, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const size_t K,
                     const list &jlist) {
  const mat M = as_mat(M_r);
  const mat v = as_mat(v_r);
  const mat w = as_mat(w_r);

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  mat result = capybara::group_sums_spectral(M, v, w, K, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  const mat M = as_mat(M_r);

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  mat result = capybara::group_sums_var(M, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  const mat M = as_mat(M_r);
  const mat N = as_mat(N_r);

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  mat result = capybara::group_sums_cov(M, N, group_indices);

  return as_doubles_matrix(result);
}

// Separation detection for Poisson models
// Based on Correia, Guimarães, Zylkin (2019)
[[cpp4r::register]] list
check_separation_(const doubles &y_r, const doubles_matrix<> &X_r,
                  const doubles &w_r, const double &tol, const double &zero_tol,
                  const size_t &max_iter, const size_t &simplex_max_iter,
                  const bool &use_relu, const bool &use_simplex,
                  const bool &verbose) {
  vec y = as_col(y_r);
  mat X = as_mat(X_r);
  vec w = as_col(w_r);

  capybara::SeparationParameters params;
  params.tol = tol;
  params.zero_tol = zero_tol;
  params.max_iter = max_iter;
  params.simplex_max_iter = simplex_max_iter;
  params.use_relu = use_relu;
  params.use_simplex = use_simplex;
  params.verbose = verbose;

  capybara::SeparationResult result =
      capybara::check_separation(y, X, w, params);

  // Convert 0-based indices to 1-based for R
  vec separated_obs_r(result.separated_obs.n_elem);
  for (size_t i = 0; i < result.separated_obs.n_elem; ++i) {
    separated_obs_r(i) = static_cast<double>(result.separated_obs(i) + 1);
  }

  auto out = writable::list(
      {"separated_obs"_nm = as_doubles(separated_obs_r),
       "num_separated"_nm =
           writable::integers({static_cast<int>(result.num_separated)}),
       "converged"_nm = writable::logicals({result.converged}),
       "iterations"_nm =
           writable::integers({static_cast<int>(result.iterations)})});

  if (result.certificate.n_elem > 0) {
    out.push_back({"certificate"_nm = as_doubles(result.certificate)});
  }

  return out;
}
