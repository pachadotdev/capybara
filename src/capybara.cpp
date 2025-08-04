#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <cpp11armadillo.hpp>

#include <chrono>
#include <iostream>
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

// Passing parameters from R to C++ functions
struct CapybaraParameters {
  double dev_tol;
  double center_tol;
  double collin_tol;
  size_t iter_max;
  size_t iter_demean_max;
  size_t iter_inner_max;
  size_t iter_interrupt;
  size_t iter_ssr;
  double step_halving_factor;
  bool keep_tx;
  double alpha_convergence_tol;
  size_t alpha_iter_max;

  CapybaraParameters()
      : dev_tol(1.0e-8), center_tol(1.0e-8), collin_tol(1.0e-7), iter_max(25),
        iter_demean_max(10000), iter_inner_max(50), iter_interrupt(1000), iter_ssr(10),
        step_halving_factor(0.5), keep_tx(false), alpha_convergence_tol(1.0e-8),
        alpha_iter_max(10000) {}

  explicit CapybaraParameters(const cpp11::list &control) {
    dev_tol = as_cpp<double>(control["dev_tol"]);
    center_tol = as_cpp<double>(control["center_tol"]);
    collin_tol = as_cpp<double>(control["collin_tol"]);
    iter_max = as_cpp<size_t>(control["iter_max"]);
    iter_demean_max = as_cpp<size_t>(control["iter_demean_max"]);
    iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
    iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
    iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
    step_halving_factor = as_cpp<double>(control["step_halving_factor"]);
    keep_tx = as_cpp<bool>(control["keep_tx"]);
    alpha_convergence_tol = as_cpp<double>(control["alpha_convergence_tol"]);
    alpha_iter_max = as_cpp<size_t>(control["alpha_iter_max"]);
  }
};

// Modular code structure

// Include all the necessary headers
#include "01_center.h"
#include "02_params.h"
#include "03_lm.h"
#include "04_glm.h"
#include "05_sums.h"

// Type aliases for easier access
using LMResult = capybara::InferenceLM;
using GLMResult = capybara::InferenceGLM;
using NegBinResult = capybara::InferenceNegBin;

// Helper function to convert R indices to C++ uvec (R uses 1-based, C++ uses
// 0-based)
inline uvec R_1based_to_Cpp_0based_indices(const integers &r_indices) {
  uvec cpp_indices(r_indices.size());

  // Use std::transform for efficient vectorized conversion
  std::transform(r_indices.begin(), r_indices.end(), cpp_indices.begin(),
                 [](int r_val) -> uword {
                   return static_cast<uword>(r_val - 1); // Convert to 0-based
                 });

  return cpp_indices;
}

// Convert R FEs to field<field<uvec>> format using 0-based indexing
inline field<field<uvec>> R_list_to_Armadillo_field(const list &FEs) {
  const size_t K = FEs.size();
  field<field<uvec>> group_indices(K);

  for (size_t k = 0; k < K; ++k) {
    const list group_list = as_cpp<list>(FEs[k]);
    const size_t n_groups = group_list.size();

    group_indices(k).set_size(n_groups);
    for (size_t g = 0; g < n_groups; ++g) {
      const integers group_obs = as_cpp<integers>(group_list[g]);

      // Print indices for debugging (without arbitrary limit)
      if (group_obs.size() > 0) {
        // Debugging can be added here if needed
      }

      // Create indices vector with validation
      uvec indices(group_obs.size());
      size_t I = group_obs.size();
      for (size_t i = 0; i < I; ++i) {
        // R uses 1-based indexing, C++ uses 0-based
        int r_idx = group_obs[i];
        // if (r_idx < 1) {
        //   r_idx = 1; // Set to first element if invalid
        // }
        indices[i] = static_cast<uword>(r_idx - 1); // Convert to 0-based
      }

      group_indices(k)(g) = indices;
    }
  }

  return group_indices;
}

// this function is not visible by the end-user, so we use multiple parameters
// instead of a CapybaraParameters object
[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &klist, const double &tol, const size_t &max_iter,
                  const size_t &iter_interrupt, const size_t &iter_ssr) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  // Convert R list to Armadillo field
  field<field<uvec>> group_indices = R_list_to_Armadillo_field(klist);

  // Call the C++ version with Armadillo types and proper namespace
  capybara::center_variables(V, w, group_indices, tol, max_iter, iter_interrupt,
                             iter_ssr);

  return as_doubles_matrix(V);
}

[[cpp11::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                   const doubles &y_r, const doubles &w_r,
                                   const list &FEs, const list &control) {
  // Create CapybaraParameters from control list
  CapybaraParameters params(control);

  // Convert R types to C++ types
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);

  // Convert FEs list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  // Call the main function with parameters
  capybara::InferenceLM result = capybara::felm_fit(X, y, w, fe_groups, params);

  // Extract names from the FEs list if available
  field<std::string> fe_names(FEs.size());
  field<field<std::string>> fe_levels(FEs.size());

  // Check if names attribute exists on the FEs list
  if (!FEs.names().empty()) {
    cpp11::strings fe_names_r = FEs.names();
    for (size_t i = 0; i < static_cast<size_t>(fe_names_r.size()); i++) {
      fe_names(i) = std::string(fe_names_r[i]);
    }
  }

  // Extract level names from each FE group
  for (size_t k = 0; k < static_cast<size_t>(FEs.size()); k++) {
    const list &group_list = as_cpp<list>(FEs[k]);
    fe_levels(k).set_size(group_list.size());

    // Check if names attribute exists on this group
    if (!group_list.names().empty()) {
      cpp11::strings level_names = group_list.names();
      for (size_t j = 0; j < static_cast<size_t>(level_names.size()); j++) {
        fe_levels(k)(j) = std::string(level_names[j]);
      }
    }
  }

  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

  // Create the return list using initializer list syntax
  auto ret = writable::list(
      {"coefficients"_nm = as_doubles(result.coefficients),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "residuals"_nm = as_doubles(result.residuals),
       "weights"_nm = as_doubles(result.weights),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "coef_status"_nm = as_integers(result.coef_status),
       "success"_nm = result.success,
       "has_fe"_nm = result.has_fe});

  // Add fixed effects information if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);

    // Create a vector of names for the list elements
    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      // Create a doubles object with the fixed effects values
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      // Add level names as row names if available
      if (k < fe_levels.n_elem && fe_levels(k).n_elem > 0) {
        writable::strings level_names(fe_levels(k).n_elem);
        for (size_t j = 0; j < fe_levels(k).n_elem; j++) {
          if (!fe_levels(k)(j).empty()) {
            level_names[j] = fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1); // Default numeric names
          }
        }
        fe_values.attr("names") = level_names;
      }

      // Store the fixed effect values in the list
      fe_list[k] = fe_values;

      // Save the name for this fixed effect
      if (!FEs.names().empty() && k < static_cast<size_t>(FEs.names().size())) {
        fe_list_names[k] = FEs.names()[k];
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    // Set the names on the list
    fe_list.names() = fe_list_names;

    // Add the fixed effects list to the output
    ret.push_back({"fixed_effects"_nm = fe_list});

    ret.push_back({"has_fe"_nm = result.has_fe});
  }

  // Add iterations if available
  if (!result.iterations.is_empty()) {
    ret.push_back({"iterations"_nm = as_integers(result.iterations)});
  }

  // Add design matrix if kept
  if (params.keep_tx && result.has_tx) {
    ret.push_back({"TX"_nm = as_doubles_matrix(result.X_dm)});
  }

  return ret;
}

// Wrapper function for R interface
[[cpp11::register]] list feglm_fit_(const doubles &beta_r, const doubles &eta_r,
                                    const doubles &y_r,
                                    const doubles_matrix<> &x_r,
                                    const doubles &wt_r, const double &theta,
                                    const std::string &family,
                                    const list &control, const list &k_list) {
  // Type conversion
  mat X = as_mat(x_r);
  vec beta = as_col(beta_r);
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec w = as_col(wt_r);

  // Get family type using proper namespace
  std::string fam = capybara::tidy_family_(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  // Create CapybaraParameters from control list
  CapybaraParameters params(control);

  // Convert FEs list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);

  capybara::InferenceGLM result = capybara::feglm_fit(
      beta, eta, y, X, w, theta, family_type, fe_groups, params);

  // Extract names from the k_list if available
  field<std::string> fe_names(k_list.size());
  field<field<std::string>> fe_levels(k_list.size());

  // Check if names attribute exists on the k_list
  if (!k_list.names().empty()) {
    cpp11::strings fe_names_r = k_list.names();
    for (size_t i = 0; i < static_cast<size_t>(fe_names_r.size()); i++) {
      fe_names(i) = std::string(fe_names_r[i]);
    }
  }

  // Extract level names from each FE group
  for (size_t k = 0; k < static_cast<size_t>(k_list.size()); k++) {
    const list &group_list = as_cpp<list>(k_list[k]);
    fe_levels(k).set_size(group_list.size());

    // Check if names attribute exists on this group
    if (!group_list.names().empty()) {
      cpp11::strings level_names = group_list.names();
      for (size_t j = 0; j < static_cast<size_t>(level_names.size()); j++) {
        fe_levels(k)(j) = std::string(level_names[j]);
      }
    }
  }

  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

  // Create the return list using initializer list syntax
  auto out = writable::list(
      {"coefficients"_nm = as_doubles(result.coefficients),
       "eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)})});

  // Add fixed effects information if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);

    // Create a vector of names for the list elements
    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      // Create a doubles object with the fixed effects values
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      // Add level names as row names if available
      if (k < fe_levels.n_elem && fe_levels(k).n_elem > 0) {
        writable::strings level_names(fe_levels(k).n_elem);
        for (size_t j = 0; j < fe_levels(k).n_elem; j++) {
          if (!fe_levels(k)(j).empty()) {
            level_names[j] = fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1); // Default numeric names
          }
        }
        fe_values.attr("names") = level_names;
      }

      // Store the fixed effect values in the list
      fe_list[k] = fe_values;

      // Save the name for this fixed effect
      if (!k_list.names().empty() &&
          k < static_cast<size_t>(k_list.names().size())) {
        fe_list_names[k] = k_list.names()[k];
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    // Set the names on the list
    fe_list.names() = fe_list_names;

    // Add the fixed effects list to the output
    out.push_back({"fixed_effects"_nm = fe_list});
  }

  // Add design matrix if kept
  if (params.keep_tx && result.has_tx) {
    out.push_back({"TX"_nm = as_doubles_matrix(result.X_dm)});
  }

  return out;
}

// R-facing wrapper function
[[cpp11::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
                  const list &k_list) {
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec offset = as_col(offset_r);
  vec w = as_col(wt_r);

  // Create CapybaraParameters from control list
  CapybaraParameters params(control);

  // Convert R list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);

  // Get family type
  std::string fam = capybara::tidy_family_(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  // Call the C++ implementation
  vec result = capybara::feglm_offset_fit(eta, y, offset, w, family_type,
                                          fe_groups, params);

  return as_doubles(result);
}

// R-facing wrapper function for negative binomial models - fix parameter type
[[cpp11::register]] list
fenegbin_fit_(const doubles_matrix<> &X_r, const doubles &y_r,
              const doubles &w_r, const list &FEs, const std::string &link,
              const doubles &beta_r, const doubles &eta_r,
              const double &init_theta, const list &control) {
  // Convert R types to Armadillo types
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);

  // Create CapybaraParameters from control list
  CapybaraParameters params(control);

  // Convert R list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  // Call the C++ implementation
  capybara::InferenceNegBin result =
      capybara::fenegbin_fit(X, y, w, fe_groups, params, init_theta);

  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

  // Create return list using initializer list syntax
  auto out = writable::list(
      {"coefficients"_nm = as_doubles(result.coefficients),
       "eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)}),
       "theta"_nm = writable::doubles({result.theta}),
       "iter.outer"_nm =
           writable::integers({static_cast<int>(result.iter_outer)}),
       "conv_outer"_nm = writable::logicals({result.conv_outer})});

  // Add fixed effects if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      fe_list[k] = as_doubles(result.fixed_effects(k));
    }
    out.push_back({"fixed_effects"_nm = fe_list});

    out.push_back({"has_fe"_nm = result.has_fe});
  }

  // Add design matrix if requested
  if (result.has_tx) {
    out.push_back({"TX"_nm = as_doubles_matrix(result.X_dm)});
  }

  return out;
}

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  // Convert R types to C++ types
  const mat M = as_mat(M_r);
  const mat w = as_mat(w_r);

  // Convert R list of indices to C++ field of uvec
  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  // Call the C++ implementation
  mat result = capybara::group_sums(M, w, group_indices);

  // Convert result back to R type
  return as_doubles_matrix(result);
}

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const int K,
                     const list &jlist) {
  // Convert R types to C++ types
  const mat M = as_mat(M_r);
  const mat v = as_mat(v_r);
  const mat w = as_mat(w_r);

  // Convert R list of indices to C++ field of uvec
  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  // Call the C++ implementation
  mat result = capybara::group_sums_spectral(M, v, w, K, group_indices);

  // Convert result back to R type
  return as_doubles_matrix(result);
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  // Convert R types to C++ types
  const mat M = as_mat(M_r);

  // Convert R list of indices to C++ field of uvec
  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  // Call the C++ implementation
  mat result = capybara::group_sums_var(M, group_indices);

  // Convert result back to R type
  return as_doubles_matrix(result);
}

[[cpp11::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  // Convert R types to C++ types
  const mat M = as_mat(M_r);
  const mat N = as_mat(N_r);

  // Convert R list of indices to C++ field of uvec
  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  // Call the C++ implementation
  mat result = capybara::group_sums_cov(M, N, group_indices);

  // Convert result back to R type
  return as_doubles_matrix(result);
}
