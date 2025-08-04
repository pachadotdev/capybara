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
  // Core tolerance parameters
  double dev_tol;
  double demean_tol;
  double collin_tol;

  // Iteration parameters
  size_t iter_max;
  size_t iter_max_cluster;
  size_t iter_full_dicho;
  size_t iter_demean_max;
  size_t iter_inner_max;
  size_t iter_interrupt;
  size_t iter_ssr;

  // Convergence parameters
  double rel_tol_denom;
  double irons_tuck_eps;
  double safe_division_min;
  double safe_log_min;
  double newton_raphson_tol;
  size_t convergence_iter_max;
  size_t convergence_iter_full_dicho;

  // GLM parameters
  double step_halving_factor;
  double binomial_mu_min;
  double binomial_mu_max;
  double safe_clamp_min;
  double safe_clamp_max;
  double glm_init_eta;

  // Negative binomial parameters
  size_t iter_nb_theta;
  double nb_theta_tol;
  double nb_info_min;
  double nb_overdispersion_threshold;
  double nb_theta_min;
  double nb_theta_max;
  double nb_step_max_decrease;
  double nb_step_max_increase;

  // Algorithm configuration
  double direct_qr_threshold;
  double qr_collin_tol_multiplier;
  double chol_stability_threshold;

  // Alpha computation
  double alpha_convergence_tol;
  size_t alpha_iter_max;

  // Demean algorithm
  size_t demean_extra_projections;
  size_t demean_warmup_iterations;
  size_t demean_projections_after_acc;
  size_t demean_grand_acc_frequency;
  size_t demean_ssr_check_frequency;

  // 2-FEs specific parameters
  size_t demean_2fe_max_iter;
  double demean_2fe_tolerance;

  // Configuration flags
  bool keep_dmx;
  bool use_weights;

  // Constructor with default initialization (matching fit_control defaults)
  CapybaraParameters()
      : dev_tol(1.0e-8), demean_tol(1.0e-8), collin_tol(1.0e-7), iter_max(25),
        iter_max_cluster(100), iter_full_dicho(10), iter_demean_max(10000),
        iter_inner_max(50), iter_interrupt(1000), iter_ssr(10),
        rel_tol_denom(0.1), irons_tuck_eps(1.0e-14), safe_division_min(1.0e-12),
        safe_log_min(1.0e-12), newton_raphson_tol(1.0e-8),
        convergence_iter_max(100), convergence_iter_full_dicho(10),
        step_halving_factor(0.5), binomial_mu_min(0.001),
        binomial_mu_max(0.999), safe_clamp_min(1.0e-15), safe_clamp_max(1.0e12),
        glm_init_eta(1.0e-5), iter_nb_theta(10), nb_theta_tol(1.0e-6),
        nb_info_min(1.0e-12), nb_overdispersion_threshold(0.01),
        nb_theta_min(0.1), nb_theta_max(1.0e6), nb_step_max_decrease(0.1),
        nb_step_max_increase(0.5), direct_qr_threshold(0.9),
        qr_collin_tol_multiplier(1.0e-7), chol_stability_threshold(1.0e-12),
        alpha_convergence_tol(1.0e-8), alpha_iter_max(10000),
        demean_extra_projections(0), demean_warmup_iterations(5),
        demean_projections_after_acc(5), demean_grand_acc_frequency(20),
        demean_ssr_check_frequency(40), demean_2fe_max_iter(100),
        demean_2fe_tolerance(1.0e-12), keep_dmx(false), use_weights(true) {}

  // Constructor from R control list
  explicit CapybaraParameters(const cpp11::list &control) {
    // Extract all parameters from R control list
    dev_tol = as_cpp<double>(control["dev_tol"]);
    demean_tol = as_cpp<double>(control["demean_tol"]);
    collin_tol = as_cpp<double>(control["collin_tol"]);

    iter_max = as_cpp<size_t>(control["iter_max"]);
    iter_max_cluster = as_cpp<size_t>(control["iter_max_cluster"]);
    iter_full_dicho = as_cpp<size_t>(control["iter_full_dicho"]);
    iter_demean_max = as_cpp<size_t>(control["iter_demean_max"]);
    iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
    iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
    iter_ssr = as_cpp<size_t>(control["iter_ssr"]);

    rel_tol_denom = as_cpp<double>(control["rel_tol_denom"]);
    irons_tuck_eps = as_cpp<double>(control["irons_tuck_eps"]);
    safe_division_min = as_cpp<double>(control["safe_division_min"]);
    safe_log_min = as_cpp<double>(control["safe_log_min"]);
    newton_raphson_tol = as_cpp<double>(control["newton_raphson_tol"]);

    convergence_iter_max = as_cpp<size_t>(control["convergence_iter_max"]);
    convergence_iter_full_dicho =
        as_cpp<size_t>(control["convergence_iter_full_dicho"]);

    step_halving_factor = as_cpp<double>(control["step_halving_factor"]);
    binomial_mu_min = as_cpp<double>(control["binomial_mu_min"]);
    binomial_mu_max = as_cpp<double>(control["binomial_mu_max"]);
    safe_clamp_min = as_cpp<double>(control["safe_clamp_min"]);
    safe_clamp_max = as_cpp<double>(control["safe_clamp_max"]);
    glm_init_eta = as_cpp<double>(control["glm_init_eta"]);

    iter_nb_theta = as_cpp<size_t>(control["iter_nb_theta"]);
    nb_theta_tol = as_cpp<double>(control["nb_theta_tol"]);
    nb_info_min = as_cpp<double>(control["nb_info_min"]);
    nb_overdispersion_threshold =
        as_cpp<double>(control["nb_overdispersion_threshold"]);
    nb_theta_min = as_cpp<double>(control["nb_theta_min"]);
    nb_theta_max = as_cpp<double>(control["nb_theta_max"]);
    nb_step_max_decrease = as_cpp<double>(control["nb_step_max_decrease"]);
    nb_step_max_increase = as_cpp<double>(control["nb_step_max_increase"]);

    direct_qr_threshold = as_cpp<double>(control["direct_qr_threshold"]);
    qr_collin_tol_multiplier =
        as_cpp<double>(control["qr_collin_tol_multiplier"]);
    chol_stability_threshold =
        as_cpp<double>(control["chol_stability_threshold"]);

    alpha_convergence_tol = as_cpp<double>(control["alpha_convergence_tol"]);
    alpha_iter_max = as_cpp<size_t>(control["alpha_iter_max"]);

    demean_extra_projections =
        as_cpp<size_t>(control["demean_extra_projections"]);
    demean_warmup_iterations =
        as_cpp<size_t>(control["demean_warmup_iterations"]);
    demean_projections_after_acc =
        as_cpp<size_t>(control["demean_projections_after_acc"]);
    demean_grand_acc_frequency =
        as_cpp<size_t>(control["demean_grand_acc_frequency"]);
    demean_ssr_check_frequency =
        as_cpp<size_t>(control["demean_ssr_check_frequency"]);

    demean_2fe_max_iter = as_cpp<size_t>(control["demean_2fe_max_iter"]);
    demean_2fe_tolerance = as_cpp<double>(control["demean_2fe_tolerance"]);

    keep_dmx = as_cpp<bool>(control["keep_dmx"]);
    use_weights = as_cpp<bool>(control["use_weights"]);
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

[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                   const list &klist, const double &tol, const size_t &max_iter,
                   const size_t &iter_interrupt, const size_t &iter_ssr) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);
  
  // Convert R list to Armadillo field
  field<field<uvec>> group_indices = R_list_to_Armadillo_field(klist);
  
  // Call the C++ version with Armadillo types and proper namespace
  capybara::center_variables(V, w, group_indices, tol, max_iter, iter_interrupt, iter_ssr);
  
  return as_doubles_matrix(V);
}

[[cpp11::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                  const doubles &y_r, const doubles &w_r,
                                  const list &FEs, const list &control) {
  // Extract control parameters
  double center_tol = as_cpp<double>(control["center_tol"]);
  size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  size_t iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  
  // Convert R types to C++ types
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);
  
  // Convert FEs list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  // Call the main function - it already computes fixed effects correctly!
  capybara::InferenceLM result = capybara::felm_fit(X, y, w, fe_groups, 
                               center_tol, iter_center_max, 
                               iter_interrupt, iter_ssr);
    
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
  
  // DON'T RECALCULATE FIXED EFFECTS - felm_fit already did this correctly!
  // Just use what felm_fit computed
  
  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }
  
  // Create the return list using initializer list syntax
  auto ret = writable::list(
    {"coefficients"_nm = as_doubles(result.coefficients),
     "fitted.values"_nm = as_doubles(result.fitted_values),
     "residuals"_nm = as_doubles(result.residuals),
     "weights"_nm = as_doubles(result.weights),
     "hessian"_nm = as_doubles_matrix(result.hessian),
     "coef_status"_nm = as_integers(result.coef_status),
     "success"_nm = result.success,
     "has_fe"_nm = result.has_fe}
  );
  
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
            level_names[j] = std::to_string(j+1); // Default numeric names
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
        fe_list_names[k] = std::to_string(k+1);
      }
    }
    
    // Set the names on the list
    fe_list.names() = fe_list_names;
    
    // Add the fixed effects list to the output
    ret.push_back({"fixed.effects"_nm = fe_list});
    
    if (!result.nb_references.is_empty()) {
      ret.push_back({"nb_references"_nm = as_integers(result.nb_references)});
    }
    
    ret.push_back({"is_regular"_nm = result.is_regular});
  }
  
  // Add iterations if available
  if (!result.iterations.is_empty()) {
    ret.push_back({"iterations"_nm = as_integers(result.iterations)});
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
  mat MX = as_mat(x_r);
  vec beta = as_col(beta_r);
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec w = as_col(wt_r);

  // Get family type using proper namespace
  std::string fam = capybara::tidy_family_(family);
  capybara::Family family_type = capybara::get_family_type(fam);
  
  // Extract control parameters
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  size_t iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
  size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  size_t iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  
  // Convert FEs list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);
  
  // Call the implementation function with proper namespace
  capybara::InferenceGLM result = capybara::feglm_fit(beta, eta, y, MX, w, theta, family_type,
                                fe_groups, center_tol, iter_max, iter_center_max, 
                                iter_inner_max, iter_interrupt, iter_ssr, 
                                dev_tol, keep_mx);
  
  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }
  
  // Create the return list using initializer list syntax
  auto out = writable::list(
    {"coefficients"_nm = as_doubles(result.coefficients),
     "eta"_nm = as_doubles(result.eta),
     "fitted.values"_nm = as_doubles(result.fitted_values),
     "weights"_nm = as_doubles(result.weights),
     "hessian"_nm = as_doubles_matrix(result.hessian),
     "deviance"_nm = writable::doubles({result.deviance}),
     "null_deviance"_nm = writable::doubles({result.null_deviance}),
     "conv"_nm = writable::logicals({result.conv}),
     "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)})}
  );
  
  // Add fixed effects if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      fe_list[k] = as_doubles(result.fixed_effects(k));
    }
    out.push_back({"fixed.effects"_nm = fe_list});
    
    if (!result.nb_references.is_empty()) {
      out.push_back({"nb_references"_nm = as_integers(result.nb_references)});
    }
    
    out.push_back({"is_regular"_nm = result.is_regular});
  }
  
  // Add design matrix if kept
  if (keep_mx) {
    out.push_back({"MX"_nm = as_doubles_matrix(result.X_dm)});
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
  
  // Get control parameters
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  size_t iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
  size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  size_t iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  
  // Convert R list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);
  
  // Get family type
  std::string fam = capybara::tidy_family_(family);
  capybara::Family family_type = capybara::get_family_type(fam);
  
  // Call the C++ implementation
  vec result = capybara::feglm_offset_fit(eta, y, offset, w, family_type, 
                               fe_groups, center_tol, iter_max, iter_center_max, 
                               iter_inner_max, iter_interrupt, iter_ssr, dev_tol);
  
  return as_doubles(result);
}

// R-facing wrapper function for negative binomial models - fix parameter type
[[cpp11::register]] list fenegbin_fit_(const doubles_matrix<> &X_r,
                                     const doubles &y_r, 
                                     const doubles &w_r,
                                     const list &FEs, 
                                     const std::string &link,
                                     const doubles &beta_r,
                                     const doubles &eta_r,
                                     const double &init_theta,
                                     const list &control) {
  // Convert R types to Armadillo types
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);
  
  // Get control parameters
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  size_t iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
  size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  size_t iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
  bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  
  // Convert R list to Armadillo field
  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);
  
  // Call the C++ implementation
  capybara::InferenceNegBin result = capybara::fenegbin_fit(X, y, w, fe_groups, 
                                      center_tol, iter_max, iter_center_max, 
                                      iter_inner_max, iter_interrupt, iter_ssr, 
                                      dev_tol, keep_mx, init_theta);
  
  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }
  
  // Create return list using initializer list syntax
  auto out = writable::list(
    {"coefficients"_nm = as_doubles(result.coefficients),
     "eta"_nm = as_doubles(result.eta),
     "fitted.values"_nm = as_doubles(result.fitted_values),
     "weights"_nm = as_doubles(result.weights),
     "hessian"_nm = as_doubles_matrix(result.hessian),
     "deviance"_nm = writable::doubles({result.deviance}),
     "null_deviance"_nm = writable::doubles({result.null_deviance}),
     "conv"_nm = writable::logicals({result.conv}),
     "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)}),
     "theta"_nm = writable::doubles({result.theta}),
     "iter.outer"_nm = writable::integers({static_cast<int>(result.iter_outer)}),
     "conv.outer"_nm = writable::logicals({result.conv_outer})}
  );
  
  // Add fixed effects if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      fe_list[k] = as_doubles(result.fixed_effects(k));
    }
    out.push_back({"fixed.effects"_nm = fe_list});
    
    if (!result.nb_references.is_empty()) {
      out.push_back({"nb_references"_nm = as_integers(result.nb_references)});
    }
    
    out.push_back({"is_regular"_nm = writable::logicals({result.is_regular})});
  }
  
  // Add design matrix if requested
  if (result.has_mx) {
    out.push_back({"MX"_nm = as_doubles_matrix(result.X_dm)});
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
    group_indices(j) = R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
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
    group_indices(j) = R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
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
    group_indices(j) = R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
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
    group_indices(j) = R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }
  
  // Call the C++ implementation
  mat result = capybara::group_sums_cov(M, N, group_indices);
  
  // Convert result back to R type
  return as_doubles_matrix(result);
}
