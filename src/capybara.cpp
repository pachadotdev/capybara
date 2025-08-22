#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <cpp11armadillo.hpp>

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
  size_t iter_ssr;
  bool return_fe;
  bool keep_tx;

  // Step-halving parameters
  double step_halving_memory;
  size_t max_step_halving;
  double start_inner_tol;

  // CG acceleration parameters
  bool use_cg;
  size_t accel_start;

  CapybaraParameters()
      : dev_tol(1.0e-08), center_tol(1.0e-08), collin_tol(1.0e-10),
        step_halving_factor(0.5), alpha_tol(1.0e-08), iter_max(25),
        iter_center_max(10000), iter_inner_max(50), iter_alpha_max(10000),
        iter_interrupt(1000), iter_ssr(10), return_fe(true), keep_tx(false),
        step_halving_memory(0.9), max_step_halving(2), start_inner_tol(1e-06),
        use_cg(true), accel_start(6) {}

  explicit CapybaraParameters(const cpp11::list &control) {
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
    iter_ssr = as_cpp<size_t>(control["iter_ssr"]);
    return_fe = as_cpp<bool>(control["return_fe"]);
    keep_tx = as_cpp<bool>(control["keep_tx"]);
    step_halving_memory = as_cpp<double>(control["step_halving_memory"]);
    max_step_halving = as_cpp<size_t>(control["max_step_halving"]);
    start_inner_tol = as_cpp<double>(control["start_inner_tol"]);
    use_cg = as_cpp<bool>(control["use_cg"]);
    accel_start = as_cpp<size_t>(control["accel_start"]);
  }
};

#include "01_center.h"
#include "02_beta.h"
#include "03_alpha.h"
#include "04_lm.h"
#include "05_glm_helpers.h"
#include "06_glm.h"
#include "07_negbin.h"
#include "08_sums.h"

using LMResult = capybara::InferenceLM;
using GLMResult = capybara::InferenceGLM;
using NegBinResult = capybara::InferenceNegBin;

// Convert R indexing to C++ indexing
inline uvec R_1based_to_Cpp_0based_indices(const integers &r_indices) {
  uvec cpp_indices(r_indices.size());

  std::transform(
      r_indices.begin(), r_indices.end(), cpp_indices.begin(),
      [](int r_val) -> uword { return static_cast<uword>(r_val - 1); });

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
        int r_idx = group_obs[i];
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
[[cpp11::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &klist, const double &tol, const size_t &max_iter,
                  const size_t &iter_interrupt, const size_t &iter_ssr,
                  const size_t &accel_start, const bool &use_cg) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  field<field<uvec>> group_indices = R_list_to_Armadillo_field(klist);

  capybara::center_variables(V, w, group_indices, tol, max_iter, iter_interrupt,
                             iter_ssr, accel_start, use_cg);

  return as_doubles_matrix(V);
}

[[cpp11::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                   const doubles &y_r, const doubles &w_r,
                                   const list &FEs, const list &control) {
  CapybaraParameters params(control);

  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  capybara::InferenceLM result = capybara::felm_fit(X, y, w, fe_groups, params);

  field<std::string> fe_names(FEs.size());
  field<field<std::string>> fe_levels(FEs.size());

  if (!FEs.names().empty()) {
    cpp11::strings fe_names_r = FEs.names();
    for (size_t i = 0; i < static_cast<size_t>(fe_names_r.size()); i++) {
      fe_names(i) = std::string(fe_names_r[i]);
    }
  }

  for (size_t k = 0; k < static_cast<size_t>(FEs.size()); k++) {
    const list &group_list = as_cpp<list>(FEs[k]);
    fe_levels(k).set_size(group_list.size());

    if (!group_list.names().empty()) {
      cpp11::strings level_names = group_list.names();
      for (size_t j = 0; j < static_cast<size_t>(level_names.size()); j++) {
        fe_levels(k)(j) = std::string(level_names[j]);
      }
    }
  }

  // Replace collinear coefficients (NaN) with R's NA_REAL
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

  auto ret = writable::list(
      {"coefficients"_nm = as_doubles(result.coefficients),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "residuals"_nm = as_doubles(result.residuals),
       "weights"_nm = as_doubles(result.weights),
       "hessian"_nm = as_doubles_matrix(result.hessian),
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

[[cpp11::register]] list feglm_fit_(const doubles &beta_r, const doubles &eta_r,
                                    const doubles &y_r,
                                    const doubles_matrix<> &x_r,
                                    const doubles &wt_r, const double &theta,
                                    const std::string &family,
                                    const list &control, const list &k_list) {
  mat X = as_mat(x_r);
  vec beta = as_col(beta_r);
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec w = as_col(wt_r);

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  CapybaraParameters params(control);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(k_list);

  capybara::InferenceGLM result = capybara::feglm_fit(
      beta, eta, y, X, w, theta, family_type, fe_groups, params);

  field<std::string> fe_names(k_list.size());
  field<field<std::string>> fe_levels(k_list.size());

  if (!k_list.names().empty()) {
    cpp11::strings fe_names_r = k_list.names();
    for (size_t i = 0; i < static_cast<size_t>(fe_names_r.size()); i++) {
      fe_names(i) = std::string(fe_names_r[i]);
    }
  }

  for (size_t k = 0; k < static_cast<size_t>(k_list.size()); k++) {
    const list &group_list = as_cpp<list>(k_list[k]);
    fe_levels(k).set_size(group_list.size());

    if (!group_list.names().empty()) {
      cpp11::strings level_names = group_list.names();
      for (size_t j = 0; j < static_cast<size_t>(level_names.size()); j++) {
        fe_levels(k)(j) = std::string(level_names[j]);
      }
    }
  }

  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

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

[[cpp11::register]] doubles
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

[[cpp11::register]] list
fenegbin_fit_(const doubles_matrix<> &X_r, const doubles &y_r,
              const doubles &w_r, const list &FEs, const std::string &link,
              const doubles &beta_r, const doubles &eta_r,
              const double &init_theta, const list &control) {
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);

  CapybaraParameters params(control);

  field<field<uvec>> fe_groups = R_list_to_Armadillo_field(FEs);

  capybara::InferenceNegBin result =
      capybara::fenegbin_fit(X, y, w, fe_groups, params, init_theta);

  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    result.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

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

[[cpp11::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
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

[[cpp11::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const int K,
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

[[cpp11::register]] doubles_matrix<>
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

[[cpp11::register]] doubles_matrix<>
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
