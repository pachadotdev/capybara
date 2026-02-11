#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <armadillo4r.hpp>

#ifdef CAPYBARA_DEBUG
#include <chrono>
#endif

using cpp4r::doubles;
using cpp4r::doubles_matrix;
using cpp4r::integers;
using cpp4r::list;
using cpp4r::strings;

// Configure OpenMP threads from configure-time macro
namespace capybara {
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
#ifndef CAPYBARA_DEFAULT_OMP_THREADS
#define CAPYBARA_DEFAULT_OMP_THREADS -1
#endif
inline void set_omp_threads_from_config() {
  static bool done = false;
  if (!done) {
#if defined(_OPENMP) && (CAPYBARA_DEFAULT_OMP_THREADS > 0)
    omp_set_num_threads(CAPYBARA_DEFAULT_OMP_THREADS);
#endif
    done = true;
  }
}
#endif
} // namespace capybara

// Passing parameters from R to C++ functions
// see R/fit_control.R
struct CapybaraParameters {
  double dev_tol;
  double center_tol;
  double center_tol_loose;
  double collin_tol;
  double step_halving_factor;
  double alpha_tol;

  // Separation detection parameters
  double sep_tol;              // Convergence tolerance
  double sep_zero_tol;         // Tolerance for treating values as zero
  size_t sep_max_iter;         // Max iterations for ReLU algorithm
  size_t sep_simplex_max_iter; // Max iterations for simplex algorithm
  bool check_separation;       // Whether to perform separation detection
  bool sep_use_relu;           // Use ReLU algorithm
  bool sep_use_simplex;        // Use simplex algorithm

  size_t iter_max;
  size_t iter_center_max;
  size_t iter_inner_max;
  size_t iter_alpha_max;
  bool return_fe;
  bool keep_tx;

  // Step-halving parameters
  double step_halving_memory;
  size_t max_step_halving;
  double start_inner_tol;

  // Centering acceleration parameters
  size_t grand_acc_period;

  CapybaraParameters()
      : dev_tol(1.0e-08), center_tol(1.0e-08), center_tol_loose(1.0e-04),
        collin_tol(1.0e-10), step_halving_factor(0.5), alpha_tol(1.0e-08),
        sep_tol(1.0e-08), sep_zero_tol(1.0e-12), sep_max_iter(200),
        sep_simplex_max_iter(2000), check_separation(true), sep_use_relu(true),
        sep_use_simplex(true), iter_max(25), iter_center_max(10000),
        iter_inner_max(50), iter_alpha_max(10000), return_fe(true),
        keep_tx(false), step_halving_memory(0.9), max_step_halving(2),
        start_inner_tol(1e-06), grand_acc_period(10) {}

  explicit CapybaraParameters(const cpp4r::list &control) {
    dev_tol = as_cpp<double>(control["dev_tol"]);
    center_tol = as_cpp<double>(control["center_tol"]);
    center_tol_loose = as_cpp<double>(control["center_tol_loose"]);
    collin_tol = as_cpp<double>(control["collin_tol"]);
    step_halving_factor = as_cpp<double>(control["step_halving_factor"]);
    alpha_tol = as_cpp<double>(control["alpha_tol"]);

    // Separation detection parameters
    sep_tol = as_cpp<double>(control["sep_tol"]);
    sep_zero_tol = as_cpp<double>(control["sep_zero_tol"]);
    sep_max_iter = as_cpp<size_t>(control["sep_max_iter"]);
    sep_simplex_max_iter = as_cpp<size_t>(control["sep_simplex_max_iter"]);
    check_separation = as_cpp<bool>(control["check_separation"]);
    sep_use_relu = as_cpp<bool>(control["sep_use_relu"]);
    sep_use_simplex = as_cpp<bool>(control["sep_use_simplex"]);

    iter_max = as_cpp<size_t>(control["iter_max"]);
    iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
    iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
    iter_alpha_max = as_cpp<size_t>(control["iter_alpha_max"]);
    return_fe = as_cpp<bool>(control["return_fe"]);
    keep_tx = as_cpp<bool>(control["keep_tx"]);
    step_halving_memory = as_cpp<double>(control["step_halving_memory"]);
    max_step_halving = as_cpp<size_t>(control["max_step_halving"]);
    start_inner_tol = as_cpp<double>(control["start_inner_tol"]);
    grand_acc_period = as_cpp<size_t>(control["grand_acc_period"]);
  }
};

#include "01_center.h"
#include "02_chol.h"
#include "03_beta.h"
#include "04_alpha.h"
#include "05_separation.h"
#include "06_fit_helpers.h"
#include "07_lm.h"
#include "08_glm.h"
#include "09_negbin.h"

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

// Build FlatFEMap directly from R integer code vectors (the lean path).
// fe_codes is a list of K integer vectors, each of length N, with 0-based
// group codes. This is O(N*K) with zero intermediate allocation — no more
// field<field<uvec>> of hundreds of small heap-allocated uvecs.
inline capybara::FlatFEMap R_codes_to_FlatFEMap(const list &fe_codes) {
  capybara::FlatFEMap map;
  const size_t K = fe_codes.size();
  if (K == 0)
    return map;

  map.K = K;
  map.n_groups.resize(K);
  map.fe_map.resize(K);
  map.n_obs = 0;

  for (size_t k = 0; k < K; ++k) {
    const integers codes_k = as_cpp<integers>(fe_codes[k]);
    const size_t N = codes_k.size();
    if (k == 0)
      map.n_obs = N;

    // Find max code to determine n_groups, and copy into fe_map in one pass
    map.fe_map[k].resize(N);
    uword *map_k = map.fe_map[k].data();
    uword max_code = 0;
    for (size_t i = 0; i < N; ++i) {
      const uword c = static_cast<uword>(codes_k[i]);
      map_k[i] = c;
      if (c > max_code)
        max_code = c;
    }
    map.n_groups[k] = max_code + 1;
  }

  map.structure_built = true;
  return map;
}

// Extract FE level names from an R list of character vectors.
// Returns field<field<string>> for output labeling.
inline void extract_fe_names_and_levels(const list &fe_codes,
                                        const list &fe_levels_r,
                                        field<std::string> &fe_names,
                                        field<field<std::string>> &fe_levels) {
  const size_t K = fe_codes.size();
  fe_names.set_size(K);
  fe_levels.set_size(K);

  // FE variable names
  if (!fe_codes.names().empty()) {
    cpp4r::strings names_r = fe_codes.names();
    for (R_xlen_t i = 0; i < names_r.size(); i++) {
      fe_names(i) = std::string(names_r[i]);
    }
  }

  // Level names per FE
  for (size_t k = 0; k < K; ++k) {
    const cpp4r::strings lvl_k = as_cpp<cpp4r::strings>(fe_levels_r[k]);
    fe_levels(k).set_size(lvl_k.size());
    for (R_xlen_t j = 0; j < lvl_k.size(); j++) {
      fe_levels(k)(j) = std::string(lvl_k[j]);
    }
  }
}

[[cpp4r::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &fe_codes, const double &tol,
                  const size_t &max_iter, const size_t &grand_acc_period) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  capybara::FlatFEMap map = R_codes_to_FlatFEMap(fe_codes);
  map.update_weights(w);
  capybara::center_variables(V, w, map, tol, max_iter, grand_acc_period);

  return as_doubles_matrix(V);
}

[[cpp4r::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                   const doubles &y_r, const doubles &w_r,
                                   const list &fe_codes,
                                   const list &fe_levels_r, const list &control,
                                   const list &cl_list) {
  CapybaraParameters params(control);

  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);

  capybara::FlatFEMap fe_map = R_codes_to_FlatFEMap(fe_codes);

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
      capybara::felm_fit(X, y, w, fe_map, params, nullptr, cluster_ptr);

  field<std::string> fe_names;
  field<field<std::string>> fe_levels;
  extract_fe_names_and_levels(fe_codes, fe_levels_r, fe_names, fe_levels);

  // Replace collinear coefficients (NaN) with R's NA_REAL in all columns of
  // coef_table
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    uvec collinear_idx = find(collinear_mask);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      uword idx = collinear_idx(i);
      result.coef_table(idx, 0) = NA_REAL; // Estimate
      result.coef_table(idx, 1) = NA_REAL; // Std. Error
      result.coef_table(idx, 2) = NA_REAL; // z value
      result.coef_table(idx, 3) = NA_REAL; // Pr(>|z|)
    }
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

      if (k < fe_names.n_elem && !fe_names(k).empty()) {
        fe_list_names[k] = fe_names(k);
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
           const std::string &family, const list &control, const list &fe_codes,
           const list &fe_levels_r, const list &cl_list) {
  mat X = as_mat(x_r);
  vec beta = as_col(beta_r);
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec w = as_col(wt_r);
  vec offset = as_col(offset_r);

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  CapybaraParameters params(control);

  capybara::FlatFEMap fe_map = R_codes_to_FlatFEMap(fe_codes);

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
      beta, eta, y, X, w, theta, family_type, fe_map, params, nullptr,
      has_clusters ? &cluster_groups : nullptr, offset_ptr);

  field<std::string> fe_names;
  field<field<std::string>> fe_levels;
  extract_fe_names_and_levels(fe_codes, fe_levels_r, fe_names, fe_levels);

  // Replace collinear coefficients (NaN) with R's NA_REAL in all columns of
  // coef_table
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    uvec collinear_idx = find(collinear_mask);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      uword idx = collinear_idx(i);
      result.coef_table(idx, 0) = NA_REAL; // Estimate
      result.coef_table(idx, 1) = NA_REAL; // Std. Error
      result.coef_table(idx, 2) = NA_REAL; // z value
      result.coef_table(idx, 3) = NA_REAL; // Pr(>|z|)
    }
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

  // Add separation detection results for Poisson models
  if (family_type == capybara::POISSON && result.has_separation) {
    out.push_back({"has_separation"_nm = writable::logicals({true})});

    // Convert 0-based indices to 1-based for R
    vec separated_obs_r(result.separated_obs.n_elem);
    for (size_t i = 0; i < result.separated_obs.n_elem; ++i) {
      separated_obs_r(i) = static_cast<double>(result.separated_obs(i) + 1);
    }
    out.push_back({"separated_obs"_nm = as_doubles(separated_obs_r)});

    if (result.separation_support.n_elem > 0) {
      out.push_back(
          {"separation_support"_nm = as_doubles(result.separation_support)});
    }
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

      if (k < fe_names.n_elem && !fe_names(k).empty()) {
        fe_list_names[k] = fe_names(k);
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
                  const list &fe_codes) {
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec offset = as_col(offset_r);
  vec w = as_col(wt_r);

  CapybaraParameters params(control);

  capybara::FlatFEMap fe_map = R_codes_to_FlatFEMap(fe_codes);

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  vec result = capybara::feglm_offset_fit(eta, y, offset, w, family_type,
                                          fe_map, params);

  return as_doubles(result);
}

[[cpp4r::register]] list
fenegbin_fit_(const doubles_matrix<> &X_r, const doubles &y_r,
              const doubles &w_r, const list &fe_codes, const list &fe_levels_r,
              const std::string &link, const doubles &beta_r,
              const doubles &eta_r, const double &init_theta,
              const doubles &offset_r, const list &control) {
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  vec w = as_col(w_r);
  vec offset_vec = as_col(offset_r);

  CapybaraParameters params(control);

  capybara::FlatFEMap fe_map = R_codes_to_FlatFEMap(fe_codes);

  capybara::InferenceNegBin result =
      capybara::fenegbin_fit(X, y, w, fe_map, params, offset_vec, init_theta);

  // Replace collinear coefficients (NaN) with R's NA_REAL in all columns of
  // coef_table
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    uvec collinear_idx = find(collinear_mask);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      uword idx = collinear_idx(i);
      result.coef_table(idx, 0) = NA_REAL; // Estimate
      result.coef_table(idx, 1) = NA_REAL; // Std. Error
      result.coef_table(idx, 2) = NA_REAL; // z value
      result.coef_table(idx, 3) = NA_REAL; // Pr(>|z|)
    }
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
  const vec w = vectorise(as_mat(w_r));

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  vec result = capybara::group_sums(M, w, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const size_t K,
                     const list &jlist) {
  const mat M = as_mat(M_r);
  const vec v = vectorise(as_mat(v_r));
  const vec w = vectorise(as_mat(w_r));

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  vec result = capybara::group_sums_spectral(M, v, w, K, group_indices);

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
