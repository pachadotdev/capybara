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
        direct_qr_threshold(0.9), qr_collin_tol_multiplier(1.0e-7),
        chol_stability_threshold(1.0e-12), alpha_convergence_tol(1.0e-8),
        alpha_iter_max(10000), demean_extra_projections(0),
        demean_warmup_iterations(15), demean_projections_after_acc(5),
        demean_grand_acc_frequency(20), demean_ssr_check_frequency(40),
        keep_dmx(false), use_weights(true) {}

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

    keep_dmx = as_cpp<bool>(control["keep_dmx"]);
    use_weights = as_cpp<bool>(control["use_weights"]);
  }
};

// Modular code structure

#include "01_convergence.h"
#include "02_demean.h"
#include "03_parameters.h"
#include "04_lm.h"
#include "05_glm.h"
#include "06_glm_offset.h"
#include "07_group_sums.h"

// Type aliases for easier access
using LMResult = capybara::lm::InferenceLM;
using GLMResult = capybara::glm::InferenceGLM;

// Helper function to convert R indices to C++ uvec (R uses 1-based, C++ uses
// 0-based)
inline uvec R_1based_to_Cpp_0based_indices(const integers &r_indices) {
  uvec cpp_indices(r_indices.size());

  // Use std::transform for efficient vectorized conversion
  std::transform(r_indices.begin(), r_indices.end(), cpp_indices.begin(),
                 [](int r_val) -> uword {
                   // if (r_val < 1) {
                   //   cpp11::stop("Invalid R index: %d (R indices must be >=
                   //   1)", r_val);
                   // }
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
      group_indices(k)(g) =
          R_1based_to_Cpp_0based_indices(group_obs); // Convert to 0-based
    }
  }

  return group_indices;
}

// Convert field<field<uvec>> to umat format for modern functions
inline umat convert_field_to_umat(const field<field<uvec>> &group_indices,
                                  size_t n_obs) {
  const size_t K = group_indices.n_elem;

  if (K == 0) {
    return umat(n_obs, 0);
  }

  umat fe_matrix(n_obs, K);
  fe_matrix.zeros();

  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &groups = group_indices(k);
    const size_t n_groups = groups.n_elem;

    for (size_t g = 0; g < n_groups; ++g) {
      const uvec &group_obs = groups(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_matrix(group_obs(i), k) = g; // 0-based group IDs
      }
    }
  }

  return fe_matrix;
}

// Helper function to convert field<field<uvec>> to field<uvec> format for
// joint_demean
inline field<uvec>
convert_groupindices_to_feids(const field<field<uvec>> &group_indices,
                              size_t n_obs) {
  const size_t K = group_indices.n_elem;
  field<uvec> fe_ids(K);

  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &groups = group_indices(k);
    const size_t n_groups = groups.n_elem;

    fe_ids(k).set_size(n_obs);
    fe_ids(k).fill(0); // Initialize with zeros

    // Assign group IDs
    for (size_t g = 0; g < n_groups; ++g) {
      const uvec &group_obs = groups(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_ids(k)(group_obs(i)) = g; // Use 0-based group IDs
      }
    }
  }

  return fe_ids;
}

[[cpp11::register]] doubles_matrix<>
demean_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &FEs, const double &tol, const int &max_iter,
                  const int &iter_interrupt, const int &iter_ssr,
                  const std::string &family) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  // Convert R list to field<field<uvec>> format
  field<field<uvec>> group_indices = R_list_to_Armadillo_field(FEs);

  // Convert to the format expected by demean_variables
  field<uvec> fe_ids = convert_groupindices_to_feids(group_indices, V.n_rows);

  // Prepare variables for demeaning - convert matrix columns to field<vec>
  field<vec> variables_to_demean(V.n_cols);
  for (size_t j = 0; j < V.n_cols; ++j) {
    variables_to_demean(j) = V.col(j);
  }

  // Create dummy nb_ids and fe_id_tables for the demean function
  uvec nb_ids(fe_ids.n_elem);
  field<uvec> fe_id_tables(fe_ids.n_elem);
  for (size_t k = 0; k < fe_ids.n_elem; ++k) {
    uvec unique_ids = unique(fe_ids(k));
    nb_ids(k) = unique_ids.n_elem;
    fe_id_tables(k).set_size(nb_ids(k));
    for (size_t id = 0; id < nb_ids(k); ++id) {
      fe_id_tables(k)(id) = sum(fe_ids(k) == unique_ids(id));
    }
  }

  // Create CapybaraParameters object with custom values
  CapybaraParameters params;

  // Use the demean_variables function
  capybara::demean::DemeanResult demean_result =
      capybara::demean::demean_variables(variables_to_demean, w, fe_ids, nb_ids,
                                         fe_id_tables, false, params);

  // Convert back to matrix
  mat result_mat(V.n_rows, V.n_cols);
  for (size_t j = 0; j < V.n_cols; ++j) {
    result_mat.col(j) = demean_result.demeaned_vars(j);
  }

  return as_doubles_matrix(result_mat);
}

[[cpp11::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                   const doubles &y_r, const doubles &w_r,
                                   const list &FE, const list &control) {
  mat X = as_Mat(X_r);
  const vec y = as_Col(y_r);
  const vec w = as_Col(w_r);

  // Convert R list to efficient field format
  field<field<uvec>> group_indices = R_list_to_Armadillo_field(FE);

  // Parameters passed from R's capybara::fit_control()
  CapybaraParameters params(control);

  // Convert to field<uvec> format for the new API
  field<uvec> fe_indices(group_indices.n_elem);
  uvec nb_ids(group_indices.n_elem);
  field<uvec> fe_id_tables(group_indices.n_elem);

  for (size_t k = 0; k < group_indices.n_elem; ++k) {
    // Create a single uvec for all observations in this FE dimension
    fe_indices(k).set_size(y.n_elem);
    const field<uvec> &groups_k = group_indices(k);
    nb_ids(k) = groups_k.n_elem;

    // Create frequency table
    fe_id_tables(k).set_size(nb_ids(k));

    for (size_t g = 0; g < groups_k.n_elem; ++g) {
      const uvec &group_obs = groups_k(g);
      fe_id_tables(k)(g) = group_obs.n_elem;

      // Assign group ID to all observations in this group
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_indices(k)(group_obs(i)) = g;
      }
    }
  }

  LMResult res =
      capybara::lm::felm_fit(X, y, w, fe_indices, nb_ids, fe_id_tables, params);

  // Replace collinear coefficients with R's NA_REAL
  uvec collinear_mask = (res.coef_status == 0);
  if (any(collinear_mask)) {
    res.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }

  return res.to_list();
}

[[cpp11::register]] list feglm_fit_(const doubles_matrix<> &X_r,
                                    const doubles &y_r, const doubles &w_r,
                                    const list &FE, const std::string &family,
                                    const doubles &beta_r, const doubles &eta_r,
                                    const double &theta, const list &control) {
  mat X = as_Mat(X_r);
  const vec y = as_Col(y_r);
  const vec w = as_Col(w_r);
  const std::string fam = capybara::glm::tidy_family(family);
  vec beta = as_Col(beta_r);
  vec eta = as_Col(eta_r);

  // Parameters passed from R's capybara::fit_control()
  CapybaraParameters params(control);

  // Convert R list to field<uvec> format for modern API
  field<field<uvec>> group_indices = R_list_to_Armadillo_field(FE);

  // Convert to field<uvec> format for the new API
  field<uvec> fe_indices(group_indices.n_elem);
  uvec nb_ids(group_indices.n_elem);
  field<uvec> fe_id_tables(group_indices.n_elem);

  for (size_t k = 0; k < group_indices.n_elem; ++k) {
    // Create a single uvec for all observations in this FE dimension
    fe_indices(k).set_size(y.n_elem);
    const field<uvec> &groups_k = group_indices(k);
    nb_ids(k) = groups_k.n_elem;

    // Create frequency table
    fe_id_tables(k).set_size(nb_ids(k));

    for (size_t g = 0; g < groups_k.n_elem; ++g) {
      const uvec &group_obs = groups_k(g);
      fe_id_tables(k)(g) = group_obs.n_elem;

      // Assign group ID to all observations in this group
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_indices(k)(group_obs(i)) = g;
      }
    }
  }

  GLMResult res = capybara::glm::feglm_fit(X, y, w, fe_indices, nb_ids,
                                           fe_id_tables, fam, params);

  // Replace collinear coefficients with R's NA_REAL using vectorized approach
  uvec collinear_mask = (res.coef_status == 0);
  if (any(collinear_mask)) {
    res.coefficients.elem(find(collinear_mask)).fill(NA_REAL);
  }
  return res.to_list(params.keep_dmx);
}

[[cpp11::register]] doubles
feglm_offset_fit_(const doubles &y_r, const doubles &offset_r,
                  const doubles &w_r, const list &FEs,
                  const std::string &family, const doubles &eta_r,
                  const list &control) {
  vec y = as_Col(y_r);
  vec offset = as_Col(offset_r);
  const vec w = as_Col(w_r);
  const std::string fam = capybara::glm_offset::tidy_family(family);
  vec eta = as_Col(eta_r);

  // Parameters passed from R's capybara::fit_control()
  CapybaraParameters params(control);

  // Convert R list to field<uvec> format for modern API
  field<field<uvec>> group_indices = R_list_to_Armadillo_field(FEs);

  // Convert to field<uvec> format for the new API
  field<uvec> fe_indices(group_indices.n_elem);
  uvec nb_ids(group_indices.n_elem);
  field<uvec> fe_id_tables(group_indices.n_elem);

  for (size_t k = 0; k < group_indices.n_elem; ++k) {
    // Create a single uvec for all observations in this FE dimension
    fe_indices(k).set_size(y.n_elem);
    const field<uvec> &groups_k = group_indices(k);
    nb_ids(k) = groups_k.n_elem;

    // Create frequency table
    fe_id_tables(k).set_size(nb_ids(k));

    for (size_t g = 0; g < groups_k.n_elem; ++g) {
      const uvec &group_obs = groups_k(g);
      fe_id_tables(k)(g) = group_obs.n_elem;

      // Assign group ID to all observations in this group
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        fe_indices(k)(group_obs(i)) = g;
      }
    }
  }

  capybara::glm_offset::InferenceGLMOffset res =
      capybara::glm_offset::feglm_offset_fit(eta, y, offset, w, fe_indices,
                                             nb_ids, fe_id_tables, fam, params);
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
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  capybara::group_sums::GroupSums res =
      capybara::group_sums::group_sums(M, w, group_indices);
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
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  capybara::group_sums::GroupSums res =
      capybara::group_sums::group_sums_spectral(M, v, w, K, group_indices);
  return res.to_matrix();
}

[[cpp11::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  const mat M = as_Mat(M_r);

  // Convert R list to single-level Armadillo field
  const size_t J = jlist.size();
  field<uvec> group_indices(J);
  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  capybara::group_sums::GroupSums res =
      capybara::group_sums::group_sums_var(M, group_indices);
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
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  capybara::group_sums::GroupSums res =
      capybara::group_sums::group_sums_cov(M, N, group_indices);
  return res.to_matrix();
}
