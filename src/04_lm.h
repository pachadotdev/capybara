// Computing linear models with fixed effects
// Y = alpha + X beta + epsilon

#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {
namespace lm {

using demean::demean_variables;
using demean::DemeanResult;
using parameters::get_alpha;
using parameters::get_beta;
using parameters::InferenceAlpha;
using parameters::InferenceBeta;

//////////////////////////////////////////////////////////////////////////////
// RESULT STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// LM fitting result structure
struct InferenceLM {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  // Fixed effects info
  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
  bool has_fe = true;
  uvec iterations;

  InferenceLM(size_t n, size_t p)
      : coefficients(p, fill::none), fitted_values(n, fill::none),
        residuals(n, fill::none), weights(n, fill::none),
        hessian(p, p, fill::none), coef_status(p, fill::none), success(false),
        is_regular(true), has_fe(false) {}

  cpp11::list to_list() const {
    auto out = writable::list({"coefficients"_nm = as_doubles(coefficients),
                               "fitted.values"_nm = as_doubles(fitted_values),
                               "weights"_nm = as_doubles(weights),
                               "residuals"_nm = as_doubles(residuals),
                               "hessian"_nm = as_doubles_matrix(hessian)});

    if (has_fe && fixed_effects.n_elem > 0) {
      writable::list fe_list(fixed_effects.n_elem);
      for (size_t k = 0; k < fixed_effects.n_elem; ++k) {
        fe_list[k] = as_doubles(fixed_effects(k));
      }
      out.push_back({"fixed.effects"_nm = fe_list});
      out.push_back({"nb_references"_nm = as_integers(nb_references)});
      out.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    }

    return out;
  }
};

//////////////////////////////////////////////////////////////////////////////
// LM FITTING
//////////////////////////////////////////////////////////////////////////////

inline InferenceLM felm_fit(
    const mat &X_orig, const vec &y_orig, const vec &w,
    const field<uvec> &fe_indices, const uvec &nb_ids,
    const field<uvec> &fe_id_tables, double center_tol, size_t iter_center_max,
    size_t iter_interrupt, size_t iter_ssr, double collin_tol,
    bool use_weights = true, double direct_qr_threshold = 0.9,
    double qr_collin_tol_multiplier = 1.0,
    double chol_stability_threshold = 1e-12,
    size_t demean_extra_projections = 0, size_t demean_warmup_iterations = 15,
    size_t demean_projections_after_acc = 5,
    size_t demean_grand_acc_frequency = 20,
    size_t demean_ssr_check_frequency = 40, double safe_division_min = 1e-12,
    double alpha_convergence_tol = 1e-8, size_t alpha_iter_max = 10000) {
  const size_t n = y_orig.n_elem;
  const size_t p_orig = X_orig.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceLM result(n, p_orig);

  mat X_demean;
  vec Y_demean;
  DemeanResult y_demean_result(0); // DECLARE HERE!

  if (has_fixed_effects) {
    // STEP 1: Demean Y and save the fixed effects
    field<vec> y_to_demean(1);
    y_to_demean(0) = y_orig;

    y_demean_result = demean_variables(
        y_to_demean, w, fe_indices, nb_ids, fe_id_tables, iter_center_max,
        center_tol, demean_extra_projections, demean_warmup_iterations,
        demean_projections_after_acc, demean_grand_acc_frequency,
        demean_ssr_check_frequency, true, safe_division_min);

    Y_demean = y_demean_result.demeaned_vars(0);

    // Demean X columns (without saving FE)
    if (p_orig > 0) {
      X_demean.set_size(n, p_orig);
      for (size_t j = 0; j < p_orig; ++j) {
        field<vec> x_to_demean(1);
        x_to_demean(0) = X_orig.col(j);

        DemeanResult x_demean_result = demean_variables(
            x_to_demean, w, fe_indices, nb_ids, fe_id_tables, iter_center_max,
            center_tol, demean_extra_projections, demean_warmup_iterations,
            demean_projections_after_acc, demean_grand_acc_frequency,
            demean_ssr_check_frequency, false, safe_division_min);

        X_demean.col(j) = x_demean_result.demeaned_vars(0);
      }
    } else {
      X_demean = mat(n, 0);
    }

    result.has_fe = true;
  } else {
    // No fixed effects
    Y_demean = y_orig;
    X_demean = X_orig;
    result.has_fe = false;
  }

  // STEP 2: Run regression on demeaned data
  if (use_weights) {
    use_weights = !all(w == 1.0);
  }

  InferenceBeta beta_result = get_beta(
      X_demean, Y_demean, y_orig, w, collin_tol, use_weights, has_fixed_effects,
      direct_qr_threshold, qr_collin_tol_multiplier, chol_stability_threshold);

  if (!beta_result.success) {
    result.success = false;
    return result;
  }

  // Direct assignment to result fields
  result.coefficients = beta_result.coefficients;
  result.fitted_values = beta_result.fitted_values;
  result.residuals = beta_result.residuals;
  result.weights = beta_result.weights;
  result.hessian = beta_result.hessian;
  result.coef_status = beta_result.coef_status;

  // STEP 3: Extract fixed effects if present
  if (has_fixed_effects) {
    // The sum of fixed effects is: fitted_values - X*beta
    vec sum_fe = result.fitted_values - X_orig * result.coefficients;

    // Convert field<uvec> to field<field<uvec>> format
    field<field<uvec>> group_indices(fe_indices.n_elem);
    for (size_t k = 0; k < fe_indices.n_elem; ++k) {
      group_indices(k).set_size(nb_ids(k));

      // Create groups from fe_indices
      for (size_t g = 0; g < nb_ids(k); ++g) {
        uvec group_obs = find(fe_indices(k) == g);
        group_indices(k)(g) = group_obs;
      }
    }

    // Use fixest algorithm instead of get_alpha
    InferenceAlpha alpha_result =
        get_alpha(sum_fe, group_indices, alpha_convergence_tol, alpha_iter_max);
    result.fixed_effects = alpha_result.Alpha;
    result.nb_references = alpha_result.nb_references;
    result.is_regular = alpha_result.is_regular;
  }

  result.success = true;
  return result;
}

} // namespace lm
} // namespace capybara

#endif // CAPYBARA_LM_H
