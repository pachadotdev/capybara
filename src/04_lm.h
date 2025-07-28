// Computing linear models with fixed effects
// Y = alpha + X beta + epsilon

#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {
namespace lm {

using demean::demean_variables;
using demean::DemeanResult;
using parameters::check_collinearity;
using parameters::CollinearityResult;
using parameters::get_alpha;
using parameters::get_beta;
using parameters::InferenceAlpha;
using parameters::InferenceBeta;

struct InferenceLM {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
  bool has_fe = true;
  uvec iterations;

  InferenceLM(size_t n, size_t p)
      : coefficients(p, fill::none), fitted_values(n, fill::none),
        residuals(n, fill::none), weights(n, fill::none),
        hessian(p, p, fill::none), coef_status(p, fill::none), success(false),
        is_regular(true), has_fe(false) {
    CAPYBARA_TIME_FUNCTION("InferenceLM::InferenceLM");
  }

  cpp11::list to_list() const {
    CAPYBARA_TIME_FUNCTION("InferenceLM::to_list");
    
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

inline InferenceLM felm_fit(const mat &X, const vec &y, const vec &w,
                            const field<uvec> &fe_indices, const uvec &nb_ids,
                            const field<uvec> &fe_id_tables,
                            const CapybaraParameters &params) {
  CAPYBARA_TIME_FUNCTION("felm_fit");
  
  const size_t n = y.n_elem;
  const size_t p_orig = X.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceLM result(n, p_orig);

  // Step 1: Check collinearity and modify X in place
  bool use_weights = params.use_weights && !all(w == 1.0);
  double tolerance = params.collin_tol;

  // TODO: avoid working copy that will be modified
  mat X_work = X;
  CollinearityResult collin_result =
      check_collinearity(X_work, w, use_weights, tolerance, false);

  // Step 2: Demean variables
  mat X_demean;
  vec y_demean;

  if (has_fixed_effects) {
    // Demean Y
    field<vec> y_to_demean(1);
    y_to_demean(0) = y;

    DemeanResult y_demean_result = demean_variables(
        y_to_demean, w, fe_indices, nb_ids, fe_id_tables, true, params);
    y_demean = y_demean_result.demeaned_vars(0);

    // Demean only non-collinear X columns
    if (X_work.n_cols > 0) {
      field<vec> x_columns_to_demean(X_work.n_cols);
      for (size_t j = 0; j < X_work.n_cols; ++j) {
        x_columns_to_demean(j) = X_work.unsafe_col(j);
      }

      // Demean all X columns in a single batch call
      DemeanResult x_demean_result =
          demean_variables(x_columns_to_demean, w, fe_indices, nb_ids,
                           fe_id_tables, false, params);

      X_demean.set_size(n, X_work.n_cols);
      for (size_t j = 0; j < X_work.n_cols; ++j) {
        X_demean.unsafe_col(j) = std::move(x_demean_result.demeaned_vars(j));
      }
    } else {
      X_demean = mat(n, 0);
    }

    result.has_fe = true;
  } else {
    y_demean = y;
    X_demean = X_work;
    result.has_fe = false;
  }

  // Step 3: Solve normal equations using the reduced matrix
  InferenceBeta beta_result = get_beta(X_demean, y_demean, y, w, collin_result,
                                       use_weights, has_fixed_effects);

  if (!beta_result.success) {
    result.success = false;
    return result;
  }

  // Step 4: Copy results
  result.coefficients = beta_result.coefficients;
  result.fitted_values = beta_result.fitted_values;
  result.residuals = beta_result.residuals;
  result.weights = beta_result.weights;
  result.hessian = beta_result.hessian;
  result.coef_status = beta_result.coef_status;

  // Step 5: Extract fixed effects using reduced X and coefficients
  if (has_fixed_effects) {
    vec coef_reduced;
    if (collin_result.has_collinearity) {
      coef_reduced = result.coefficients(collin_result.non_collinear_cols);
    } else {
      coef_reduced = result.coefficients;
    }

    vec sum_fe = result.fitted_values - X_work * coef_reduced;

    field<field<uvec>> group_indices(fe_indices.n_elem);
    for (size_t k = 0; k < fe_indices.n_elem; ++k) {
      group_indices(k).set_size(nb_ids(k));

      field<uvec> temp_groups(nb_ids(k));
      const uvec &fe_idx = fe_indices(k);

      // Initialize each group as empty
      for (size_t g = 0; g < nb_ids(k); ++g) {
        temp_groups(g).reset();
      }

      // Count group sizes first
      uvec group_sizes(nb_ids(k), fill::zeros);
      for (size_t obs = 0; obs < fe_idx.n_elem; ++obs) {
        group_sizes(fe_idx(obs))++;
      }

      // Pre-allocate and fill groups
      for (size_t g = 0; g < nb_ids(k); ++g) {
        if (group_sizes(g) > 0) {
          temp_groups(g).set_size(group_sizes(g));
        }
      }

      uvec group_counters(nb_ids(k), fill::zeros);
      for (size_t obs = 0; obs < fe_idx.n_elem; ++obs) {
        uword group_id = fe_idx(obs);
        temp_groups(group_id)(group_counters(group_id)++) = obs;
      }

      group_indices(k) = std::move(temp_groups);
    }
    InferenceAlpha alpha_result =
        get_alpha(sum_fe, group_indices, params.alpha_convergence_tol,
                  params.alpha_iter_max);

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
