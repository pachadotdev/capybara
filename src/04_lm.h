// Computing linear models with fixed effects
// Y = alpha + X beta + epsilon

#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {

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
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), coef_status(p, fill::ones), success(false),
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

inline InferenceLM felm_fit(mat &X, const vec &y, const vec &w,
                            const field<uvec> &fe_indices, const uvec &nb_ids,
                            const field<uvec> &fe_id_tables,
                            const CapybaraParameters &params) {
  CAPYBARA_TIME_FUNCTION("felm_fit");

  const size_t n = y.n_elem;
  const size_t p_orig = X.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceLM result(n, p_orig);

  // Step 1: Check collinearity
  bool use_weights = params.use_weights && !all(w == 1.0);
  double tolerance = params.collin_tol;

  CollinearityResult collin_result =
      check_collinearity(X, w, use_weights, tolerance, false);

  // Step 2: Demean variables
  mat X_demean;
  vec y_demean;

  if (has_fixed_effects) {

    field<vec> y_to_demean(1);
    y_to_demean(0) = y;

    DemeanResult y_demean_result = demean_variables(
        y_to_demean, w, fe_indices, nb_ids, fe_id_tables, true, params);
    y_demean = std::move(y_demean_result.demeaned_vars(0));

    if (X.n_cols > 0) {
      field<vec> x_columns_to_demean(X.n_cols);

      for (size_t j = 0; j < X.n_cols; ++j) {
        x_columns_to_demean(j) = X.unsafe_col(j);
      }

      DemeanResult x_demean_result =
          demean_variables(x_columns_to_demean, w, fe_indices, nb_ids,
                           fe_id_tables, false, params);

      X_demean.set_size(n, X.n_cols);
      for (size_t j = 0; j < X.n_cols; ++j) {
        X_demean.unsafe_col(j) = std::move(x_demean_result.demeaned_vars(j));
      }
    } else {
      X_demean.set_size(n, 0);
    }

    result.has_fe = true;
  } else {

    y_demean = y;
    X_demean = std::move(X);
    result.has_fe = false;
  }

  // Step 3: Solve normal equation
  InferenceBeta beta_result = get_beta(X_demean, y_demean, y, w, collin_result,
                                       use_weights, has_fixed_effects);

  if (!beta_result.success) {
    result.success = false;
    return result;
  }

  // Step 4: Copy results
  result.coefficients = std::move(beta_result.coefficients);
  result.fitted_values = std::move(beta_result.fitted_values);
  result.residuals = std::move(beta_result.residuals);
  result.weights = std::move(beta_result.weights);
  result.hessian = std::move(beta_result.hessian);
  result.coef_status = std::move(beta_result.coef_status);

  // Step 5: Extract fixed effects
  if (has_fixed_effects) {

    vec coef_reduced;
    if (collin_result.has_collinearity) {
      coef_reduced = result.coefficients(collin_result.non_collinear_cols);
    } else {
      coef_reduced = result.coefficients;
    }

    vec sum_fe = result.fitted_values - X * coef_reduced;

    field<field<uvec>> group_indices(fe_indices.n_elem);
    for (size_t k = 0; k < fe_indices.n_elem; ++k) {
      const uvec &fe_idx = fe_indices(k);
      const size_t n_obs = fe_idx.n_elem;
      const size_t n_groups = nb_ids(k);

      group_indices(k).set_size(n_groups);

      uvec group_sizes(n_groups, fill::zeros);
      const uword *fe_idx_ptr = fe_idx.memptr();
      uword *group_sizes_ptr = group_sizes.memptr();

      for (size_t obs = 0; obs < n_obs; ++obs) {
        group_sizes_ptr[fe_idx_ptr[obs]]++;
      }

      for (size_t g = 0; g < n_groups; ++g) {
        if (group_sizes_ptr[g] > 0) {
          group_indices(k)(g).set_size(group_sizes_ptr[g]);
        } else {
          group_indices(k)(g).reset();
        }
      }

      uvec group_counters(n_groups, fill::zeros);
      uword *group_counters_ptr = group_counters.memptr();

      for (size_t obs = 0; obs < n_obs; ++obs) {
        uword group_id = fe_idx_ptr[obs];
        group_indices(k)(group_id)(group_counters_ptr[group_id]++) = obs;
      }
    }
    InferenceAlpha alpha_result =
        get_alpha(sum_fe, group_indices, params.alpha_convergence_tol,
                  params.alpha_iter_max);

    result.fixed_effects = std::move(alpha_result.Alpha);
    result.nb_references = std::move(alpha_result.nb_references);
    result.is_regular = alpha_result.is_regular;
  }

  result.success = true;
  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
