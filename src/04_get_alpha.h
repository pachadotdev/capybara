#ifndef CAPYBARA_ALPHA
#define CAPYBARA_ALPHA

// Fixed effects recovery - Enhanced version with fixest compatibility
struct GetAlphaResult {
  field<vec> Alpha;
  uvec nb_references; // Number of references per dimension (fixest
                      // compatibility)
  bool is_regular;    // Whether fixed effects are regular
  bool success;       // Whether extraction succeeded

  GetAlphaResult() : is_regular(true), success(false) {}

  cpp11::list to_list() const {
    writable::list Alpha_r(Alpha.n_elem);
    for (size_t k = 0; k < Alpha.n_elem; ++k) {
      Alpha_r[k] = as_doubles_matrix(Alpha(k).eval());
    }

    // Add fixest-style metadata
    writable::list result;
    result.push_back({"fixed_effects"_nm = Alpha_r});
    result.push_back({"nb_references"_nm = as_integers(nb_references)});
    result.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    result.push_back({"success"_nm = writable::logicals({success})});

    return result;
  }
};

// Extract fixed effects using for single FE (Q=1)
inline GetAlphaResult extract_fixef_single(const vec &sum_fe,
                                           const uvec &fe_id) {
  GetAlphaResult result;

  uvec myOrder = sort_index(fe_id);
  uvec sorted_id = fe_id(myOrder);

  // Find positions where ID changes (first occurrence of each unique ID)
  uvec select;
  select.resize(0);

  if (sorted_id.n_elem > 0) {
    select.resize(1);
    select(0) = myOrder(0); // First element

    for (size_t i = 1; i < sorted_id.n_elem; ++i) {
      if (sorted_id(i) != sorted_id(i - 1)) {
        select.resize(select.n_elem + 1);
        select(select.n_elem - 1) = myOrder(i);
      }
    }
  }

  // Extract fixed effects at selected positions
  result.Alpha.set_size(1);
  result.Alpha(0) = sum_fe(select);

  // For single FE, no references needed
  result.nb_references.set_size(1);
  result.nb_references(0) = 0;
  result.is_regular = true;
  result.success = true;

  return result;
}

inline GetAlphaResult get_alpha(const vec &p,
                                const field<field<uvec>> &group_indices,
                                double tol, size_t iter_max) {
  const size_t K = group_indices.n_elem;
  GetAlphaResult result;

  if (K == 0) {
    // No fixed effects => return intercept
    result.Alpha.set_size(1);
    result.Alpha(0) = vec(1);
    result.Alpha(0)(0) = mean(p);
    result.nb_references.set_size(1);
    result.nb_references(0) = 0;
    result.is_regular = true;
    result.success = true;
    return result;
  }

  if (K == 1) {
    // Single FE case
    // Convert field<field<uvec>> to uvec for single FE
    const field<uvec> &groups = group_indices(0);
    uvec fe_id(p.n_elem);

    for (size_t g = 0; g < groups.n_elem; ++g) {
      const uvec &group_obs = groups(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        // TODO: use 0-based indexing consistently
        fe_id(group_obs(i)) = g + 1; // 1-based indexing like fixest
      }
    }

    return extract_fixef_single(p, fe_id);
  }

  // Multi-FE case

  // Initialize fixed effects storage
  field<vec> Alpha(K);
  for (size_t k = 0; k < K; ++k) {
    const size_t n_groups = group_indices(k).n_elem;
    Alpha(k).zeros(n_groups);
  }

  // TODO: find a more efficient way to avoid converting
  // Convert to matrix format like fixest
  const size_t N = p.n_elem;
  umat dumMat(N, K);

  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &groups_k = group_indices(k);
    for (size_t g = 0; g < groups_k.n_elem; ++g) {
      const uvec &group_obs = groups_k(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        dumMat(group_obs(i), k) = g; // 0-based for internal computation
      }
    }
  }

  // Alpaca-like alternating projections
  field<vec> Alpha_old(K);
  for (size_t k = 0; k < K; ++k) {
    Alpha_old(k).zeros(Alpha(k).n_elem);
  }

  double ratio = 0.0;
  size_t iter = 0;

  for (; iter < iter_max; ++iter) {
    Alpha_old = Alpha;

    // Update each FE dimension
    for (size_t k = 0; k < K; ++k) {
      const size_t n_groups_k = Alpha(k).n_elem;

      // Compute residual: p - sum of other FEs
      vec resid = p;
      for (size_t l = 0; l < K; ++l) {
        if (l == k)
          continue;

        for (size_t obs = 0; obs < N; ++obs) {
          resid(obs) -= Alpha(l)(dumMat(obs, l));
        }
      }

      // Update FE k
      Alpha(k).zeros();
      uvec group_counts(n_groups_k, fill::zeros);

      for (size_t obs = 0; obs < N; ++obs) {
        size_t group_id = dumMat(obs, k);
        Alpha(k)(group_id) += resid(obs);
        group_counts(group_id)++;
      }

      // Convert sums to means
      for (size_t g = 0; g < n_groups_k; ++g) {
        if (group_counts(g) > 0) {
          Alpha(k)(g) /= group_counts(g);
        }
      }
    }

    // Check convergence
    double num = 0.0, denom = 0.0;
    for (size_t k = 0; k < K; ++k) {
      const vec &diff = Alpha(k) - Alpha_old(k);
      num += dot(diff, diff);
      denom += dot(Alpha_old(k), Alpha_old(k));
    }
    ratio = sqrt(num / (denom + 1e-16));
    if (ratio < tol)
      break;
  }

  // By construction, the elements of the first fixed-effect dimension
  // are never set as references
  result.nb_references.set_size(K);
  result.nb_references.zeros();

  if (K >= 2) {
    // Set references for all FE dimensions except the first
    // In the presence of regular fixed-effects, there should be Q-1 references
    for (size_t k = 1; k < K; ++k) {
      result.nb_references(k) = 1;

      if (Alpha(k).n_elem > 0) {
        double reference_value = Alpha(k)(Alpha(k).n_elem - 1);
        Alpha(k) -= reference_value;
      }
    }
  }

  result.Alpha = Alpha;
  result.success = (iter < iter_max);
  result.is_regular = true; // TODO: Assume regular for now

  return result;
}

// Extract fixed effects from fitted model object structure
inline GetAlphaResult
extract_model_fixef(const vec &fitted_values, const vec &linear_predictor,
                    const mat &X, const vec &beta,
                    const field<field<uvec>> &group_indices,
                    const std::string &family = "gaussian", double tol = 1e-8,
                    size_t iter_max = 10000) {
  // This is the sum of all fixed effects per observation
  // For linear models: sum_fe = fitted_values - X * beta
  // For GLM: sum_fe = linear_predictor - X * beta

  vec S;
  if (family == "gaussian" || family == "linear") {
    // Linear model case
    if (X.n_cols > 0 && beta.n_elem > 0) {
      S = fitted_values - X * beta;
    } else {
      S = fitted_values; // No covariates case
    }
  } else {
    // GLM case - use linear predictor (eta)
    if (X.n_cols > 0 && beta.n_elem > 0) {
      S = linear_predictor - X * beta;
    } else {
      S = linear_predictor; // No covariates case
    }
  }

  GetAlphaResult result = get_alpha(S, group_indices, tol, iter_max);

  // Set family-specific information
  result.is_regular = (group_indices.n_elem <= 2);

  return result;
}

#endif // CAPYBARA_ALPHA
