#ifndef CAPYBARA_ALPHA
#define CAPYBARA_ALPHA

// Fixed effects recovery - Enhanced version with fixest compatibility
struct GetAlphaResult {
  field<vec> Alpha;
  uvec nb_references;          // Number of references per dimension (fixest compatibility)
  bool is_regular;             // Whether fixed effects are regular
  bool success;                // Whether extraction succeeded
  
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

struct AlphaGroupInfo {
  field<uvec> indices;
  size_t n_groups;

  AlphaGroupInfo() = default;
  AlphaGroupInfo(const field<uvec> &group_field) {
    n_groups = group_field.n_elem;
    indices = group_field;
  }
};

inline GetAlphaResult get_alpha(const vec &p,
                                const field<field<uvec>> &group_indices,
                                double tol, size_t iter_max) {
  const size_t K = group_indices.n_elem;
  field<AlphaGroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info(k) = AlphaGroupInfo(group_indices(k));
  }
  field<vec> Alpha(K);
  for (size_t k = 0; k < K; ++k) {
    if (group_info(k).n_groups > 0) {
      Alpha(k).zeros(group_info(k).n_groups);
    }
  }
  field<vec> Alpha0(K), Alpha1(K), Alpha2(K);
  for (size_t k = 0; k < K; ++k) {
    if (group_info(k).n_groups > 0) {
      Alpha0(k).zeros(group_info(k).n_groups);
      Alpha1(k).zeros(group_info(k).n_groups);
      Alpha2(k).zeros(group_info(k).n_groups);
    }
  }
  double ratio = 0.0;
  if (K == 2) {
    // Ultra-optimized 2-way FE recovery: direct alternating (no acceleration
    // needed)
    const AlphaGroupInfo &gi0 = group_info(0);
    const AlphaGroupInfo &gi1 = group_info(1);

    for (size_t iter = 0; iter < iter_max; ++iter) {
      Alpha0 = Alpha;

      // Update Alpha0: residual = p - Alpha1
      vec resid = p;
      for (size_t j = 0; j < gi1.n_groups; ++j) {
        resid.elem(gi1.indices(j)) -= Alpha(1)(j);
      }
      for (size_t j = 0; j < gi0.n_groups; ++j) {
        const uvec &idx = gi0.indices(j);
        if (idx.n_elem == 0)
          continue;
        Alpha(0)(j) = mean(resid.elem(idx));
      }

      // Update Alpha1: residual = p - Alpha0
      resid = p;
      for (size_t j = 0; j < gi0.n_groups; ++j) {
        resid.elem(gi0.indices(j)) -= Alpha(0)(j);
      }
      for (size_t j = 0; j < gi1.n_groups; ++j) {
        const uvec &idx = gi1.indices(j);
        if (idx.n_elem == 0)
          continue;
        Alpha(1)(j) = mean(resid.elem(idx));
      }

      // Convergence check
      double num = 0.0, denom = 0.0;
      for (size_t k = 0; k < 2; ++k) {
        const vec &diff = Alpha(k) - Alpha0(k);
        num += dot(diff, diff);
        denom += dot(Alpha0(k), Alpha0(k));
      }
      ratio = sqrt(num / (denom + 1e-16));
      if (ratio < tol)
        break;
    }
  } else {
    // K>2
    const size_t warmup = 15, grand_acc = 40;
    size_t iter = 0;

    for (; iter < std::min<size_t>(warmup, iter_max); ++iter) {
      Alpha0 = Alpha;
      for (size_t k = 0; k < K; ++k) {
        vec resid = p;
        const AlphaGroupInfo &gi_k = group_info(k);

        // Subtract other FEs
        for (size_t l = 0; l < K; ++l) {
          if (l == k || group_info(l).n_groups == 0)
            continue;
          const AlphaGroupInfo &gi_l = group_info(l);
          for (size_t j = 0; j < gi_l.n_groups; ++j) {
            resid.elem(gi_l.indices(j)) -= Alpha(l)(j);
          }
        }

        // Update current FE
        for (size_t j = 0; j < gi_k.n_groups; ++j) {
          const uvec &idx = gi_k.indices(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }

      // Convergence check
      double num = 0.0, denom = 0.0;
      for (size_t k = 0; k < K; ++k) {
        const vec &diff = Alpha(k) - Alpha0(k);
        num += dot(diff, diff);
        denom += dot(Alpha0(k), Alpha0(k));
      }
      ratio = sqrt(num / (denom + 1e-16));
      if (ratio < tol)
        break;
    }

    // Main loop - Alternate projections with Irons-Tuck acceleration
    size_t acc_count = 0;
    while (iter < iter_max && ratio >= tol) {
      // Save previous states
      Alpha2 = Alpha1;
      Alpha1 = Alpha0;
      Alpha0 = Alpha;

      // Simple projection
      for (size_t k = 0; k < K; ++k) {
        vec resid = p;
        const AlphaGroupInfo &gi_k = group_info(k);

        for (size_t l = 0; l < K; ++l) {
          if (l == k || group_info(l).n_groups == 0)
            continue;
          const AlphaGroupInfo &gi_l = group_info(l);
          for (size_t j = 0; j < gi_l.n_groups; ++j) {
            resid.elem(gi_l.indices(j)) -= Alpha(l)(j);
          }
        }

        for (size_t j = 0; j < gi_k.n_groups; ++j) {
          const uvec &idx = gi_k.indices(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }
      ++iter;

      // Irons-Tuck acceleration every grand_acc iterations
      if (++acc_count == grand_acc) {
        acc_count = 0;
        for (size_t k = 0; k < K; ++k) {
          if (group_info(k).n_groups == 0)
            continue;
          vec delta1 = Alpha0(k) - Alpha1(k);
          vec delta2 = Alpha1(k) - Alpha2(k);
          vec delta_diff = delta1 - delta2;
          double denom_acc = dot(delta_diff, delta_diff);
          if (denom_acc > 1e-16) {
            double coef = dot(delta1, delta_diff) / denom_acc;
            Alpha(k) = Alpha0(k) - coef * delta1;
          }
        }
      }

      // Convergence check
      double num = 0.0, denom = 0.0;
      for (size_t k = 0; k < K; ++k) {
        const vec &diff = Alpha(k) - Alpha0(k);
        num += dot(diff, diff);
        denom += dot(Alpha0(k), Alpha0(k));
      }
      ratio = sqrt(num / (denom + 1e-16));
    }
  }

  // Handle the case of no fixed effects (K = 0)
  if (K == 0) {
    // For no fixed effects, we need to return the intercept
    // The intercept is the mean of the residuals p
    Alpha.set_size(1);
    Alpha(0) = vec(1);
    Alpha(0)(0) = mean(p);
  }

  GetAlphaResult res;
  res.Alpha = Alpha;
  res.success = true;
  
  // Set reference count (simplified for now)
  res.nb_references.set_size(K);
  res.nb_references.zeros();
  if (K >= 2) {
    // First FE has 0 references, others have at least 1
    for (size_t q = 1; q < K; ++q) {
      res.nb_references(q) = 1;
    }
  }
  
  // Check if fixed effects are regular (simplified)
  res.is_regular = (K <= 2);
  
  return res;
}

// Enhanced fixed effects extraction for fixest compatibility
// Extract fixed effects using the simple single FE method (Q=1)
inline GetAlphaResult extract_fixef_single(const vec &sum_fe, 
                                          const uvec &fe_id) {
  GetAlphaResult result;
  
  // Sort by FE ID to find unique groups
  uvec sorted_order = sort_index(fe_id);
  uvec sorted_id = fe_id(sorted_order);
  
  // Find first occurrence of each unique ID
  uvec group_starts;
  group_starts.resize(0);
  
  if (sorted_id.n_elem > 0) {
    group_starts.resize(1);
    group_starts(0) = sorted_order(0);
    
    for (size_t i = 1; i < sorted_id.n_elem; ++i) {
      if (sorted_id(i) != sorted_id(i-1)) {
        group_starts.resize(group_starts.n_elem + 1);
        group_starts(group_starts.n_elem - 1) = sorted_order(i);
      }
    }
  }
  
  // Extract fixed effects at group starts
  result.Alpha.set_size(1);
  result.Alpha(0) = sum_fe(group_starts);
  
  // No references for single FE
  result.nb_references.set_size(1);
  result.nb_references(0) = 0;
  result.is_regular = true;
  result.success = true;
  
  return result;
}

// Extract fixed effects from fitted model object structure
// This function takes the typical capybara model structure and extracts fixed effects
inline GetAlphaResult extract_model_fixef(const vec &fitted_values,
                                          const vec &linear_predictor,
                                          const mat &X,
                                          const vec &beta,
                                          const field<field<uvec>> &group_indices,
                                          const std::string &family = "gaussian",
                                          double tol = 1e-8,
                                          size_t iter_max = 10000) {
  // Calculate sum of fixed effects from fitted model
  // For GLM: sum_fe = linear_predictor - X * beta
  // For LM: sum_fe = fitted_values - X * beta
  
  vec sum_fe;
  if (family == "gaussian" || family == "linear") {
    // Linear model case
    sum_fe = fitted_values - X * beta;
  } else {
    // GLM case - use linear predictor
    sum_fe = linear_predictor - X * beta;
  }
  
  GetAlphaResult result = get_alpha(sum_fe, group_indices, tol, iter_max);
  
  // Add family-specific information
  result.is_regular = (group_indices.n_elem <= 2);
  
  return result;
}

#endif // CAPYBARA_ALPHA
