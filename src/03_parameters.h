// Computing beta and then alpha in a model with fixed effects
// Y = alpha + X beta

#ifndef CAPYBARA_PARAMETERS_H
#define CAPYBARA_PARAMETERS_H

namespace capybara {
namespace parameters {

struct InferenceBeta {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;
  bool success;

  InferenceBeta(size_t n, size_t p)
      : coefficients(p, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), coef_status(p, fill::ones), success(false) {
    CAPYBARA_TIME_FUNCTION("InferenceBeta::InferenceBeta");
  }
};

struct InferenceAlpha {
  field<vec> Alpha;
  uvec nb_references;
  bool is_regular;
  bool success;

  InferenceAlpha() : is_regular(true), success(false) {}

  cpp11::list to_list() const {
    CAPYBARA_TIME_FUNCTION("InferenceAlpha::to_list");
    
    writable::list Alpha_r(Alpha.n_elem);
    for (size_t k = 0; k < Alpha.n_elem; ++k) {
      Alpha_r[k] = as_doubles_matrix(Alpha(k).eval());
    }

    writable::list result;
    result.push_back({"fixed_effects"_nm = Alpha_r});
    result.push_back({"nb_references"_nm = as_integers(nb_references)});
    result.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    result.push_back({"success"_nm = writable::logicals({success})});

    return result;
  }
};

struct CollinearityResult {
  uvec coef_status;
  bool has_collinearity;
  size_t n_valid;
  uvec non_collinear_cols;
  mat Q, R;
  bool has_qr;

  CollinearityResult(size_t p)
      : coef_status(p, fill::ones), has_collinearity(false), n_valid(p),
        has_qr(false) {
    CAPYBARA_TIME_FUNCTION("CollinearityResult::CollinearityResult");
  }
};

inline bool rank_revealing_cholesky(uvec &excluded, const mat &XtX, double tol) {
  const size_t p = XtX.n_cols;
  excluded.zeros(p);
  
  if (p == 0) return true;
  
  mat R(p, p, fill::zeros);
  
  for (size_t j = 0; j < p; ++j) {
    // R(j,j) = sqrt(X(j,j) - sum(R(k,j)^2 for k < j))
    double R_jj = XtX(j, j);
    
    if (j > 0) {
      // Dot product of column R(.,j) with itself (excluding elements)
      uvec valid_k = find(excluded.head(j) == 0);
      if (!valid_k.is_empty()) {
        vec R_col_j = R.submat(valid_k, uvec{j});
        R_jj -= dot(R_col_j, R_col_j);
      }
    }
    
    // Check for rank deficiency
    if (R_jj < tol) {
      excluded(j) = 1;
      if (accu(excluded) == p) return false;
      continue;
    }
    
    R_jj = std::sqrt(R_jj);
    R(j, j) = R_jj;
    
    // Row R(j, j+1:p-1)
    if (j < p - 1) {
      // Remaining column indices
      uvec remaining_cols = regspace<uvec>(j + 1, p - 1);
      
      vec R_row_j = XtX.submat(remaining_cols, uvec{j});
      
      if (j > 0) {
        // R_row_j -= R(valid_k, j+1:p-1).t() * R(valid_k, j)
        uvec valid_k = find(excluded.head(j) == 0);
        if (!valid_k.is_empty()) {
          mat R_prev_cols = R.submat(valid_k, remaining_cols);  // R(valid_k, j+1:p-1)
          vec R_prev_j = R.submat(valid_k, uvec{j});            // R(valid_k, j)
          
          R_row_j -= R_prev_cols.t() * R_prev_j;
        }
      }
      
      R_row_j /= R_jj;
      R.submat(uvec{j}, remaining_cols) = R_row_j.t();
    }
  }
  
  return accu(excluded) < p;
}

inline CollinearityResult check_collinearity(mat &X, const vec &w,
                                             bool has_weights, double tolerance,
                                             bool store_qr = false) {
  CAPYBARA_TIME_FUNCTION("check_collinearity");
  
  const size_t p = X.n_cols;
  CollinearityResult result(p);

  if (p == 0) {
    result.coef_status = uvec();
    return result;
  }

  // Early exit for single column
  if (p == 1) {
    // Check if the single column is all zeros or has very small variance
    vec col = X.col(0);
    if (has_weights) {
      col = col % sqrt(w);
    }
    double var_col = var(col);
    if (var_col < tolerance * tolerance) {
      result.coef_status.zeros();
      result.has_collinearity = true;
      result.n_valid = 0;
      result.non_collinear_cols = uvec();
      X.reset();
    }
    return result;
  }

  // Build X'X matrix efficiently
  mat XtX;
  if (has_weights) {
    // Compute X'WX efficiently without creating NxN diagonal matrix
    mat X_weighted = X;
    X_weighted.each_col() %= sqrt(w);
    XtX = X_weighted.t() * X_weighted;
  } else {
    XtX = X.t() * X;
  }

  // Use fixest's rank-revealing approach (exact match for compatibility)
  uvec excluded;
  bool success = rank_revealing_cholesky(excluded, XtX, tolerance);
  
  if (!success) {
    // All variables are collinear
    result.coef_status.zeros();
    result.has_collinearity = true;
    result.n_valid = 0;
    result.non_collinear_cols = uvec();
    X.reset();
    return result;
  }

  // Find non-excluded columns
  const uvec indep = find(excluded == 0);
  
  result.coef_status.zeros();
  if (!indep.is_empty()) {
    result.coef_status.elem(indep).ones();
  }
  result.has_collinearity = (indep.n_elem < p);
  result.n_valid = indep.n_elem;
  result.non_collinear_cols = indep;

  // Store decomposition if requested (for future use in beta computation)
  if (store_qr && !result.has_collinearity && result.n_valid > 0) {
    // Store the Cholesky factor of X'X for non-collinear columns
    mat XtX_reduced = XtX.submat(indep, indep);
    mat L;
    if (chol(L, XtX_reduced, "lower")) {
      result.R = std::move(L);
      result.has_qr = true;
    }
  }

  // Remove collinear columns from X
  if (result.has_collinearity && !indep.is_empty()) {
    X = X.cols(indep);
  } else if (result.has_collinearity && indep.is_empty()) {
    X.reset();
  }

  return result;
}

struct BetaWorkspace {
  mat XtX;
  vec XtY;
  mat L;
  vec beta_work;
  mat X_weighted;

  BetaWorkspace(size_t n, size_t p) {
    CAPYBARA_TIME_FUNCTION("BetaWorkspace::BetaWorkspace");

    size_t safe_n = std::max(n, size_t(1));
    size_t safe_p = std::max(p, size_t(1));

    XtX.set_size(safe_p, safe_p);
    XtY.set_size(safe_p);
    L.set_size(safe_p, safe_p);
    beta_work.set_size(safe_p);
    X_weighted.set_size(safe_n, safe_p);
  }
};

inline InferenceBeta get_beta(const mat &X, const vec &y, const vec &y_orig,
                              const vec &w,
                              const CollinearityResult &collin_result,
                              bool has_weights, bool has_fixed_effects,
                              BetaWorkspace *ws = nullptr) {
  CAPYBARA_TIME_FUNCTION("get_beta");
  
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  const size_t p_orig = collin_result.coef_status.n_elem;

  InferenceBeta result(n, p_orig);

  if (p == 0 || collin_result.n_valid == 0) {
    result.success = true;
    result.coefficients.zeros();
    result.coef_status = collin_result.coef_status;
    result.fitted_values = has_fixed_effects ? y_orig : zeros<vec>(n);
    result.residuals = y_orig - result.fitted_values;
    result.weights = w;
    result.hessian.zeros();
    return result;
  }

  std::unique_ptr<BetaWorkspace> temp_ws;
  if (!ws) {
    temp_ws = std::make_unique<BetaWorkspace>(n, std::max(p, size_t(1)));
    ws = temp_ws.get();
  }

  if (p == 0) {
    result.success = false;
    return result;
  }

  mat XtX = ws->XtX.submat(0, 0, p - 1, p - 1);
  vec XtY = ws->XtY.subvec(0, p - 1);
  if (has_weights) {
    mat X_w = ws->X_weighted.submat(0, 0, n - 1, p - 1);

    X_w = X;
    X_w.each_col() %= sqrt(w);

    XtX = X_w.t() * X_w;
    XtY = X.t() * (w % y);
  } else {

    XtX = X.t() * X;
    XtY = X.t() * y;
  }

  mat L = ws->L.submat(0, 0, p - 1, p - 1);
  if (!chol(L, XtX, "lower")) {
    result.success = false;
    return result;
  }

  vec beta_reduced = ws->beta_work.subvec(0, p - 1);
  beta_reduced = solve(trimatl(L), XtY);
  beta_reduced = solve(trimatu(L.t()), beta_reduced);

  result.coefficients.zeros();
  if (collin_result.has_collinearity) {
    result.coefficients.elem(collin_result.non_collinear_cols) = beta_reduced;
  } else {
    result.coefficients = beta_reduced;
  }

  result.coef_status = collin_result.coef_status;

  if (has_fixed_effects) {
    vec pred_demeaned = X * beta_reduced;
    result.fitted_values = y_orig - (y - pred_demeaned);
  } else {
    result.fitted_values = X * beta_reduced;
  }

  result.residuals = y_orig - result.fitted_values;
  if (has_weights) {
    result.residuals /= sqrt(w);
  }

  result.weights = w;

  result.hessian.zeros();
  if (collin_result.has_collinearity) {
    result.hessian(collin_result.non_collinear_cols,
                   collin_result.non_collinear_cols) = XtX;
  } else {
    result.hessian = XtX;
  }

  result.success = true;
  return result;
}

struct AlphaWorkspace {
  vec group_sums;
  uvec group_counts;
  vec residual;
  vec current_fe;
  vec old_fe;
  umat obs_to_group;

  AlphaWorkspace(size_t n_obs, size_t n_fe, size_t max_groups) {
    CAPYBARA_TIME_FUNCTION("AlphaWorkspace::AlphaWorkspace");

    size_t safe_n_obs = std::max(n_obs, size_t(1));
    size_t safe_n_fe = std::max(n_fe, size_t(1));
    size_t safe_max_groups = std::max(max_groups, size_t(1));

    group_sums.set_size(safe_max_groups);
    group_counts.set_size(safe_max_groups);
    residual.set_size(safe_n_obs);
    current_fe.set_size(safe_n_obs);
    old_fe.set_size(safe_n_obs);
    obs_to_group.set_size(safe_n_obs, safe_n_fe);
  }
};

inline InferenceAlpha get_alpha(const vec &sumFE,
                                const field<field<uvec>> &group_indices,
                                double tol, size_t iter_max,
                                AlphaWorkspace *ws = nullptr) {
  CAPYBARA_TIME_FUNCTION("get_alpha");
  
  const size_t Q = group_indices.n_elem;
  const size_t N = sumFE.n_elem;

  InferenceAlpha result;

  if (Q == 0) {
    result.Alpha.set_size(1);
    result.Alpha(0) = vec(1, fill::value(mean(sumFE)));
    result.nb_references.set_size(1);
    result.nb_references.zeros();
    result.is_regular = true;
    result.success = true;
    return result;
  }

  uvec cluster_sizes(Q);
  size_t max_groups = 0;
  for (size_t q = 0; q < Q; ++q) {
    cluster_sizes(q) = group_indices(q).n_elem;
    max_groups = std::max(max_groups, static_cast<size_t>(cluster_sizes(q)));
  }

  std::unique_ptr<AlphaWorkspace> temp_ws;
  if (!ws) {
    temp_ws = std::make_unique<AlphaWorkspace>(N, Q, max_groups);
    ws = temp_ws.get();
  }

  if (max_groups == 0) {
    result.Alpha.set_size(Q);
    for (size_t q = 0; q < Q; ++q) {
      result.Alpha(q) = vec(0, arma::fill::zeros);
    }
    result.nb_references.set_size(Q);
    result.nb_references.zeros();
    result.is_regular = true;
    result.success = true;
    return result;
  }

  // Use simple alternating projection method (like fixest)
  result.Alpha.set_size(Q);
  for (size_t q = 0; q < Q; ++q) {
    result.Alpha(q) = vec(cluster_sizes(q), fill::zeros);
  }

  // Build observation-to-group mapping using workspace
  umat &obs_to_group = ws->obs_to_group;
  if (obs_to_group.n_rows < N || obs_to_group.n_cols < Q) {
    obs_to_group.set_size(N, Q);
  }
  
  for (size_t q = 0; q < Q; ++q) {
    for (size_t g = 0; g < group_indices(q).n_elem; ++g) {
      const uvec &group_obs = group_indices(q)(g);
      for (uword obs : group_obs) {
        obs_to_group(obs, q) = g;
      }
    }
  }

  // Alternating projection algorithm (similar to fixest approach)
  vec &current_fe = ws->current_fe;
  vec &old_fe = ws->old_fe;
  vec &residual = ws->residual;
  vec &group_sums = ws->group_sums;
  uvec &group_counts = ws->group_counts;
  
  if (current_fe.n_elem < N) current_fe.set_size(N);
  if (old_fe.n_elem < N) old_fe.set_size(N);
  if (residual.n_elem < N) residual.set_size(N);
  if (group_sums.n_elem < max_groups) group_sums.set_size(max_groups);
  if (group_counts.n_elem < max_groups) group_counts.set_size(max_groups);
  
  current_fe.subvec(0, N-1).zeros();
  bool converged = false;
  size_t iter = 0;

  // Set first group in each dimension as reference (except first dimension)
  uvec nb_ref(Q, fill::zeros);
  for (size_t q = 1; q < Q; ++q) {
    if (cluster_sizes(q) > 0) {
      result.Alpha(q)(0) = 0.0;  // First group is reference
      nb_ref(q) = 1;
    }
  }

  while (iter < iter_max && !converged) {
    iter++;
    old_fe.subvec(0, N-1) = current_fe.subvec(0, N-1);
    
    // For each dimension, solve for fixed effects
    for (size_t q = 0; q < Q; ++q) {
      if (cluster_sizes(q) == 0) continue;
      
      // Compute residual after removing other dimensions
      residual.subvec(0, N-1) = sumFE;
      for (size_t other_q = 0; other_q < Q; ++other_q) {
        if (other_q != q) {
          for (size_t obs = 0; obs < N; ++obs) {
            residual(obs) -= result.Alpha(other_q)(obs_to_group(obs, other_q));
          }
        }
      }
      
      // Compute group averages for dimension q
      group_sums.subvec(0, cluster_sizes(q)-1).zeros();
      group_counts.subvec(0, cluster_sizes(q)-1).zeros();
      
      for (size_t obs = 0; obs < N; ++obs) {
        uword g = obs_to_group(obs, q);
        group_sums(g) += residual(obs);
        group_counts(g)++;
      }
      
      // Set averages (with reference constraint if not first dimension)
      for (size_t g = 0; g < cluster_sizes(q); ++g) {
        if (group_counts(g) > 0) {
          result.Alpha(q)(g) = group_sums(g) / group_counts(g);
        }
      }
      
      // Apply reference constraint (all groups sum to zero, except first dimension)
      if (q > 0) {
        double mean_fe = 0.0;
        size_t total_count = 0;
        for (size_t g = 0; g < cluster_sizes(q); ++g) {
          mean_fe += result.Alpha(q)(g) * group_counts(g);
          total_count += group_counts(g);
        }
        if (total_count > 0) {
          mean_fe /= total_count;
          result.Alpha(q) -= mean_fe;
        }
      }
    }
    
    // Check convergence by computing current fitted values
    current_fe.subvec(0, N-1).zeros();
    for (size_t obs = 0; obs < N; ++obs) {
      for (size_t q = 0; q < Q; ++q) {
        current_fe(obs) += result.Alpha(q)(obs_to_group(obs, q));
      }
    }
    
    // Convergence check
    if (iter > 1) {
      double diff = norm(current_fe.subvec(0, N-1) - old_fe.subvec(0, N-1), 2);
      if (diff < tol) {
        converged = true;
      }
    }
  }

  result.nb_references = nb_ref;
  result.is_regular = (Q <= 2) || (accu(nb_ref) == Q - 1);
  result.success = converged || (iter < iter_max);

  return result;
}

} // namespace parameters
} // namespace capybara

#endif // CAPYBARA_PARAMETERS_H