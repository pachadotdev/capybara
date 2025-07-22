// Computing beta and then alpha in a model with fixed effects
// Y = alpha + X beta

#ifndef CAPYBARA_PARAMETERS_H
#define CAPYBARA_PARAMETERS_H

namespace capybara {
namespace parameters {

//////////////////////////////////////////////////////////////////////////////
// RESULT STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// Beta computation result
struct InferenceBeta {
  vec coefficients;
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  uvec coef_status;
  bool success;

  InferenceBeta(size_t n, size_t p)
      : coefficients(p, fill::none), fitted_values(n, fill::none),
        residuals(n, fill::none), weights(n, fill::none),
        hessian(p, p, fill::none), coef_status(p, fill::none), success(false) {}
};

// Fixed effects extraction result
struct InferenceAlpha {
  field<vec> Alpha;
  uvec nb_references; // Number of references per dimension (fixest
                      // compatibility)
  bool is_regular;    // Whether fixed effects are regular
  bool success;       // Whether extraction succeeded

  InferenceAlpha() : is_regular(true), success(false) {}

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

//////////////////////////////////////////////////////////////////////////////
// STRUCTURAL PARAMETERS ESTIMATION
//////////////////////////////////////////////////////////////////////////////

// Result structure for collinearity detection
struct CollinearityResult {
  uvec coef_status;
  bool has_collinearity;
  size_t n_valid;
  uvec non_collinear_cols; // Indices of non-collinear columns
  mat Q, R; // QR decomposition for reuse
  bool has_qr; // Whether QR decomposition is available

  CollinearityResult(size_t p)
      : coef_status(p, fill::ones), has_collinearity(false), n_valid(p), 
        has_qr(false) {}
};

// Collinearity detection that directly modifies X
inline CollinearityResult check_collinearity(mat &X,
                                                       const vec &w,
                                                       bool has_weights,
                                                       double tolerance,
                                                       bool store_qr = false) {
  const size_t p = X.n_cols;
  CollinearityResult result(p);

  if (p == 0) {
    result.coef_status = uvec();
    return result;
  }

  mat Q, R;

  if (has_weights) {
    mat X_weighted = X.each_col() % sqrt(w);
    qr_econ(Q, R, X_weighted);
  } else {
    qr_econ(Q, R, X);
  }

  const vec diag_abs = abs(R.diag());
  const double max_diag = diag_abs.max();
  const double tol = tolerance * max_diag;
  const uvec indep = find(diag_abs > tol);

  result.coef_status.zeros();
  result.coef_status(indep).ones();
  result.has_collinearity = (indep.n_elem < p);
  result.n_valid = indep.n_elem;
  result.non_collinear_cols = indep;

  // Store QR decomposition if requested
  if (store_qr) {
    result.Q = std::move(Q);
    result.R = std::move(R);
    result.has_qr = true;
  }

  // Directly modify X to keep only non-collinear columns
  if (result.has_collinearity && !indep.is_empty()) {
    X = X.cols(indep);
  }

  return result;
}

// Cholesky-based beta estimation (assumes X has already been trimmed for collinearity)
inline InferenceBeta get_beta_cholesky(const mat &X, const vec &y, const vec &y_orig,
                                       const vec &w, const CollinearityResult &collin_result,
                                       bool has_weights = false,
                                       bool has_fixed_effects = false) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  const size_t p_orig = collin_result.coef_status.n_elem;

  InferenceBeta result(n, p_orig);

  if (p == 0) {
    result.success = true;
    result.coef_status = collin_result.coef_status;
    result.fitted_values = has_fixed_effects ? y_orig : zeros<vec>(n);
    result.residuals = y_orig - result.fitted_values;
    result.weights = w;
    return result;
  }

  // Form normal equations
  mat XtX, XtY;
  if (has_weights) {
    const vec sqrt_w = sqrt(w);
    const mat X_weighted = X.each_col() % sqrt_w;
    XtX = X_weighted.t() * X_weighted;
    XtY = X.t() * (w % y);
  } else {
    XtX = X.t() * X;
    XtY = X.t() * y;
  }

  // Cholesky decomposition
  mat L;
  if (!chol(L, XtX, "lower")) {
    result.success = false;
    return result;
  }

  // Solve normal equations
  vec work = solve(trimatl(L), XtY, solve_opts::fast);
  vec beta_reduced = solve(trimatu(L.t()), work, solve_opts::fast);

  // Expand coefficients back to original size
  result.coefficients.fill(datum::nan);
  if (collin_result.has_collinearity) {
    uvec valid_cols = find(collin_result.coef_status == 1);
    result.coefficients(valid_cols) = beta_reduced;
    // Set collinear coefficients to 0
    uvec collinear_mask = find(collin_result.coef_status == 0);
    if (!collinear_mask.is_empty()) {
      result.coefficients(collinear_mask).zeros();
    }
  } else {
    result.coefficients = beta_reduced;
  }

  result.coef_status = collin_result.coef_status;

  // Compute fitted values and residuals using original X matrix size
  if (has_fixed_effects) {
    // For fixed effects models:
    // fitted_values = y_orig - (y_demeaned - X_demeaned * beta)
    const vec pred_demeaned = X * beta_reduced;
    result.fitted_values = y_orig - (y - pred_demeaned);
  } else {
    // Standard case: fitted_values = X_full * beta_full
    // Need to reconstruct with original X dimensions
    if (collin_result.has_collinearity) {
      // This would require original X, so we'll approximate
      result.fitted_values = X * beta_reduced;
    } else {
      result.fitted_values = X * beta_reduced;
    }
  }

  result.residuals = y_orig - result.fitted_values;

  if (has_weights) {
    result.residuals = result.residuals / sqrt(w);
  }

  result.weights = w;

  // Hessian computation
  if (has_weights) {
    const vec sqrt_w = sqrt(w);
    const mat X_weighted = X.each_col() % sqrt_w;
    result.hessian.zeros();
    if (collin_result.has_collinearity) {
      uvec valid_cols = find(collin_result.coef_status == 1);
      result.hessian.submat(valid_cols, valid_cols) = X_weighted.t() * X_weighted;
    } else {
      result.hessian = X_weighted.t() * X_weighted;
    }
  } else {
    result.hessian.zeros();
    if (collin_result.has_collinearity) {
      uvec valid_cols = find(collin_result.coef_status == 1);
      result.hessian.submat(valid_cols, valid_cols) = X.t() * X;
    } else {
      result.hessian = X.t() * X;
    }
  }

  result.success = true;
  return result;
}

// QR-based beta estimation that reuses QR decomposition from collinearity detection
inline InferenceBeta get_beta_qr(const mat &X, const vec &y, const vec &y_orig,
                                 const vec &w, const CollinearityResult &collin_result,
                                 bool has_weights = false,
                                 bool has_fixed_effects = false) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;
  const size_t p_orig = collin_result.coef_status.n_elem;

  InferenceBeta result(n, p_orig);

  if (p == 0) {
    result.success = true;
    result.coef_status = collin_result.coef_status;
    result.fitted_values = has_fixed_effects ? y_orig : zeros<vec>(n);
    result.residuals = y_orig - result.fitted_values;
    result.weights = w;
    return result;
  }

  // Use stored QR decomposition if available
  if (!collin_result.has_qr) {
    // Fallback to Cholesky method
    return get_beta_cholesky(X, y, y_orig, w, collin_result, has_weights, has_fixed_effects);
  }

  const mat &Q = collin_result.Q;
  const mat &R = collin_result.R;
  const uvec &indep = collin_result.non_collinear_cols;

  // Compute Q^T * y
  vec QTy;
  if (has_weights) {
    vec y_weighted = y % sqrt(w);
    QTy = Q.t() * y_weighted;
  } else {
    QTy = Q.t() * y;
  }

  // Initialize coefficients
  result.coefficients.fill(datum::nan);
  result.coef_status = collin_result.coef_status;

  // Solve using QR decomposition
  if (collin_result.has_collinearity) {
    if (!indep.is_empty()) {
      const mat Rr = R.submat(indep, indep);
      const vec Yr = QTy.elem(indep);
      const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
      result.coefficients(indep) = br;
      // Set collinear coefficients to 0
      uvec collinear_mask = find(collin_result.coef_status == 0);
      if (!collinear_mask.is_empty()) {
        result.coefficients(collinear_mask).zeros();
      }
    }
  } else {
    result.coefficients = solve(trimatu(R), QTy, solve_opts::fast);
  }

  // Compute fitted values and residuals
  if (has_fixed_effects) {
    const vec pred_demeaned = X * (collin_result.has_collinearity ? 
      result.coefficients(indep) : result.coefficients);
    result.fitted_values = y_orig - (y - pred_demeaned);
  } else {
    if (collin_result.has_collinearity) {
      result.fitted_values = X * result.coefficients(indep);
    } else {
      result.fitted_values = X * result.coefficients;
    }
  }

  result.residuals = y_orig - result.fitted_values;

  if (has_weights) {
    result.residuals = result.residuals / sqrt(w);
  }

  result.weights = w;

  // Hessian computation using QR
  if (has_weights) {
    const vec sqrt_w = sqrt(w);
    const mat X_weighted = X.each_col() % sqrt_w;
    result.hessian.zeros();
    if (collin_result.has_collinearity) {
      uvec valid_cols = find(collin_result.coef_status == 1);
      result.hessian.submat(valid_cols, valid_cols) = X_weighted.t() * X_weighted;
    } else {
      result.hessian = X_weighted.t() * X_weighted;
    }
  } else {
    result.hessian.zeros();
    if (collin_result.has_collinearity) {
      uvec valid_cols = find(collin_result.coef_status == 1);
      result.hessian.submat(valid_cols, valid_cols) = X.t() * X;
    } else {
      result.hessian = X.t() * X;
    }
  }

  result.success = true;
  return result;
}

// Unified beta estimation function that chooses between QR and Cholesky
inline InferenceBeta get_beta(mat X, const vec &y, const vec &y_orig,
                              const vec &w, const CapybaraParameters &params,
                              bool has_weights = false,
                              bool has_fixed_effects = false,
                              bool first_iter = true) {
  // Detect and trim collinearity
  double tolerance = params.qr_collin_tol_multiplier * 1e-7;
  CollinearityResult collin_result = check_collinearity(X, w, has_weights, tolerance, first_iter);
  
  // Call the appropriate function based on iteration preference
  if (first_iter && collin_result.has_qr) {
    return get_beta_qr(X, y, y_orig, w, collin_result, has_weights, has_fixed_effects);
  } else {
    return get_beta_cholesky(X, y, y_orig, w, collin_result, has_weights, has_fixed_effects);
  }
}

//////////////////////////////////////////////////////////////////////////////
// FIXED EFFECTS ESTIMATION
//////////////////////////////////////////////////////////////////////////////

inline InferenceAlpha get_alpha(const vec &sumFE,
                                const field<field<uvec>> &group_indices,
                                double tol, size_t iter_max) {
  const size_t Q = group_indices.n_elem;
  const size_t N = sumFE.n_elem;
  InferenceAlpha result;

  if (Q == 0) {
    // No fixed effects => return intercept
    result.Alpha.set_size(1);
    result.Alpha(0) = vec(1);
    result.Alpha(0)(0) = mean(sumFE);
    result.nb_references.set_size(1);
    result.nb_references(0) = 0;
    result.is_regular = true;
    result.success = true;
    return result;
  }

  // Convert group_indices to dumMat format (N x Q matrix)
  umat dumMat(N, Q);
  uvec cluster_sizes(Q);
  for (size_t q = 0; q < Q; ++q) {
    cluster_sizes(q) = group_indices(q).n_elem;
    for (size_t g = 0; g < group_indices(q).n_elem; ++g) {
      const uvec &group_obs = group_indices(q)(g);
      for (size_t i = 0; i < group_obs.n_elem; ++i) {
        dumMat(group_obs(i), q) = g;
      }
    }
  }

  // Total number of coefficients
  size_t nb_coef = accu(cluster_sizes);
  vec cluster_values(nb_coef, fill::zeros);

  // Index mapping for clusters
  uvec cluster_starts(Q);
  cluster_starts(0) = 0;
  for (size_t q = 1; q < Q; ++q) {
    cluster_starts(q) = cluster_starts(q - 1) + cluster_sizes(q - 1);
  }

  // Create observation lists for each cluster coefficient
  field<uvec> obs_by_cluster(nb_coef);
  for (size_t q = 0; q < Q; ++q) {
    for (size_t k = 0; k < cluster_sizes(q); ++k) {
      uvec obs_in_cluster = find(dumMat.col(q) == k);
      obs_by_cluster(cluster_starts(q) + k) = obs_in_cluster;
    }
  }

  // Matrix tracking which clusters have been computed
  umat mat_done(N, Q, fill::zeros);
  uvec rowsums(N, fill::zeros);
  uvec nb_ref(Q, fill::zeros);

  // Main algorithm loop
  size_t iter = 0;
  uvec id_todo = regspace<uvec>(0, N - 1);
  size_t nb_todo = N;

  while (iter < iter_max && nb_todo > 0) {
    iter++;

    // Find observation with maximum rowsum (most FEs already computed)
    uword qui_max = 0;
    uword rs_max = 0;

    if (iter == 1) {
      qui_max = 0;
    } else {
      for (size_t i = 0; i < nb_todo; ++i) {
        uword obs = id_todo(i);
        uword rs = rowsums(obs);

        if (rs == Q - 2) {
          qui_max = obs;
          break;
        } else if (rs < Q && rs > rs_max) {
          qui_max = obs;
          rs_max = rs;
        }
      }
    }

    // Set references for this observation
    bool first = true;
    for (size_t q = 0; q < Q; ++q) {
      if (mat_done(qui_max, q) == 0) {
        if (first) {
          first = false; // Skip first dimension
        } else {
          // Set this cluster coefficient as reference (= 0)
          uword id_cluster = dumMat(qui_max, q);
          size_t index = cluster_starts(q) + id_cluster;
          cluster_values(index) = 0;

          // Mark all observations in this cluster as done for dimension q
          const uvec &obs_in_cluster = obs_by_cluster(index);
          for (size_t i = 0; i < obs_in_cluster.n_elem; ++i) {
            mat_done(obs_in_cluster(i), q) = 1;
            rowsums(obs_in_cluster(i))++;
          }

          nb_ref(q)++;
        }
      }
    }

    // Update loop: compute values for observations with Q-1 dimensions done
    bool changed = true;
    size_t iter_loop = 0;

    while (changed && iter_loop < iter_max) {
      iter_loop++;
      changed = false;

      std::vector<uword> new_todo_vec;
      new_todo_vec.reserve(nb_todo);

      for (size_t i = 0; i < nb_todo; ++i) {
        uword obs = id_todo(i);
        uword rs = rowsums(obs);

        if (rs < Q - 1) {
          // Still need to process later
          new_todo_vec.push_back(obs);
        } else if (rs == Q - 1) {
          // Can compute the remaining FE for this observation
          changed = true;

          // Find which dimension needs to be computed
          size_t q_missing = 0;
          for (size_t q = 0; q < Q; ++q) {
            if (mat_done(obs, q) == 0) {
              q_missing = q;
              break;
            }
          }

          // Compute sum of other FE values
          double other_values = 0;
          for (size_t q = 0; q < Q; ++q) {
            if (q != q_missing) {
              size_t index = cluster_starts(q) + dumMat(obs, q);
              other_values += cluster_values(index);
            }
          }

          // Set the missing cluster value
          size_t index_missing =
              cluster_starts(q_missing) + dumMat(obs, q_missing);
          cluster_values(index_missing) = sumFE(obs) - other_values;

          // Mark all observations in this cluster as done
          const uvec &obs_in_cluster = obs_by_cluster(index_missing);
          for (size_t i = 0; i < obs_in_cluster.n_elem; ++i) {
            mat_done(obs_in_cluster(i), q_missing) = 1;
            rowsums(obs_in_cluster(i))++;
          }
        }
      }

      // Convert std::vector back to uvec
      uvec new_todo(new_todo_vec.size());
      for (size_t i = 0; i < new_todo_vec.size(); ++i) {
        new_todo(i) = new_todo_vec[i];
      }

      id_todo = new_todo;
      nb_todo = new_todo.n_elem;
    }

    if (nb_todo == 0)
      break;
  }

  // Extract results into separate vectors for each FE dimension
  result.Alpha.set_size(Q);
  for (size_t q = 0; q < Q; ++q) {
    result.Alpha(q) = cluster_values.subvec(
        cluster_starts(q), cluster_starts(q) + cluster_sizes(q) - 1);
  }

  result.nb_references = nb_ref;
  result.is_regular = (Q <= 2) || (accu(nb_ref) == Q - 1);
  result.success = (iter < iter_max);

  return result;
}

} // namespace parameters
} // namespace capybara

#endif // CAPYBARA_PARAMETERS_H
