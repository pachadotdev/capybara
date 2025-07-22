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
  uvec non_collinear_cols;
  uvec coef_status;
  bool has_collinearity;

  CollinearityResult(size_t p)
      : non_collinear_cols(p), coef_status(p, fill::ones),
        has_collinearity(false) {
    non_collinear_cols = regspace<uvec>(0, p - 1);
  }
};

// Shared QR-based collinearity detection helper
inline CollinearityResult detect_collinearity_qr(const mat &X, const vec &w,
                                                 bool has_weights,
                                                 double tolerance) {
  const size_t p = X.n_cols;
  CollinearityResult result(p);

  if (p == 0) {
    result.non_collinear_cols = uvec();
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

  result.non_collinear_cols = indep;
  result.coef_status.zeros();
  result.coef_status(indep).ones();
  result.has_collinearity = (indep.n_elem < p);

  return result;
}

inline void get_beta_qr(mat &X, const vec &y, const vec &w,
                        InferenceBeta &result, bool has_weights,
                        double qr_collin_tol_multiplier) {
  const size_t p = X.n_cols;

  // Use shared collinearity detection
  double tolerance = qr_collin_tol_multiplier * 1e-7;
  CollinearityResult collin_result =
      detect_collinearity_qr(X, w, has_weights, tolerance);

  // QR decomposition for coefficient computation
  mat Q, R;
  if (has_weights) {
    mat X_weighted = X.each_col() % sqrt(w);
    qr_econ(Q, R, X_weighted);
  } else {
    qr_econ(Q, R, X);
  }

  vec QTy = Q.t() * y;
  const uvec &indep = collin_result.non_collinear_cols;

  result.coefficients.fill(datum::nan);
  result.coef_status = collin_result.coef_status;

  if (indep.n_elem == p) {
    result.coefficients = solve(trimatu(R), QTy, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = R.submat(indep, indep);
    const vec Yr = QTy.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    result.coefficients(indep) = br;
    // Keep NaN for invalid coefficients
  }
}

inline InferenceBeta get_beta(const mat &X, const vec &y, const vec &y_orig,
                              const vec &w, const CapybaraParameters &params,
                              bool has_weights = false,
                              bool has_fixed_effects = false) {
  const size_t n = X.n_rows;
  const size_t p = X.n_cols;

  InferenceBeta result(n, p);

  if (p == 0) {
    result.success = true;
    return result;
  }

  // For very wide matrices, use QR directly
  const bool direct_qr = (p > params.direct_qr_threshold * n);

  if (direct_qr) {
    mat X_copy = X; // QR modifies the matrix
    get_beta_qr(X_copy, y, w, result, has_weights,
                params.qr_collin_tol_multiplier);
  } else {
    // Try Cholesky first (faster), fall back to QR if needed
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

    mat L;
    const bool chol_ok = chol(L, XtX, "lower");

    if (chol_ok) {
      const vec d = abs(L.diag());
      const double mind = d.min();
      const double avgd = mean(d);

      if (mind > params.chol_stability_threshold * avgd) {
        vec work = solve(trimatl(L), XtY, solve_opts::fast);
        result.coefficients = solve(trimatu(L.t()), work, solve_opts::fast);
        result.coef_status.ones();
      } else {
        // Cholesky failed due to conditioning, fall back to QR
        mat X_copy = X;
        get_beta_qr(X_copy, y, w, result, has_weights,
                    params.qr_collin_tol_multiplier);
      }
    } else {
      // Cholesky failed, fall back to QR
      mat X_copy = X;
      get_beta_qr(X_copy, y, w, result, has_weights,
                  params.qr_collin_tol_multiplier);
    }
  }

  // Replace collinear coefficients with 0 using coef_status
  uvec collinear_mask = find(result.coef_status == 0);
  if (!collinear_mask.is_empty()) {
    result.coefficients.elem(collinear_mask).zeros();
  }

  // Compute fitted values and residuals
  if (has_fixed_effects) {
    // For fixed effects models:
    // fitted_values = y_orig - (y_demeaned - X_demeaned * beta)
    const vec pred_demeaned = X * result.coefficients;
    result.fitted_values = y_orig - (y - pred_demeaned);
  } else {
    // Standard case: fitted_values = X * beta
    result.fitted_values = X * result.coefficients;
  }

  result.residuals = y_orig - result.fitted_values;

  if (has_weights) {
    result.residuals = result.residuals / sqrt(w);
  }

  result.weights = w;

  if (has_weights) {
    const vec sqrt_w = sqrt(w);
    const mat X_weighted = X.each_col() % sqrt_w;
    result.hessian = X_weighted.t() * X_weighted;
  } else {
    result.hessian = X.t() * X;
  }
  result.success = true;

  return result;
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
