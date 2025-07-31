// Computing beta and then alpha in a model with fixed effects
// Y = alpha + X beta

#ifndef CAPYBARA_PARAMETERS_H
#define CAPYBARA_PARAMETERS_H

namespace capybara {

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
    
  }
};

struct InferenceAlpha {
  field<vec> Alpha;
  uvec nb_references;
  bool is_regular;
  bool success;

  InferenceAlpha() : is_regular(true), success(false) {}

  cpp11::list to_list() const {
    

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
    
  }
};

inline bool rank_revealing_cholesky(uvec &excluded, const mat &XtX,
                                    double tol) {
  const size_t p = XtX.n_cols;
  excluded.zeros(p);

  if (p == 0)
    return true;

  mat R(p, p, fill::zeros);

  double *R_ptr = R.memptr();
  uword *excluded_ptr = excluded.memptr();

  for (size_t j = 0; j < p; ++j) {

    double R_jj = XtX(j, j);

    if (j > 0) {

      double sum_squares = 0.0;
      for (size_t k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          double R_kj = R_ptr[k + j * p];
          sum_squares += R_kj * R_kj;
        }
      }
      R_jj -= sum_squares;
    }

    if (R_jj < tol) {
      excluded_ptr[j] = 1;

      bool all_excluded = true;
      for (size_t k = 0; k < p; ++k) {
        if (excluded_ptr[k] == 0) {
          all_excluded = false;
          break;
        }
      }
      if (all_excluded)
        return false;
      continue;
    }

    R_jj = std::sqrt(R_jj);
    R_ptr[j + j * p] = R_jj;

    for (size_t col = j + 1; col < p; ++col) {
      double R_j_col = XtX(j, col);

      for (size_t k = 0; k < j; ++k) {
        if (excluded_ptr[k] == 0) {
          R_j_col -= R_ptr[k + j * p] * R_ptr[k + col * p];
        }
      }

      R_ptr[j + col * p] = R_j_col / R_jj;
    }
  }

  size_t n_excluded = 0;
  for (size_t j = 0; j < p; ++j) {
    if (excluded_ptr[j] == 1)
      n_excluded++;
  }

  return n_excluded < p;
}

inline CollinearityResult check_collinearity(mat &X, const vec &w,
                                             bool has_weights, double tolerance,
                                             bool store_qr = false) {
  

  const size_t p = X.n_cols;
  const size_t n = X.n_rows;
  CollinearityResult result(p);

  if (p == 0) {
    result.coef_status = uvec();
    return result;
  }

  if (p == 1) {
    const double *col_ptr = X.colptr(0);
    const double *w_ptr = has_weights ? w.memptr() : nullptr;

    double mean_val = 0.0, sum_sq = 0.0, sum_w = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double val = col_ptr[i];
      double weight = has_weights ? w_ptr[i] : 1.0;

      if (has_weights) {
        val *= std::sqrt(weight);
        sum_w += weight;
      } else {
        sum_w += 1.0;
      }
      mean_val += val;
      sum_sq += val * val;
    }

    mean_val /= sum_w;
    double variance = (sum_sq / sum_w) - (mean_val * mean_val);

    if (variance < tolerance * tolerance) {
      result.coef_status.zeros();
      result.has_collinearity = true;
      result.n_valid = 0;
      result.non_collinear_cols = uvec();
      X.reset();
    }
    return result;
  }

  mat XtX(p, p);
  if (has_weights) {

    const double *w_ptr = w.memptr();

    for (size_t i = 0; i < p; ++i) {
      const double *Xi_ptr = X.colptr(i);
      for (size_t j = i; j < p; ++j) {
        const double *Xj_ptr = X.colptr(j);

        double sum = 0.0;
        for (size_t obs = 0; obs < n; ++obs) {
          sum += Xi_ptr[obs] * Xj_ptr[obs] * w_ptr[obs];
        }

        XtX(i, j) = sum;
        if (i != j) {
          XtX(j, i) = sum;
        }
      }
    }
  } else {
    XtX = X.t() * X;
  }

  uvec excluded;
  bool success = rank_revealing_cholesky(excluded, XtX, tolerance);

  if (!success) {

    result.coef_status.zeros();
    result.has_collinearity = true;
    result.n_valid = 0;
    result.non_collinear_cols = uvec();
    X.reset();
    return result;
  }

  const uvec indep = find(excluded == 0);

  result.coef_status.zeros();
  if (!indep.is_empty()) {
    result.coef_status.elem(indep).ones();
  }
  result.has_collinearity = (indep.n_elem < p);
  result.n_valid = indep.n_elem;
  result.non_collinear_cols = indep;

  if (store_qr && !result.has_collinearity && result.n_valid > 0) {
    mat XtX_reduced = XtX.submat(indep, indep);
    mat L;
    if (chol(L, XtX_reduced, "lower")) {
      result.R = std::move(L);
      result.has_qr = true;
    }
  }

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

    const double *w_ptr = w.memptr();

    for (size_t i = 0; i < p; ++i) {
      const double *Xi_ptr = X.colptr(i);
      for (size_t j = i; j < p; ++j) {
        const double *Xj_ptr = X.colptr(j);

        double sum = 0.0;
        for (size_t obs = 0; obs < n; ++obs) {
          sum += Xi_ptr[obs] * Xj_ptr[obs] * w_ptr[obs];
        }

        XtX(i, j) = sum;
        if (i != j) {
          XtX(j, i) = sum;
        }
      }
    }

    const double *y_ptr = y.memptr();
    for (size_t i = 0; i < p; ++i) {
      const double *Xi_ptr = X.colptr(i);
      double sum = 0.0;
      for (size_t obs = 0; obs < n; ++obs) {
        sum += Xi_ptr[obs] * y_ptr[obs] * w_ptr[obs];
      }
      XtY(i) = sum;
    }
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

  result.Alpha.set_size(Q);
  for (size_t q = 0; q < Q; ++q) {
    result.Alpha(q) = vec(cluster_sizes(q), fill::zeros);
  }

  umat &obs_to_group = ws->obs_to_group;
  if (obs_to_group.n_rows < N || obs_to_group.n_cols < Q) {
    obs_to_group.set_size(N, Q);
  }

  for (size_t q = 0; q < Q; ++q) {
    uword *obs_group_ptr = obs_to_group.colptr(q);
    for (size_t g = 0; g < group_indices(q).n_elem; ++g) {
      const uvec &group_obs = group_indices(q)(g);
      const uword *group_obs_ptr = group_obs.memptr();
      size_t n_group_obs = group_obs.n_elem;

      for (size_t i = 0; i < n_group_obs; ++i) {
        obs_group_ptr[group_obs_ptr[i]] = g;
      }
    }
  }

  vec &current_fe = ws->current_fe;
  vec &old_fe = ws->old_fe;
  vec &residual = ws->residual;
  vec &group_sums = ws->group_sums;
  uvec &group_counts = ws->group_counts;

  if (current_fe.n_elem < N)
    current_fe.set_size(N);
  if (old_fe.n_elem < N)
    old_fe.set_size(N);
  if (residual.n_elem < N)
    residual.set_size(N);
  if (group_sums.n_elem < max_groups)
    group_sums.set_size(max_groups);
  if (group_counts.n_elem < max_groups)
    group_counts.set_size(max_groups);

  current_fe.subvec(0, N - 1).zeros();
  bool converged = false;
  size_t iter = 0;

  uvec nb_ref(Q, fill::zeros);
  for (size_t q = 1; q < Q; ++q) {
    if (cluster_sizes(q) > 0) {
      result.Alpha(q)(0) = 0.0;
      nb_ref(q) = 1;
    }
  }

  const double *sumFE_ptr = sumFE.memptr();
  double *current_fe_ptr = current_fe.memptr();
  double *old_fe_ptr = old_fe.memptr();
  double *residual_ptr = residual.memptr();
  double *group_sums_ptr = group_sums.memptr();
  uword *group_counts_ptr = group_counts.memptr();

  while (iter < iter_max && !converged) {
    iter++;

    for (size_t obs = 0; obs < N; ++obs) {
      old_fe_ptr[obs] = current_fe_ptr[obs];
    }

    for (size_t q = 0; q < Q; ++q) {
      if (cluster_sizes(q) == 0)
        continue;

      for (size_t obs = 0; obs < N; ++obs) {
        residual_ptr[obs] = sumFE_ptr[obs];
      }

      for (size_t other_q = 0; other_q < Q; ++other_q) {
        if (other_q != q) {
          const uword *obs_group_ptr = obs_to_group.colptr(other_q);
          const double *alpha_ptr = result.Alpha(other_q).memptr();

          for (size_t obs = 0; obs < N; ++obs) {
            residual_ptr[obs] -= alpha_ptr[obs_group_ptr[obs]];
          }
        }
      }

      size_t n_groups = cluster_sizes(q);
      for (size_t g = 0; g < n_groups; ++g) {
        group_sums_ptr[g] = 0.0;
        group_counts_ptr[g] = 0;
      }

      const uword *obs_group_ptr = obs_to_group.colptr(q);
      for (size_t obs = 0; obs < N; ++obs) {
        uword g = obs_group_ptr[obs];
        group_sums_ptr[g] += residual_ptr[obs];
        group_counts_ptr[g]++;
      }

      double *alpha_q_ptr = result.Alpha(q).memptr();
      for (size_t g = 0; g < n_groups; ++g) {
        if (group_counts_ptr[g] > 0) {
          alpha_q_ptr[g] = group_sums_ptr[g] / group_counts_ptr[g];
        }
      }

      if (q > 0) {
        double weighted_mean = 0.0;
        size_t total_count = 0;
        for (size_t g = 0; g < n_groups; ++g) {
          weighted_mean += alpha_q_ptr[g] * group_counts_ptr[g];
          total_count += group_counts_ptr[g];
        }
        if (total_count > 0) {
          weighted_mean /= total_count;
          for (size_t g = 0; g < n_groups; ++g) {
            alpha_q_ptr[g] -= weighted_mean;
          }
        }
      }
    }

    for (size_t obs = 0; obs < N; ++obs) {
      current_fe_ptr[obs] = 0.0;
    }

    for (size_t q = 0; q < Q; ++q) {
      const uword *obs_group_ptr = obs_to_group.colptr(q);
      const double *alpha_q_ptr = result.Alpha(q).memptr();

      for (size_t obs = 0; obs < N; ++obs) {
        current_fe_ptr[obs] += alpha_q_ptr[obs_group_ptr[obs]];
      }
    }

    if (iter > 1) {
      double diff_sq = 0.0;
      for (size_t obs = 0; obs < N; ++obs) {
        double d = current_fe_ptr[obs] - old_fe_ptr[obs];
        diff_sq += d * d;
      }
      if (std::sqrt(diff_sq) < tol) {
        converged = true;
      }
    }
  }

  result.nb_references = nb_ref;
  result.is_regular = (Q <= 2) || (accu(nb_ref) == Q - 1);
  result.success = converged || (iter < iter_max);

  return result;
}

} // namespace capybara

#endif // CAPYBARA_PARAMETERS_H
