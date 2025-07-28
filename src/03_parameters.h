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
        has_qr(false) {}
};

inline CollinearityResult check_collinearity(mat &X, const vec &w,
                                             bool has_weights, double tolerance,
                                             bool store_qr = false) {
  const size_t p = X.n_cols;
  CollinearityResult result(p);

  if (p == 0) {
    result.coef_status = uvec();
    return result;
  }

  mat Q, R;
  if (has_weights) {

    vec sqrt_w = sqrt(w);
    mat X_weighted = X.each_col() % sqrt_w;
    qr_econ(Q, R, X_weighted);
  } else {
    qr_econ(Q, R, X);
  }

  const vec diag_abs = abs(R.diag());
  const double tol = tolerance * diag_abs.max();
  const uvec indep = find(diag_abs > tol);

  result.coef_status.zeros();
  result.coef_status.elem(indep).ones();
  result.has_collinearity = (indep.n_elem < p);
  result.n_valid = indep.n_elem;
  result.non_collinear_cols = indep;

  if (store_qr && !result.has_collinearity) {
    result.Q = std::move(Q);
    result.R = std::move(R);
    result.has_qr = true;
  }

  if (result.has_collinearity && !indep.is_empty()) {
    X = X.cols(indep);
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

    mat X_w = ws->X_weighted.submat(0, 0, n - 1, p - 1);
    vec sqrt_w = sqrt(w);

    X_w = X;
    X_w.each_col() %= sqrt_w;

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
  vec cluster_values;
  vec sum_values;
  uvec cluster_counts;
  uvec obs_indices;
  umat dumMat;
  umat mat_done;
  uvec rowsums;
  uvec id_todo;
  vec other_values_vec;

  AlphaWorkspace(size_t n_obs, size_t n_fe, size_t total_clusters) {

    size_t safe_n_obs = std::max(n_obs, size_t(1));
    size_t safe_n_fe = std::max(n_fe, size_t(1));
    size_t safe_total_clusters = std::max(total_clusters, size_t(1));

    cluster_values.set_size(safe_total_clusters);
    cluster_values.zeros();
    sum_values.set_size(safe_total_clusters);
    cluster_counts.set_size(safe_total_clusters);
    cluster_counts.zeros();
    obs_indices.set_size(safe_n_obs);
    dumMat.set_size(safe_n_obs, safe_n_fe);
    mat_done.set_size(safe_n_obs, safe_n_fe);
    mat_done.zeros();
    rowsums.set_size(safe_n_obs);
    rowsums.zeros();
    id_todo.set_size(safe_n_obs);
    other_values_vec.set_size(safe_n_fe);
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
  for (size_t q = 0; q < Q; ++q) {
    cluster_sizes(q) = group_indices(q).n_elem;
  }

  size_t nb_coef = accu(cluster_sizes);

  std::unique_ptr<AlphaWorkspace> temp_ws;
  if (!ws) {
    temp_ws = std::make_unique<AlphaWorkspace>(N, Q, nb_coef);
    ws = temp_ws.get();
  }

  if (nb_coef == 0) {
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

  size_t workspace_size = std::max(nb_coef, size_t(1));

  if (ws->cluster_values.n_elem < workspace_size) {
    ws->cluster_values.set_size(workspace_size);
  }
  if (ws->dumMat.n_rows < N || ws->dumMat.n_cols < Q) {
    ws->dumMat.set_size(N, Q);
  }

  vec cluster_values;
  if (nb_coef > 0) {
    cluster_values = ws->cluster_values.subvec(0, nb_coef - 1);
  } else {
    cluster_values.set_size(0);
  }
  cluster_values.zeros();

  umat dumMat = ws->dumMat.submat(0, 0, N - 1, Q - 1);

  for (size_t q = 0; q < Q; ++q) {
    for (size_t g = 0; g < group_indices(q).n_elem; ++g) {
      const uvec &group_obs = group_indices(q)(g);
      for (uword idx : group_obs) {
        if (idx >= N) {
          throw std::runtime_error("Index out of bounds in dumMat");
        }
        dumMat(idx, q) = g;
      }
    }
  }

  uvec cluster_starts(Q);
  cluster_starts(0) = 0;
  for (size_t q = 1; q < Q; ++q) {
    cluster_starts(q) = cluster_starts(q - 1) + cluster_sizes(q - 1);
  }

  field<uvec> obs_by_cluster(nb_coef);
  for (size_t q = 0; q < Q; ++q) {
    for (size_t k = 0; k < cluster_sizes(q); ++k) {
      obs_by_cluster(cluster_starts(q) + k) = find(dumMat.col(q) == k);
    }
  }

  umat mat_done = ws->mat_done.submat(0, 0, N - 1, Q - 1);
  mat_done.zeros();

  uvec rowsums = ws->rowsums.subvec(0, N - 1);
  rowsums.zeros();

  uvec &nb_ref = result.nb_references;
  nb_ref.set_size(Q);
  nb_ref.zeros();

  size_t iter = 0;
  uvec id_todo = ws->id_todo.subvec(0, N - 1);
  id_todo = regspace<uvec>(0, N - 1);
  size_t nb_todo = N;

  while (iter < iter_max && nb_todo > 0) {
    iter++;

    uword qui_max = id_todo(0);
    uword rs_max = rowsums(qui_max);

    for (size_t i = 0; i < nb_todo; ++i) {
      uword obs = id_todo(i);
      uword rs = rowsums(obs);

      if (rs == Q - 2) {
        qui_max = obs;
        break;
      } else if (rs > rs_max) {
        qui_max = obs;
        rs_max = rs;
      }
    }

    bool first = true;
    for (size_t q = 0; q < Q; ++q) {
      if (mat_done(qui_max, q) == 0) {
        if (first) {
          first = false;
        } else {
          uword id_cluster = dumMat(qui_max, q);
          size_t index = cluster_starts(q) + id_cluster;
          cluster_values(index) = 0;

          const uvec &obs_in_cluster = obs_by_cluster(index);
          for (uword idx : obs_in_cluster) {
            mat_done(idx, q) = 1;
            rowsums(idx) += 1;
          }

          nb_ref(q)++;
        }
      }
    }

    bool changed = true;
    size_t iter_loop = 0;

    while (changed && iter_loop < iter_max) {
      iter_loop++;
      changed = false;

      uvec new_todo_mask(nb_todo, fill::ones);

      for (size_t i = 0; i < nb_todo; ++i) {
        uword obs = id_todo(i);
        uword rs = rowsums(obs);

        if (rs == Q - 1) {
          changed = true;
          new_todo_mask(i) = 0;

          uvec missing = find(mat_done.row(obs) == 0);
          size_t q_missing = missing(0);

          double other_sum = 0;
          for (size_t q = 0; q < Q; ++q) {
            if (q != q_missing) {
              size_t index = cluster_starts(q) + dumMat(obs, q);
              other_sum += cluster_values(index);
            }
          }

          size_t index_missing =
              cluster_starts(q_missing) + dumMat(obs, q_missing);
          cluster_values(index_missing) = sumFE(obs) - other_sum;

          const uvec &obs_in_cluster = obs_by_cluster(index_missing);
          for (uword idx : obs_in_cluster) {
            mat_done(idx, q_missing) = 1;
            rowsums(idx) += 1;
          }
        }
      }

      uvec mask_indices = find(new_todo_mask);
      id_todo = id_todo(mask_indices);
      nb_todo = id_todo.n_elem;
    }

    if (nb_todo == 0)
      break;
  }

  result.Alpha.set_size(Q);
  for (size_t q = 0; q < Q; ++q) {
    if (cluster_sizes(q) > 0) {
      result.Alpha(q) = cluster_values.subvec(
          cluster_starts(q), cluster_starts(q) + cluster_sizes(q) - 1);
    } else {
      result.Alpha(q) = vec(0, arma::fill::zeros);
    }
  }

  result.is_regular = (Q <= 2) || (accu(nb_ref) == Q - 1);
  result.success = (iter < iter_max);

  return result;
}

} // namespace parameters
} // namespace capybara

#endif // CAPYBARA_PARAMETERS_H