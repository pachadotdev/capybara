// Linear model with fixed effects Y = alpha + X beta
#ifndef CAPYBARA_LM_H
#define CAPYBARA_LM_H

namespace capybara {

struct InferenceLM {
  mat coef_table; // Coefficient table: [estimate, std.error, z, p-value]
  vec fitted_values;
  vec residuals;
  vec weights;
  mat hessian;
  mat vcov;         // Covariance matrix (inverse Hessian or sandwich)
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  field<vec> fixed_effects;
  bool has_fe = true;
  uvec iterations;

  mat TX; // Centered design matrix T(X)
  double r_squared;
  double adj_r_squared;
  bool has_tx = false;

  InferenceLM(uword n, uword p)
      : coef_table(p, 4, fill::zeros), fitted_values(n, fill::zeros),
        residuals(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), vcov(p, p, fill::zeros),
        coef_status(p, fill::ones), success(false), has_fe(false),
        r_squared(0.0), adj_r_squared(0.0), has_tx(false) {}
};

struct FelmWorkspace {
  // Contiguous memory block holding [y_demeaned | X_centered] column-major.
  // A single allocation of N*(P+1) doubles, with non-owning Armadillo
  // views pointing into it.  This lets center_variables() operate on
  // all P+1 columns in a single call — no temporary yX matrix, no memcpy.
  std::vector<double> center_buf; // owns the memory: N * (P+1) doubles
  vec y_demeaned;                 // non-owning view of center_buf[0..N-1]
  mat X_centered;                 // non-owning view of center_buf[N..N*(P+1)-1]
  mat yX_view; // non-owning view of entire buffer as N x (P+1)

  vec x_beta;
  vec pi;
  mat X_original; // Uncentered X for fitted values computation
  vec y_original;

  // Persistent FE structures (survive across IRLS iterations)
  FlatFEMap fe_map;           // FE group structure (invariant)
  CenterWarmStart warm_start; // Warm-start for centering across IRLS
  bool fe_map_initialized;    // Has fe_map.build() been called?

  // Reusable InferenceLM for IRLS inner calls (avoids re-allocating N-length
  // vectors + P×P matrices every iteration)
  std::unique_ptr<InferenceLM> glm_result;

  uword cached_N, cached_P;

  FelmWorkspace() : fe_map_initialized(false), cached_N(0), cached_P(0) {}

  void ensure_size(uword N, uword P) {
    if (N != cached_N || P != cached_P) {
      // Allocate one contiguous block: [y (N doubles) | X (N*P doubles)]
      center_buf.resize(N * (P + 1));
      double *buf = center_buf.data();

      // Non-owning Armadillo views into the buffer
      // copy_aux_mem=false, strict=true -> Armadillo won't reallocate
      y_demeaned = vec(buf, N, false, true);
      X_centered = mat(buf + N, N, P, false, true);
      yX_view = mat(buf, N, P + 1, false, true);

      cached_N = N;
      cached_P = P;
    }
    x_beta.set_size(N);
    pi.set_size(N);
    y_original.set_size(N);
    X_original.set_size(N, P);
  }

  // Build FE map structure once; only update weights on subsequent calls
  void ensure_fe_map(const FlatFEMap &source_map, const vec &w) {
    if (!fe_map_initialized) {
      fe_map = source_map; // copy structure
      fe_map_initialized = true;
    }
    fe_map.update_weights(w);
  }
};

// Fixed effects accumulation using FlatFEMap (no more FEAccumInfo)
inline void accumulate_fixed_effects(vec &fitted_values,
                                     const field<vec> &fixed_effects,
                                     const FlatFEMap &map) {
  const uword K = map.K;
  const uword N = map.n_obs;

  for (uword k = 0; k < K; ++k) {
    const uword *gk = map.fe_map[k].data();
    const double *fek = fixed_effects(k).memptr();
    double *fv = fitted_values.memptr();
    for (uword i = 0; i < N; ++i) {
      fv[i] += fek[gk[i]];
    }
  }
}

inline void fitted_values_(FelmWorkspace *ws, InferenceLM &result,
                           const CollinearityResult &collin_result,
                           const FlatFEMap &fe_map,
                           const CapybaraParameters &params,
                           bool skip_alpha = false) {
  const uword N = ws->y_original.n_elem;
  const vec &coef = result.coef_table.col(0);
  const bool has_fixed_effects = fe_map.K > 0;

  if (has_fixed_effects && skip_alpha) {
    // Fast path for IRLS: recover fitted values from centered quantities
    // directly, avoiding the expensive iterative get_alpha() call.
    //
    // After centering [y|X] we have:
    //   y_demeaned = y - M_fe(y)      (FE means removed from y)
    //   X_centered = X - M_fe(X)      (FE means removed from X)
    //
    // OLS on centered data gives beta, then:
    //   residual = y_demeaned - X_centered * beta
    //   fitted   = y - residual = y - y_demeaned + X_centered * beta
    //
    // This is exact (same as X*beta + FE_means(y - X*beta)) because
    // centering is linear: M_fe(y - X*beta) = M_fe(y) - M_fe(X)*beta.
    // Compute fitted = y - y_demeaned + X_centered * beta in-place
    // to avoid temporary vector allocation
    double *fv = result.fitted_values.memptr();
    const double *yo = ws->y_original.memptr();
    const double *yd = ws->y_demeaned.memptr();
    const uword Pc = ws->X_centered.n_cols;

    if (collin_result.has_collinearity &&
        !collin_result.non_collinear_cols.is_empty()) {
      const uvec &cols = collin_result.non_collinear_cols;
      const vec x_centered_beta = ws->X_centered.cols(cols) * coef.elem(cols);
      const double *xb = x_centered_beta.memptr();
      for (uword i = 0; i < N; ++i) {
        fv[i] = yo[i] - yd[i] + xb[i];
      }
    } else if (Pc > 0) {
      const vec x_centered_beta = ws->X_centered * coef;
      const double *xb = x_centered_beta.memptr();
      for (uword i = 0; i < N; ++i) {
        fv[i] = yo[i] - yd[i] + xb[i];
      }
    } else {
      for (uword i = 0; i < N; ++i) {
        fv[i] = yo[i] - yd[i];
      }
    }
    result.has_fe = true;
    return;
  }

  // Full path: need X_original * beta
  if (collin_result.has_collinearity &&
      !collin_result.non_collinear_cols.is_empty()) {
    const uvec &cols = collin_result.non_collinear_cols;
    ws->x_beta = ws->X_original.cols(cols) * coef.elem(cols);
  } else if (!collin_result.has_collinearity && ws->X_original.n_cols > 0) {
    ws->x_beta = ws->X_original * coef;
  } else {
    ws->x_beta.zeros(N);
  }

  if (has_fixed_effects) {
    // Full path: recover FE coefficients via get_alpha
    ws->pi = ws->y_original - ws->x_beta;

    const vec *coef_weights = nullptr;
    coef_weights = &result.weights;

    result.fixed_effects = get_alpha(ws->pi, fe_map, params.alpha_tol,
                                     params.iter_alpha_max, coef_weights);
    result.has_fe = true;

    result.fitted_values = ws->x_beta;

    accumulate_fixed_effects(result.fitted_values, result.fixed_effects,
                             fe_map);
  } else {
    result.fitted_values = ws->x_beta;
  }
}

inline void r_squared_(InferenceLM &result, const vec &y, const vec &residuals,
                       const vec &w, uword n_coef, const FlatFEMap &fe_map) {
  const uword N = y.n_elem;
  const double w_sum = accu(w);
  const double y_mean = dot(w, y) / w_sum;

  const vec y_centered = y - y_mean;

  const double tss = dot(w % y_centered, y_centered);
  const double rss = dot(w % residuals, residuals);

  result.r_squared = (tss > 1e-12) ? (1.0 - rss / tss) : 0.0;

  // Count degrees of freedom for FE
  uword k = n_coef;
  const uword K = fe_map.K;
  for (uword j = 0; j < K; ++j) {
    k += fe_map.n_groups[j] - 1;
  }

  const double denom = std::max(1.0, static_cast<double>(N - k));
  result.adj_r_squared =
      1.0 - (1.0 - result.r_squared) * (static_cast<double>(N - 1)) / denom;
}

// Expand reduced vcov matrix to full size accounting for collinearity
inline void expand_vcov(mat &vcov_full, const mat &vcov_reduced,
                        const uvec &non_collinear_cols, uword n_coef) {
  vcov_full.set_size(n_coef, n_coef);
  vcov_full.fill(datum::nan);

  const uword n_idx = non_collinear_cols.n_elem;
  if (n_idx > 0 && n_idx == vcov_reduced.n_rows &&
      n_idx == vcov_reduced.n_cols) {
    vcov_full.submat(non_collinear_cols, non_collinear_cols) = vcov_reduced;
  }
}

InferenceLM felm_fit(const mat &X, const vec &y, const vec &w,
                     const FlatFEMap &fe_map, const CapybaraParameters &params,
                     FelmWorkspace *workspace = nullptr,
                     const field<uvec> *cluster_groups = nullptr,
                     bool run_from_glm = false,
                     double adaptive_center_tol = 0.0,
                     const field<uvec> *entity1_groups = nullptr,
                     const field<uvec> *entity2_groups = nullptr) {
  const uword N = y.n_elem;
  const uword P_input = X.n_cols;
  const bool has_fixed_effects = fe_map.K > 0;

  // Determine final P (with or without intercept)
  const uword P = (!has_fixed_effects && !run_from_glm) ? P_input + 1 : P_input;

  FelmWorkspace local_workspace;
  FelmWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N, P);

  // Reuse or create result object.
  // In the IRLS path (run_from_glm=true) the workspace keeps a persistent
  // InferenceLM so we don't re-allocate N-length vectors + P×P matrices
  // every iteration.
  InferenceLM *res_ptr;
  if (run_from_glm) {
    if (!ws->glm_result || ws->glm_result->fitted_values.n_elem != N ||
        ws->glm_result->coef_table.n_rows != P) {
      ws->glm_result = std::make_unique<InferenceLM>(N, P);
    }
    res_ptr = ws->glm_result.get();
    // Reset only the fields that matter for each iteration
    res_ptr->success = false;
    res_ptr->has_fe = has_fixed_effects;
  } else {
    // Standalone felm: allocate fresh (returned to caller)
    res_ptr = nullptr; // will use local below
  }

  // For standalone felm we need a local result that we return
  InferenceLM local_result(run_from_glm ? 0 : N, run_from_glm ? 0 : P);
  InferenceLM &result = run_from_glm ? *res_ptr : local_result;
  if (!run_from_glm) {
    result.weights = w;
    result.has_fe = has_fixed_effects;
  }

  ws->y_original = y;

  // When called standalone (not from GLM), guard against non-finite
  // values in y, w, or X that would produce NaN in X'WX and crash
  // the LAPACK Cholesky solver (DLASCL error -4).
  // When called from GLM, the guard is in feglm_fit's IRLS loop.
  // We create local working copies so the rest of the function uses clean data.
  vec w_work = w;
  if (!run_from_glm) {
    // Build a mask of observations where y, w, or any X column is non-finite
    uvec bad = find_nonfinite(y);
    {
      uvec bad_w = find_nonfinite(w);
      if (bad_w.n_elem > 0) {
        bad = (bad.n_elem > 0) ? unique(join_cols(bad, bad_w)) : bad_w;
      }
    }
    for (uword j = 0; j < P_input && bad.n_elem < N; ++j) {
      uvec bad_j = find_nonfinite(X.col(j));
      if (bad_j.n_elem > 0)
        bad = unique(join_cols(bad, bad_j));
    }
    if (bad.n_elem > 0 && bad.n_elem < N) {
      // Zero the weight for non-finite rows so they contribute nothing
      // to the cross-product X'WX and X'Wy
      w_work.elem(bad).zeros();
      // Replace non-finite y values with 0 to prevent NaN propagation
      vec y_clean = y;
      y_clean.elem(bad).zeros();
      ws->y_original = y_clean;
    }
    result.weights = w_work;
  }

  const bool use_weights = any(w_work != 1.0);

#ifdef CAPYBARA_DEBUG
  auto tcenter0 = std::chrono::high_resolution_clock::now();
#endif

  if (has_fixed_effects) {
    // Build FE map structure once, only update weights each call
    ws->ensure_fe_map(fe_map, w_work);

    // Use adaptive tolerance if provided, otherwise use params.center_tol
    const double effective_tol =
        (adaptive_center_tol > 0.0) ? adaptive_center_tol : params.center_tol;

    // X_original is only needed for the full get_alpha() path (standalone
    // felm).  In the IRLS path (run_from_glm=true) we use the skip_alpha
    // fast path which computes fitted values from centered quantities
    // directly, so we skip this N*P copy.
    if (!run_from_glm) {
      ws->X_original = X;
    }

    // Copy y and X into the contiguous center_buf via the non-owning views.
    // y_demeaned and X_centered point into the same buffer, so
    // center_variables(yX_view, ...) demeans all P+1 columns in-place
    // with zero extra allocation.
    std::memcpy(ws->y_demeaned.memptr(), ws->y_original.memptr(),
                N * sizeof(double));
    if (P > 0) {
      std::memcpy(ws->X_centered.memptr(), X.memptr(), N * P * sizeof(double));
    }

    center_variables(ws->yX_view, w_work, ws->fe_map, effective_tol,
                     params.iter_center_max, params.grand_acc_period,
                     &ws->warm_start);
  } else {
    // Copy X to workspace buffers
    if (!run_from_glm) {
      // Standalone felm: add intercept column
      ws->X_original.col(0).ones();
      if (P_input > 0) {
        ws->X_original.cols(1, P - 1) = X;
      }
    } else {
      ws->X_original = X;
    }
    // No FE: y_demeaned = y (unmodified), X_centered = X_original
    std::memcpy(ws->y_demeaned.memptr(), ws->y_original.memptr(),
                N * sizeof(double));
    std::memcpy(ws->X_centered.memptr(), ws->X_original.memptr(),
                N * P * sizeof(double));
  }

#ifdef CAPYBARA_DEBUG
  auto tcenter1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> center_duration = tcenter1 - tcenter0;
  std::ostringstream center_msg;
  center_msg << "Centering time: " << center_duration.count() << " seconds.\n";
  cpp4r::message(center_msg.str());
#endif

  if (params.keep_tx && P > 0) {
    result.TX = ws->X_centered;
    result.has_tx = true;
  }

  const uword P_final = P;

// Solve normal equations using rank-revealing Cholesky
#ifdef CAPYBARA_DEBUG
  auto tsolve0 = std::chrono::high_resolution_clock::now();
#endif

  mat XtX;
  vec XtY_vec;

  if (use_weights) {
    XtX = crossprod(ws->X_centered, w_work);
    XtY_vec = ws->X_centered.t() * (w_work % ws->y_demeaned);
  } else {
    XtX = crossprod(ws->X_centered);
    XtY_vec = ws->X_centered.t() * ws->y_demeaned;
  }

  mat R_rank;
  uvec excl;
  uword rank;

  if (!chol_rank(R_rank, excl, rank, XtX, "upper", params.collin_tol)) {
    throw std::runtime_error("chol_rank failed in felm_fit");
  }

  // Populate collinearity result from excl vector
  CollinearityResult collin_result(P_final);
  collin_result.has_collinearity = any(excl);
  collin_result.non_collinear_cols = find(excl == 0);
  collin_result.collinear_cols = find(excl != 0);
  collin_result.coef_status = 1 - excl;

  // Solve reduced system for non-excluded columns
  const uword n_coef = P_final;
  vec beta_solved(n_coef, fill::value(datum::nan));
  mat hessian_reduced;

  if (rank > 0) {
    const uvec keep_idx = collin_result.non_collinear_cols;
    const mat R_sub = R_rank.submat(keep_idx, keep_idx);
    const vec XtY_sub = XtY_vec.elem(keep_idx);

    // Solve triangular systems: R'y = XtY, then Rb = y
    const vec y_sub = solve(trimatl(R_sub.t()), XtY_sub);
    const vec beta_sub = solve(trimatu(R_sub), y_sub);

    // Scatter into full beta (excluded entries remain NaN)
    beta_solved.elem(keep_idx) = beta_sub;

    // Hessian from Cholesky factor: X'X = R'R
    hessian_reduced = R_sub.t() * R_sub;
  }

  if (result.coef_table.n_rows != n_coef) {
    result.coef_table.set_size(n_coef, 4);
  }
  result.coef_status = std::move(collin_result.coef_status);

  result.coef_table.col(0) = beta_solved;
  if (collin_result.has_collinearity) {
    expand_vcov(result.hessian, hessian_reduced,
                collin_result.non_collinear_cols, n_coef);
  } else {
    result.hessian = hessian_reduced;
  }
  result.success = true;

  fitted_values_(ws, result, collin_result, fe_map, params, run_from_glm);

  if (run_from_glm) {
    // Only return coefficients
    return result;
  }

#ifdef CAPYBARA_DEBUG
  auto tsolve1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_duration = tsolve1 - tsolve0;
  std::ostringstream solve_msg;
  solve_msg << "Solving time: " << solve_duration.count() << " seconds.\n";
  cpp4r::message(solve_msg.str());
#endif

  result.residuals = ws->y_original - result.fitted_values;

  r_squared_(result, ws->y_original, result.residuals, w_work, n_coef, fe_map);

  mat vcov_reduced;
  const double rss = dot(w_work % result.residuals, result.residuals);

  if (cluster_groups != nullptr && cluster_groups->n_elem > 0) {
    if (params.vcov_type == "m-estimator-dyadic" && entity1_groups != nullptr &&
        entity2_groups != nullptr) {
      // For dyadic clustering, compute observation-level scores
      // Score_i = X_i * (y_i - fitted_i) for OLS
      const vec resid = ws->y_original - result.fitted_values;
      mat scores(ws->X_centered.n_rows, ws->X_centered.n_cols);
      for (uword i = 0; i < ws->X_centered.n_rows; ++i) {
        scores.row(i) = resid(i) * ws->X_centered.row(i);
      }
      vcov_reduced = sandwich_vcov_mestimator_dyadic_(
          result.hessian, scores, *entity1_groups, *entity2_groups);
    } else if (params.vcov_type == "m-estimator") {
      // For standard M-estimator clustering
      const vec resid = ws->y_original - result.fitted_values;
      mat scores(ws->X_centered.n_rows, ws->X_centered.n_cols);
      for (uword i = 0; i < ws->X_centered.n_rows; ++i) {
        scores.row(i) = resid(i) * ws->X_centered.row(i);
      }
      vcov_reduced =
          sandwich_vcov_mestimator_(result.hessian, scores, *cluster_groups);
    } else {
      vcov_reduced =
          sandwich_vcov_(ws->X_centered, ws->y_original, result.fitted_values,
                         result.hessian, *cluster_groups);
    }
  } else {
    mat H_inv;
    bool success = inv(H_inv, result.hessian);
    if (!success) {
      H_inv.set_size(result.hessian.n_rows, result.hessian.n_cols);
      H_inv.fill(datum::inf);
    }

    const double sigma2 = rss / std::max(1.0, static_cast<double>(N - P));
    vcov_reduced = sigma2 * H_inv;
  }

  // Expand vcov if there's collinearity
  if (collin_result.has_collinearity &&
      collin_result.non_collinear_cols.n_elem > 0) {
    expand_vcov(result.vcov, vcov_reduced, collin_result.non_collinear_cols,
                n_coef);
  } else {
    result.vcov = std::move(vcov_reduced);
  }

  // Coefficients table
  const vec se = sqrt(diagvec(result.vcov));
  const vec &coefficients = result.coef_table.col(0);
  const vec z_values = coefficients / se;

  result.coef_table.col(1) = se;
  result.coef_table.col(2) = z_values;
  result.coef_table.col(3) = 2.0 * normcdf(-abs(z_values));

  // Mark collinear coefficients as NaN
  // if (collin_result.has_collinearity) {
  //   const uvec collinear_idx = find(result.coef_status == 0);
  //   for (uword i = 0; i < collinear_idx.n_elem; ++i) {
  //     result.coef_table.row(collinear_idx(i)).fill(datum::nan);
  //   }
  // }

  return result;
}

} // namespace capybara

#endif // CAPYBARA_LM_H
