#ifndef CAPYBARA_BETA
#define CAPYBARA_BETA

// Cache for storing precomputed matrices for 2-way models
struct TwoWayBetaCache {
  mat cached_XtX;
  uvec cached_group_i;
  uvec cached_group_j;
  bool is_valid = false;

  void invalidate() { is_valid = false; }

  bool matches_structure(const uvec &group_i, const uvec &group_j) {
    return is_valid && group_i.n_elem == cached_group_i.n_elem &&
           group_j.n_elem == cached_group_j.n_elem &&
           all(group_i == cached_group_i) && all(group_j == cached_group_j);
  }
};

// Global cache instance (one per thread would be better for parallel use)
static TwoWayBetaCache g_twoway_cache;

struct beta_results {
  mat XtX;
  vec XtY;
  mat decomp;
  vec work;
  mat Xt;
  mat Q;
  mat XW;
  vec coefficients;
  uvec valid_coefficients;

  beta_results(size_t n, size_t p)
      : XtX(p, p, fill::none), XtY(p, fill::none), decomp(p, p, fill::none),
        work(p, fill::none), Xt(p, n, fill::none), Q(p, 0, fill::none),
        XW(n, 0, fill::none), coefficients(p, fill::zeros),
        valid_coefficients(p, fill::ones) {}
};

// Solve for regression coefficients using QR decomposition (handles
// collinearity)
inline void get_beta_qr(mat &MX, const vec &MNU, const vec &w, beta_results &ws,
                        const uword p, bool use_weights) {
  if (use_weights) {
    if (ws.XW.n_rows != MX.n_rows || ws.XW.n_cols != MX.n_cols) {
      ws.XW.set_size(MX.n_rows, MX.n_cols);
    }

    ws.XW = MX.each_col() % sqrt(w);
    qr_econ(ws.Q, ws.decomp, ws.XW);
  } else {
    qr_econ(ws.Q, ws.decomp, MX);
  }

  ws.work = ws.Q.t() * MNU;

  const vec diag_abs = abs(ws.decomp.diag());
  const double max_diag = diag_abs.max();
  // Use R's default tolerance for collinearity detection
  const double tol = 1e-7 * max_diag;
  const uvec indep = find(diag_abs > tol);

  ws.coefficients.fill(datum::nan);
  ws.valid_coefficients.zeros();
  ws.valid_coefficients(indep).ones();

  if (indep.n_elem == p) {
    ws.coefficients = solve(trimatu(ws.decomp), ws.work, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = ws.decomp.submat(indep, indep);
    const vec Yr = ws.work.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    ws.coefficients(indep) = br;
    // Keep NaN for invalid coefficients
  }
}

// Main beta solver: uses Cholesky if possible, otherwise falls back to QR
inline vec get_beta(mat &MX, const vec &MNU, const vec &w, const uword n,
                    const uword p, beta_results &ws, bool use_weights) {
  // TIME_FUNCTION;
  ws.coefficients.set_size(p);
  ws.coefficients.fill(datum::nan);
  ws.valid_coefficients.zeros(
      p); // Initialize all as invalid, will be set to 1 for valid ones

  if (ws.work.n_elem != p) {
    ws.work.set_size(p);
  }

  const bool direct_qr = (p > 0.9 * n);

  if (direct_qr) {
    get_beta_qr(MX, MNU, w, ws, p, use_weights);
    return ws.coefficients;
  }

  if (use_weights) {
    const vec sqrt_w = sqrt(w);
    ws.XW = MX.each_col() % sqrt_w;
    ws.XtX = ws.XW.t() * ws.XW;

    ws.XtY = MX.t() * (w % MNU);
  } else {
    ws.XtX = MX.t() * MX;
    ws.XtY = MX.t() * MNU;
  }

  const bool chol_ok = chol(ws.decomp, ws.XtX, "lower");

  if (chol_ok) {
    const vec d = abs(ws.decomp.diag());
    const double mind = d.min();
    const double avgd = mean(d);
    
    if (mind > 1e-12 * avgd) {
      ws.work = solve(trimatl(ws.decomp), ws.XtY, solve_opts::fast);
      ws.coefficients = solve(trimatu(ws.decomp.t()), ws.work, solve_opts::fast);
      ws.valid_coefficients.ones();
      return ws.coefficients;
    }
  }

  get_beta_qr(MX, MNU, w, ws, p, use_weights);

  return ws.coefficients;
}

// Optimized beta computation for 2-way fixed effects models (disabled for now due to type issues)
// This avoids recomputing X'X when the fixed effects structure hasn't changed
inline vec get_beta_twoway_optimized(mat &MX, const vec &MNU, const vec &w,
                                     const list &k_list, beta_results &ws,
                                     bool use_weights) {
  // For now, fall back to standard computation to avoid type issues
  const uword p = MX.n_cols;
  const uword n = MX.n_rows;
  return get_beta(MX, MNU, w, n, p, ws, use_weights);
}

// Enhanced get_beta function that uses optimizations when appropriate
inline vec get_beta_fast(mat &MX, const vec &MNU, const vec &w, uword n,
                         uword p, beta_results &ws, bool use_weights,
                         const list *k_list = nullptr) {

  // Use specialized 2-way optimization if applicable
  if (k_list != nullptr && k_list->size() == 2 && p > 1) {
    return get_beta_twoway_optimized(MX, MNU, w, *k_list, ws, use_weights);
  }

  // Fall back to standard computation
  return get_beta(MX, MNU, w, n, p, ws, use_weights);
}

#endif // CAPYBARA_BETA
