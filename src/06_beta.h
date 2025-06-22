#ifndef CAPYBARA_BETA_OPTIMIZED_H
#define CAPYBARA_BETA_OPTIMIZED_H

inline void solve_beta_qr(mat &MX, const vec &MNU, const vec &w,
                          beta_results &ws, const uword p, bool use_weights) {
  if (use_weights) {
    // Pre-allocate workspace once
    if (ws.XW.n_rows != MX.n_rows || ws.XW.n_cols != MX.n_cols) {
      ws.XW.set_size(MX.n_rows, MX.n_cols);
    }

    // Element-wise multiplication
    ws.XW = MX.each_col() % sqrt(w);
    qr_econ(ws.Q, ws.decomp, ws.XW);
  } else {
    qr_econ(ws.Q, ws.decomp, MX);
  }

  ws.work = ws.Q.t() * MNU;

  // Rank detection
  vec diag_abs = abs(ws.decomp.diag());
  double max_diag = diag_abs.max();
  double tol = 1e-7 * max_diag;
  uvec indep = find(diag_abs > tol);

  // Reset coefficients
  ws.coefficients.zeros();
  ws.valid_coefficients.zeros();
  ws.valid_coefficients(indep).ones();

  // Solve based on rank
  if (indep.n_elem == p) {
    ws.coefficients = solve(trimatu(ws.decomp), ws.work, solve_opts::fast);
  } else if (!indep.is_empty()) {
    mat Rr = ws.decomp.submat(indep, indep);
    vec Yr = ws.work.elem(indep);
    vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
    ws.coefficients(indep) = br;
  }
}

inline vec solve_beta(mat &MX, const vec &MNU, const vec &w, const uword n,
                      const uword p, beta_results &ws, bool use_weights) {
  ws.coefficients.zeros(p);
  ws.valid_coefficients.ones(p);

  if (ws.work.n_elem != p) {
    ws.work.set_size(p);
  }

  // Use QR for wide matrices
  bool direct_qr = (p > 0.9 * n);

  if (direct_qr) {
    solve_beta_qr(MX, MNU, w, ws, p, use_weights);
    return ws.coefficients;
  }

  // Solve normal equations
  if (use_weights) {
    // if (ws.XW.n_rows != n || ws.XW.n_cols != p) {
    //   ws.XW.set_size(n, p);
    // }

    // Store sqrt(w) to then benefit from BLAS multiplication
    // this was
    // ws.XtX = ws.X.t() * (ws.X.each_col() % sqrt(w));
    vec sqrt_w = sqrt(w);
    ws.XW = MX.each_col() % sqrt_w;
    ws.XtX = ws.XW.t() * ws.XW;

    vec MNU_weighted = MNU % sqrt_w;
    ws.XtY = ws.XW.t() * MNU_weighted;
  } else {
    ws.XtX = MX.t() * MX;
    ws.XtY = MX.t() * MNU;
  }

  // if (ws.decomp.n_rows != p || ws.decomp.n_cols != p) {
  //   ws.decomp.set_size(p, p);
  // }

  // Try Cholesky decomposition
  bool chol_ok = chol(ws.decomp, ws.XtX, "lower");

  if (chol_ok) {
    // Estimate condition number via diagonal elements
    vec d = abs(ws.decomp.diag());
    double mind = d.min();
    double avgd = mean(d);

    // Fail if diagonal element is extremely small relative to average
    if (mind > 1e-12 * avgd) {
      ws.work = solve(trimatl(ws.decomp), ws.XtY, solve_opts::fast);
      ws.coefficients =
          solve(trimatu(ws.decomp.t()), ws.work, solve_opts::fast);
      return ws.coefficients;
    }
  }

  // QR fallback
  solve_beta_qr(MX, MNU, w, ws, p, use_weights);

  return ws.coefficients;
}

#endif // CAPYBARA_BETA_OPTIMIZED_H
