#ifndef CAPYBARA_BETA_OPTIMIZED_H
#define CAPYBARA_BETA_OPTIMIZED_H

inline void solve_beta_qr(mat &MX, const vec &MNU, const vec &w,
                          beta_results &ws, const uword p, bool use_weights) {
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
  const double tol = 1e-7 * max_diag;
  const uvec indep = find(diag_abs > tol);

  ws.coefficients.zeros();
  ws.valid_coefficients.zeros();
  ws.valid_coefficients(indep).ones();

  if (indep.n_elem == p) {
    ws.coefficients = solve(trimatu(ws.decomp), ws.work, solve_opts::fast);
  } else if (!indep.is_empty()) {
    const mat Rr = ws.decomp.submat(indep, indep);
    const vec Yr = ws.work.elem(indep);
    const vec br = solve(trimatu(Rr), Yr, solve_opts::fast);
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

  const bool direct_qr = (p > 0.9 * n);

  if (direct_qr) {
    solve_beta_qr(MX, MNU, w, ws, p, use_weights);
    return ws.coefficients;
  }

  if (use_weights) {
    const vec sqrt_w = sqrt(w);
    ws.XW = MX.each_col() % sqrt_w;
    ws.XtX = ws.XW.t() * ws.XW;

    const vec MNU_weighted = MNU % sqrt_w;
    ws.XtY = ws.XW.t() * MNU_weighted;
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
      ws.coefficients =
          solve(trimatu(ws.decomp.t()), ws.work, solve_opts::fast);
      return ws.coefficients;
    }
  }

  solve_beta_qr(MX, MNU, w, ws, p, use_weights);

  return ws.coefficients;
}

#endif // CAPYBARA_BETA_OPTIMIZED_H
