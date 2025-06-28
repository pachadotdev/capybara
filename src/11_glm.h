#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

// Line search for general GLM step, with damping and deviance checks
inline bool line_search_glm(double &dev, double dev_old, double &damping,
                            double dev_tol, size_t inner_max, vec &eta,
                            vec &beta, const vec &eta_old, const vec &beta_old,
                            const vec &eta_upd, const vec &beta_upd, vec &mu,
                            const vec &mu_full, const vec &y, const vec &wt,
                            family_type family, double theta,
                            glm_workspace &ws) {
  const double eps = 0.1;
  const double min_damping = 1e-4;

  for (size_t i = 0; i < inner_max; ++i) {
    if (i > 0 && damping < min_damping) {
      break;
    }

    if (i == 0) {
      ws.eta_candidate = eta_old;
      ws.eta_candidate += eta_upd;
      ws.beta_candidate = beta_old;
      ws.beta_candidate += beta_upd;
      ws.mu_candidate = mu_full;
    } else {
      ws.eta_candidate = eta_old;
      ws.eta_candidate += damping * eta_upd;
      ws.beta_candidate = beta_old;
      ws.beta_candidate += damping * beta_upd;
      link_inv(ws.mu_candidate, ws.eta_candidate, family);
    }

    if (!valid_eta_mu(ws.eta_candidate, ws.mu_candidate, family)) {
      damping *= 0.5;
      continue;
    }

    const double tmp = dev_resids(y, ws.mu_candidate, theta, wt, family);
    if (!std::isfinite(tmp)) {
      damping *= 0.5;
      continue;
    }

    const double ratio = (tmp - dev_old) / (eps + std::fabs(tmp));
    if (ratio <= -dev_tol) {
      dev = tmp;
      eta = ws.eta_candidate;
      beta = ws.beta_candidate;
      mu = ws.mu_candidate;
      return true;
    }

    damping *= 0.5;
  }
  return false;
}

// Main FE-GLM fitting routine for all families
feglm_results feglm(mat &MX, vec &beta, vec &eta, const vec &y, const vec &wt,
                    double theta, family_type family, double center_tol,
                    double dev_tol, size_t iter_max, size_t iter_center_max,
                    size_t iter_inner_max, size_t iter_interrupt,
                    size_t iter_ssr, const indices_info &indices,
                    glm_workspace &ws, const bool &use_acceleration) {
  const uword N = y.n_elem;
  const uword P = MX.n_cols;

  if (family == POISSON) {
    return feglm_poisson(MX, beta, eta, y, wt, center_tol, dev_tol, iter_max,
                         iter_center_max, iter_inner_max, iter_interrupt,
                         iter_ssr, indices, ws, N, P, use_acceleration);
  }

  reserve_glm_workspace(ws, N, P);

  const mat &MX_orig = MX;
  const bool has_fe = (indices.fe_sizes.n_elem > 0);

  if (eta.is_empty()) {
    eta.set_size(N);
    eta.zeros();
  } else if (all(eta == 0.0)) {
    smart_initialize_glm(eta, y, family);
  }
  link_inv(ws.mu, eta, family);

  const vec ymean(N, fill::value(mean(y)));
  double dev = dev_resids(y, ws.mu, theta, wt, family);
  const double null_dev = dev_resids(y, ymean, theta, wt, family);

  std::vector<double> hist;
  hist.reserve(5);
  bool conv = false;
  size_t actual_iters = 0;

  for (size_t it = 0; it < iter_max; ++it) {
    actual_iters = it + 1;
    const double dev_old = dev;
    const vec eta_old = eta;
    const vec beta_old = beta;

    get_mu(ws.xi, eta, family);
    variance(ws.var_mu, ws.mu, theta, family);

    ws.w = wt;
    ws.w %= square(ws.xi);
    ws.w /= ws.var_mu;
    // Combine elementwise operations for ws.nu
    ws.nu = (y - ws.mu) / ws.xi;

    if (it == 0) {
      ws.MNU_accum = ws.nu;
    } else {
      ws.MNU_accum += ws.nu;
      ws.MNU_accum -= ws.nu_old;
    }
    ws.nu_old = ws.nu;
    ws.MNU = ws.MNU_accum;

    if (has_fe) {
      MX = MX_orig;
      vec MNU(ws.MNU.colptr(0), N, false, false);
      center_variables(MX, MNU, ws.w, MX_orig, indices, center_tol,
                       iter_center_max, iter_interrupt, iter_ssr,
                       use_acceleration);
    }

    ws.beta_upd = solve_beta(MX, ws.MNU, ws.w, N, P, ws.beta_ws, true);

    const uvec valid = find(ws.beta_ws.valid_coefficients);
    if (valid.n_elem < P) {
      ws.eta_upd = MX.cols(valid) * ws.beta_upd.elem(valid);
    } else {
      ws.eta_upd = MX * ws.beta_upd;
    }
    ws.eta_upd += ws.nu;
    ws.eta_upd -= ws.MNU;

    double damping = adaptive_damping(hist);

    ws.eta_full = eta_old;
    ws.eta_full += ws.eta_upd;
    ws.beta_full = beta_old;
    ws.beta_full += ws.beta_upd;
    link_inv(ws.mu_full, ws.eta_full, family);

    const bool ok =
        line_search_glm(dev, dev_old, damping, dev_tol, iter_inner_max, eta,
                        beta, eta_old, beta_old, ws.eta_upd, ws.beta_upd, ws.mu,
                        ws.mu_full, y, wt, family, theta, ws);

    if (!ok) {
      eta = eta_old;
      beta = beta_old;
      dev = dev_old;
      link_inv(ws.mu, eta, family);
    } else {
      hist.push_back(dev);
      if (hist.size() > 5)
        hist.erase(hist.begin());
    }

    const double diff = std::fabs(dev - dev_old) / (0.1 + std::fabs(dev));
    if (diff < dev_tol) {
      conv = true;
      break;
    }

    if ((it % iter_interrupt) == 0 && it > 0)
      check_user_interrupt();
  }

  if (!conv)
    stop("FE-GLM failed to converge");

  mat H = crossproduct(MX, ws.w, ws.cross_ws, true);

  for (uword j = 0; j < P; ++j) {
    if (!ws.beta_ws.valid_coefficients(j)) {
      beta(j) = datum::nan;
    }
  }

  return feglm_results(std::move(beta),
                       std::move(ws.beta_ws.valid_coefficients), std::move(eta),
                       wt, std::move(H), dev, null_dev, conv, actual_iters);
}

#endif // CAPYBARA_GLM_H
