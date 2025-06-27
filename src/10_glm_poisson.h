#ifndef CAPYBARA_GLM_POISSON_H
#define CAPYBARA_GLM_POISSON_H

// Compute adaptive damping factor for line search based on deviance history
inline double adaptive_damping(const std::vector<double> &dev_hist) {
  if (dev_hist.size() < 3)
    return 1.0;
  const double d1 =
      dev_hist[dev_hist.size() - 1] - dev_hist[dev_hist.size() - 2];
  const double d2 =
      dev_hist[dev_hist.size() - 2] - dev_hist[dev_hist.size() - 3];
  return (d1 * d2 < 0.0) ? 0.8 : 1.0;
}

// Line search for Poisson GLM step, with damping and deviance checks
inline bool line_search_poisson(double &dev, double dev_old, double &damping,
                                double dev_tol, size_t inner_max, vec &eta,
                                vec &beta, const vec &eta_old,
                                const vec &beta_old, const vec &eta_upd,
                                const vec &beta_upd, vec &mu, vec &exp_eta,
                                const vec &y, const vec &wt,
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
    } else {
      ws.eta_candidate = eta_old;
      ws.eta_candidate += damping * eta_upd;
      ws.beta_candidate = beta_old;
      ws.beta_candidate += damping * beta_upd;
    }

    ws.exp_eta_candidate = exp(ws.eta_candidate);
    ws.mu_candidate = ws.exp_eta_candidate;

    if (!ws.mu_candidate.is_finite()) {
      damping *= 0.5;
      continue;
    }

    const double tmp = dev_resids_poisson(y, ws.mu_candidate, wt,
                                          ws.dev_vec_work, ws.ratio_work);

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
      exp_eta = ws.exp_eta_candidate;
      return true;
    }

    damping *= 0.5;
  }
  return false;
}

// Main Poisson FE-GLM fitting routine (with centering and line search)
feglm_results feglm_poisson(mat &MX, vec &beta, vec &eta, const vec &y,
                            const vec &wt, const double &center_tol,
                            const double &dev_tol, size_t iter_max,
                            size_t iter_center_max, size_t iter_inner_max,
                            size_t iter_interrupt, size_t iter_ssr,
                            const indices_info &indices, glm_workspace &ws,
                            uword N, uword P, const bool &use_acceleration) {
  reserve_glm_workspace(ws, N, P);

  ws.MX_work = MX; // Avoid reallocation
  const mat MX_orig = ws.MX_work;
  const bool has_fe = (indices.fe_sizes.n_elem > 0);

  if (eta.is_empty() || all(eta == 0.0)) {
    eta.set_size(N);
    eta = log(y + 0.1);
  }

  const vec ymean(N, fill::value(mean(y)));
  ws.exp_eta = exp(eta);
  ws.mu = ws.exp_eta;

  if (ws.dev_vec_work.n_elem != N) {
    ws.dev_vec_work.set_size(N);
    ws.ratio_work.set_size(N);
  }

  double dev = dev_resids_poisson(y, ws.mu, wt, ws.dev_vec_work, ws.ratio_work);
  const double null_dev =
      dev_resids_poisson(y, ymean, wt, ws.dev_vec_work, ws.ratio_work);

  bool conv = false;
  double best_dev = dev;
  size_t no_imp = 0;
  size_t actual_iters = 0;

  for (size_t it = 0; it < iter_max; ++it) {
    actual_iters = it + 1;
    const double dev_old = dev;
    const vec eta_old = eta;
    const vec beta_old = beta;

    ws.w = wt;
    ws.w %= ws.exp_eta;

    ws.nu = y;
    ws.nu -= ws.mu;
    ws.nu /= ws.mu;

    if (it == 0) {
      ws.MNU_accum = ws.nu;
    } else {
      ws.MNU_accum += ws.nu;
      ws.MNU_accum -= ws.nu_old;
    }
    ws.nu_old = ws.nu;
    ws.MNU = ws.MNU_accum;

    if (has_fe) {
      ws.MX_work = MX_orig;

      vec MNU(ws.MNU.colptr(0), N, false, false);
      center_variables(ws.MX_work, MNU, ws.w, MX_orig, indices, center_tol,
                       iter_center_max, iter_interrupt, iter_ssr,
                       use_acceleration);
    }

    ws.beta_upd = solve_beta(ws.MX_work, ws.MNU, ws.w, N, P, ws.beta_ws, true);

    const uvec valid = find(ws.beta_ws.valid_coefficients);
    if (valid.n_elem < P) {
      ws.eta_upd = ws.MX_work.cols(valid) * ws.beta_upd.elem(valid);
    } else {
      ws.eta_upd = ws.MX_work * ws.beta_upd;
    }
    ws.eta_upd += ws.nu;
    ws.eta_upd -= ws.MNU;

    double damping = 1.0;
    const bool ok = line_search_poisson(
        dev, dev_old, damping, dev_tol, iter_inner_max, eta, beta, eta_old,
        beta_old, ws.eta_upd, ws.beta_upd, ws.mu, ws.exp_eta, y, wt, ws);

    if (!ok) {
      eta = eta_old;
      beta = beta_old;
      ws.exp_eta = exp(eta);
      ws.mu = ws.exp_eta;
      dev = dev_old;

      if (++no_imp >= 3) {
        conv = true;
        break;
      }
    } else {
      if (dev < best_dev - dev_tol * std::fabs(best_dev)) {
        best_dev = dev;
        no_imp = 0;
      } else {
        ++no_imp;
        if (no_imp >= 3) {
          conv = true;
          break;
        }
      }
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
    stop("Poisson FE-GLM failed to converge");

  crossproduct_results cross_ws(N, P);
  crossproduct(ws.MX_work, ws.w, cross_ws, true);
  mat &H = cross_ws.M;

  for (uword j = 0; j < P; ++j) {
    if (!ws.beta_ws.valid_coefficients(j)) {
      beta(j) = arma::datum::nan;
    }
  }

  return feglm_results(std::move(beta),
                       std::move(ws.beta_ws.valid_coefficients), std::move(eta),
                       wt, std::move(H), dev, null_dev, conv, actual_iters);
}

#endif // CAPYBARA_GLM_POISSON_H
