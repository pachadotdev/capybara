#ifndef CAPYBARA_GLM_OFFSET_H
#define CAPYBARA_GLM_OFFSET_H

// FE-GLM offset-only routine (for models with only offset and FEs)
feglm_offset_results feglm_offset(vec eta, const vec &y, const vec &offset,
                                  const vec &wt, family_type family,
                                  double center_tol, double dev_tol,
                                  size_t iter_max, size_t iter_center_max,
                                  size_t iter_inner_max, size_t iter_interrupt,
                                  size_t iter_ssr, const indices_info &indices,
                                  glm_workspace &ws,
                                  const bool &use_acceleration) {
  const uword N = y.n_elem;
  reserve_glm_workspace(ws, N, 1);

  const bool has_fe = (indices.fe_sizes.n_elem > 0);

  if (eta.is_empty() || all(eta == 0.0)) {
    smart_initialize_glm(eta, y, family);
    eta += offset;
  }
  link_inv(ws.mu, eta, family);

  double dev = dev_resids(y, ws.mu, 0.0, wt, family);
  std::vector<double> dev_hist;
  dev_hist.reserve(5);

  uvec valid_eta(N, fill::value(1));

  for (size_t it = 0; it < iter_max; ++it) {
    const double dev_old = dev;
    const vec eta_old = eta;

    get_mu(ws.xi, eta, family);
    variance(ws.var_mu, ws.mu, 0.0, family);

    ws.w = wt;
    ws.w %= square(ws.xi);
    ws.w /= ws.var_mu;

    ws.yadj = y;
    ws.yadj -= ws.mu;
    ws.yadj /= ws.xi;
    ws.yadj += eta;
    ws.yadj -= offset;

    const uvec bad = find_nonfinite(ws.w);
    if (!bad.is_empty()) {
      valid_eta.elem(bad).zeros();
      ws.w.elem(bad).zeros();
    }

    if (has_fe) {
      vec yc = ws.yadj;
      center_variables(yc, ws.w, indices, center_tol, iter_center_max,
                       iter_interrupt, iter_ssr, use_acceleration);
      ws.yadj = yc;
      ws.eta_upd = ws.yadj;
      ws.eta_upd -= eta;
    } else {
      ws.eta_upd = offset;
      ws.eta_upd -= eta;
    }

    double damping = adaptive_damping(dev_hist);
    bool ok = false;

    for (size_t inner = 0; inner < iter_inner_max; ++inner) {
      if (inner == 0) {
        ws.eta_candidate = eta;
        ws.eta_candidate += ws.eta_upd;
      } else {
        ws.eta_candidate = eta;
        ws.eta_candidate += damping * ws.eta_upd;
      }

      link_inv(ws.mu, ws.eta_candidate, family);
      if (!valid_eta_mu(ws.eta_candidate, ws.mu, family)) {
        damping *= 0.5;
        continue;
      }

      const double tmp = dev_resids(y, ws.mu, 0.0, wt, family);
      if (!std::isfinite(tmp)) {
        damping *= 0.5;
        continue;
      }

      const double ratio = (tmp - dev_old) / (0.1 + std::fabs(tmp));
      if (ratio <= -dev_tol) {
        eta = ws.eta_candidate;
        dev = tmp;
        ok = true;
        break;
      }
      damping *= 0.5;
      if (damping < 1e-4)
        break;
    }

    if (!ok) {
      eta = eta_old;
      link_inv(ws.mu, eta, family);
      dev = dev_old;
    } else {
      dev_hist.push_back(dev);
      if (dev_hist.size() > 5)
        dev_hist.erase(dev_hist.begin());
    }

    const uvec bad2 = find_nonfinite(eta);
    if (!bad2.is_empty())
      valid_eta.elem(bad2).zeros();

    const double diff = std::fabs(dev - dev_old) / (0.1 + std::fabs(dev));
    if (diff < dev_tol)
      break;

    if ((it % iter_interrupt) == 0 && it > 0)
      check_user_interrupt();
  }

  return feglm_offset_results(std::move(eta), std::move(valid_eta));
}

#endif // CAPYBARA_GLM_OFFSET_H
