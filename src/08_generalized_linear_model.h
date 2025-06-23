#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <vector>

using namespace arma;

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

    ws.nu = y;
    ws.nu -= ws.mu;
    ws.nu /= ws.xi;

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
      beta(j) = arma::datum::nan;
    }
  }

  return feglm_results(std::move(beta),
                       std::move(ws.beta_ws.valid_coefficients), std::move(eta),
                       wt, std::move(H), dev, null_dev, conv, actual_iters);
}

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

#endif // CAPYBARA_GLM_H
