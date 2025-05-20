#include "00_main.h"

[[cpp11::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
                  const list &k_list) {
  // Type conversion

  vec eta = as_Col(eta_r);
  vec y = as_Col(y_r);
  const vec offset = as_Col(offset_r);
  vec Myadj = vec(y.n_elem, fill::zeros);
  const vec wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const std::string fam = tidy_family_(family);
  const FamilyType family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const size_t iter_max = as_cpp<int>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               iter_ssr = as_cpp<size_t>(control["iter_ssr"]), n = y.n_elem;

  // Auxiliary variables (storage)

  size_t iter, iter_inner;
  vec mu = link_inv_(eta, family_type), mu_eta(n, fill::none),
      yadj(n, fill::none), w(n, fill::none), eta_upd(n, fill::none),
      eta_old(n, fill::none);
  double dev = dev_resids_(y, mu, 0.0, wt, family_type), dev_old, dev_ratio,
         dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  // Maximize the log-likelihood

  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta, dev_old = dev;

    // Compute weights and dependent variable

    mu_eta = mu_eta_(eta, family_type);
    w = (wt % square(mu_eta)) / variance_(mu, 0.0, family_type);
    yadj = (y - mu) / mu_eta + eta - offset;

    // Center variables

    Myadj += yadj;
    center_variables_(Myadj, w, k_list, center_tol, iter_center_max,
                      iter_interrupt, iter_ssr);

    // Compute update step and update eta

    // Step-halving with three checks:
    // 1. finite deviance
    // 2. valid eta and mu
    // 3. improvement as in glm2

    eta_upd = yadj - Myadj + offset - eta;

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, 0.0, wt, family_type);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev_old));

      dev_crit = is_finite(dev);
      val_crit = (valid_eta_(eta, family_type) && valid_mu_(mu, family_type));
      imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit == true && val_crit == true && imp_crit == true) {
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed (deviance and invalid eta or mu)

    if (dev_crit == false || val_crit == false) {
      stop("Inner loop failed; cannot correct step size.");
    }

    // Check convergence

    dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));

    if (dev_ratio < dev_tol) {
      break;
    }

    // Update starting guesses for acceleration

    Myadj = Myadj - yadj;
  }

  return as_doubles(eta);
}
