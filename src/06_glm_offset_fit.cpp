#include "00_main.h"

[[cpp11::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
                  const list &k_list) {
  // Type conversion

  Col<double> eta = as_Col(eta_r);
  Col<double> y = as_Col(y_r);
  Col<double> offset = as_Col(offset_r);
  Mat<double> Myadj = Mat<double>(y.n_elem, 1, fill::zeros);
  Col<double> wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  std::string fam = tidy_family_(family);
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  int iter, iter_max = as_cpp<int>(control["iter_max"]);
  int iter_center_max = 10000;
  int iter_inner, iter_inner_max = 50;

  // Auxiliary variables (storage)

  Col<double> mu = link_inv_(eta, fam);
  double dev = dev_resids_(y, mu, 0.0, wt, fam);

  const int n = y.n_elem;
  Col<double> mu_eta(n), yadj(n);
  Mat<double> w(n, 1);

  bool dev_crit, val_crit, imp_crit;
  double dev_old, dev_ratio, dev_ratio_inner, rho;
  Col<double> eta_upd(n), eta_old(n);

  // Maximize the log-likelihood

  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta, dev_old = dev;

    // Compute weights and dependent variable

    mu_eta = mu_eta_(eta, fam);
    w = (wt % square(mu_eta)) / variance_(mu, 0.0, fam);
    yadj = (y - mu) / mu_eta + eta - offset;

    // Center variables

    Myadj =
        center_variables_(Myadj + yadj, w, k_list, center_tol, iter_center_max);

    // Compute update step and update eta

    // Step-halving with three checks:
    // 1. finite deviance
    // 2. valid eta and mu
    // 3. improvement as in glm2

    eta_upd = yadj - Myadj + offset - eta;

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      mu = link_inv_(eta, fam);
      dev = dev_resids_(y, mu, 0.0, wt, fam);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev_old));

      dev_crit = is_finite(dev);
      val_crit = (valid_eta_(eta, fam) && valid_mu_(mu, fam));
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
