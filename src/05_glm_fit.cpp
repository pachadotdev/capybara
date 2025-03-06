#include "00_main.h"

std::string tidy_family_(const std::string &family) {
  // tidy family param
  std::string fam = family;

  // 1. put all in lowercase
  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // 2. remove numbers
  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isdigit), fam.end());

  // 3. remove parentheses and everything inside
  size_t pos = fam.find("(");
  if (pos != std::string::npos) {
    fam.erase(pos, fam.size());
  }

  // 4. replace spaces and dots
  std::replace(fam.begin(), fam.end(), ' ', '_');
  std::replace(fam.begin(), fam.end(), '.', '_');

  // 5. trim
  fam.erase(std::remove_if(fam.begin(), fam.end(), ::isspace), fam.end());

  return fam;
}

Col<double> link_inv_gaussian_(const Col<double> &eta) { return eta; }

Col<double> link_inv_poisson_(const Col<double> &eta) { return exp(eta); }

Col<double> link_inv_logit_(const Col<double> &eta) {
  Col<double> expeta = exp(eta);
  return expeta / (1 + expeta);
}

Col<double> link_inv_gamma_(const Col<double> &eta) { return 1 / eta; }

Col<double> link_inv_invgaussian_(const Col<double> &eta) {
  return 1 / sqrt(eta);
}

Col<double> link_inv_negbin_(const Col<double> &eta) { return exp(eta); }

double dev_resids_gaussian_(const Col<double> &y, const Col<double> &mu,
                            const Col<double> &wt) {
  return accu(wt % square(y - mu));
}

double dev_resids_poisson_(const Col<double> &y, const Col<double> &mu,
                           const Col<double> &wt) {
  Col<double> r = mu % wt;

  uvec p = find(y > 0);
  r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));

  return 2 * accu(r);
}

// Adapted from binomial_dev_resids()
// in R base it can be found in src/library/stats/src/family.c
// unfortunately the functions that work with a SEXP won't work with a Col<>
double dev_resids_logit_(const Col<double> &y, const Col<double> &mu,
                         const Col<double> &wt) {
  Col<double> r(y.n_elem, fill::zeros);
  Col<double> s(y.n_elem, fill::zeros);

  uvec p = find(y == 1);
  uvec q = find(y == 0);
  r(p) = y(p) % log(y(p) / mu(p));
  s(q) = (1 - y(q)) % log((1 - y(q)) / (1 - mu(q)));

  return 2 * accu(wt % (r + s));
}

double dev_resids_gamma_(const Col<double> &y, const Col<double> &mu,
                         const Col<double> &wt) {
  Col<double> r = y / mu;

  uvec p = find(y == 0);
  r.elem(p).fill(1.0);
  r = wt % (log(r) - (y - mu) / mu);

  return -2 * accu(r);
}

double dev_resids_invgaussian_(const Col<double> &y, const Col<double> &mu,
                               const Col<double> &wt) {
  return accu(wt % square(y - mu) / (y % square(mu)));
}

double dev_resids_negbin_(const Col<double> &y, const Col<double> &mu,
                          const double &theta, const Col<double> &wt) {
  Col<double> r = y;

  uvec p = find(y < 1);
  r.elem(p).fill(1.0);
  r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2 * accu(r);
}

Col<double> variance_gaussian_(const Col<double> &mu) {
  return ones<Col<double>>(mu.n_elem);
}

Col<double> link_inv_(const Col<double> &eta, const std::string &fam) {
  Col<double> res(eta.n_elem);

  if (fam == "gaussian") {
    res = link_inv_gaussian_(eta);
  } else if (fam == "poisson") {
    res = link_inv_poisson_(eta);
  } else if (fam == "binomial") {
    res = link_inv_logit_(eta);
  } else if (fam == "gamma") {
    res = link_inv_gamma_(eta);
  } else if (fam == "inverse_gaussian") {
    res = link_inv_invgaussian_(eta);
  } else if (fam == "negative_binomial") {
    res = link_inv_negbin_(eta);
  } else {
    stop("Unknown family");
  }

  return res;
}

double dev_resids_(const Col<double> &y, const Col<double> &mu,
                   const double &theta, const Col<double> &wt,
                   const std::string &fam) {
  double res;

  if (fam == "gaussian") {
    res = dev_resids_gaussian_(y, mu, wt);
  } else if (fam == "poisson") {
    res = dev_resids_poisson_(y, mu, wt);
  } else if (fam == "binomial") {
    res = dev_resids_logit_(y, mu, wt);
  } else if (fam == "gamma") {
    res = dev_resids_gamma_(y, mu, wt);
  } else if (fam == "inverse_gaussian") {
    res = dev_resids_invgaussian_(y, mu, wt);
  } else if (fam == "negative_binomial") {
    res = dev_resids_negbin_(y, mu, theta, wt);
  } else {
    stop("Unknown family");
  }

  return res;
}

bool valid_eta_(const Col<double> &eta, const std::string &fam) {
  bool res;

  if (fam == "gaussian") {
    res = true;
  } else if (fam == "poisson") {
    res = true;
  } else if (fam == "binomial") {
    res = true;
  } else if (fam == "gamma") {
    res = is_finite(eta) && all(eta != 0.0);
  } else if (fam == "inverse_gaussian") {
    res = is_finite(eta) && all(eta > 0.0);
  } else if (fam == "negative_binomial") {
    res = true;
  } else {
    stop("Unknown family");
  }

  return res;
}

bool valid_mu_(const Col<double> &mu, const std::string &fam) {
  bool res;

  if (fam == "gaussian") {
    res = true;
  } else if (fam == "poisson") {
    res = is_finite(mu) && all(mu > 0);
  } else if (fam == "binomial") {
    res = is_finite(mu) && all(mu > 0 && mu < 1);
  } else if (fam == "gamma") {
    res = is_finite(mu) && all(mu > 0.0);
  } else if (fam == "inverse_gaussian") {
    res = true;
  } else if (fam == "negative_binomial") {
    return all(mu > 0.0);
  } else {
    stop("Unknown family");
  }

  return res;
}

// mu_eta = d link_inv / d eta = d mu / d eta

enum FamilyType {
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN,
  UNKNOWN
};

FamilyType get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, FamilyType> family_map = {
      {"gaussian", GAUSSIAN},
      {"poisson", POISSON},
      {"binomial", BINOMIAL},
      {"gamma", GAMMA},
      {"inverse_gaussian", INV_GAUSSIAN},
      {"negative_binomial", NEG_BIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

Col<double> mu_eta_(const Col<double> &eta, const std::string &fam) {
  Col<double> result(eta.n_elem);

  switch (get_family_type(fam)) {
    case GAUSSIAN:
      result.ones();
      break;
    case POISSON:
    case NEG_BIN:
      result = arma::exp(eta);
      break;
    case BINOMIAL: {
      Col<double> exp_eta = arma::exp(eta);
      result = exp_eta / arma::square(1 + exp_eta);
      break;
    }
    case GAMMA:
      result = -1 / arma::square(eta);
      break;
    case INV_GAUSSIAN:
      result = -1 / (2 * arma::pow(eta, 1.5));
      break;
    default:
      stop("Unknown family");
  }

  return result;
}

Col<double> variance_(const Col<double> &mu, const double &theta,
                      const std::string &fam) {
  switch (get_family_type(fam)) {
    case GAUSSIAN:
      return ones<Col<double>>(mu.n_elem);
    case POISSON:
      return mu;
    case BINOMIAL:
      return mu % (1 - mu);
    case GAMMA:
      return square(mu);
    case INV_GAUSSIAN:
      return pow(mu, 3.0);
    case NEG_BIN:
      return mu + square(mu) / theta;
    default:
      stop("Unknown family");
  }
}

[[cpp11::register]] list feglm_fit_(const doubles &beta_r, const doubles &eta_r,
                                    const doubles &y_r,
                                    const doubles_matrix<> &x_r,
                                    const doubles &wt_r, const double &theta,
                                    const std::string &family,
                                    const list &control, const list &k_list) {
  // Type conversion

  Col<double> beta = as_Col(beta_r);
  Col<double> eta = as_Col(eta_r);
  Col<double> y = as_Col(y_r);
  Mat<double> MX = as_Mat(x_r);
  Col<double> MNU = Col<double>(y.n_elem, fill::zeros);
  Col<double> wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  std::string fam = tidy_family_(family);
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  int iter, iter_max = as_cpp<int>(control["iter_max"]);
  int iter_center_max = 10000;
  bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  int iter_inner, iter_inner_max = 50;
  const int k = beta.n_elem;

  // Auxiliary variables (storage)

  Col<double> mu = link_inv_(eta, fam);
  Col<double> ymean = mean(y) * Col<double>(y.n_elem, fill::ones);
  double dev = dev_resids_(y, mu, theta, wt, fam);
  double null_dev = dev_resids_(y, ymean, theta, wt, fam);

  const int n = y.n_elem;
  const int p = MX.n_cols;
  Col<double> mu_eta(n), nu(n), w(n);
  Mat<double> H(p, p);
  bool conv = false;

  bool dev_crit, val_crit, imp_crit;
  double dev_old, dev_ratio, dev_ratio_inner, rho;
  Col<double> eta_upd(n), beta_upd(k), eta_old(n), beta_old(k);

  // Maximize the log-likelihood

  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta, beta_old = std::move(beta), dev_old = dev;

    // Compute weights and dependent variable

    mu_eta = mu_eta_(eta, fam);
    w = (wt % square(mu_eta)) / variance_(mu, theta, fam);
    nu = (y - mu) / mu_eta;

    // Center variables

    MNU = center_variables_(MNU + nu, w, k_list, center_tol, iter_center_max);
    MX = center_variables_(MX, w, k_list, center_tol, iter_center_max);

    // Compute update step and update eta

    // Step-halving with three checks:
    // 1. finite deviance
    // 2. valid eta and mu
    // 3. improvement as in glm2

    beta_upd = solve_beta_(MX, MNU, w);
    eta_upd = MX * beta_upd;
    eta_upd += nu;
    eta_upd -= MNU;

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      beta = beta_old + (rho * beta_upd);
      mu = link_inv_(eta, fam);
      dev = dev_resids_(y, mu, theta, wt, fam);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev));

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

    // If step halving does not improve the deviance

    if (imp_crit == false) {
      eta = eta_old;
      beta = beta_old;
      dev = dev_old;
      mu = link_inv_(eta, fam);
    }

    // Check convergence

    dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));

    if (dev_ratio < dev_tol) {
      conv = true;
      break;
    }

    // Update starting guesses for acceleration

    MNU -= nu;
  }

  // Information if convergence failed

  if (conv == false) {
    stop("Algorithm did not converge.");
  }

  // Update weights and dependent variable

  mu_eta = mu_eta_(eta, fam);
  w = (wt % square(mu_eta)) / variance_(mu, theta, fam);

  // Recompute Hessian

  H = crossprod_(MX, w);

  // Generate result list

  writable::list out(8);

  out[0] = as_doubles(beta);
  out[1] = as_doubles(eta);
  out[2] = as_doubles(wt);
  out[3] = as_doubles_matrix(H);
  out[4] = writable::doubles({dev});
  out[5] = writable::doubles({null_dev});
  out[6] = writable::logicals({conv});
  out[7] = writable::integers({iter + 1});

  out.attr("names") =
      writable::strings({"coefficients", "eta", "weights", "hessian",
                         "deviance", "null_deviance", "conv", "iter"});

  if (keep_mx == true) {
    out.push_back({"MX"_nm = as_doubles_matrix(center_variables_(
                       as_Mat(x_r), w, k_list, center_tol, iter_center_max))});
  }

  return out;
}
