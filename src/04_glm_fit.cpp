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

vec link_inv_gaussian_(const vec &eta) { return eta; }

vec link_inv_poisson_(const vec &eta) { return exp(eta); }

vec link_inv_logit_(const vec &eta) {
  return 1.0 / (1.0 + exp(-eta));
}

vec link_inv_gamma_(const vec &eta) { return 1 / eta; }

vec link_inv_invgaussian_(const vec &eta) { return 1 / sqrt(eta); }

vec link_inv_negbin_(const vec &eta) { return exp(eta); }

double dev_resids_gaussian_(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu));
}

double dev_resids_poisson_(const vec &y, const vec &mu, const vec &wt) {
  vec r = mu % wt;

  uvec p = find(y > 0);
  r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));

  return 2 * accu(r);
}

// Adapted from binomial_dev_resids()
// in R base it can be found in src/library/stats/src/family.c
// unfortunately the functions that work with a SEXP won't work with a Col<>
double dev_resids_logit_(const vec &y, const vec &mu, const vec &wt) {
  vec r(y.n_elem, fill::zeros);
  vec s(y.n_elem, fill::zeros);

  uvec p = find(y == 1);
  uvec q = find(y == 0);
  r(p) = y(p) % log(y(p) / mu(p));
  s(q) = (1 - y(q)) % log((1 - y(q)) / (1 - mu(q)));

  return 2 * dot(wt, r + s);
}

double dev_resids_gamma_(const vec &y, const vec &mu, const vec &wt) {
  vec r = y / mu;

  uvec p = find(y == 0);
  r.elem(p).fill(1.0);
  r = wt % (log(r) - (y - mu) / mu);

  return -2 * accu(r);
}

double dev_resids_invgaussian_(const vec &y, const vec &mu, const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

double dev_resids_negbin_(const vec &y, const vec &mu, const double &theta,
                          const vec &wt) {
  vec r = y;

  uvec p = find(y < 1);
  r.elem(p).fill(1.0);
  r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2 * accu(r);
}

vec variance_gaussian_(const vec &mu) { return ones<vec>(mu.n_elem); }

vec link_inv_(const vec &eta, const FamilyType family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
    case GAUSSIAN:
      result = link_inv_gaussian_(eta);
      break;
    case POISSON:
      result = link_inv_poisson_(eta);
      break;
    case BINOMIAL:
      result = link_inv_logit_(eta);
      break;
    case GAMMA:
      result = link_inv_gamma_(eta);
      break;
    case INV_GAUSSIAN:
      result = link_inv_invgaussian_(eta);
      break;
    case NEG_BIN:
      result = link_inv_negbin_(eta);
      break;
    default:
      stop("Unknown family");
  }

  return result;
}

double dev_resids_(const vec &y, const vec &mu, const double &theta,
                   const vec &wt, const FamilyType family_type) {
  switch (family_type) {
    case GAUSSIAN:
      return dev_resids_gaussian_(y, mu, wt);
    case POISSON:
      return dev_resids_poisson_(y, mu, wt);
    case BINOMIAL:
      return dev_resids_logit_(y, mu, wt);
    case GAMMA:
      return dev_resids_gamma_(y, mu, wt);
    case INV_GAUSSIAN:
      return dev_resids_invgaussian_(y, mu, wt);
    case NEG_BIN:
      return dev_resids_negbin_(y, mu, theta, wt);
    default:
      stop("Unknown family");
  }
}

bool valid_eta_(const vec &eta, const FamilyType family_type) {
  switch (family_type) {
    case GAUSSIAN:
    case POISSON:
    case BINOMIAL:
    case NEG_BIN:
      return true;
    case GAMMA:
      return is_finite(eta) && all(eta != 0.0);
    case INV_GAUSSIAN:
      return is_finite(eta) && all(eta > 0.0);
    default:
      stop("Unknown family");
  }
}

bool valid_mu_(const vec &mu, const FamilyType family_type) {
  switch (family_type) {
    case GAUSSIAN:
      return true;
    case POISSON:
    case NEG_BIN:
      return is_finite(mu) && all(mu > 0);
    case BINOMIAL:
      return is_finite(mu) && all(mu > 0 && mu < 1);
    case GAMMA:
      return is_finite(mu) && all(mu > 0.0);
    case INV_GAUSSIAN:
      return true;
    default:
      stop("Unknown family");
  }
}

// mu_eta = d link_inv / d eta = d mu / d eta

vec mu_eta_(const vec &eta, const FamilyType family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case GAUSSIAN:
    result.ones();
    break;
  case POISSON:
  case NEG_BIN:
    result = arma::exp(eta);
    break;
  case BINOMIAL: {
    vec exp_eta = arma::exp(eta);
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

vec variance_(const vec &mu, const double &theta,
              const FamilyType family_type) {
  switch (family_type) {
  case GAUSSIAN:
    return ones<vec>(mu.n_elem);
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
  vec beta = as_Col(beta_r);
  vec eta = as_Col(eta_r);
  vec y = as_Col(y_r);
  mat MX = as_Mat(x_r);
  vec MNU = vec(y.n_elem, fill::zeros);
  vec wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  const std::string fam = tidy_family_(family);
  const FamilyType family_type = get_family_type(fam);
  const double center_tol = as_cpp<double>(control["center_tol"]),
               dev_tol = as_cpp<double>(control["dev_tol"]);
  const bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]),
               iter_center_max = as_cpp<size_t>(control["iter_center_max"]),
               iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]),
               iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]),
               n = y.n_elem, p = MX.n_cols, k = beta.n_elem;

  // Auxiliary variables (storage)

  size_t iter, iter_inner;
  vec mu = link_inv_(eta, family_type);
  vec ymean = mean(y) * vec(y.n_elem, fill::ones);
  double dev = dev_resids_(y, mu, theta, wt, family_type);
  double null_dev = dev_resids_(y, ymean, theta, wt, family_type);
  bool dev_crit, val_crit, imp_crit, conv = false;
  double dev_old, dev_ratio, dev_ratio_inner, rho;
  vec mu_eta(n), w(n), nu(n), beta_upd(k), eta_upd(n), eta_old(n), beta_old(k),
      nu_old = vec(n, fill::zeros);
  mat H(p, p);

  // Maximize the log-likelihood
  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta;
    beta_old = beta;
    dev_old = dev;

    // Compute weights and dependent variable
    mu_eta = mu_eta_(eta, family_type);
    w = (wt % square(mu_eta)) / variance_(mu, theta, family_type);
    nu = (y - mu) / mu_eta;

    // Center variables

    MNU += (nu - nu_old);
    nu_old = nu;

    center_variables_(MNU, w, k_list, center_tol, iter_center_max,
                      iter_interrupt);
    center_variables_(MX, w, k_list, center_tol, iter_center_max,
                      iter_interrupt);

    // Compute update step and update eta

    // Step-halving with three checks:
    // 1. finite deviance
    // 2. valid eta and mu
    // 3. improvement as in glm2

    beta_upd = solve_beta_(MX, MNU, w);
    eta_upd = MX * beta_upd + nu - MNU;

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + rho * eta_upd;
      beta = beta_old + rho * beta_upd;
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, theta, wt, family_type);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev));

      dev_crit = is_finite(dev);
      val_crit = valid_eta_(eta, family_type) && valid_mu_(mu, family_type);
      imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed (deviance and invalid eta or mu)

    if (!dev_crit || !val_crit) {
      stop("Inner loop failed; cannot correct step size.");
    }

    // If step halving does not improve the deviance

    if (!imp_crit) {
      eta = eta_old;
      beta = beta_old;
      dev = dev_old;
      mu = link_inv_(eta, family_type);
    }

    // Check convergence

    dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));

    if (dev_ratio < dev_tol) {
      conv = true;
      break;
    }
  }

  // Information if convergence failed
  if (!conv) {
    stop("Algorithm did not converge.");
  }

  // Update weights and dependent variable

  mu_eta = mu_eta_(eta, family_type);
  w = (wt % square(mu_eta)) / variance_(mu, theta, family_type);

  // Compute Hessian

  H = crossprod_(MX, w);

  // Generate result list

  writable::list out = writable::list({
      "coefficients"_nm = as_doubles(beta),
      "eta"_nm = as_doubles(eta),
      "weights"_nm = as_doubles(wt),
      "hessian"_nm = as_doubles_matrix(H),
      "deviance"_nm = writable::doubles({dev}),
      "null_deviance"_nm = writable::doubles({null_dev}),
      "conv"_nm = writable::logicals({conv}),
      "iter"_nm = writable::integers({static_cast<int>(iter + 1)}),
  });

  if (keep_mx) {
    mat x_cpp = as_Mat(x_r);
    center_variables_(x_cpp, w, k_list, center_tol, iter_center_max,
                      iter_interrupt);
    out.push_back({"MX"_nm = as_doubles_matrix(x_cpp)});
  }

  return out;
}
