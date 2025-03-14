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

// Compute link function inverse
vec link_inv_(const vec &eta, const FamilyType &fam) {
  switch (fam) {
  case GAUSSIAN:
    return eta;
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  case BINOMIAL: {
    vec expeta = exp(eta);
    return expeta / (1 + expeta);
  }
  case GAMMA:
    return 1 / eta;
  case INV_GAUSSIAN:
    return 1 / sqrt(eta);
  default:
    stop("Unknown family");
  }
}

// Compute deviance residuals
double dev_resids_(const vec &y, const vec &mu, const double &theta,
                   const vec &wt, const FamilyType &fam) {
  switch (fam) {
  case GAUSSIAN:
    return dot(wt, square(y - mu));
  case POISSON: {
    vec r = mu % wt;
    uvec p = find(y > 0);
    r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));
    return 2 * accu(r);
  }
  case BINOMIAL: {
    vec r(y.n_elem, fill::zeros);
    vec s(y.n_elem, fill::zeros);
    uvec p = find(y == 1);
    uvec q = find(y == 0);
    r(p) = y(p) % log(y(p) / mu(p));
    s(q) = (1 - y(q)) % log((1 - y(q)) / (1 - mu(q)));
    return 2 * dot(wt, r + s);
  }
  case GAMMA: {
    vec r = y / mu;
    uvec p = find(y == 0);
    r.elem(p).fill(1.0);
    r = wt % (log(r) - (y - mu) / mu);
    return -2 * accu(r);
  }
  case INV_GAUSSIAN:
    return dot(wt, square(y - mu) / (y % square(mu)));
  case NEG_BIN: {
    vec r = y;
    uvec p = find(y < 1);
    r.elem(p).fill(1.0);
    r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));
    return 2 * accu(r);
  }
  default:
    stop("Unknown family");
  }
}

// Compute mu_eta
vec mu_eta_(const vec &eta, const FamilyType &fam) {
  switch (fam) {
  case GAUSSIAN:
    return ones<vec>(eta.n_elem);
  case POISSON:
  case NEG_BIN:
    return exp(eta);
  case BINOMIAL: {
    vec exp_eta = exp(eta);
    return exp_eta / square(1 + exp_eta);
  }
  case GAMMA:
    return -1 / square(eta);
  case INV_GAUSSIAN:
    return -1 / (2 * pow(eta, 1.5));
  default:
    stop("Unknown family");
  }
}

// Compute variance
vec variance_(const vec &mu, const double &theta, const FamilyType &fam) {
  switch (fam) {
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

// Validate eta
bool valid_eta_(const vec &eta, const FamilyType &fam) {
  switch (fam) {
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

// Validate mu
bool valid_mu_(const vec &mu, const FamilyType &fam) {
  switch (fam) {
  case GAUSSIAN:
  case POISSON:
  case BINOMIAL:
  case NEG_BIN:
    return true;
  case GAMMA:
  case INV_GAUSSIAN:
    return all(mu > 0.0);
  default:
    stop("Unknown family");
  }
}

// Main GLM Fitting Function
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
  FamilyType fam = get_family_type(tidy_family_(family));
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  bool keep_mx = as_cpp<bool>(control["keep_mx"]);
  size_t iter, iter_inner;
  const size_t iter_max = as_cpp<size_t>(control["iter_max"]);
  const size_t iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
  const size_t iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
  const size_t n = y.n_elem;
  const size_t p = MX.n_cols;
  const size_t k = beta.n_elem;

  // Auxiliary variables (storage)
  vec mu = link_inv_(eta, fam), ymean = mean(y) * vec(y.n_elem, fill::ones),
      mu_eta(n), w(n), nu(n), beta_upd(k), eta_upd(n), eta_old(n), beta_old(k);
  double dev = dev_resids_(y, mu, theta, wt, fam),
         null_dev = dev_resids_(y, ymean, theta, wt, fam), dev_old, dev_ratio,
         dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit, conv = false;
  size_t iter_interrupt = as_cpp<size_t>(control["iter_interrupt"]);
  mat H(p, p);

  // Maximize the log-likelihood
  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta;
    beta_old = beta;
    dev_old = dev;

    // Compute weights and dependent variable
    mu_eta = mu_eta_(eta, fam);
    w = (wt % square(mu_eta)) / variance_(mu, theta, fam);
    nu = (y - mu) / mu_eta;

    // Center variables
    MNU += nu;
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
      mu = link_inv_(eta, fam);
      dev = dev_resids_(y, mu, theta, wt, fam);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev));

      dev_crit = is_finite(dev);
      val_crit = valid_eta_(eta, fam) && valid_mu_(mu, fam);
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
  if (!conv) {
    stop("Algorithm did not converge.");
  }

  // Update weights and dependent variable

  mu_eta = mu_eta_(eta, fam);
  w = (wt % square(mu_eta)) / variance_(mu, theta, fam);

  // Compute Hessian

  H = crossprod_(MX, w);

  // Generate result list

  writable::list out({
      "coefficients"_nm = as_doubles(std::move(beta)),
       "eta"_nm = as_doubles(std::move(eta)),
       "weights"_nm = as_doubles(std::move(wt)),
       "hessian"_nm = as_doubles_matrix(std::move(H)),
       "deviance"_nm = writable::doubles({dev}),
       "null_deviance"_nm = writable::doubles({null_dev}),
       "conv"_nm = writable::logicals({conv}),
       "iter"_nm = writable::integers({static_cast<int>(iter + 1)})
  });

  if (keep_mx) {
    mat x_cpp = as_Mat(x_r);
    center_variables_(x_cpp, w, k_list, center_tol, iter_center_max,
                      iter_interrupt);
    out.push_back({"MX"_nm = as_doubles_matrix(std::move(x_cpp))});
  }

  return out;
}
