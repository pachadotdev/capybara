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

// Pairwise-maximum function
// Col<double> pmax_(const Col<double> &x, const Col<double> &y) {
//   Col<double> res(x.n_elem);

//   // for (int i = 0; i < x.n_elem; ++i) {
//   //   res(i) = std::max(x(i), y(i));
//   // }

//   std::transform(x.begin(), x.end(), y.begin(), res.begin(),
//                  [](double a, double b) { return std::max(a, b); });

//   return res;
// }

Col<double> link_inv_(const Col<double> &eta, const std::string &fam) {
  Col<double> res(eta.n_elem);
  
  if (fam == "gaussian") {
    res = eta;
  } else if (fam == "poisson") {
    // Col<double> epsilon = 1e-7 * ones<Col<double>>(eta.n_elem);
    // res = pmax_(exp(eta), epsilon);
    res = exp(eta);
  } else if (fam == "binomial") {
    // res = exp(eta) / (1.0 + exp(eta));
    res = 1.0 / (1.0 + exp(-eta));
  } else if (fam == "gamma") {
    res = 1.0 / eta;
  } else if (fam == "inverse_gaussian") {
    res = 1.0 / sqrt(eta);
  } else if (fam == "negative_binomial") {
    res = exp(eta);
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
    res = accu(wt % square(y - mu));
  } else if (fam == "poisson") {
    uvec p = find(y > 0.0);
    Col<double> r = mu % wt;
    r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));
    res = 2.0 * accu(r);
  } else if (fam == "binomial") {
    uvec p = find(y != 0.0);
    uvec q = find(y != 1.0);
    Col<double> r = y / mu;
    Col<double> s = (1.0 - y) / (1.0 - mu);
    r(p) = log(r(p));
    s(q) = log(s(q));
    res = 2.0 * accu(wt % (y % r + (1.0 - y) % s));
  } else if (fam == "gamma") {
    uvec p = find(y == 0.0);
    Col<double> r = y / mu;
    r.elem(p).fill(1.0);
    res = -2.0 * accu(wt % (log(r) - (y - mu) / mu));
  } else if (fam == "inverse_gaussian") {
    res = accu(wt % square(y - mu) / (y % square(mu)));
  } else if (fam == "negative_binomial") {
    uvec p = find(y < 1.0);
    Col<double> r = y;
    r.elem(p).fill(1.0);
    res = 2.0 * accu(
        wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta))));
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
    res = is_finite(mu) && all(mu > 0.0);
  } else if (fam == "binomial") {
    res = is_finite(mu) && all(mu > 0.0 && mu < 1.0);
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

// inverse link mu = g^-1 (eta), then mu_eta = d mu / d eta

Col<double> mu_eta_(Col<double> &eta, const std::string &fam) {
  Col<double> res(eta.n_elem);

  if (fam == "gaussian") {
    res.ones();
  } else if (fam == "poisson") {
    res = exp(eta);
  } else if (fam == "binomial") {
    res = 1.0 / (2.0 + exp(eta) + exp(-eta));
  } else if (fam == "gamma") {
    res = -1.0 / square(eta);
  } else if (fam == "inverse_gaussian") {
    res = 1.0 / (2.0 * pow(eta, 1.5));
  } else if (fam == "negative_binomial") {
    res = exp(eta);
  } else {
    stop("Unknown family");
  }

  return res;
}

Col<double> variance_(const Col<double> &mu,
                      const double &theta, const std::string &fam) {
  Col<double> res(mu.n_elem);

  if (fam == "gaussian") {
    res.ones();
  } else if (fam == "poisson") {
    res = mu;
  } else if (fam == "binomial") {
    res = mu % (1.0 - mu);
  } else if (fam == "gamma") {
    res = square(mu);
  } else if (fam == "inverse_gaussian") {
    res = pow(mu, 3.0);
  } else if (fam == "negative_binomial") {
    res = mu + square(mu) / theta;
  } else {
    stop("Unknown family");
  }

  return res;
}

[[cpp11::register]] list feglm_fit_(
    const doubles &beta_r, const doubles &eta_r, const doubles &y_r,
    const doubles_matrix<> &x_r, const double &nt, const doubles &wt_r,
    const double &theta, const std::string &family, const list &control,
    const list &k_list) {
  // Type conversion

  Col<double> beta = as_Col(beta_r);
  Col<double> eta = as_Col(eta_r);
  Col<double> y = as_Col(y_r);
  Mat<double> MX = as_Mat(x_r);
  // Mat<double> MNU = nt * Mat<double>(y.n_elem, 1, fill::ones);
  Mat<double> MNU(y.n_elem, 1, fill::ones);
  Col<double> wt = as_Col(wt_r);

  // Auxiliary variables (fixed)

  std::string fam = tidy_family_(family);
  double center_tol = as_cpp<double>(control["center_tol"]);
  double dev_tol = as_cpp<double>(control["dev_tol"]);
  // std::cout << "dev_tol: " << dev_tol << std::endl;
  int iter;
  int iter_max = as_cpp<int>(control["iter_max"]);
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
  Col<double> mu_eta(n), nu(n);
  Mat<double> H(p, p), w(n, 1);
  bool conv = false;

  bool dev_crit, val_crit, imp_crit;
  double dev_old, dev_crit_ratio, rho;
  Col<double> eta_upd(n), beta_upd(k), eta_old(n), beta_old(k);

  // Maximize the log-likelihood

  for (iter = 0; iter < iter_max; ++iter) {
    std::cout << "iter: " << iter << std::endl;
    std::cout << "dev: " << dev << std::endl;
    rho = 1.0;
    dev_crit = false, val_crit = false, imp_crit = false;
    eta_old = eta, beta_old = beta, dev_old = dev;

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
    eta_upd = solve_eta_(MX, MNU, nu, beta_upd);

    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      beta = beta_old + (rho * beta_upd);
      mu = link_inv_(eta, fam);
      dev = dev_resids_(y, mu, theta, wt, fam);
      dev_crit = is_finite(dev);
      val_crit = (valid_eta_(eta, fam) && valid_mu_(mu, fam));
      imp_crit = ((dev - dev_old) / (0.1 + abs(dev)) <=  -1.0 * dev_tol);

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

    dev_crit_ratio = abs(dev - dev_old) / (0.1 + abs(dev));
    if (dev_crit_ratio < dev_tol) {
      conv = true;
      break;
    }

    // Update starting guesses for acceleration

    MNU = MNU - nu;
  }

  // Information if convergence failed

  if (conv == false) {
    stop("Algorithm did not converge.");
  }

  // Update weights and dependent variable

  mu_eta = mu_eta_(eta, fam);
  w = (wt % square(mu_eta)) / variance_(mu, theta, fam);

  // Center variables

  MX = center_variables_(as_Mat(x_r), w, k_list, center_tol, iter_center_max);

  // Recompute Hessian

  H = crossprod_(MX, w, n, p, true, true);

  // Generate result list

  writable::list out(8);

  out.push_back({"coefficients"_nm = as_doubles(beta)});
  out.push_back({"eta"_nm = as_doubles(eta)});
  out.push_back({"weights"_nm = as_doubles(wt)});
  out.push_back({"Hessian"_nm = as_doubles_matrix(H)});
  out.push_back({"deviance"_nm = dev});
  out.push_back({"null.deviance"_nm = null_dev});
  out.push_back({"conv"_nm = conv});
  out.push_back({"iter"_nm = iter});

  if (keep_mx == true) {
    out.push_back({"MX"_nm = as_doubles_matrix(MX)});
  }

  return out;
}
