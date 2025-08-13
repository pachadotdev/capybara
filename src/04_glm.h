// Computing generalized linear models with fixed effects
// eta = X beta + alpha + offset

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

#include <algorithm>
#include <string>
#include <unordered_map>

namespace capybara {

// Define the family types enum
enum Family {
  UNKNOWN = 0,
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN
};

// Utility function to clamp scalar values
template <typename T>
inline T clamp(const T &value, const T &lower, const T &upper) {
  return (value < lower) ? lower : ((value > upper) ? upper : value);
}

struct InferenceGLM {
  vec coefficients;
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  size_t iter;
  uvec coef_status; // 1 = estimable, 0 = collinear

  field<vec> fixed_effects;
  bool has_fe = false;
  uvec iterations;

  mat TX;
  bool has_tx = false;

  vec means;

  // Separation info
  bool has_separation = false;
  uvec separated_obs;
  vec separation_certificate;

  InferenceGLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), eta(n, fill::zeros),
        fitted_values(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), deviance(0.0), null_deviance(0.0),
        conv(false), iter(0), coef_status(p, fill::ones), has_fe(false),
        has_tx(false), has_separation(false) {}
};

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

Family get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, Family> family_map = {
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

vec link_inv_logit_(const vec &eta) { return 1.0 / (1.0 + exp(-eta)); }

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
// in base R it can be found in src/library/stats/src/family.c
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

vec link_inv_(const vec &eta, const Family family_type) {
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
                   const vec &wt, const Family family_type) {
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

bool valid_eta_(const vec &eta, const Family family_type) {
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

bool valid_mu_(const vec &mu, const Family family_type) {
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

vec mu_eta_(const vec &eta, const Family family_type) {
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

vec variance_(const vec &mu, const double &theta, const Family family_type) {
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

// Poisson-only separation propery
// Here we use a rectified linear unit (ReLU) activation function
struct SeparationResult {
  bool has_separation;
  uvec separated_obs;
  vec certificate; // z values from ReLU algorithm
  size_t iterations;

  SeparationResult() : has_separation(false), iterations(0) {}
};

SeparationResult detect_poisson_separation(
    const vec &y, const mat &X, const vec &w,
    const field<field<uvec>> &fe_groups, const CapybaraParameters &params,
    const CollinearityResult &collin_result, double tol = 1e-6,
    size_t max_iter = 100, bool accelerate = true) {
  SeparationResult result;

  const size_t n = y.n_elem;
  const size_t p = X.n_cols;

  // Identify boundary observations (y == 0)
  uvec boundary_obs = find(y == 0);
  uvec interior_obs = find(y > 0);
  size_t num_boundary = boundary_obs.n_elem;

  if (num_boundary == 0) {
    return result; // No separation possible
  }

  // Initialize u = (y == 0)
  vec u = conv_to<vec>::from(y == 0);
  vec u_last = u;
  vec utilde = u;
  mat xtilde = X;

  // Working variables
  vec xbd(n, fill::zeros);
  vec xbd_prev1(n, fill::zeros);
  vec xbd_prev2(n, fill::zeros);
  vec resid(n);
  double epsilon = 0.0;
  double delta;

  // Acceleration variables
  uvec accelerated_obs;
  double acceleration_weight = 1.0;
  bool convergence_stuck = false;

  // Create workspace for weighted regression
  BetaWorkspace beta_workspace(n, p);

  for (size_t iter = 1; iter <= max_iter; ++iter) {
    // Rotate previous xbd values
    xbd_prev2 = xbd_prev1;
    xbd_prev1 = xbd;

    // Update utilde for acceleration
    if (iter > 1) {
      utilde = u + utilde - u_last;
    }
    u_last = u;

    // Solve least squares with equalities (LSE) using method of weighting
    // Give very high weights to interior observations (y > 0)
    vec w_lse = w;
    w_lse.elem(interior_obs) *=
        1e10; // Large weight to enforce u ~= 0 when y > 0

    // Add acceleration weights if stuck
    if (convergence_stuck && accelerated_obs.n_elem > 0) {
      w_lse.elem(accelerated_obs) *= acceleration_weight;
    }

    // Center variables if fixed effects present
    vec u_centered = utilde;
    mat X_centered = xtilde;
    if (fe_groups.n_elem > 0) {
      center_variables(u_centered, w_lse, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       params.iter_ssr);
      center_variables(X_centered, w_lse, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       params.iter_ssr);
    }

    // Solve weighted least squares
    InferenceBeta beta_result =
        get_beta(X_centered, u_centered, u_centered, w_lse, collin_result,
                 false, false, &beta_workspace);

    // Compute predictions and residuals using original matrix and handling
    // collinearity
    vec beta_coef = beta_result.coefficients;

    // Handle mapping from centered space back to original space
    if (collin_result.has_collinearity &&
        collin_result.non_collinear_cols.n_elem > 0) {
      if (beta_coef.n_elem == collin_result.coef_status.n_elem) {
        // get_beta returned full-size vector, extract non-collinear part for X
        // multiplication
        vec beta_reduced = beta_coef.elem(collin_result.non_collinear_cols);
        xbd = X * beta_reduced;
      } else if (beta_coef.n_elem == collin_result.non_collinear_cols.n_elem) {
        // get_beta returned reduced vector, use directly
        xbd = X * beta_coef;
      } else {
        // Fallback: create properly sized vector for X
        vec beta_safe(X.n_cols, fill::zeros);
        size_t copy_size = std::min(beta_coef.n_elem, X.n_cols);
        beta_safe.head(copy_size) = beta_coef.head(copy_size);
        xbd = X * beta_safe;
      }
    } else {
      // No collinearity, use coefficients directly
      if (beta_coef.n_elem == X.n_cols) {
        xbd = X * beta_coef;
      } else {
        // Size mismatch, create properly sized vector
        vec beta_safe(X.n_cols, fill::zeros);
        size_t copy_size = std::min(beta_coef.n_elem, X.n_cols);
        beta_safe.head(copy_size) = beta_coef.head(copy_size);
        xbd = X * beta_safe;
      }
    }
    resid = u - xbd;

    // Update epsilon for tolerance
    epsilon = max(abs(resid.elem(interior_obs)));
    delta = epsilon + tol;

    // Set xbd = 0 for interior observations
    xbd.elem(interior_obs).zeros();

    // Set xbd = 0 for boundary obs within tolerance
    uvec near_zero = find(abs(xbd.elem(boundary_obs)) <= delta);
    if (near_zero.n_elem > 0) {
      xbd.elem(boundary_obs.elem(near_zero)).zeros();
    }

    // Check convergence: all boundary predictions are non-negative
    if (all(xbd.elem(boundary_obs) >= 0)) {
      uvec sep_idx = find(xbd.elem(boundary_obs) > 0);
      if (sep_idx.n_elem > 0) {
        result.has_separation = true;
        result.separated_obs = boundary_obs.elem(sep_idx);
        result.certificate = xbd;
        result.iterations = iter;
      }
      break;
    }

    // Check for no negative residuals (alternative convergence criterion)
    if (min(resid.elem(boundary_obs)) >= -tol) {
      uvec pos_resid = find(resid.elem(boundary_obs) > delta);
      if (pos_resid.n_elem > 0) {
        xbd.elem(boundary_obs.elem(pos_resid)).zeros();
      }
      uvec sep_idx = find(xbd.elem(boundary_obs) > 0);
      if (sep_idx.n_elem > 0) {
        result.has_separation = true;
        result.separated_obs = boundary_obs.elem(sep_idx);
        result.certificate = xbd;
        result.iterations = iter;
      }
      break;
    }

    // Check for stuck convergence and apply acceleration
    if (accelerate && iter > 3) {
      size_t num_candidates = sum(xbd.elem(boundary_obs) > delta);
      // Simple stuck detection: little progress in candidates
      if (!convergence_stuck && num_candidates > 0) {
        convergence_stuck =
            true; // Simplified - you can add more sophisticated detection
      }

      if (convergence_stuck) {
        // Find observations to accelerate: y=0 and xbd consistently negative
        accelerated_obs =
            find((y == 0) && (xbd_prev2 < 1.01 * xbd_prev1) &&
                 (xbd_prev1 < 1.01 * xbd) && (xbd < -0.1 * delta));
        if (accelerated_obs.n_elem > 0) {
          acceleration_weight = std::min(256.0, 4.0 * acceleration_weight);
        }
      }
    }

    // Apply ReLU: u = max(xbd, 0) for boundary observations
    u.elem(boundary_obs) =
        arma::max(xbd.elem(boundary_obs), zeros(num_boundary));
  }

  return result;
}

// Core implementation function using pure C++/Armadillo types
InferenceGLM feglm_fit(vec &beta, vec &eta, const vec &y, mat &X, const vec &w,
                       const double &theta, const Family family_type,
                       const field<field<uvec>> &fe_groups,
                       const CapybaraParameters &params) {
  const size_t n = y.n_elem;
  const size_t p = X.n_cols;
  const size_t k = beta.n_elem;
  const bool has_fixed_effects = fe_groups.n_elem > 0;

  // Keep a copy of original X before centering for fixed effects computation
  mat X0 = X;

  // Initialize result object
  InferenceGLM result(n, p);

  // Check input data
  if (!is_finite(y) || !is_finite(X)) {
    result.conv = false;
    return result;
  }

  // Auxiliary variables (storage)
  vec MNU = vec(n, fill::zeros);
  vec mu = link_inv_(eta, family_type);
  vec ymean = mean(y) * vec(n, fill::ones);
  vec mu_eta(n, fill::none), w_working(n, fill::none);
  vec nu(n, fill::none), beta_upd(k, fill::none);
  vec eta_upd(n, fill::none), eta0(n, fill::none);
  vec beta0(k, fill::none), nu0 = vec(n, fill::zeros);
  mat H(p, p, fill::none);

  // Create a workspace for get_beta
  BetaWorkspace beta_workspace(n, p);

  // Initial deviance
  double dev = dev_resids_(y, mu, theta, w, family_type);
  double null_dev = dev_resids_(y, ymean, theta, w, family_type);
  double dev0, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit, conv = false;

  CollinearityResult collin_result =
      check_collinearity(X, w, /*use_weights =*/true, params.collin_tol);

  // Early separation detection for Poisson models
  if (family_type == POISSON && params.check_separation) {
    // Use the same X matrix that collinearity was checked on, not X0
    SeparationResult sep_result = detect_poisson_separation(
        y, X, w, fe_groups, params, collin_result, params.sep_tol,
        params.sep_max_iter, params.sep_accelerate);

    if (sep_result.has_separation) {
      result.has_separation = true;
      result.separated_obs = sep_result.separated_obs;
      result.separation_certificate = sep_result.certificate;
      result.conv = false; // Cannot fit GLM with separation
      return result;
    }
  }

  // Maximize the log-likelihood
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    beta0 = beta;
    dev0 = dev;

    // Compute weights and dependent variable
    mu_eta = mu_eta_(eta, family_type);
    w_working = (w % square(mu_eta)) / variance_(mu, theta, family_type);
    nu = (y - mu) / mu_eta;

    // Center variables
    MNU += (nu - nu0);
    nu0 = nu;

    if (has_fixed_effects) {
      center_variables(MNU, w_working, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       params.iter_ssr);
      center_variables(X, w_working, fe_groups, params.center_tol,
                       params.iter_center_max, params.iter_interrupt,
                       params.iter_ssr);
    }

    // Use the full version of get_beta that returns InferenceBeta
    InferenceBeta beta_result = get_beta(X, MNU, MNU, w_working, collin_result,
                                         false, false, &beta_workspace);

    // Handle collinearity properly - work with reduced coefficients throughout
    vec beta_upd_reduced;
    if (collin_result.has_collinearity &&
        collin_result.non_collinear_cols.n_elem > 0) {
      // Extract only non-collinear coefficients for the reduced system
      beta_upd_reduced =
          beta_result.coefficients.elem(collin_result.non_collinear_cols);
    } else {
      // No collinearity, use all coefficients
      beta_upd_reduced = beta_result.coefficients;
    }

    // Ensure beta has the right size for the full parameter vector
    const size_t full_p =
        collin_result.has_collinearity ? collin_result.coef_status.n_elem : p;
    if (beta.n_elem != full_p) {
      beta.resize(full_p);
    }

    // Compute eta update using reduced coefficients with reduced X
    if (X.n_cols > 0) {
      eta_upd = X * beta_upd_reduced + nu - MNU;
    } else {
      eta_upd = nu - MNU;
    }

    // Step-halving with three checks
    for (size_t iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + rho * eta_upd;

      // Update beta by expanding the reduced coefficients back to full size
      vec beta_new = beta0;
      if (collin_result.has_collinearity &&
          collin_result.non_collinear_cols.n_elem > 0) {
        // Update only the non-collinear coefficients
        vec beta0_reduced = beta0.elem(collin_result.non_collinear_cols);
        vec beta_upd_step = beta0_reduced + rho * beta_upd_reduced;
        beta_new.elem(collin_result.non_collinear_cols) = beta_upd_step;
      } else {
        // No collinearity, update all coefficients
        beta_new = beta0 + rho * beta_upd_reduced;
      }
      beta = beta_new;

      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, theta, w, family_type);
      dev_ratio_inner = (dev - dev0) / (0.1 + fabs(dev));

      dev_crit = is_finite(dev);
      val_crit = valid_eta_(eta, family_type) && valid_mu_(mu, family_type);
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= params.step_halving_factor;
    }

    // Check if step-halving failed
    if (!dev_crit || !val_crit) {
      result.conv = false;
      return result;
    }

    // If step halving does not improve the deviance
    if (!imp_crit) {
      eta = eta0;
      beta = beta0;
      dev = dev0;
      mu = link_inv_(eta, family_type);
    }

    // Check convergence
    dev_ratio = fabs(dev - dev0) / (0.1 + fabs(dev));
    if (dev_ratio < params.dev_tol) {
      conv = true;
      break;
    }

    result.iter = iter + 1;
  }

  // Final computations if converged
  if (conv) {
    // Compute final Hessian
    H = crossprod(X, w_working);

    // Recover fixed effects if needed
    if (has_fixed_effects) {
      // Following alpaca's getFE approach exactly for GLMs:
      // pi = eta - X %*% beta where eta is the linear predictor
      // We use the original (non-centered) X matrix, just like in felm_fit

      // Compute X * beta using original (non-centered) data and handling
      // collinearity
      vec x_beta(n, fill::zeros);
      if (X0.n_cols > 0) {
        if (collin_result.has_collinearity &&
            collin_result.non_collinear_cols.n_elem > 0) {
          // Use only non-collinear columns and coefficients
          x_beta = X0.cols(collin_result.non_collinear_cols) *
                   beta.elem(collin_result.non_collinear_cols);
        } else {
          // No collinearity, use all columns and coefficients
          x_beta = X0 * beta;
        }
      } else {
        x_beta.zeros(n);
      }

      // Compute pi = eta - X*beta (using original data, matching alpaca's
      // getFE)
      vec pi = eta - x_beta;

      // Store fixed effects results
      result.has_fe = true;
      if (params.return_fe) {
        result.fixed_effects =
            get_alpha(pi, fe_groups, params.alpha_tol, params.iter_alpha_max);
      }
    }

    // Populate result
    result.coefficients = beta;
    result.coef_status =
        collin_result.coef_status; // Include collinearity status
    result.eta = eta;
    result.fitted_values = mu;
    result.weights = w;
    result.hessian = H;
    result.deviance = dev;
    result.null_deviance = null_dev;
    result.conv = true;

    // Keep design matrix if requested
    if (params.keep_tx) {
      result.TX = X;
      result.has_tx = true;
    }
  }

  return result;
}

vec feglm_offset_fit(vec &eta, const vec &y, const vec &offset, const vec &w,
                     const Family family_type,
                     const field<field<uvec>> &fe_groups,
                     const CapybaraParameters &params) {

  const size_t n = y.n_elem;

  // Auxiliary variables (storage)
  vec Myadj = vec(n, fill::zeros);
  vec mu = link_inv_(eta, family_type);
  vec mu_eta(n, fill::none), yadj(n, fill::none);
  vec w_working(n, fill::none), eta_upd(n, fill::none), eta0(n, fill::none);

  double dev = dev_resids_(y, mu, 0.0, w, family_type);
  double dev0, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  // Maximize the log-likelihood
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    rho = 1.0;
    eta0 = eta;
    dev0 = dev;

    // Compute weights and dependent variable
    mu_eta = mu_eta_(eta, family_type);
    w_working = (w % square(mu_eta)) / variance_(mu, 0.0, family_type);
    yadj = (y - mu) / mu_eta + eta - offset;

    // Center variables
    Myadj += yadj;

    // Use C++/Armadillo types for centering
    center_variables(Myadj, w_working, fe_groups, params.center_tol,
                     params.iter_center_max, params.iter_interrupt,
                     params.iter_ssr);

    // Compute update step and update eta
    eta_upd = yadj - Myadj + offset - eta;

    for (size_t iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta0 + (rho * eta_upd);
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, 0.0, w, family_type);
      dev_ratio_inner = (dev - dev0) / (0.1 + fabs(dev0));

      dev_crit = is_finite(dev);
      val_crit = (valid_eta_(eta, family_type) && valid_mu_(mu, family_type));
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= params.step_halving_factor;
    }

    // Check if step-halving failed
    if (!dev_crit || !val_crit) {
      eta = eta0;
      mu = link_inv_(eta, family_type);
      break;
    }

    // Check convergence
    dev_ratio = fabs(dev - dev0) / (0.1 + fabs(dev));
    if (dev_ratio < params.dev_tol) {
      break;
    }

    // Update starting guesses for acceleration
    Myadj = Myadj - yadj;
  }

  return eta;
}

// Dedicated implementation for negative binomial models
struct InferenceNegBin : public InferenceGLM {
  double theta;
  size_t iter_outer;
  bool conv_outer;

  InferenceNegBin(size_t n, size_t p)
      : InferenceGLM(n, p), theta(1.0), iter_outer(0), conv_outer(false) {}
};

// Method of moments: theta = mu^2 / (var - mu)
inline double estimate_theta(const vec &y, const vec &mu,
                             double theta_min = 0.1, double theta_max = 1.0e6,
                             double overdispersion_threshold = 0.01) {
  const double y_mean = mean(y);
  const double y_var = var(y);
  const double overdispersion = y_var - y_mean;

  // Very little overdispersion => return very large theta (Poisson-like)
  if (overdispersion <= overdispersion_threshold * y_mean) {
    return theta_max;
  }

  // Estimate theta using method of moments
  double theta = y_mean * y_mean / overdispersion;

  // Ensure theta is within reasonable bounds
  return clamp(theta, theta_min, theta_max);
}

// Dedicated implementation for negative binomial models
InferenceNegBin fenegbin_fit(mat &X, const vec &y, const vec &w,
                             const field<field<uvec>> &fe_groups,
                             const CapybaraParameters &params,
                             double init_theta = 0.0) {
  const size_t n = y.n_elem;
  const size_t p = X.n_cols;

  InferenceNegBin result(n, p);

  // Initialize theta if not provided
  double theta = (init_theta > 0) ? init_theta : 1.0;

  // Start with Poisson for initialization
  Family poisson_family = POISSON;

  // Initialize beta and eta with Poisson fit
  vec beta(p, fill::zeros);
  vec eta(n, fill::zeros);

  // Get initial fit using Poisson
  InferenceGLM poisson_fit =
      feglm_fit(beta, eta, y, X, w, 0.0, poisson_family, fe_groups, params);

  if (!poisson_fit.conv) {
    result.conv = false;
    result.conv_outer = false;
    return result;
  }

  // Transfer initial values from Poisson fit
  beta = poisson_fit.coefficients;
  eta = poisson_fit.eta;
  vec mu = poisson_fit.fitted_values;

  // Estimate initial theta if not provided
  if (init_theta <= 0) {
    theta = estimate_theta(y, mu);
  }

  double dev0 = poisson_fit.deviance;
  double theta0 = theta;
  bool converged = false;

  // Alternate between fitting GLM and updating theta
  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    result.iter_outer = iter + 1;

    // Save old theta
    theta0 = theta;

    // Fit GLM with current theta
    Family negbin_family = NEG_BIN;
    InferenceGLM glm_fit =
        feglm_fit(beta, eta, y, X, w, theta, negbin_family, fe_groups, params);

    if (!glm_fit.conv) {
      break;
    }

    // Update mu and compute new theta
    mu = glm_fit.fitted_values;
    double dev = glm_fit.deviance;

    // Estimate new theta based on current fit
    double theta_new = estimate_theta(y, mu);

    // Check validity
    if (!is_finite(theta_new) || theta_new <= 0) {
      theta_new = theta;
    }

    // Check convergence criteria
    double dev_crit = std::abs(dev - dev0) / (0.1 + std::abs(dev));
    double theta_crit = std::abs(theta_new - theta0) / (0.1 + std::abs(theta0));

    if (dev_crit <= params.dev_tol && theta_crit <= params.dev_tol) {
      converged = true;
      theta = theta_new;

      // Transfer results from glm_fit to result
      result.coefficients = beta;
      result.eta = eta;
      result.fitted_values = mu;
      result.weights = w;
      result.hessian = glm_fit.hessian;
      result.deviance = dev;
      result.null_deviance = glm_fit.null_deviance;
      result.conv = glm_fit.conv;
      result.iter = glm_fit.iter;
      result.coef_status = std::move(glm_fit.coef_status);
      result.fixed_effects = std::move(glm_fit.fixed_effects);
      result.has_fe = glm_fit.has_fe;
      result.TX = std::move(glm_fit.TX);
      result.has_tx = glm_fit.has_tx;
      result.theta = theta;
      result.conv_outer = true;

      break;
    }

    // Update values for next iteration
    theta = theta_new;
    dev0 = dev;

    // Save latest results for next iteration
    beta = glm_fit.coefficients;
    eta = glm_fit.eta;

    // Save results to return in case we hit max iterations
    result.coefficients = std::move(glm_fit.coefficients);
    result.eta = std::move(glm_fit.eta);
    result.fitted_values = std::move(glm_fit.fitted_values);
    result.weights = std::move(glm_fit.weights);
    result.hessian = std::move(glm_fit.hessian);
    result.deviance = glm_fit.deviance;
    result.null_deviance = glm_fit.null_deviance;
    result.conv = glm_fit.conv;
    result.iter = glm_fit.iter;
    result.coef_status = std::move(glm_fit.coef_status);
    result.fixed_effects = std::move(glm_fit.fixed_effects);
    result.has_fe = glm_fit.has_fe;
    result.TX = std::move(glm_fit.TX);
    result.has_tx = glm_fit.has_tx;
  }

  // Set final theta and convergence status
  result.theta = theta;
  result.conv_outer = converged;

  return result;
}

} // namespace capybara

#endif // CAPYBARA_GLM_H
