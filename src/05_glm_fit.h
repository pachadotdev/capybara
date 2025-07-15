#ifndef CAPYBARA_GLM
#define CAPYBARA_GLM

enum FamilyType {
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN,
  UNKNOWN
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

vec link_inv_invgaussian_(const vec &eta) { return 1 / sqrt(abs(eta)); }

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
  vec r(y.n_elem, fill::none);

  uvec p = find(y == 1);
  uvec q = find(y == 0);
  vec y_p = y(p), y_q = y(q);

  r(p) = y_p % log(y_p / mu(p));
  r(q) = (1 - y_q) % log((1 - y_q) / (1 - mu(q)));

  return 2 * dot(wt, r);
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

inline bool stopping_criterion(double a, double b, double diffMax) {
  double diff = fabs(a - b);
  return ((diff < diffMax) || (diff / (0.1 + fabs(a)) < diffMax));
}

// TODO: Direct port of fixest CCC_logit to Armadillo
inline void ccc_logit_fixest(vec &cluster_coef, const vec &mu, const vec &sum_y, 
                             const field<uvec> &group_indices, double diff_max_nr = 1e-6) {
  const int iter_max = 100, iter_full_dicho = 10;
  const size_t nb_cluster = group_indices.n_elem;
  
  // First find the min/max values of mu for each cluster to get the bounds
  std::vector<double> borne_inf(nb_cluster);
  std::vector<double> borne_sup(nb_cluster);
  
  for (size_t m = 0; m < nb_cluster; ++m) {
    const uvec &group_m = group_indices(m);
    if (group_m.n_elem == 0) continue;
    
    // Find min/max of mu
    double mu_min = mu(group_m(0));
    double mu_max = mu(group_m(0));
    for (size_t u = 1; u < group_m.n_elem; ++u) {
      double value = mu(group_m(u));
      if (value < mu_min) {
        mu_min = value;
      } else if (value > mu_max) {
        mu_max = value;
      }
    }
    
    // Computing the bounds (protecting against log(0))
    double sum_y_m = sum_y(m);
    double table_m = static_cast<double>(group_m.n_elem);
    borne_inf[m] = log(std::max(sum_y_m, 1e-10)) - log(std::max(table_m - sum_y_m, 1e-10)) - mu_max;
    borne_sup[m] = borne_inf[m] + (mu_max - mu_min);
  }
  
  // Main loop over each cluster
  for (size_t m = 0; m < nb_cluster; ++m) {
    const uvec &group_m = group_indices(m);
    if (group_m.n_elem == 0) continue;
    
    // Initialize the cluster coefficient at 0
    double x1 = 0;
    bool keepGoing = true;
    int iter = 0;
    
    double value, x0, derivee = 0, exp_mu;
    
    // The bounds
    double lower_bound = borne_inf[m];
    double upper_bound = borne_sup[m];
    
    // Update if x1 goes out of boundaries
    if (x1 >= upper_bound || x1 <= lower_bound) {
      x1 = (lower_bound + upper_bound) / 2;
    }
    
    while (keepGoing) {
      ++iter;
      
      // Computing the value of f(x)
      value = sum_y(m);
      for (size_t u = 0; u < group_m.n_elem; ++u) {
        value -= 1.0 / (1.0 + exp(-x1 - mu(group_m(u))));
      }
      
      // Update of the bounds
      if (value > 0) {
        lower_bound = x1;
      } else {
        upper_bound = x1;
      }
      
      // Newton-Raphson iteration or Dichotomy
      x0 = x1;
      if (value == 0) {
        keepGoing = false;
      } else if (iter <= iter_full_dicho) {
        // Computing the derivative
        derivee = 0;
        for (size_t u = 0; u < group_m.n_elem; ++u) {
          exp_mu = exp(x1 + mu(group_m(u)));
          derivee -= 1.0 / ((1.0 / exp_mu + 1.0) * (1.0 + exp_mu));
        }
        
        x1 = x0 - value / derivee;
        
        // Dichotomy if necessary
        if (x1 >= upper_bound || x1 <= lower_bound) {
          x1 = (lower_bound + upper_bound) / 2;
        }
      } else {
        x1 = (lower_bound + upper_bound) / 2;
      }
      
      // Stopping criteria
      if (iter == iter_max) {
        keepGoing = false;
        // TODO: Could add warning here like fixest does
      }
      
      if (stopping_criterion(x0, x1, diff_max_nr)) {
        keepGoing = false;
      }
    }
    
    // After convergence: update cluster coefficient
    cluster_coef(m) = x1;
  }
}

struct FeglmFitResult {
  vec coefficients;
  vec eta;
  vec fitted_values;  // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  int iter;
  mat mx; // optional, only if keep_mx
  bool has_mx = false;
  uvec coef_status; // 1 = valid, 0 = collinear

  cpp11::list to_list(bool keep_mx = false) const {
    auto out = writable::list(
        {"coefficients"_nm = as_doubles(coefficients),
         "eta"_nm = as_doubles(eta), 
         "fitted.values"_nm = as_doubles(fitted_values),
         "weights"_nm = as_doubles(weights),
         "hessian"_nm = as_doubles_matrix(hessian),
         "deviance"_nm = writable::doubles({deviance}),
         "null_deviance"_nm = writable::doubles({null_deviance}),
         "conv"_nm = writable::logicals({conv}),
         "iter"_nm = writable::integers({iter}),
         "coef_status"_nm =
             as_integers(arma::conv_to<ivec>::from(coef_status))});
    if (keep_mx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(mx)});
    }
    return out;
  }
};

// Family-specific GLM algorithms matching Python pyfixest EXACTLY
// Poisson uses specialized PPML algorithm (fepois_.py)
// Binomial and others use standard GLM IRLS (feglm_.py, felogit_.py)

inline FeglmFitResult feglm_poisson(mat MX, vec beta, vec eta, const vec &y, const vec &wt,
                                             const field<field<uvec>> &group_indices,
                                             double center_tol, double dev_tol, bool keep_mx,
                                             size_t iter_max, size_t iter_center_max,
                                             size_t iter_interrupt, size_t iter_ssr, const std::string &fam) {
  FeglmFitResult res;
  size_t p = MX.n_cols;  // Remove unused n variable
  bool has_fixed_effects = group_indices.n_elem > 0;
  
  // Initialize starting values for Poisson PPML (Python fepois_.py approach)
  beta.zeros();
  double y_mean = mean(y);
  vec mu = (y + y_mean) / 2.0;  // Python: mu = (_Y + _mean) / 2
  eta = log(mu);                // Python: eta = np.log(mu)
  vec Z = eta + y / mu - 1.0;   // Python: Z = eta + _Y / mu - 1
  vec reg_Z = Z;                // Python: reg_Z = Z.copy()
  
  double last_deviance = 2.0 * sum(y % (log(y / mu) - 1.0) + mu); // Poisson deviance
  bool conv = false;
  size_t iter;
  
  // Declare variables outside loop to use after convergence
  vec delta_new(p, fill::zeros);
  mat final_WX;

  // PPML algorithm (Python fepois_.py get_fit method)
  for (iter = 0; iter < iter_max; ++iter) {
    if (iter > 0) {
      Z = eta + y / mu - 1.0;   // eq (8) in Python
      reg_Z = Z;                // eq (9) in Python
    }

    // Step 1: weighted demeaning (Python fepois_.py)
    mat ZX;
    if (MX.n_cols > 0) {
      ZX = join_horiz(reg_Z, MX);
    } else {
      ZX = reg_Z;  // Only Z column if no covariates
    }
    vec mu_weights = mu;  // Python: weights=mu.flatten() (mu is already a vector)
    
    if (has_fixed_effects) {
      demean_variables(ZX, mu_weights, group_indices, center_tol, iter_center_max, fam);
    }
    
    vec Z_resid = ZX.col(0);
    mat X_resid;
    // Python: X_resid = ZX_resid[:, 1:]  # takes all columns from index 1 onwards
    if (ZX.n_cols > 1) {
      X_resid = ZX.cols(1, ZX.n_cols - 1);
    } else {
      X_resid = mat(ZX.n_rows, 0);  // Empty matrix with correct number of rows
    }

    // Step 2: estimate WLS (Python fepois_.py)
    if (X_resid.n_cols > 0) {  // Has covariates
      mat WX = X_resid.each_col() % sqrt(mu_weights);  // Python: WX = np.sqrt(mu) * X_resid
      vec WZ = Z_resid % sqrt(mu_weights);             // Python: WZ = np.sqrt(mu) * Z_resid
      final_WX = WX;  // Store for Hessian computation

      mat XWX = WX.t() * WX;  // Python: XWX = WX.transpose() @ WX
      vec XWZ = WX.t() * WZ;  // Python: XWZ = WX.transpose() @ WZ

      try {
        delta_new = solve(XWX, XWZ, solve_opts::fast);  // Python: delta_new = solve_ols(XWX, XWZ)
        if (!is_finite(delta_new)) {
          delta_new = solve(XWX, XWZ, solve_opts::likely_sympd);
        }
      } catch (...) {
        delta_new.zeros();
      }

      vec resid = Z_resid - X_resid * delta_new;  // Python: resid = Z_resid - X_resid @ delta_new
      
      // Update eta and mu (Python fepois_.py)
      eta = Z - resid;    // Python: eta = Z - resid
      mu = exp(eta);      // Python: mu = np.exp(eta)
    } else {
      // No covariates case
      final_WX = mat(mu_weights.n_elem, 0);  // Empty matrix for Hessian
      delta_new.zeros();
      
      // For intercept-only model, use the demeaned Z directly
      eta = Z - Z_resid;
      mu = exp(eta);
    }

    // Convergence check (same criterion as fixest)
    double deviance = 2.0 * sum(y % (log(y / mu) - 1.0) + mu);
    double crit = fabs(deviance - last_deviance) / (0.1 + fabs(last_deviance));
    last_deviance = deviance;

    if (crit < dev_tol) {
      conv = true;
      break;
    }
  }

  // Final results (Python fepois_.py style)
  beta = delta_new;
  
  // Compute Hessian (Python: _hessian = XWX)
  mat H;
  if (final_WX.n_cols > 0) {
    H = final_WX.t() * final_WX;
  } else {
    H = mat(0, 0, fill::zeros);
  }

  res.coefficients = beta;
  res.eta = eta;
  res.fitted_values = mu;
  res.weights = wt;
  res.hessian = H;
  res.deviance = last_deviance;
  res.null_deviance = 0.0; // Will be computed elsewhere if needed
  res.conv = conv;
  res.iter = static_cast<int>(iter + 1);
  res.coef_status = ones<uvec>(p);  // All coefficients are valid in Poisson
  if (keep_mx) {
    res.mx = MX;
    res.has_mx = true;
  }

  return res;
}

inline FeglmFitResult feglm_irls(mat MX, vec beta, vec eta, const vec &y, const vec &wt,
                                     double theta, const field<field<uvec>> &group_indices,
                                     double center_tol, double dev_tol, bool keep_mx,
                                     size_t iter_max, size_t iter_center_max,
                                     size_t iter_inner_max, size_t iter_interrupt,
                                     size_t iter_ssr, const std::string &fam,
                                     FamilyType family_type) {
  FeglmFitResult res;
  size_t n = MX.n_rows, p = MX.n_cols;
  bool has_fixed_effects = group_indices.n_elem > 0;
  
  // Initialize starting values following Python pyfixest feglm_.py EXACTLY
  beta.zeros(); // Start with zero coefficients (Python: beta = np.zeros(_X.shape[1]))
  
  // For families that don't allow eta=0, use valid starting values
  if (family_type == GAMMA) {
    eta.fill(1.0 / mean(y));  // For Gamma: eta = 1/mu, start with mu = mean(y)
  } else if (family_type == INV_GAUSSIAN) {
    eta.fill(1.0 / sqrt(mean(y)));  // For Inverse Gaussian: eta = 1/sqrt(mu)
  } else {
    eta.zeros();  // Start with zero eta for other families (Python: eta = np.zeros(_N))
  }
  
  vec mu = link_inv_(eta, family_type); // mu = self._get_mu(theta=eta)
  
  vec ymean = mean(y) * vec(y.n_elem, fill::ones);
  mat H(p, p, fill::none);
  uvec coef_status; // Will be set from first iteration's collinearity detection
  double dev = dev_resids_(y, mu, theta, wt, family_type),
         null_dev = dev_resids_(y, ymean, theta, wt, family_type), 
         dev_old, dev_evol;
  bool conv = false;
  size_t iter;

  // Standard GLM IRLS iterations following Python pyfixest feglm_.py EXACTLY
  for (iter = 0; iter < iter_max; ++iter) {
    dev_old = dev;

    // Step 1: Compute GLM quantities (following Python pyfixest _update_W method)
    vec detadmu = mu_eta_(eta, family_type);    // mu.eta in Python
    vec V = variance_(mu, theta, family_type);  // V in Python
    vec W = ones<vec>(n) / (square(detadmu) % V); // Pure IRLS weights (W in Python, no user weights!)
    
    // Step 2: Compute working residuals and transformed quantities 
    vec v = (y - mu) % detadmu;                 // Working residuals: v = (y - mu) * mu.eta (Python formula!)
    
    // W_tilde computation (Python _update_W_tilde)
    vec W_tilde = sqrt(W);  // Python: W_tilde = np.sqrt(W) (no user weights in base GLM)
    
    vec v_tilde = W_tilde % v;                  // v_tilde = W_tilde * v (Python: W_tilde * ((y - mu) * detadmu))
    mat X_tilde = MX.each_col() % W_tilde;      // X_tilde = X * W_tilde
    
    // Step 3: Demean v_tilde and X_tilde using W_tilde weights
    vec v_dotdot = v_tilde;  // v_dotdot in Python (will be demeaned)
    mat X_dotdot = X_tilde;  // X_dotdot in Python (will be demeaned)
    
    if (has_fixed_effects) {
      // Use W_tilde for demeaning weights (Python approach)
      vec demean_weights = W_tilde;
      
      // Demean v_tilde 
      mat v_mat = v_dotdot;
      demean_variables(v_mat, demean_weights, group_indices, center_tol, iter_center_max, fam);
      v_dotdot = v_mat.col(0);
      
      // Demean X_tilde 
      demean_variables(X_dotdot, demean_weights, group_indices, center_tol, iter_center_max, fam);
    }
    
    // Step 4: Solve WLS on demeaned data 
    vec beta_diff(p, fill::zeros);
    if (p > 0) {
      // Collinearity detection (only on first iteration)
      if (iter == 0) {
        mat Q, R;
        qr(Q, R, X_dotdot);
        double tol = 1e-10;
        coef_status = ones<uvec>(p);
        for (uword j = 0; j < R.n_cols && j < R.n_rows; ++j) {
          if (std::abs(R(j, j)) < tol) {
            coef_status(j) = 0;
          }
        }
      }
      
      // Python approach: X_dotdot and v_dotdot are already weighted by W_tilde
      mat XtX = X_dotdot.t() * X_dotdot;  // Python: _update_beta_diff uses lstsq
      vec Xty = X_dotdot.t() * v_dotdot;
      
      try {
        beta_diff = solve(XtX, Xty, solve_opts::fast);
        if (!is_finite(beta_diff)) {
          beta_diff = solve(XtX, Xty, solve_opts::likely_sympd);
        }
      } catch (...) {
        beta_diff = vec(p, fill::zeros);
      }
    }
    
    // Step 5: Update using step-halving (Python _update_eta_step_halfing)
    double alpha = 1.0;
    double step_halfing_tolerance = 1e-12;  // Python uses 1e-12
    bool step_accepted = false;
    
    // If beta_diff is all zeros, no update needed - accept step immediately
    if (p == 0 || norm(beta_diff) < 1e-15) {
      step_accepted = true;
    } else {
      while (alpha > step_halfing_tolerance) {
        vec beta_try = beta + alpha * beta_diff;
        
        // Python exact formula: eta = eta + X_dotdot @ (alpha * beta_diff) / W_tilde
        vec eta_try = eta;
        if (p > 0) {
          vec eta_increment = X_dotdot * (alpha * beta_diff);
          eta_increment /= W_tilde; // Element-wise division by W_tilde (Python formula!)
          eta_try += eta_increment;
        }
        
        // Check validity
        if (!valid_eta_(eta_try, family_type)) {
          alpha /= 2.0;
          continue;
        }
        
        vec mu_try = link_inv_(eta_try, family_type);
        
        if (!valid_mu_(mu_try, family_type)) {
          alpha /= 2.0;
          continue;
        }
        
        double dev_try = dev_resids_(y, mu_try, theta, wt, family_type);
        
        // Accept step if deviance improves (Python's criterion)
        if (is_finite(dev_try) && dev_try < dev_old) {
          beta = beta_try;
          eta = eta_try;
          mu = mu_try;
          dev = dev_try;
          step_accepted = true;
          break;
        } else {
          alpha /= 2.0;
        }
      }
    }
    
    if (!step_accepted) {
      // Python raises RuntimeError("Step-halving failed to find improvement.")
      conv = false;
      break;
    }

    // Convergence check (Python style)
    dev_evol = dev - dev_old;
    double crit = fabs(dev_evol) / (0.1 + fabs(dev_old));
    if (crit < dev_tol) {
      conv = true;
      break;
    }
  }

  // Compute final Hessian using standard IRLS approach
  vec final_detadmu = mu_eta_(eta, family_type);
  vec final_V = variance_(mu, theta, family_type);
  vec final_W = (ones<vec>(n) / (square(final_detadmu) % final_V)) % wt;  // IRLS weights * user weights
  
  if (p > 0) {
    H = MX.t() * diagmat(final_W) * MX;
  } else {
    H = mat(0, 0, fill::zeros);
  }

  res.coefficients = beta;
  res.eta = eta;
  res.fitted_values = mu;
  res.weights = wt;
  res.hessian = H;
  res.deviance = dev;
  res.null_deviance = null_dev;
  res.conv = conv;
  res.iter = static_cast<int>(iter + 1);
  res.coef_status = coef_status;

  if (keep_mx) {
    res.mx = MX;
    res.has_mx = true;
  }

  return res;
}

// Main dispatcher function - uses family-specific algorithms like Python pyfixest
inline FeglmFitResult feglm_fit(mat MX, // copy for in-place centering
                                vec beta, vec eta, const vec &y, const vec &wt,
                                double theta, const field<field<uvec>> &group_indices,
                                double center_tol, double dev_tol, bool keep_mx,
                                size_t iter_max, size_t iter_center_max,
                                size_t iter_inner_max, size_t iter_interrupt,
                                size_t iter_ssr, const std::string &fam,
                                FamilyType family_type) {
  
  // Family-specific algorithm dispatch matching Python pyfixest structure
  if (family_type == POISSON) {
    // Use specialized PPML algorithm like Python fepois_.py
    return feglm_poisson(MX, beta, eta, y, wt, group_indices,
                                  center_tol, dev_tol, keep_mx, iter_max, iter_center_max,
                                  iter_interrupt, iter_ssr, fam);
  } else {
    // Use standard GLM IRLS like Python feglm_.py (for binomial, gamma, etc.)
    return feglm_irls(MX, beta, eta, y, wt, theta, group_indices,
                          center_tol, dev_tol, keep_mx, iter_max, iter_center_max,
                          iter_inner_max, iter_interrupt, iter_ssr, fam, family_type);
  }
}

#endif // CAPYBARA_GLM
