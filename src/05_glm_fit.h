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

// Forward declaration for concentration function


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
  vec result(eta.n_elem);
  for (size_t i = 0; i < eta.n_elem; ++i) {
    double x = eta(i);
    if (x < -30) {
      result(i) = std::numeric_limits<double>::epsilon();
    } else if (x > 30) {
      result(i) = 1 - std::numeric_limits<double>::epsilon();
    } else {
      result(i) = 1.0 / (1.0 + 1.0 / exp(x));
    }
  }
  return result;
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
    for (size_t i = 0; i < eta.n_elem; ++i) {
      double x = eta(i);
      if (std::abs(x) > 30) {
        result(i) = std::numeric_limits<double>::epsilon();
      } else {
        double exp_x = exp(x);
        result(i) = 1.0 / ((1.0 + 1.0 / exp_x) * (1.0 + exp_x));
      }
    }
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

inline FeglmFitResult feglm_fit(mat MX, // copy for in-place centering
                                vec beta, vec eta, const vec &y, const vec &wt,
                                double theta, const field<field<uvec>> &group_indices,
                                double center_tol, double dev_tol, bool keep_mx,
                                size_t iter_max, size_t iter_center_max,
                                size_t iter_inner_max, size_t iter_interrupt,
                                size_t iter_ssr, const std::string &fam,
                                FamilyType family_type) {
  FeglmFitResult res;
  size_t n = y.n_elem, p = MX.n_cols, k = beta.n_elem;
  vec mu(n, fill::none), ymean = mean(y) * vec(n, fill::ones),
      mu_eta(n, fill::none), w(n, fill::none), z(n, fill::none),
      beta_upd(k, fill::none), eta_upd(n, fill::none), eta_old(n, fill::none),
      beta_old(k, fill::none);
  mat H(p, p, fill::none);
  double dev, null_dev, rho;
  bool dev_crit, val_crit, conv = false;
  size_t iter, iter_inner;

  beta_results ws(n, p);
  
  vec mustart = vec(n, fill::zeros);
  
  if (family_type == POISSON) {
    mustart = (y + 0.1) / 2.0;
  } else if (family_type == BINOMIAL) {
    mustart = (wt % y + 0.5) / (wt + 1.0);
  } else if (family_type == GAMMA) {
    mustart = y;  // Gamma family: mustart = y
  } else if (family_type == INV_GAUSSIAN) {
    mustart = clamp(abs(y), 0.1, 100.0);  // Inverse Gaussian: bounded positive values
  } else {
    mustart.fill(0.0);
  }
  
  if (family_type == POISSON) {
    eta = log(mustart);
  } else if (family_type == BINOMIAL) {
    eta = log(mustart / (1 - mustart));  // logit link
  } else if (family_type == GAMMA) {
    eta = 1.0 / mustart;  // Gamma inverse link
  } else if (family_type == INV_GAUSSIAN) {
    eta = 1.0 / square(mustart);  // Inverse Gaussian link: 1/mu^2
  } else {
    eta = mustart;
  }
  
  mu = link_inv_(eta, family_type);
  
  // Starting deviance with mu values computed from eta
  double devold = dev_resids_(y, mu, theta, wt, family_type);
  
  // Null deviance
  null_dev = dev_resids_(y, ymean, theta, wt, family_type);
  
  // Initialize wols_means to zeros
  vec wols_means = vec(n, fill::zeros);
  
  // Main GLM iteration loop
  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta;
    beta_old = beta;
    
    // Compute mu.eta and variance
    mu_eta = mu_eta_(eta, family_type);
    vec var_mu = variance_(mu, theta, family_type);
    
    // Error checks
    if (any(var_mu <= 0) || !var_mu.is_finite()) {
      stop("Problems with variance function at iteration ", iter);
    }
    if (!mu_eta.is_finite()) {
      stop("Problems with mu.eta function at iteration ", iter);
    }
    
    // Working response: z = eta + (y - mu)/mu.eta
    z = eta + (y - mu) / mu_eta;
    
    // Weights: w = weights * mu.eta^2 / var_mu
    w = (wt % square(mu_eta)) / var_mu;
    
    // Check for zero weights
    uvec is_0w = find(w == 0);
    if (is_0w.n_elem == n) {
      stop("All weights are zero at iteration ", iter);
    }
    
    // Solve weighted least squares
    // This solves: min_beta,fe sum(w * (z - X*beta - fe)^2)
    
    vec z_demean = z - wols_means;  // subtract previous fixed effects
    mat MX_demean = MX;
    
    // Simple iterative weighted demeaning
    for (size_t iter_demean = 0; iter_demean < 10; ++iter_demean) {
      // Demean X and z using current weights
      for (size_t g = 0; g < group_indices.n_elem; ++g) {
        const field<uvec> &groups_g = group_indices(g);
        
        for (size_t j = 0; j < groups_g.n_elem; ++j) {
          const uvec &group_j = groups_g(j);
          if (group_j.n_elem > 0) {
            vec group_weights = w.elem(group_j);
            double weight_sum = sum(group_weights);
            if (weight_sum > 0) {
              // Weighted mean for z
              double z_wmean = dot(group_weights, z_demean.elem(group_j)) / weight_sum;
              z_demean.elem(group_j) -= z_wmean;
              
              // Weighted mean for each column of X
              for (size_t col = 0; col < MX.n_cols; ++col) {
                vec x_col = MX_demean.col(col);
                double x_wmean = dot(group_weights, x_col.elem(group_j)) / weight_sum;
                x_col.elem(group_j) -= x_wmean;
                MX_demean.col(col) = x_col;
              }
            }
          }
        }
      }
    }
    
    // Solve weighted regression on demeaned data
    mat XtW = MX_demean.t() * diagmat(w);
    mat XtWX = XtW * MX_demean;
    vec XtWz = XtW * z_demean;
    
    // Collinearity detection via Cholesky decomposition (like fixest)
    mat L;
    bool chol_success = chol(L, XtWX);
    uvec is_excluded = zeros<uvec>(p);  // Use p (MX.n_cols) not k (beta.n_elem)
    
    if (chol_success) {
      // Check diagonal elements for collinearity
      vec diag_L = L.diag();
      double collin_tol = 1e-10;  // Same as fixest default
      for (size_t i = 0; i < p; ++i) {
        if (diag_L(i) < collin_tol) {
          is_excluded(i) = 1;
        }
      }
    } else {
      // Cholesky failed, use SVD fallback
      mat U, V;
      vec s;
      bool svd_success = svd(U, s, V, XtWX);
      if (svd_success) {
        double collin_tol = 1e-10;
        for (size_t i = 0; i < p; ++i) {
          if (s(i) < collin_tol) {
            is_excluded(i) = 1;
          }
        }
      }
    }
    
    // Solve with collinearity handling
    beta_upd = zeros<vec>(p);  // Use p to match MX dimensions
    if (any(is_excluded == 0)) {  // If any variables are not excluded
      uvec good_vars = find(is_excluded == 0);
      mat XtWX_good = XtWX.submat(good_vars, good_vars);
      vec XtWz_good = XtWz.elem(good_vars);
      
      try {
        vec beta_good = solve(XtWX_good, XtWz_good);
        beta_upd.elem(good_vars) = beta_good;
      } catch (...) {
        vec beta_good = pinv(XtWX_good) * XtWz_good;
        beta_upd.elem(good_vars) = beta_good;
      }
    }
    
    // Ensure beta is properly sized and copy relevant coefficients
    if (beta.n_elem != p) {
      beta.resize(p);
    }
    beta = beta_upd;
    
    // Store collinearity info
    ws.valid_coefficients = 1 - is_excluded;  // 1 for valid, 0 for excluded
    
    // Update fixed effects by computing residuals and taking weighted group means
    vec residuals = z - MX * beta_upd;
    vec fe_new = vec(n, fill::zeros);
    
    for (size_t g = 0; g < group_indices.n_elem; ++g) {
      const field<uvec> &groups_g = group_indices(g);
      
      for (size_t j = 0; j < groups_g.n_elem; ++j) {
        const uvec &group_j = groups_g(j);
        if (group_j.n_elem > 0) {
          vec group_weights = w.elem(group_j);
          vec group_residuals = residuals.elem(group_j);
          double weight_sum = sum(group_weights);
          if (weight_sum > 0) {
            double fe_value = dot(group_weights, group_residuals) / weight_sum;
            fe_new.elem(group_j).fill(fe_value);
          }
        }
      }
    }
    
    wols_means = fe_new;
    
    // Compute eta update (fitted values)
    eta_upd = MX * beta_upd + wols_means;
    
    // Step halving loop
    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + rho * (eta_upd - eta_old);
      beta = beta_old + rho * (beta_upd - beta_old);
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, theta, wt, family_type);
      
      // Check criteria
      dev_crit = is_finite(dev);
      val_crit = valid_eta_(eta, family_type) && valid_mu_(mu, family_type);
      
      double dev_evol = dev - devold;
      bool no_SH = is_finite(dev) && (fabs(dev_evol) < dev_tol || fabs(dev_evol)/(0.1 + fabs(dev)) < dev_tol);
      
      if (no_SH) break;
      
      // Step halving condition
      if (!dev_crit || dev_evol > 0 || !val_crit) {
        rho *= 0.5;
        if (rho < 1e-8) {
          if (iter == 0) {
            stop("Algorithm failed at first iteration. Step-halving could not find valid parameters.");
          }
          eta = eta_old;
          beta = beta_old;
          dev = devold;
          mu = link_inv_(eta, family_type);
          break;
        }
      } else {
        break;
      }
    }
    
    // Convergence check
    double dev_evol = dev - devold;
    double rel_dev_change = fabs(dev_evol)/(0.1 + fabs(dev));
    
    if (rel_dev_change < dev_tol) {
      conv = true;
      break;
    } else {
      devold = dev;
    }
  }

  if (!conv) stop("Algorithm did not converge.");

  H = crossprod_(MX, w);

  // Set excluded coefficients to NaN (will become NA in R)
  for (size_t i = 0; i < p; ++i) {
    if (ws.valid_coefficients(i) == 0) {
      beta(i) = datum::nan;
    }
  }
  
  res.coefficients = beta;
  res.eta = eta;
  res.fitted_values = mu;  // Store final mu values (response scale)
  res.weights = wt;
  res.hessian = H;
  res.deviance = dev;
  res.null_deviance = null_dev;
  res.conv = conv;
  res.iter = static_cast<int>(iter + 1);
  res.coef_status = ws.valid_coefficients;

  if (keep_mx) {
    res.mx = MX;
    res.has_mx = true;
  }

  return res;
}

#endif // CAPYBARA_GLM
