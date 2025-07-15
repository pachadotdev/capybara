#ifndef CAPYBARA_GLM
#define CAPYBARA_GLM

struct SeparationResult {
  uvec separated_indices;
  bool has_separation;
  int n_separated;
};

struct GLMResult {
  vec coefficients;
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool conv;
  int iter;
  mat mx; // only if keep_mx = true
  bool has_mx = false;
  uvec coef_status; // 1 = valid, 0 = collinear
  
  // PPML-only
  vec residuals_working;
  vec residuals_response;
  mat scores;

  cpp11::list to_list(bool keep_mx = false) const {
    auto out =
        writable::list({"coefficients"_nm = as_doubles(coefficients),
                        "eta"_nm = as_doubles(eta),
                        "fitted.values"_nm = as_doubles(fitted_values),
                        "weights"_nm = as_doubles(weights),
                        "hessian"_nm = as_doubles_matrix(hessian),
                        "deviance"_nm = writable::doubles({deviance}),
                        "null_deviance"_nm = writable::doubles({null_deviance}),
                        "conv"_nm = writable::logicals({conv}),
                        "iter"_nm = writable::integers({iter}),
                        "coef_status"_nm = as_integers({coef_status})});
    if (keep_mx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(mx)});
    }
    return out;
  }
};

inline bool stopping_criterion(double a, double b, double diffMax) {
  double diff = fabs(a - b);
  return ((diff < diffMax) || (diff / (0.1 + fabs(a)) < diffMax));
}

inline bool convergence_check(const vec &x_new, const vec &x_old,
                              const vec &weights, double tol,
                              const std::string &family = "gaussian") {
  if (family == "poisson") {
    double ssr_new = dot(weights, square(x_new));
    double ssr_old = dot(weights, square(x_old));
    return std::abs(ssr_new - ssr_old) / (0.1 + std::abs(ssr_new)) < tol;
  } else {
    // Gaussian and others: weighted absolute difference
    double diff = dot(abs(x_new - x_old), weights);
    double total_weight = accu(weights);
    return (diff / total_weight) < tol;
  }
}

// Binomial GLM helpers

inline vec get_detadmu_binomial(const vec &mu) {
  vec mu_safe = clamp(mu, 1e-15, 1.0 - 1e-15);
  return 1.0 / (mu_safe % (1.0 - mu_safe));
}

inline vec get_V_binomial(const vec &mu) {
  vec mu_safe = clamp(mu, 1e-15, 1.0 - 1e-15);
  return mu_safe % (1.0 - mu_safe);
}

inline vec get_mu_binomial(const vec &eta) { return 1.0 / (1.0 + exp(-eta)); }

inline double get_deviance_binomial(const vec &y, const vec &mu,
                                    const vec &wt) {
  vec mu_safe = clamp(mu, 1e-15, 1.0 - 1e-15);
  vec log_mu = log(mu_safe);
  vec log_1_minus_mu = log(1.0 - mu_safe);
  return -2.0 * accu(wt % (y % log_mu + (1.0 - y) % log_1_minus_mu));
}

// PPML separation helpers

// Separation checking functions for Poisson GLM
inline uvec check_separation_fe(const vec &y,
                                const field<field<uvec>> &group_indices) {
  // No fixed effects, no separation
  if (group_indices.n_elem == 0) {
    return uvec();
  }

  // No boundary observations, no separation
  if (all(y > 0)) {
    return uvec();
  }

  // Create indicator for y > 0
  uvec y_positive = (y > 0);
  uvec separated_indices;

  // Loop over each fixed effect category
  for (size_t fe_idx = 0; fe_idx < group_indices.n_elem; ++fe_idx) {
    const field<uvec> &current_fe = group_indices(fe_idx);

    // For each level of the fixed effect
    for (size_t level = 0; level < current_fe.n_elem; ++level) {
      const uvec &group_obs = current_fe(level);

      if (group_obs.n_elem == 0) continue;

      // Check if this group has only positive y values
      uvec y_group = y_positive.elem(group_obs);
      arma::uword sum_positive = sum(y_group);
      arma::uword group_size = group_obs.n_elem;

      // Separation condition:
      // fixed effect level has only observations with y > 0
      // while the overall sample has some Y = 0 observations
      if (sum_positive == group_size && sum_positive > 0) {
        separated_indices = join_cols(separated_indices, group_obs);
      }
    }
  }

  // Remove duplicates and sort
  if (separated_indices.n_elem > 0) {
    separated_indices = unique(separated_indices);
  }

  return separated_indices;
}

SeparationResult check_separation_fe(const vec &y, const mat &X, const umat &fe,
                                     const std::vector<std::string> &fe_names) {
  SeparationResult result;
  result.has_separation = false;
  result.n_separated = 0;

  if (fe.n_cols == 0) {
    result.separated_indices = uvec();
    return result;
  }

  // Check if all y > 0 (no boundary observations)
  if (all(y > 0)) {
    result.separated_indices = uvec();
    return result;
  }

  // Create indicator for Y > 0
  uvec y_positive = (y > 0);
  uvec separated_indices;

  // Loop over all fixed effect dimensions
  for (size_t fe_dim = 0; fe_dim < fe.n_cols; fe_dim++) {
    const uvec &fe_col = fe.col(fe_dim);

    // unique FE levels
    uvec unique_fe = unique(fe_col);

    for (size_t fe_level : unique_fe) {
      uvec fe_mask = (fe_col == fe_level);

      // Count observations with y=0 and y>0 for this FE level
      int n_zero = sum(fe_mask % (1 - y_positive));
      int n_positive = sum(fe_mask % y_positive);

      // Separation occurs if FE level has only y=0 or only y>0 observations
      if ((n_zero > 0 && n_positive == 0) || (n_zero == 0 && n_positive > 0)) {
        uvec separated_obs = find(fe_col == fe_level);
        separated_indices = join_cols(separated_indices, separated_obs);
        result.has_separation = true;
      }
    }
  }

  if (separated_indices.n_elem > 0) {
    result.separated_indices = unique(separated_indices);
  } else {
    result.separated_indices = uvec();
  }

  result.n_separated = result.separated_indices.n_elem;
  return result;
}

// Iterative Rectifier separation check
SeparationResult check_separation_ir(const vec &y, const mat &X, const umat &fe,
                                     double tol = 1e-4, int maxiter = 100) {
  SeparationResult result;
  result.has_separation = false;
  result.n_separated = 0;

  // Check if all Y > 0 (no boundary observations)
  uvec is_interior = (y > 0);
  if (all(is_interior)) {
    return result;
  }
  vec U = conv_to<vec>::from(1 - is_interior);  // U = (y == 0)

  // Initialize weights using vectorized operations
  int N0 = sum(is_interior);
  double K = static_cast<double>(N0) / (tol * tol);
  vec omega = conv_to<vec>::from(is_interior) * K +
              (1.0 - conv_to<vec>::from(is_interior));

  bool has_converged = false;

  for (int iter = 0; iter < maxiter; iter++) {
    // Solve weighted regression: U ~ X + fe with weights omega
    // This is a simplified version - in practice would need full FE solver
    mat X_weighted = X.each_col() % sqrt(omega);
    vec U_weighted = U % sqrt(omega);

    // Simple OLS solution (ignoring FE for now - would need proper FE solver)
    vec beta_hat;
    if (X.n_cols > 0) {
      beta_hat =
          solve(X_weighted.t() * X_weighted, X_weighted.t() * U_weighted);
    } else {
      beta_hat = vec(0, fill::none);
    }

    // Predict U_hat
    vec U_hat = X * beta_hat;

    // Update U_hat based on tolerance
    uvec within_zero = (U_hat > -0.1 * tol) && (U_hat < tol);
    for (size_t i = 0; i < U_hat.n_elem; i++) {
      if (is_interior(i) || within_zero(i)) {
        U_hat(i) = 0;
      }
    }

    // Check convergence
    if (all(U_hat >= 0)) {
      has_converged = true;

      // Find separated observations using Armadillo
      uvec separated_obs = find(U_hat > 0);
      result.separated_indices = separated_obs;
      result.has_separation = !separated_obs.is_empty();
      break;
    }

    // Update U with ReLU activation using vectorized operations
    uvec exterior_mask = (1 - is_interior);
    U = U % conv_to<vec>::from(is_interior) +
        clamp(U_hat, 0.0, datum::inf) % conv_to<vec>::from(exterior_mask);
  }

  if (!has_converged) {
    // TODO: could add logging here
  }

  result.n_separated = result.separated_indices.n_elem;
  return result;
}

// Combined separation checking function
SeparationResult check_for_separation(
    const vec &y, const mat &X, const umat &fe,
    const std::vector<std::string> &fe_names,
    const std::vector<std::string> &methods = {"fe", "ir"}) {
  SeparationResult combined_result;
  combined_result.has_separation = false;
  combined_result.n_separated = 0;

  uvec all_separated;  // Use uvec instead of std::vector

  for (const std::string &method : methods) {
    SeparationResult method_result;

    if (method == "fe") {
      method_result = check_separation_fe(y, X, fe, fe_names);
    } else if (method == "ir") {
      method_result = check_separation_ir(y, X, fe);
    } else {
      continue;  // Skip unknown methods
    }

    // Combine results using Armadillo join
    all_separated = join_cols(all_separated, method_result.separated_indices);

    if (method_result.has_separation) {
      combined_result.has_separation = true;
    }
  }

  // Remove duplicates using Armadillo unique
  if (all_separated.n_elem > 0) {
    combined_result.separated_indices = unique(all_separated);
  } else {
    combined_result.separated_indices = uvec();
  }

  combined_result.n_separated = combined_result.separated_indices.n_elem;

  return combined_result;
}

// Main separation checking function
inline uvec check_separation(const vec &y,
                             const field<field<uvec>> &group_indices,
                             const std::vector<std::string> &methods = {"fe"}) {
  uvec all_separated;

  for (const auto &method : methods) {
    if (method == "fe") {
      uvec fe_separated = check_separation_fe(y, group_indices);
      all_separated = join_cols(all_separated, fe_separated);
    }
    // TODO: add the "ir" method for iterative rectifier separation checking
  }

  if (all_separated.n_elem > 0) {
    all_separated = unique(all_separated);
  }

  return all_separated;
}

// PPML helpers

// Compute PPML deviance (porting compute_deviance from fepois_.py)
double compute_ppml_deviance(const vec &y, const vec &mu) {
  vec dev_terms(y.n_elem);

  for (size_t i = 0; i < y.n_elem; i++) {
    if (y(i) == 0) {
      dev_terms(i) = 0;
    } else {
      dev_terms(i) = y(i) * std::log(y(i) / mu(i));
    }
    dev_terms(i) -= (y(i) - mu(i));
  }

  return 2.0 * sum(dev_terms);
}

// Core PPML IRLS algorithm based on ppmlhdfe
GLMResult
fit_ppml_irls(const vec &y, const mat &X, const umat &fe,
              const std::vector<std::string> &fe_names, double tol, int maxiter,
              double fixef_tol, int fixef_maxiter,
              const std::vector<std::string> &separation_methods = {}) {
  GLMResult result;
  result.conv = false;
  result.iter = 0;

  size_t p = X.n_cols;

  uvec coef_status = ones<uvec>(p);

  // Check for separation
  if (!separation_methods.empty()) {
    SeparationResult sep_result =
        check_for_separation(y, X, fe, fe_names, separation_methods);
    if (sep_result.has_separation) {
      // TODO: would need to remove separated observations
    }
  }

  // y must be non-negative for Poisson regression
  if (any(y < 0)) {
    return result;
  }

  double mean_y = mean(y);
  vec fitted_values = (y + mean_y) / 2.0; // Initial fitted_values
  vec eta = log(fitted_values);           // Initial eta (log link)

  double last_deviance = compute_ppml_deviance(y, fitted_values);
  double crit = 1.0;
  bool stop_iterating = false;

  // IRLS loop
  for (int iter = 0; iter < maxiter; iter++) {
    result.iter = iter + 1;

    if (stop_iterating) {
      result.conv = true;
      break;
    }

    // Step 1: Compute working dependent variable Z
    vec Z = eta + y / fitted_values - 1.0;
    vec reg_Z = Z;             

    // Step 2: Weighted demeaning
    mat ZX = join_rows(reg_Z, X); // Concatenate Z and X

    WeightedDemeanResult demean_result =
        weighted_demean(ZX, fe, fitted_values, fixef_tol, fixef_maxiter);
    if (!demean_result.success) {
      // Demeaning failed
      return result;
    }

    vec Z_resid = demean_result.demeaned_data.col(0);
    mat X_resid = demean_result.demeaned_data.cols(1, p);

    // Step 3: Weighted least squares estimation
    mat WX = X_resid.each_col() % sqrt(fitted_values);
    vec WZ = Z_resid % sqrt(fitted_values);           

    mat XWX = WX.t() * WX;
    vec XWZ = WX.t() * WZ;

    // Solve for coefficient update
    beta_results ws(WX.n_rows, WX.n_cols);
    vec delta_new = get_beta(WX, WZ, fitted_values, WX.n_rows, WX.n_cols, ws, true);

    // Track collinearity status
    for (size_t i = 0; i < coef_status.n_elem; ++i) {
      if (ws.valid_coefficients(i) == 0) {
        coef_status(i) = 0;
        result.coefficients(i) = 0.0;
      }
    }

    vec resid = Z_resid - X_resid * delta_new;

    // Step 4: Update mu and eta
    vec mu_old = fitted_values;
    eta = Z - resid;

    fitted_values = exp(eta);

    // Step 5: Check convergence using same criterion as fixest
    double deviance = compute_ppml_deviance(y, fitted_values);
    crit = std::abs(deviance - last_deviance) / (0.1 + std::abs(last_deviance));
    last_deviance = deviance;

    stop_iterating = (crit < tol);

    // Store intermediate results
    result.coefficients = delta_new;
    result.fitted_values = fitted_values;
    result.eta = eta;
    result.weights = mu_old;
    result.deviance = deviance;
    result.hessian = XWX;

    result.residuals_working = WZ - WX * delta_new;
    result.residuals_response = y - exp(eta);

    result.scores = WX.each_col() % result.residuals_working;
  }

  if (!result.conv && result.iter == maxiter) {
    // TODO: add algorithm did not converge warning
  }

  // Set final collinearity status
  result.coef_status = coef_status;

  // Compute null deviance for consistency with GLMResult
  double y_mean = mean(y);
  vec fitted_values_null(y.n_elem, fill::value(y_mean));
  result.null_deviance = compute_ppml_deviance(y, fitted_values_null);

  return result;
}

// Helper function to initialize PPML with better starting values
// vec get_ppml_starting_values(const vec &y, const mat &X) {
//   // Use simple Poisson regression for starting values
//   vec log_y_plus = log(y + 0.1); // Add small constant to avoid log(0)

//   if (X.n_cols > 0) {
//     return solve(X.t() * X, X.t() * log_y_plus);
//   } else {
//     return vec(0, fill::none);
//   }
// }

// // Compute robust variance-covariance matrix for PPML
// mat compute_ppml_vcov(const mat &X_weighted, const vec &residuals,
//                       const mat &hessian) {
//   // Sandwich variance estimator: (X'WX)^{-1} (X'WUU'WX) (X'WX)^{-1}
//   mat meat = X_weighted.each_col() % residuals;
//   meat = meat.t() * meat;

//   mat bread_inv = pinv(hessian);
//   return bread_inv * meat * bread_inv;
// }

// GLM fitting

// Main GLM fitting function with family-specific algorithm dispatch
inline GLMResult
feglm_fit(mat &MX, vec &beta, vec &eta, const vec &y, const vec &wt,
          const double &theta, const field<field<uvec>> &group_indices,
          double center_tol, double dev_tol, bool keep_mx, size_t iter_max,
          size_t iter_center_max, size_t iter_inner_max, size_t iter_interrupt,
          size_t iter_ssr, const std::string &fam, FamilyType family_type) {
  GLMResult result;

  if (family_type == POISSON) {
    // Convert group_indices to format expected by PPML functions
    std::vector<std::string> fe_names;
    umat fe_matrix;

    // Create FE matrix and names from group_indices
    if (group_indices.n_elem > 0) {
      size_t n_obs = y.n_elem;
      fe_matrix.set_size(n_obs, group_indices.n_elem);

      for (size_t k = 0; k < group_indices.n_elem; k++) {
        fe_names.push_back("fe_" + std::to_string(k));

        // Initialize this FE dimension to 0
        fe_matrix.col(k).fill(0);

        // Set FE levels based on group indices
        for (size_t g = 0; g < group_indices(k).n_elem; g++) {
          const uvec &group_obs = group_indices(k)(g);
          for (size_t i = 0; i < group_obs.n_elem; i++) {
            fe_matrix(group_obs(i), k) = g;
          }
        }
      }
    }

    // Run PPML algorithm and return result directly
    GLMResult result = fit_ppml_irls(
        y, MX, fe_matrix, fe_names, dev_tol, static_cast<int>(iter_max),
        center_tol, static_cast<int>(iter_center_max));

    if (keep_mx) {
      result.mx = MX;
      result.has_mx = true;
    }

    return result;
  } else {
    // Use standard IRLS algorithm for non-Poisson families

    size_t n = y.n_elem;
    size_t p = MX.n_cols;

    // for collinearity tracking
    uvec coef_status = ones<uvec>(p);

    vec mu = link_inv_(eta, family_type);
    vec mu_eta(n), w(n), nu(n), beta_upd(p), eta_upd(n), eta_old(n),
        beta_old(p);
    mat H(p, p);

    double dev = dev_resids_(y, mu, theta, wt, family_type);
    double ymean = sum(wt % y) / sum(wt);
    vec mu_null(n, fill::value(ymean));
    double null_dev = dev_resids_(y, mu_null, theta, wt, family_type);

    double dev_old, dev_ratio, dev_ratio_inner, rho;
    bool dev_crit, val_crit, imp_crit, conv = false;
    size_t iter, iter_inner;

    // IRLS main loop
    for (iter = 0; iter < iter_max; ++iter) {
      rho = 1.0;
      eta_old = eta;
      beta_old = beta;
      dev_old = dev;

      // Compute weights and working response
      mu_eta = mu_eta_(eta, family_type);
      w = (wt % square(mu_eta)) / variance_(mu, theta, family_type);
      nu = (y - mu) / mu_eta + eta;

      // Center variables for fixed effects
      vec MNU = nu;
      mat MNU_mat = MNU;
      demean_variables(MNU_mat, w, group_indices, center_tol, iter_center_max,
                       fam);
      MNU = MNU_mat.col(0);
      demean_variables(MX, w, group_indices, center_tol, iter_center_max, fam);

      // Compute coefficient update
      beta_results ws(MX.n_rows, MX.n_cols);
      beta_upd = get_beta(MX, MNU, w, MX.n_rows, MX.n_cols, ws, true);

      // Track collinearity status - consistent with PPML approach
      for (size_t i = 0; i < coef_status.n_elem; ++i) {
        if (ws.valid_coefficients(i) == 0) {
          coef_status(i) = 0;
          beta_upd(i) = 0.0;
        }
      }

      eta_upd = MX * beta_upd + nu - MNU;

      // Step-halving with convergence checks
      for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
        eta = eta_old + rho * eta_upd;
        beta = beta_old + rho * beta_upd;
        mu = link_inv_(eta, family_type);
        dev = dev_resids_(y, mu, theta, wt, family_type);
        dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev_old));

        dev_crit = is_finite(dev);
        val_crit = valid_eta_(eta, family_type) && valid_mu_(mu, family_type);
        imp_crit = (dev_ratio_inner <= -dev_tol);

        if (dev_crit && val_crit && imp_crit)
          break;
        rho *= 0.5;
      }

      if (!dev_crit || !val_crit) {
        conv = false;
        break;
      }

      // Check outer loop convergence
      dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));
      if (dev_ratio < dev_tol) {
        conv = true;
        break;
      }
    }

    // Compute final quantities
    mu_eta = mu_eta_(eta, family_type);
    w = (wt % square(mu_eta)) / variance_(mu, theta, family_type);

    // Hessian computation
    H = crossprod_(MX, w);

    // Set up result
    result.coefficients = beta;
    result.eta = eta;
    result.fitted_values = mu;
    result.weights = w;
    result.hessian = H;
    result.deviance = dev;
    result.null_deviance = null_dev;
    result.conv = conv;
    result.iter = static_cast<int>(iter);
    result.coef_status = coef_status;

    if (keep_mx) {
      result.mx = MX;
      result.has_mx = true;
    }

    return result;
  }
}

#endif // CAPYBARA_GLM
