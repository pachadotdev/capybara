// Computing generalized linear models with fixed effects
// eta = X beta + alpha + offset

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {

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

  // Adaptive tolerance variables - more aggressive for large models
  double hdfe_tolerance = params.center_tol;
  double highest_inner_tol = std::max(1e-12, std::min(params.center_tol, 0.1 * params.dev_tol));
  double start_inner_tol = params.start_inner_tol;
  
  // Model size-based initial tolerance
  bool is_large_model = (n > 100000) || (p > 1000) || 
                        (has_fixed_effects && fe_groups.n_elem > 1);
  
  if (has_fixed_effects) {
    if (is_large_model) {
      // Much looser initial tolerance for large models
      hdfe_tolerance = std::max(1e-3, std::max(start_inner_tol, params.dev_tol * 10));
    } else {
      hdfe_tolerance = std::max(start_inner_tol, params.dev_tol);
    }
  }

  vec MNU_increment = vec(n, fill::zeros);
  bool use_partial = false; // Partial out variables
  
  // Step halving with memory variables
  double step_halving_memory = params.step_halving_memory;
  size_t num_step_halving = 0;
  size_t max_step_halving = params.max_step_halving;
  
  // Convergence history prediction
  vec eps_history(3, fill::value(datum::inf));
  double predicted_eps = datum::inf;
  
  // Adaptive tolerance scheduling
  size_t convergence_count = 0;
  double last_dev_ratio = datum::inf;

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

    // Calculate current HDFE tolerance with model size awareness
    bool iter_solver = has_fixed_effects && (hdfe_tolerance > params.dev_tol * 11) && 
                           (predicted_eps > params.dev_tol);
    
    // More aggressive tolerance management for large models
    double current_hdfe_tol;
    if (is_large_model) {
      if (iter < 5) {
        // Very loose tolerance in early iterations for large models
        current_hdfe_tol = std::max(hdfe_tolerance, 1e-2);
      } else if (iter_solver) {
        // Still use fast solver but with tighter bound
        current_hdfe_tol = std::min(hdfe_tolerance, 1e-3);
      } else {
        // Tighten as we converge
        current_hdfe_tol = std::min(hdfe_tolerance, highest_inner_tol);
      }
    } else {
      // Standard tolerance scheduling for smaller models
      current_hdfe_tol = iter_solver ? hdfe_tolerance : 
                         std::min(hdfe_tolerance, highest_inner_tol);
    }

    // Partial out: only update the increment after first iteration
    // Following ppmlhdfe.mata approach: use incremental updates for speed
    if (use_partial && iter > 1) {
      MNU_increment = nu - nu0;
      MNU += MNU_increment;
      
      if (has_fixed_effects) {
        center_variables(MNU, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start, params.use_cg);
        center_variables(X, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start, params.use_cg);
      }
      nu0 = nu;
    } else {
      // Full update: data = (z, x) in ppmlhdfe.mata
      MNU += (nu - nu0);
      nu0 = nu;
      
      if (has_fixed_effects) {
        center_variables(MNU, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start, params.use_cg);
        center_variables(X, w_working, fe_groups, current_hdfe_tol,
                         params.iter_center_max, params.iter_interrupt,
                         params.iter_ssr, params.accel_start, params.use_cg);
      }
    }
    
    // Enable partial out after first iteration
    if (params.use_acceleration) {
      use_partial = true;
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

    // Check convergence with history tracking
    dev_ratio = fabs(dev - dev0) / (0.1 + fabs(dev));
    double delta_deviance = dev0 - dev;
    
    // Track convergence for large models
    if (is_large_model && dev_ratio < last_dev_ratio * 0.5) {
      convergence_count++;
    } else {
      convergence_count = 0;
    }
    last_dev_ratio = dev_ratio;
    
    // Update convergence history if acceleration is enabled
    if (params.use_acceleration) {
      eps_history(0) = eps_history(1);
      eps_history(1) = eps_history(2);
      eps_history(2) = dev_ratio;
      
      // Predict next epsilon for adaptive tolerance decisions
      predicted_eps = predict_convergence(eps_history, dev_ratio);
    }
     
    if (dev_ratio < params.dev_tol) {
      conv = true;
      break;
    }
    
    // Post-convergence step-halving following ppmlhdfe.mata exactly
    if (delta_deviance < 0 && num_step_halving < max_step_halving) {
      // Run step-halving after checking for convergence as in ppmlhdfe.mata
      if (params.use_acceleration) {
        eta = step_halving_memory * eta0 + (1.0 - step_halving_memory) * eta;
      } else {
        eta = params.step_halving_factor * eta0 + (1.0 - params.step_halving_factor) * eta;
      }
      
      // If the first step halving was not enough, clip very low values of eta
      if (num_step_halving > 0 && family_type == POISSON) {
        eta = arma::max(eta, vec(n, fill::value(-10.0)));
      }
      
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, theta, w, family_type);
      num_step_halving++;
      
      // Continue to next iteration without breaking
      result.iter = iter + 1;
      continue;
    } else {
      // Reset step halving counter 
      num_step_halving = 0;
    }

    // Adaptive HDFE tolerance update with model size awareness
    if (has_fixed_effects && params.use_acceleration) {
      if (is_large_model) {
        // More aggressive tolerance scheduling for large models
        if (convergence_count >= 3 || dev_ratio < hdfe_tolerance * 0.1) {
          // Rapid convergence detected, tighten significantly
          hdfe_tolerance = std::max(highest_inner_tol, hdfe_tolerance * 0.01);
        } else if (dev_ratio < hdfe_tolerance) {
          // Normal tightening for large models
          double alt_tol = std::pow(10.0, -std::ceil(std::log10(1.0 / std::max(0.1 * dev_ratio, 1e-16))));
          hdfe_tolerance = std::max(std::min(0.1 * hdfe_tolerance, alt_tol), highest_inner_tol);
        }
      } else {
        // Standard tolerance update for smaller models
        if (dev_ratio < hdfe_tolerance) {
          double alt_tol = std::pow(10.0, -std::ceil(std::log10(1.0 / std::max(0.1 * dev_ratio, 1e-16))));
          hdfe_tolerance = std::max(std::min(0.1 * hdfe_tolerance, alt_tol), highest_inner_tol);
        }
      }
    }

    result.iter = iter + 1;
  }

  // Final computations if converged
  if (conv) {
    // Compute final Hessian using optimized helper
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

  // Adaptive tolerance for offset fitting
  double adaptive_tol = params.center_tol;
  if (n > 100000) {
    adaptive_tol = std::max(params.center_tol, 1e-3);
  }

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

    // Use adaptive tolerance for centering
    center_variables(Myadj, w_working, fe_groups, adaptive_tol,
                     params.iter_center_max, params.iter_interrupt,
                     params.iter_ssr, params.accel_start, params.use_cg);

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
    
    // Tighten tolerance as we converge
    if (n > 100000 && iter > 5 && dev_ratio < 0.1) {
      adaptive_tol = params.center_tol;
    }
    
    if (dev_ratio < params.dev_tol) {
      break;
    }

    // Update starting guesses for acceleration
    Myadj = Myadj - yadj;
  }

  return eta;
}

} // namespace capybara

#endif // CAPYBARA_GLM_H
