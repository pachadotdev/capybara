// Computing generalized linear models with fixed effects
// eta = X beta + alpha + offset

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {
namespace glm {

using demean::demean_variables;
using demean::DemeanResult;
using lm::felm_fit;
using lm::InferenceLM;
using parameters::get_alpha;
using parameters::get_beta;
using parameters::InferenceAlpha;
using parameters::InferenceBeta;

// Use convergence family types to avoid duplication
using convergence::Family;
using convergence::utils::is_poisson_family;
using convergence::utils::safe_divide;
using convergence::utils::safe_log;

//////////////////////////////////////////////////////////////////////////////
// FAMILY TYPE MAPPING
//////////////////////////////////////////////////////////////////////////////

// Map convergence::Family to the GLM FamilyType for compatibility
inline Family string_to_family(const std::string &fam) {
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", Family::GAUSSIAN},
      {"poisson", Family::POISSON},
      {"binomial", Family::BINOMIAL},
      {"gamma", Family::GAMMA},
      {"inverse_gaussian", Family::INV_GAUSSIAN},
      {"negative_binomial", Family::NEGBIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second
                                  : Family::GAUSSIAN; // Default fallback
}

//////////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

inline std::string tidy_family(const std::string &family) {
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

inline Family get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", Family::GAUSSIAN},
      {"poisson", Family::POISSON},
      {"binomial", Family::BINOMIAL},
      {"gamma", Family::GAMMA},
      {"inverse_gaussian", Family::INV_GAUSSIAN},
      {"negative_binomial", Family::NEGBIN}};

  auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : Family::GAUSSIAN;
}

//////////////////////////////////////////////////////////////////////////////
// LINK FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Link inverse functions (eta -> mu)
inline vec link_inv_gaussian(const vec &eta) {
  return eta; // identity link
}

inline vec link_inv_poisson(const vec &eta) {
  return exp(eta); // log link
}

inline vec link_inv_logit(const vec &eta) {
  return 1.0 / (1.0 + exp(-eta)); // logit link
}

inline vec link_inv_gamma(const vec &eta) {
  return 1.0 / eta; // reciprocal link: mu = 1/eta
}

inline vec link_inv_invgaussian(const vec &eta) {
  return 1.0 / sqrt(eta); // inverse squared link (matching old code)
}

inline vec link_inv_negbin(const vec &eta) {
  return exp(eta); // log link
}

inline vec link_inv(const vec &eta, const Family family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case Family::GAUSSIAN:
    result = link_inv_gaussian(eta);
    break;
  case Family::POISSON:
    result = link_inv_poisson(eta);
    break;
  case Family::BINOMIAL:
    result = link_inv_logit(eta);
    break;
  case Family::GAMMA:
    result = link_inv_gamma(eta);
    break;
  case Family::INV_GAUSSIAN:
    result = link_inv_invgaussian(eta);
    break;
  case Family::NEGBIN:
    result = link_inv_negbin(eta);
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

// d mu / d eta (derivative of inverse link function)
inline vec d_inv_link(const vec &eta, const Family family_type) {
  vec result(eta.n_elem);

  switch (family_type) {
  case Family::GAUSSIAN:
    result.ones();
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    result = arma::exp(eta);
    break;
  case Family::BINOMIAL: {
    vec exp_eta = arma::exp(eta);
    result = exp_eta / arma::square(1 + exp_eta);
    break;
  }
  case Family::GAMMA:
    result = -1.0 / arma::square(eta);
    break;
  case Family::INV_GAUSSIAN:
    result = -1.0 / (2.0 * arma::pow(abs(eta), 1.5));
    break;
  default:
    stop("Unknown family");
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////
// INITIALIZATION FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Initialize mu and eta for different families
inline void initialize_family(vec &mu, vec &eta, const vec &y_orig,
                              double mean_y, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    mu = y_orig; // Identity link: mu = eta for Gaussian
    eta = y_orig;
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    mu = (y_orig + mean_y) / 2.0;
    eta = log(clamp(mu, 1e-12, 1e12));
    break;
  case Family::BINOMIAL:
    mu = clamp((y_orig + mean_y) / 2.0, 0.001, 0.999);
    eta = log(mu / (1.0 - mu)); // logit link
    break;
  case Family::GAMMA:
    mu = clamp(y_orig, 1e-12, 1e12);
    eta = 1.0 / mu; // reciprocal link
    break;
  case Family::INV_GAUSSIAN:
    mu = clamp(y_orig, 1e-12, 1e12);
    eta = 1.0 / square(mu); // inverse squared link
    break;
  default:
    stop("Unknown family");
  }
}

//////////////////////////////////////////////////////////////////////////////
// DEVIANCE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Deviance functions - matching Python exactly
inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &wt) {
  // Python fegaussian_.py: np.sum((y - mu) ** 2)
  return dot(wt, square(y - mu));
}

inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &wt) {
  // Standard Poisson deviance
  vec r = mu % wt;

  uvec p = find(y > 0);
  r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));

  return 2 * accu(r);
}

// Logit deviance matching R's base implementation exactly
inline double dev_resids_logit(const vec &y, const vec &mu, const vec &wt) {
  // Adapted from binomial_dev_resids() in R base src/library/stats/src/family.c
  vec r(y.n_elem, fill::none);

  uvec p = find(y == 1);
  uvec q = find(y == 0);

  if (p.n_elem > 0) {
    vec y_p = y(p);
    r(p) = y_p % log(y_p / mu(p));
  }

  if (q.n_elem > 0) {
    vec y_q = y(q);
    r(q) = (1 - y_q) % log((1 - y_q) / (1 - mu(q)));
  }

  return 2.0 * dot(wt, r);
}

inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &wt) {
  // Standard Gamma deviance: -2 * sum(log(y/mu) - (y-mu)/mu)
  vec y_safe = clamp(y, 1e-15, arma::datum::inf);
  vec mu_safe = clamp(mu, 1e-15, arma::datum::inf);

  vec terms = log(y_safe / mu_safe) - (y - mu) / mu_safe;
  return -2.0 * dot(wt, terms);
}

inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &wt) {
  return dot(wt, square(y - mu) / (y % square(mu)));
}

inline double dev_resids_negbin(const vec &y, const vec &mu,
                                const double &theta, const vec &wt) {
  vec r = y;

  uvec p = find(y < 1);
  r.elem(p).fill(1.0);
  r = wt % (y % log(r / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2 * accu(r);
}

inline double dev_resids(const vec &y, const vec &mu, const double &theta,
                         const vec &wt, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return dev_resids_gaussian(y, mu, wt);
  case Family::POISSON:
    return dev_resids_poisson(y, mu, wt);
  case Family::BINOMIAL:
    return dev_resids_logit(y, mu, wt);
  case Family::GAMMA:
    return dev_resids_gamma(y, mu, wt);
  case Family::INV_GAUSSIAN:
    return dev_resids_invgaussian(y, mu, wt);
  case Family::NEGBIN:
    return dev_resids_negbin(y, mu, theta, wt);
  default:
    stop("Unknown family");
  }
}

//////////////////////////////////////////////////////////////////////////////
// VALIDATION FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

inline bool valid_eta(const vec &eta, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
  case Family::POISSON:
  case Family::BINOMIAL:
  case Family::NEGBIN:
    return is_finite(eta);
  case Family::GAMMA:
    return is_finite(eta) &&
           all(eta != 0.0); // reciprocal link can't have eta=0
  case Family::INV_GAUSSIAN:
    return is_finite(eta) &&
           all(eta > 0.0); // inverse squared link needs eta > 0
  default:
    stop("Unknown family");
  }
}

inline bool valid_mu(const vec &mu, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return is_finite(mu);
  case Family::POISSON:
  case Family::NEGBIN:
    return is_finite(mu) && all(mu > 0);
  case Family::BINOMIAL:
    return is_finite(mu) && all(mu > 0 && mu < 1);
  case Family::GAMMA:
    return is_finite(mu) && all(mu > 0.0);
  case Family::INV_GAUSSIAN:
    return is_finite(mu) && all(mu > 0.0);
  default:
    stop("Unknown family");
  }
}

// Variance function V(mu)
inline vec variance(const vec &mu, const double &theta,
                    const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return ones<vec>(mu.n_elem);
  case Family::POISSON:
    return mu;
  case Family::BINOMIAL:
    return mu % (1 - mu);
  case Family::GAMMA:
    return square(mu);
  case Family::INV_GAUSSIAN:
    return pow(mu, 3.0);
  case Family::NEGBIN:
    return mu + square(mu) / theta;
  default:
    stop("Unknown family");
  }
}

//////////////////////////////////////////////////////////////////////////////
// RESULT STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// GLM fitting result structure
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

  // PPML-specific residuals
  vec residuals_working;
  vec residuals_response;

  // Fixed effects info
  field<vec> fixed_effects;
  uvec nb_references; // Number of references per dimension
  bool is_regular;    // Whether fixed effects are regular
  bool has_fe = false;
  uvec iterations;

  // Optional matrix storage
  mat mx;
  bool has_mx = false;

  InferenceGLM(size_t n, size_t p)
      : coefficients(p, fill::none), eta(n, fill::none),
        fitted_values(n, fill::none), weights(n, fill::none),
        hessian(p, p, fill::none), deviance(0.0), null_deviance(0.0),
        conv(false), iter(0), coef_status(p, fill::none),
        residuals_working(n, fill::none), residuals_response(n, fill::none),
        is_regular(true), has_fe(false), has_mx(false) {}

  cpp11::list to_list(bool keep_mx = true) const {
    auto out = writable::list(
        {"coefficients"_nm = as_doubles(coefficients),
         "eta"_nm = as_doubles(eta),
         "fitted.values"_nm = as_doubles(fitted_values),
         "weights"_nm = as_doubles(weights),
         "hessian"_nm = as_doubles_matrix(hessian),
         "deviance"_nm = writable::doubles({deviance}),
         "null.deviance"_nm = writable::doubles({null_deviance}),
         "conv"_nm = writable::logicals({conv}),
         "iter"_nm = writable::integers({static_cast<int>(iter)})});

    if (has_fe && fixed_effects.n_elem > 0) {
      writable::list fe_list(fixed_effects.n_elem);
      for (size_t k = 0; k < fixed_effects.n_elem; ++k) {
        fe_list[k] = as_doubles(fixed_effects(k));
      }
      out.push_back({"fixed.effects"_nm = fe_list});
      out.push_back({"nb_references"_nm = as_integers(nb_references)});
      out.push_back({"is_regular"_nm = writable::logicals({is_regular})});
    }

    if (keep_mx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(mx)});
    }

    return out;
  }
};

//////////////////////////////////////////////////////////////////////////////
// FIXED EFFECTS EXTRACTION
//////////////////////////////////////////////////////////////////////////////

// Extract fixed effects from GLM results
inline field<vec> extract_glm_fixed_effects(const vec &eta, const mat &X_orig,
                                            const vec &coefficients,
                                            const field<uvec> &fe_indices,
                                            const uvec &nb_ids,
                                            const field<uvec> &fe_id_tables) {
  const size_t Q = fe_indices.n_elem;
  field<vec> fixed_effects(Q);

  if (Q == 0 || fe_indices(0).n_elem == 0) {
    return fixed_effects;
  }

  // Compute fixed effects component
  vec fe_component;
  if (X_orig.n_cols > 0 && coefficients.n_elem > 0) {
    vec linear_part = X_orig * coefficients;
    fe_component = eta - linear_part;
  } else {
    fe_component = eta;
  }

  // Call the unified get_alpha function
  // Convert fe_indices to field<field<uvec>> format expected by get_alpha
  field<field<uvec>> group_indices(fe_indices.n_elem);
  for (size_t k = 0; k < fe_indices.n_elem; ++k) {
    group_indices(k).set_size(1);
    group_indices(k)(0) = fe_indices(k);
  }

  InferenceAlpha alpha_result = get_alpha(fe_component, group_indices);
  return alpha_result.Alpha;
}

//////////////////////////////////////////////////////////////////////////////
// GLM FITTING
//////////////////////////////////////////////////////////////////////////////

inline InferenceGLM feglm_fit(
    const mat &X_orig, const vec &y_orig, const vec &w,
    const field<uvec> &fe_indices, const uvec &nb_ids,
    const field<uvec> &fe_id_tables, double center_tol, size_t iter_center_max,
    size_t iter_interrupt, size_t iter_ssr, double collin_tol, double dev_tol,
    size_t iter_max, size_t iter_inner_max, const std::string &family,
    bool keep_mx, bool use_weights = false,
    // Algorithm parameters
    double direct_qr_threshold = 0.9, double qr_collin_tol_multiplier = 1.0,
    double chol_stability_threshold = 1e-12, double safe_division_min = 1e-12,
    double safe_log_min = 1e-12, double newton_raphson_tol = 1e-8,
    // Demean algorithm parameters
    size_t demean_extra_projections = 0, size_t demean_warmup_iterations = 15,
    size_t demean_projections_after_acc = 5,
    size_t demean_grand_acc_frequency = 20,
    size_t demean_ssr_check_frequency = 40,
    // Convergence algorithm parameters
    double irons_tuck_eps = 1e-14, double alpha_convergence_tol = 1e-8,
    size_t alpha_iter_max = 10000) {
  const size_t n = y_orig.n_elem;
  const size_t p_orig = X_orig.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceGLM result(n, p_orig);

  // Get family type
  std::string fam = tidy_family(family);
  Family family_type = get_family_type(fam);

  // Validate response for all families
  if (!valid_mu(y_orig, family_type)) {
    result.conv = false;
    return result;
  }

  // Initialize
  double mean_y = sum(w % y_orig) / sum(w);
  vec mu;
  vec eta;

  // Check if we actually have weights
  bool glm_weights = true;

  // Initialize mu and eta using helper function
  initialize_family(mu, eta, y_orig, mean_y, family_type);

  result.deviance = dev_resids(y_orig, mu, 0.0, w, family_type);

  // Storage for IRLS algorithm
  vec eta_old, z, working_weights;
  double devold = result.deviance;
  uvec coef_status = ones<uvec>(p_orig);

  mat X_demean;
  vec Y_demean;
  DemeanResult y_demean_result(0);

  // IRLS loop
  for (size_t iter = 0; iter < iter_max; iter++) {
    result.iter = iter + 1;
    eta_old = eta;

    // Compute mu.eta and variance using exponential family functions
    vec mu_eta_val = d_inv_link(eta, family_type);
    vec var_mu = variance(mu, 0.0, family_type);

    // Check variance
    if (any(var_mu <= 0) || any(var_mu != var_mu)) {
      break;
    }

    // Working response and weights
    z = eta + (y_orig - mu) / mu_eta_val;
    working_weights = w % square(mu_eta_val) / var_mu;

    // Check weights
    uvec zero_w = (working_weights <= 0);
    if (any(zero_w) && all(zero_w)) {
      break;
    }

    // Demean variables if we have fixed effects
    if (has_fixed_effects) {
      // Prepare data for demeaning: combine Y and X
      field<vec> variables_to_demean(p_orig + 1);
      variables_to_demean(0) = z; // Working response first
      for (size_t j = 0; j < p_orig; ++j) {
        variables_to_demean(j + 1) = X_orig.col(j); // Then X columns
      }

      y_demean_result = demean_variables(
          variables_to_demean, w, fe_indices, nb_ids, fe_id_tables,
          iter_center_max, center_tol, demean_extra_projections,
          demean_warmup_iterations, demean_projections_after_acc,
          demean_grand_acc_frequency, demean_ssr_check_frequency, true,
          safe_division_min);

      // Extract demeaned Y and X
      Y_demean = y_demean_result.demeaned_vars(0);
      if (p_orig > 0) {
        X_demean.set_size(n, p_orig);
        for (size_t j = 0; j < p_orig; ++j) {
          X_demean.col(j) = y_demean_result.demeaned_vars(j + 1);
        }
      } else {
        X_demean = mat(n, 0);
      }
    } else {
      // No fixed effects
      Y_demean = z;
      X_demean = X_orig;
    }

    // Regression on demeaned data
    InferenceBeta beta_result =
        get_beta(X_demean, Y_demean, y_orig, working_weights, collin_tol,
                 glm_weights, has_fixed_effects, direct_qr_threshold,
                 qr_collin_tol_multiplier, chol_stability_threshold);

    if (!beta_result.success ||
        any(beta_result.coefficients != beta_result.coefficients)) {
      if (iter == 0) {
        result.conv = false;
        return result;
      }
      break;
    }

    coef_status = beta_result.coef_status;
    vec new_eta = beta_result.fitted_values;

    // Step halving
    double rho = 1.0;
    bool step_accepted = false;

    for (size_t iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + rho * (new_eta - eta_old);
      mu = link_inv(eta, family_type);
      double deviance = dev_resids(y_orig, mu, 0.0, w, family_type);

      double dev_ratio = (deviance - devold) / (0.1 + std::abs(devold));

      bool dev_finite = is_finite(deviance);
      bool mu_valid = valid_mu(mu, family_type);
      bool eta_valid = valid_eta(eta, family_type);
      bool dev_improved = (dev_ratio <= -dev_tol);

      if (dev_finite && mu_valid && eta_valid && dev_improved) {
        result.deviance = deviance;
        step_accepted = true;
        break;
      }

      rho *= 0.5;
    }

    if (!step_accepted) {
      eta = eta_old;
      mu = link_inv(eta, family_type);
      result.deviance = devold;
    }

    // Check convergence
    double dev_ratio =
        std::abs(result.deviance - devold) / (0.1 + std::abs(result.deviance));
    if (dev_ratio < dev_tol) {
      result.conv = true;
      break;
    }

    devold = result.deviance;
  }

  // Final results
  result.coefficients =
      get_beta(X_demean, Y_demean, y_orig, working_weights, collin_tol,
               glm_weights, has_fixed_effects, direct_qr_threshold,
               qr_collin_tol_multiplier, chol_stability_threshold)
          .coefficients;
  result.eta = eta;
  result.fitted_values = mu;
  result.weights = working_weights;
  result.coef_status = coef_status;

  // Hessian
  if (has_fixed_effects) {
    if (p_orig > 0) {
      result.hessian = X_demean.t() * (X_demean.each_col() % working_weights);
    } else {
      result.hessian = mat(0, 0);
    }
  } else {
    result.hessian = X_orig.t() * (X_orig.each_col() % working_weights);
  }

  // Null deviance
  vec mu_null(y_orig.n_elem, fill::value(mean_y));
  result.null_deviance = dev_resids(y_orig, mu_null, 0.0, w, family_type);

  if (keep_mx) {
    result.mx = X_orig;
    result.has_mx = true;
  }

  // Extract fixed effects if present
  if (has_fixed_effects) {
    result.fixed_effects = extract_glm_fixed_effects(
        eta, X_orig, result.coefficients, fe_indices, nb_ids, fe_id_tables);
    result.has_fe = true;
    result.is_regular = true;
    result.nb_references.set_size(fe_indices.n_elem);
    result.nb_references.fill(1);
  } else {
    result.has_fe = false;
  }

  return result;
}

} // namespace glm
} // namespace capybara

#endif // CAPYBARA_GLM_H
