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

inline vec link_inv(const vec &eta, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return eta;
  case Family::POISSON:
  case Family::NEGBIN:
    return exp(eta);
  case Family::BINOMIAL:
    return 1.0 / (1.0 + exp(-eta));
  case Family::GAMMA:
    return 1.0 / eta;
  case Family::INV_GAUSSIAN:
    return 1.0 / sqrt(eta);
  default:
    stop("Unknown family");
  }
}

// d mu / d eta (derivative of inverse link function)
inline vec d_inv_link(const vec &eta, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return ones<vec>(eta.n_elem);
  case Family::POISSON:
  case Family::NEGBIN:
    return exp(eta);
  case Family::BINOMIAL: {
    vec exp_eta = exp(eta);
    return exp_eta / square(1.0 + exp_eta);
  }
  case Family::GAMMA:
    return -1.0 / square(eta);
  case Family::INV_GAUSSIAN:
    return -1.0 / (2.0 * pow(abs(eta), 1.5));
  default:
    stop("Unknown family");
  }
}

//////////////////////////////////////////////////////////////////////////////
// INITIALIZATION FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Initialize mu and eta for different families
void initialize_family(vec &mu, vec &eta, const vec &y_orig, double mean_y,
                       const Family family_type, double binomial_mu_min,
                       double binomial_mu_max, double safe_clamp_min,
                       double safe_clamp_max) {
  switch (family_type) {
  case Family::GAUSSIAN:
    mu = y_orig; // Identity link: mu = eta for Gaussian
    eta = y_orig;
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    mu = (y_orig + mean_y) / 2.0;
    eta = log(clamp(mu, safe_clamp_min, safe_clamp_max));
    break;
  case Family::BINOMIAL:
    // For binomial, y should be 0 or 1
    // Initialize mu away from boundaries to avoid numerical issues
    mu = clamp((y_orig + mean_y) / 2.0, binomial_mu_min, binomial_mu_max);
    eta = log(mu / (1.0 - mu));
    break;
  case Family::GAMMA:
    mu = clamp(y_orig, safe_clamp_min, safe_clamp_max);
    eta = 1.0 / mu; // reciprocal link
    break;
  case Family::INV_GAUSSIAN:
    mu = clamp(y_orig, safe_clamp_min, safe_clamp_max);
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
inline double dev_resids_gaussian(const vec &y, const vec &mu, const vec &w) {
  return dot(w, square(y - mu));
}

inline double dev_resids_poisson(const vec &y, const vec &mu, const vec &w) {
  // Standard Poisson deviance
  vec r(y.n_elem, fill::zeros);

  // Find positive y values
  uvec p = find(y > 0);
  if (p.n_elem > 0) {
    vec y_p = y(p);
    vec mu_p = mu(p);
    vec w_p = w(p);
    r(p) = w_p % (y_p % log(y_p / mu_p) - (y_p - mu_p));
  }

  return 2.0 * accu(r);
}

// Binomial deviance matching R's base implementation exactly
inline double dev_resids_binomial(const vec &y, const vec &mu, const vec &w) {
  vec r(y.n_elem, fill::zeros);

  // y = 1 cases
  uvec p = find(y == 1);
  if (p.n_elem > 0) {
    r(p) = log(y(p) / mu(p));
  }

  // y = 0 cases
  uvec q = find(y == 0);
  if (q.n_elem > 0) {
    r(q) = log((1.0 - y(q)) / (1.0 - mu(q)));
  }

  return 2.0 * dot(w, r);
}

inline double dev_resids_gamma(const vec &y, const vec &mu, const vec &w,
                               double safe_clamp_min) {
  // Standard Gamma deviance: -2 * sum(log(y/mu) - (y-mu)/mu)
  vec y_adj = clamp(y, safe_clamp_min, datum::inf);
  vec mu_adj = clamp(mu, safe_clamp_min, datum::inf);

  vec terms = log(y_adj / mu_adj) - (y - mu) / mu_adj;
  return -2.0 * dot(w, terms);
}

inline double dev_resids_invgaussian(const vec &y, const vec &mu,
                                     const vec &w) {
  return dot(w, square(y - mu) / (y % square(mu)));
}

inline double dev_resids_negbin(const vec &y, const vec &mu,
                                const double &theta, const vec &w,
                                double safe_clamp_min) {
  vec y_adj = clamp(y, safe_clamp_min, datum::inf); // Avoid log(0)
  vec r =
      w % (y % log(y_adj / mu) - (y + theta) % log((y + theta) / (mu + theta)));

  return 2.0 * accu(r);
}

inline double dev_resids(const vec &y, const vec &mu, const double &theta,
                         const vec &w, const Family family_type,
                         double safe_clamp_min) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return dev_resids_gaussian(y, mu, w);
  case Family::POISSON:
    return dev_resids_poisson(y, mu, w);
  case Family::BINOMIAL:
    return dev_resids_binomial(y, mu, w);
  case Family::GAMMA:
    return dev_resids_gamma(y, mu, w, safe_clamp_min);
  case Family::INV_GAUSSIAN:
    return dev_resids_invgaussian(y, mu, w);
  case Family::NEGBIN:
    return dev_resids_negbin(y, mu, theta, w, safe_clamp_min);
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
  mat X_dm;
  bool has_mx = false;

  InferenceGLM(size_t n, size_t p)
      : coefficients(p, fill::zeros), eta(n, fill::zeros),
        fitted_values(n, fill::zeros), weights(n, fill::ones),
        hessian(p, p, fill::zeros), deviance(0.0), null_deviance(0.0),
        conv(false), iter(0), coef_status(p, fill::ones),
        residuals_working(n, fill::zeros), residuals_response(n, fill::zeros),
        is_regular(true), has_fe(false), has_mx(false) {}

  cpp11::list to_list(bool keep_dmx = true) const {
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

    if (keep_dmx && has_mx) {
      out.push_back({"MX"_nm = as_doubles_matrix(X_dm)});
    }

    return out;
  }
};

//////////////////////////////////////////////////////////////////////////////
// GLM FITTING
//////////////////////////////////////////////////////////////////////////////

inline InferenceGLM feglm_fit(const mat &X_orig, const vec &y_orig,
                              const vec &w, const field<uvec> &fe_indices,
                              const uvec &nb_ids,
                              const field<uvec> &fe_id_tables,
                              const std::string &family,
                              const CapybaraParameters &params) {
  const size_t n = y_orig.n_elem;
  const size_t p_orig = X_orig.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  // Debug: Function entry
  std::cout << "DEBUG feglm_fit ENTRY: family='" << family << "', n=" << n
            << ", p_orig=" << p_orig
            << ", has_fixed_effects=" << has_fixed_effects << std::endl;

  // Print data dimensions
  std::cout << "X rows=" << X_orig.n_rows << ", cols=" << X_orig.n_cols
            << ", y rows=" << y_orig.n_elem << std::endl;

  // Debug: Print first few values of input data
  std::cout << "DEBUG INPUT DATA:" << std::endl;
  std::cout << "First 5 y values: ";
  for (size_t i = 0; i < 5; i++) {
    std::cout << y_orig(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "First 5 x1 values: ";
  for (size_t i = 0; i < 5; i++) {
    std::cout << X_orig(i, 0) << " ";
  }
  std::cout << std::endl;

  if (X_orig.n_cols > 1) {
    std::cout << "First 5 x2 values: ";
    for (size_t i = 0; i < 5; i++) {
      std::cout << X_orig(i, 1) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "First 5 weights: ";
  for (size_t i = 0; i < 5; i++) {
    std::cout << w(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "First 5 fe_indices: ";
  if (fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0) {
    for (size_t i = 0; i < 5; i++) {
      std::cout << fe_indices(0)(i) << " ";
    }
  }
  std::cout << std::endl;

  InferenceGLM result(n, p_orig);

  // Get family type
  std::string fam = tidy_family(family);
  Family family_type = get_family_type(fam);

  // Validate response for all families

  switch (family_type) {
  case Family::GAUSSIAN:
    if (!is_finite(y_orig)) {
      result.conv = false;
      return result;
    }
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    // For Poisson/NegBin: y must be non-negative integers (can include 0)
    if (!is_finite(y_orig) || any(y_orig < 0)) {
      result.conv = false;
      return result;
    }
    break;
  case Family::BINOMIAL:
    // For Binomial: y must be between 0 and 1
    if (!is_finite(y_orig) || any(y_orig < 0) || any(y_orig > 1)) {
      result.conv = false;
      return result;
    }
    break;
  case Family::GAMMA:
  case Family::INV_GAUSSIAN:
    // For Gamma/InvGaussian: y must be positive
    if (!is_finite(y_orig) || any(y_orig <= 0)) {
      result.conv = false;
      return result;
    }
    break;
  default:
    stop("Unknown family");
  }
  std::cout << "DEBUG: Response validation passed!" << std::endl;

  // Initialize
  double mean_y = sum(w % y_orig) / sum(w);
  vec mu(n, fill::none);
  vec eta(n, fill::none);

  // Check if we actually have weights
  bool glm_weights = true;

  // Initialize mu and eta using helper function
  initialize_family(mu, eta, y_orig, mean_y, family_type,
                    params.binomial_mu_min, params.binomial_mu_max,
                    params.safe_clamp_min, params.safe_clamp_max);

  // For fixed effects models, initialize eta conservatively
  if (has_fixed_effects) {
    double safe_mean_y =
        std::max(static_cast<double>(mean_y), params.safe_clamp_min);

    switch (family_type) {
    case Family::GAUSSIAN:
      mu.fill(mean_y);
      eta.fill(mean_y);
      break;
    case Family::POISSON:
    case Family::NEGBIN:
      mu.fill(mean_y);
      eta.fill(log(safe_mean_y));
      break;
    case Family::BINOMIAL:
      mu.fill(0.5); // Start at neutral probability
      eta.zeros();  // logit(0.5) = 0
      break;
    case Family::GAMMA:
      mu.fill(safe_mean_y);
      eta.fill(1.0 / mean_y); // reciprocal link
      break;
    case Family::INV_GAUSSIAN:
      mu.fill(safe_mean_y);
      eta.fill(1.0 / (mean_y * mean_y)); // inverse squared link
      break;
    default:
      stop("Unknown family");
    }
  }

  result.deviance =
      dev_resids(y_orig, mu, 0.0, w, family_type, params.safe_clamp_min);

  // Storage for IRLS algorithm - pre-allocate
  vec eta_old(n, fill::none);
  vec z(n, fill::none);
  vec working_weights(n, fill::none);
  double devold = result.deviance;
  uvec coef_status = ones<uvec>(p_orig);

  // Variables for convergence checking
  double deviance = result.deviance;
  vec coefficients_old = result.coefficients;
  bool converged = false;

  mat X_demean;
  vec Y_demean;

  // Initialize demean matrices properly
  if (!has_fixed_effects) {
    X_demean = X_orig;
  } else if (p_orig > 0) {
    X_demean.set_size(n, p_orig);
  } else {
    X_demean = mat(n, 0);
  }

  // Pre-allocate for fixed effects case
  field<vec> variables_to_demean;
  if (has_fixed_effects) {
    variables_to_demean.set_size(p_orig + 1);
    if (p_orig > 0) {
      X_demean.set_size(n, p_orig);
    }
  }

  DemeanResult y_demean_result(0);

  // Debug after DemeanResult initialization
  std::cout << "DEBUG: DemeanResult created successfully" << std::endl;

  // IRLS loop

  for (size_t iter = 0; iter < params.iter_max; iter++) {
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
    if (all(working_weights <= 0)) {
      break;
    }

    // Demean variables if we have fixed effects
    if (has_fixed_effects) {
      // Prepare data for demeaning: combine Y and X
      variables_to_demean(0) = z; // Working response first
      for (size_t j = 0; j < p_orig; ++j) {
        variables_to_demean(j + 1) = X_orig.col(j); // Then X columns
      }

      y_demean_result =
          demean_variables(variables_to_demean, working_weights, fe_indices,
                           nb_ids, fe_id_tables, true, params);

      // Debug: Print demeaned data
      std::cout << "DEBUG: After demeaning - First 5 Y_demean values: ";
      Y_demean = y_demean_result.demeaned_vars(0);
      for (size_t i = 0; i < 5; i++) {
        std::cout << Y_demean(i) << " ";
      }
      std::cout << std::endl;

      if (p_orig > 0) {
        std::cout << "DEBUG: After demeaning - First 5 X_demean values: ";
        for (size_t j = 0; j < p_orig; ++j) {
          X_demean.col(j) = y_demean_result.demeaned_vars(j + 1);
        }
        for (size_t i = 0; i < 5; i++) {
          std::cout << X_demean(i, 0) << " ";
        }
        std::cout << std::endl;
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
        get_beta(X_demean, Y_demean, z, working_weights, params, glm_weights,
                 has_fixed_effects);

    // Debug: Print beta results
    std::cout << "DEBUG: Iteration " << iter
              << " - Beta calculation:" << std::endl;
    std::cout << "Beta success: " << beta_result.success << std::endl;
    if (beta_result.success && beta_result.coefficients.n_elem > 0) {
      std::cout << "Coefficients: ";
      for (size_t i = 0; i < beta_result.coefficients.n_elem; i++) {
        std::cout << beta_result.coefficients(i) << " ";
      }
      std::cout << std::endl;
    }

    if (!beta_result.success ||
        any(beta_result.coefficients != beta_result.coefficients)) {
      if (iter == 0) {
        result.conv = false;
        return result;
      }
      break;
    }

    coef_status = beta_result.coef_status;
    // Ensure proper copy of coefficients
    result.coefficients = vec(beta_result.coefficients);

    // Use fitted values directly from get_beta - they already include fixed
    // effects
    vec new_eta = beta_result.fitted_values;

    // Step halving
    double rho = 1.0;
    bool step_accepted = false;

    for (size_t iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta_old + rho * (new_eta - eta_old);
      mu = link_inv(eta, family_type);
      double deviance =
          dev_resids(y_orig, mu, 0.0, w, family_type, params.safe_clamp_min);

      double dev_ratio = (deviance - devold) / (0.1 + std::abs(devold));

      bool dev_finite = is_finite(deviance);
      bool mu_valid = valid_mu(mu, family_type);
      bool eta_valid = valid_eta(eta, family_type);
      bool dev_improved = (dev_ratio <= -params.dev_tol);

      if (dev_finite && mu_valid && eta_valid && dev_improved) {
        result.deviance = deviance;
        step_accepted = true;
        break;
      }

      rho *= params.step_halving_factor;
    }

    if (!step_accepted) {
      eta = eta_old;
      mu = link_inv(eta, family_type);
      result.deviance = devold;
    }

    // Check convergence
    double dev_ratio =
        std::abs(result.deviance - devold) / (0.1 + std::abs(result.deviance));
    if (dev_ratio < params.dev_tol) {
      result.conv = true;
      break;
    }

    devold = result.deviance;
  }

  // Final processing
  result.conv = converged;
  result.deviance = deviance;
  result.eta = eta;
  result.fitted_values = mu;
  result.weights = working_weights;
  result.coef_status = coef_status;

  // Hessian
  if (has_fixed_effects) {
    if (p_orig > 0) {
      mat weighted_X = X_demean.each_col() % working_weights;
      result.hessian = X_demean.t() * weighted_X;
    } else {
      result.hessian = mat(0, 0);
    }
  } else {
    mat weighted_X = X_orig.each_col() % working_weights;
    result.hessian = X_orig.t() * weighted_X;
  }

  // Null deviance
  vec mu_null(y_orig.n_elem, fill::value(mean_y));
  result.null_deviance =
      dev_resids(y_orig, mu_null, 0.0, w, family_type, params.safe_clamp_min);

  if (params.keep_dmx) {
    result.X_dm = X_orig;
    result.has_mx = true;
  }

  // Extract fixed effects if present
  if (has_fixed_effects) {
    // Compute fixed effects component
    vec fe_component;
    if (X_orig.n_cols > 0 && result.coefficients.n_elem > 0) {
      vec linear_part = X_orig * result.coefficients;
      fe_component = eta - linear_part;
    } else {
      fe_component = eta;
    }

    // Convert fe_indices to field<field<uvec>> format expected by get_alpha
    field<field<uvec>> group_indices(fe_indices.n_elem);
    for (size_t k = 0; k < fe_indices.n_elem; ++k) {
      group_indices(k).set_size(nb_ids(k));

      // Create groups from fe_indices
      for (size_t g = 0; g < nb_ids(k); ++g) {
        uvec group_obs = find(fe_indices(k) == g);
        group_indices(k)(g) = group_obs;
      }
    }

    InferenceAlpha alpha_result =
        get_alpha(fe_component, group_indices, 1e-8, 10000);

    result.fixed_effects = alpha_result.Alpha;
    result.has_fe = true;
    result.is_regular = alpha_result.is_regular;
    result.nb_references = alpha_result.nb_references;
  } else {
    result.has_fe = false;
  }

  return result;
}

} // namespace glm
} // namespace capybara

#endif // CAPYBARA_GLM_H
