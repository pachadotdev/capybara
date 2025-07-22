// Computing generalized linear models with fixed effects
// eta = X beta + alpha + offset

#ifndef CAPYBARA_GLM_H
#define CAPYBARA_GLM_H

namespace capybara {
namespace glm {

using demean::demean_variables;
using demean::DemeanResult;
using lm::InferenceLM;
using parameters::CollinearityResult;
using parameters::check_collinearity;
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

// Derivative of inverse link function (d mu / d eta)
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

// Conservative initialization for fixed effects models
void initialize_family_fixed_effects(vec &mu, vec &eta, double mean_y,
                                     const Family family_type,
                                     double safe_clamp_min) {
  double safe_mean_y = std::max(static_cast<double>(mean_y), safe_clamp_min);

  switch (family_type) {
  case Family::GAUSSIAN:
    mu.fill(mean_y);
    eta.fill(mean_y);
    break;
  case Family::POISSON:
  case Family::NEGBIN:
    mu.fill(safe_mean_y);
    eta.fill(log(safe_mean_y));
    break;
  case Family::BINOMIAL:
    mu.fill(0.5);
    eta.zeros();
    break;
  case Family::GAMMA:
    mu.fill(safe_mean_y);
    eta.fill(1.0 / safe_mean_y);
    break;
  case Family::INV_GAUSSIAN:
    mu.fill(safe_mean_y);
    eta.fill(1.0 / (safe_mean_y * safe_mean_y));
    break;
  default:
    stop("Unknown family");
  }
}

//////////////////////////////////////////////////////////////////////////////
// DEVIANCE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

// Deviance functions
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

// Binomial deviance matching R's base implementation
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
// VARIANCE V(mu)
//////////////////////////////////////////////////////////////////////////////

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

// Validate response vector for given family
inline bool valid_response(const vec &y, const Family family_type) {
  switch (family_type) {
  case Family::GAUSSIAN:
    return is_finite(y);
  case Family::POISSON:
  case Family::NEGBIN:
    return is_finite(y) && all(y >= 0);
  case Family::BINOMIAL:
    return is_finite(y) && all(y >= 0) && all(y <= 1);
  case Family::GAMMA:
  case Family::INV_GAUSSIAN:
    return is_finite(y) && all(y > 0);
  default:
    stop("Unknown family");
  }
}

//////////////////////////////////////////////////////////////////////////////
// RESULT STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// Similar to InferenceWLM but only with what IRLS needs
struct InferenceWLM {
  vec coefficients;
  vec fitted_values;
  uvec coef_status; // 1 = estimable, 0 = collinear
  bool success;

  // Optional components (only computed when requested)
  mat hessian;
  field<vec> fixed_effects;
  uvec nb_references;
  bool is_regular;
  bool has_fe;

  InferenceWLM(size_t n, size_t p)
      : coefficients(p, fill::none), fitted_values(n, fill::none),
        coef_status(p, fill::none), success(false), hessian(p, p, fill::none),
        is_regular(true), has_fe(false) {}
};

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
// WLM FITTING
//////////////////////////////////////////////////////////////////////////////

inline InferenceWLM wlm_fit(const mat &X, const vec &y, const vec &y_orig,
                            const vec &w, const field<uvec> &fe_indices,
                            const uvec &nb_ids, const field<uvec> &fe_id_tables,
                            const CapybaraParameters &params,
                            bool compute_hessian = false,
                            bool compute_fixed_effects = false,
                            bool first_iter = true) {
  const size_t n = y.n_elem;
  const size_t p_orig = X.n_cols;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceWLM result(n, p_orig);

  mat X_demean;
  vec y_demean;
  DemeanResult y_demean_result(0);

  if (has_fixed_effects) {
    // STEP 1: Demean Y (save fixed effects if requested)
    field<vec> y_to_demean(1);
    y_to_demean(0) = y;

    y_demean_result =
        demean_variables(y_to_demean, w, fe_indices, nb_ids, fe_id_tables,
                         compute_fixed_effects, params);
    y_demean = y_demean_result.demeaned_vars(0);

    // Demean X columns (without saving FE)
    if (p_orig > 0) {
      X_demean.set_size(n, p_orig);
      for (size_t j = 0; j < p_orig; ++j) {
        field<vec> x_to_demean(1);
        x_to_demean(0) = X.col(j);

        DemeanResult x_demean_result = demean_variables(
            x_to_demean, w, fe_indices, nb_ids, fe_id_tables, false, params);

        X_demean.col(j) = x_demean_result.demeaned_vars(0);
      }
    } else {
      X_demean = mat(n, 0);
    }

    result.has_fe = true;
  } else {
    // No fixed effects
    y_demean = y;
    X_demean = X;
    result.has_fe = false;
  }

  // STEP 2: Run regression using unified get_beta
  bool use_weights = params.use_weights;
  if (use_weights) {
    use_weights = !all(w == 1.0);
  }

  InferenceBeta beta_result = get_beta(X_demean, y_demean, y, w, params,
                                       use_weights, has_fixed_effects, first_iter);

  // Extract what we need
  result.coefficients = beta_result.coefficients;
  result.fitted_values = beta_result.fitted_values;
  result.coef_status = beta_result.coef_status;

  // Optionally compute hessian
  if (compute_hessian) {
    result.hessian = beta_result.hessian;
  }

  // STEP 3: Extract fixed effects if present and requested
  if (has_fixed_effects && compute_fixed_effects) {
    // The sum of fixed effects is: fitted_values - X*beta
    vec sum_fe = result.fitted_values - X * result.coefficients;

    // Convert field<uvec> to field<field<uvec>> format
    field<field<uvec>> group_indices(fe_indices.n_elem);
    for (size_t k = 0; k < fe_indices.n_elem; ++k) {
      group_indices(k).set_size(nb_ids(k));

      // Create groups from fe_indices
      for (size_t g = 0; g < nb_ids(k); ++g) {
        uvec group_obs = find(fe_indices(k) == g);
        group_indices(k)(g) = group_obs;
      }
    }

    // Use fixest algorithm to get fixed effects
    InferenceAlpha alpha_result =
        get_alpha(sum_fe, group_indices, params.alpha_convergence_tol,
                  params.alpha_iter_max);
    result.fixed_effects = alpha_result.Alpha;
    result.nb_references = alpha_result.nb_references;
    result.is_regular = alpha_result.is_regular;
  }

  result.success = true;
  return result;
}

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

  InferenceGLM result(n, p_orig);

  // Get family type
  std::string fam = tidy_family(family);
  Family family_type = get_family_type(fam);

  // Validate response
  if (!valid_response(y_orig, family_type)) {
    result.conv = false;
    return result;
  }

  // Detect collinearity before starting iterations
  mat X; // This will be the working X matrix without collinear columns
  CollinearityResult collin_result(p_orig);

  if (p_orig > 0) {
    // Use shared collinearity detection
    X = X_orig; // Copy to allow modification
    collin_result = check_collinearity(X, w, false, params.qr_collin_tol_multiplier);
  } else {
    X = X_orig;
  }

  const size_t p = X.n_cols; // Number of non-collinear columns

  // Check if we actually have weights
  bool use_weights = params.use_weights && w.n_elem > 1 && !all(w == 1.0);

  // Create effective weights vector
  vec weights_vec;
  if (use_weights) {
    weights_vec = w;
  } else {
    weights_vec = ones<vec>(n);
  }

  // Initialize
  double mean_y;
  if (use_weights) {
    mean_y = sum(weights_vec % y_orig) / sum(weights_vec);
  } else {
    mean_y = mean(y_orig);
  }

  vec mu(n, fill::none);
  vec eta(n, fill::none);

  // Initialize mu and eta
  initialize_family(mu, eta, y_orig, mean_y, family_type,
                    params.binomial_mu_min, params.binomial_mu_max,
                    params.safe_clamp_min, params.safe_clamp_max);

  // For fixed effects models, use conservative initialization
  if (has_fixed_effects) {
    initialize_family_fixed_effects(mu, eta, mean_y, family_type,
                                    params.safe_clamp_min);
  }

  // Initial deviance
  vec mu_init =
      vec(n, fill::value(link_inv(vec(n, fill::value(1e-5)), family_type)(0)));
  double devold = dev_resids(y_orig, mu_init, 0.0, weights_vec, family_type,
                             params.safe_clamp_min);

  // Storage for IRLS algorithm
  vec eta_old(n, fill::none);
  vec z(n, fill::none);
  vec working_weights(n, fill::none);
  uvec coef_status = ones<uvec>(p); // Status for non-collinear columns only

  // For saving the old WLS results
  vec eta_old_wls = vec(n, fill::value(1e-5));

  // Variables for convergence checking
  bool converged = false;

  // IRLS loop
  for (size_t iter = 0; iter < params.iter_max; iter++) {
    result.iter = iter + 1;
    eta_old = eta;

    // Compute mu.eta and variance
    vec mu_eta_val = d_inv_link(eta, family_type);
    vec var_mu = variance(mu, 0.0, family_type);

    // Check variance
    if (any(var_mu <= 0) || any(var_mu != var_mu)) {
      break;
    }

    // Working response
    z = eta + (y_orig - mu) / mu_eta_val;

    // Working weights
    working_weights = weights_vec % square(mu_eta_val) / var_mu;

    // Check weights
    if (all(working_weights <= 0)) {
      break;
    }

    // Call wlm_fit with the reduced X matrix (already trimmed for collinearity)
    InferenceWLM felm_result =
        wlm_fit(X, z, y_orig, working_weights, fe_indices, nb_ids, fe_id_tables,
                params, false, false, iter == 0); // first_iter = true for first iteration

    if (!felm_result.success ||
        any(felm_result.coefficients != felm_result.coefficients)) {
      if (iter == 0) {
        result.conv = false;
        return result;
      }
      break;
    }

    coef_status = felm_result.coef_status;

    // Store coefficients in the right positions (accounting for removed
    // columns)
    vec coef_reduced = felm_result.coefficients;
    result.coefficients.zeros();

    if (collin_result.has_collinearity && collin_result.n_valid > 0) {
      // Map reduced coefficients back to original positions
      uvec valid_cols = find(collin_result.coef_status == 1);
      for (size_t i = 0; i < valid_cols.n_elem; i++) {
        result.coefficients(valid_cols(i)) = coef_reduced(i);
      }
    } else if (p > 0) {
      // No collinearity, direct copy
      result.coefficients = coef_reduced;
    }

    // New eta
    vec eta_new = felm_result.fitted_values;

    // New mu for deviance calculation
    vec mu_new = link_inv(eta_new, family_type);
    double dev = dev_resids(y_orig, mu_new, 0.0, weights_vec, family_type,
                            params.safe_clamp_min);
    double dev_evol = dev - devold;

    // Step halving
    bool need_step_halving = !is_finite(dev) || (dev_evol > 0) ||
                             !valid_eta(eta_new, family_type) ||
                             !valid_mu(mu_new, family_type);

    if (need_step_halving &&
        !(std::abs(dev_evol) < params.dev_tol ||
          std::abs(dev_evol) / (0.1 + std::abs(dev)) < params.dev_tol)) {
      size_t iter_sh = 0;
      bool step_accepted = false;

      while (iter_sh < params.iter_inner_max) {
        iter_sh++;

        eta_new = (eta_old_wls + felm_result.fitted_values) / 2.0;

        mu_new = link_inv(eta_new, family_type);
        dev = dev_resids(y_orig, mu_new, 0.0, weights_vec, family_type,
                         params.safe_clamp_min);
        dev_evol = dev - devold;

        if (is_finite(dev) && (dev_evol <= 0) &&
            valid_eta(eta_new, family_type) && valid_mu(mu_new, family_type)) {
          step_accepted = true;
          break;
        }

        if (iter == 0 && iter_sh >= 2 && is_finite(dev) &&
            valid_eta(eta_new, family_type) && valid_mu(mu_new, family_type)) {
          step_accepted = true;
          break;
        }
      }

      if (!step_accepted) {
        if (iter == 0) {
          result.conv = false;
          return result;
        }
        eta = eta_old;
        mu = link_inv(eta, family_type);
        result.deviance = devold;
      } else {
        eta = eta_new;
        mu = mu_new;
        result.deviance = dev;
        dev_evol = datum::inf;
      }
    } else {
      eta = eta_new;
      mu = mu_new;
      result.deviance = dev;
    }

    eta_old_wls = felm_result.fitted_values;

    // Check convergence
    if (std::abs(dev_evol) / (0.1 + std::abs(dev)) < params.dev_tol) {
      result.conv = true;
      converged = true;
      break;
    }

    devold = dev;
  }

  // Final processing
  result.conv = converged;
  result.eta = eta;
  result.fitted_values = mu;
  result.weights = working_weights;

  // Update coef_status to reflect collinear variables
  result.coef_status.zeros();
  if (collin_result.has_collinearity && collin_result.n_valid > 0) {
    uvec valid_cols = find(collin_result.coef_status == 1);
    for (size_t i = 0; i < valid_cols.n_elem; i++) {
      result.coef_status(valid_cols(i)) = coef_status(i);
    }
  } else {
    result.coef_status = coef_status;
  }

  // Compute residuals
  result.residuals_response = y_orig - mu;
  result.residuals_working = z - eta;

  // Hessian and fixed effects extraction
  if (has_fixed_effects) {
    vec final_mu_eta = d_inv_link(eta, family_type);
    vec final_var_mu = variance(mu, 0.0, family_type);
    vec final_working_weights =
        weights_vec % square(final_mu_eta) / final_var_mu;
    vec final_z = eta + (y_orig - mu) / final_mu_eta;

    InferenceWLM final_felm =
        wlm_fit(X, final_z, y_orig, final_working_weights, fe_indices, nb_ids,
                fe_id_tables, params, true,
                true, false); // Compute hessian and FE for final result, use Cholesky

    // For hessian, we need to account for the original dimensions
    if (collin_result.has_collinearity && p_orig > 0) {
      result.hessian = zeros<mat>(p_orig, p_orig);
      mat hess_reduced = final_felm.hessian;
      uvec valid_cols = find(collin_result.coef_status == 1);
      for (size_t i = 0; i < valid_cols.n_elem; i++) {
        for (size_t j = 0; j < valid_cols.n_elem; j++) {
          result.hessian(valid_cols(i), valid_cols(j)) = hess_reduced(i, j);
        }
      }
    } else {
      result.hessian = final_felm.hessian;
    }

    result.fixed_effects = final_felm.fixed_effects;
    result.has_fe = true;
    result.is_regular = final_felm.is_regular;
    result.nb_references = final_felm.nb_references;
  } else {
    mat weighted_X = X.each_col() % sqrt(working_weights);
    mat hess_reduced = weighted_X.t() * weighted_X;

    if (collin_result.has_collinearity && p_orig > 0) {
      result.hessian = zeros<mat>(p_orig, p_orig);
      uvec valid_cols = find(collin_result.coef_status == 1);
      for (size_t i = 0; i < valid_cols.n_elem; i++) {
        for (size_t j = 0; j < valid_cols.n_elem; j++) {
          result.hessian(valid_cols(i), valid_cols(j)) = hess_reduced(i, j);
        }
      }
    } else {
      result.hessian = hess_reduced;
    }

    result.has_fe = false;
  }

  // Null deviance
  vec mu_null(y_orig.n_elem, fill::value(mean_y));
  result.null_deviance = dev_resids(y_orig, mu_null, 0.0, weights_vec,
                                    family_type, params.safe_clamp_min);

  if (params.keep_dmx) {
    result.X_dm = X_orig;
    result.has_mx = true;
  }

  return result;
}

} // namespace glm
} // namespace capybara

#endif // CAPYBARA_GLM_H
