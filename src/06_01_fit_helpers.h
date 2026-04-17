// Code used for LM, GLM and NegBin model fitting

#ifndef CAPYBARA_GLM_HELPERS_H
#define CAPYBARA_GLM_HELPERS_H

namespace capybara {

struct InferenceGLM {
  mat coef_table; // Coefficient table: [estimate, std.error, z, p-value]
  vec eta;
  vec fitted_values; // mu values (response scale)
  vec weights;
  mat hessian;
  mat vcov; // inverse Hessian or sandwich)
  double deviance;
  double null_deviance;
  bool conv;
  uword iter;
  uvec coef_status; // 1 = estimable, 0 = collinear

  field<vec> fixed_effects;
  double pseudo_rsq; // pseudo R-squared for Poisson
  bool has_fe = false;
  uvec iterations;

  mat TX;
  bool has_tx = false;

  vec means;

  // Separation detection fields
  bool has_separation = false;
  uword num_separated = 0;
  uvec separated_obs;
  vec separation_support;

  // Average Partial Effects (APE) fields for binomial models
  vec ape_delta;   // APE estimates for each coefficient
  mat ape_vcov;    // Covariance matrix of APE estimates
  uvec ape_binary; // 1 = binary regressor, 0 = continuous
  bool has_apes = false;

  // Bias correction fields for binomial models (Fernández-Val & Weidner 2016)
  vec beta_corrected; // Bias-corrected coefficient estimates
  vec bias_term;      // Estimated bias term
  bool has_bias_corr = false;

  // Full constructor - allocates all fields including P*P hessian/vcov
  InferenceGLM(uword n, uword p)
      : coef_table(p, 4, fill::none), eta(n, fill::none),
        fitted_values(n, fill::none), weights(n, fill::ones),
        hessian(p, p, fill::zeros), vcov(p, p, fill::zeros), deviance(0.0),
        null_deviance(0.0), conv(false), iter(0), coef_status(p, fill::ones),
        pseudo_rsq(0.0), has_fe(false), has_tx(false), has_separation(false),
        num_separated(0) {}

  // Lite constructor - skips hessian/vcov allocation for fast paths
  // (e.g., negbin outer loop iterations where only beta/eta/mu are needed)
  InferenceGLM(uword n, uword p, bool allocate_vcov)
      : coef_table(p, 4, fill::none), eta(), fitted_values(), weights(),
        hessian(), vcov(), deviance(0.0), null_deviance(0.0), conv(false),
        iter(0), coef_status(p, fill::ones), pseudo_rsq(0.0), has_fe(false),
        has_tx(false), has_separation(false), num_separated(0) {
    // Defer N-length vector allocation until results are assigned
    // This avoids allocating 3N doubles that would be immediately overwritten
    if (allocate_vcov) {
      hessian.zeros(p, p);
      vcov.zeros(p, p);
    }
  }
};

enum Family {
  UNKNOWN = 0,
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  PROBIT,
  TOBIT,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN
};

inline double predict_convergence(const vec &eps_history, double current_eps) {
  const uword n = eps_history.n_elem;
  if (n < 3 || !eps_history.is_finite()) {
    return current_eps;
  }

  const vec last3 = eps_history.tail(3);
  if (!last3.is_finite()) {
    return current_eps;
  }

  // Vectorized regression: log(eps) = a + b*x with x = {1,2,3}
  const vec log_eps = log(last3);
  // slope = (log_eps(2) - log_eps(0)) / 2, predict at x=4: mean + 2*slope
  return std::max(std::exp(mean(log_eps) + (log_eps(2) - log_eps(0))),
                  datum::eps);
}

inline std::string tidy_family(const std::string &family) {
  std::string fam = family;

  // Lowercase
  std::transform(fam.begin(), fam.end(), fam.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Truncate at '(' if present
  const auto pos = fam.find('(');
  if (pos != std::string::npos) {
    fam.resize(pos);
  }

  // Remove digits and replace separators with underscore in one pass
  fam.erase(
      std::remove_if(fam.begin(), fam.end(),
                     [](char c) { return std::isdigit(c) || std::isspace(c); }),
      fam.end());

  std::replace(fam.begin(), fam.end(), '.', '_');

  return fam;
}

Family get_family_type(const std::string &fam) {
  static const std::unordered_map<std::string, Family> family_map = {
      {"gaussian", GAUSSIAN},
      {"poisson", POISSON},
      {"binomial", BINOMIAL},
      {"probit", PROBIT},
      {"tobit", TOBIT},
      {"gamma", GAMMA},
      {"inverse_gaussian", INV_GAUSSIAN},
      {"negative_binomial", NEG_BIN}};

  const auto it = family_map.find(fam);
  return (it != family_map.end()) ? it->second : UNKNOWN;
}

} // namespace capybara

#endif // CAPYBARA_GLM_HELPERS_H
