#ifndef CAPYBARA_EXPONENTIAL_FAMILY
#define CAPYBARA_EXPONENTIAL_FAMILY

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

// Link inverse functions (eta -> mu)
vec link_inv_gaussian_(const vec &eta) {
  return eta;  // identity link
}

vec link_inv_poisson_(const vec &eta) {
  return exp(eta);  // log link
}

vec link_inv_logit_(const vec &eta) {
  return 1.0 / (1.0 + exp(-eta));  // logit link
}

vec link_inv_gamma_(const vec &eta) {
  return 1.0 / eta;  // reciprocal link: mu = 1/eta
}

vec link_inv_invgaussian_(const vec &eta) {
  return 1.0 / sqrt(eta);  // inverse squared link (matching old code)
}

vec link_inv_negbin_(const vec &eta) {
  return exp(eta);  // log link
}

// Deviance functions - matching Python exactly
double dev_resids_gaussian_(const vec &y, const vec &mu, const vec &wt) {
  // Python fegaussian_.py: np.sum((y - mu) ** 2)
  return dot(wt, square(y - mu));
}

double dev_resids_poisson_(const vec &y, const vec &mu, const vec &wt) {
  // Standard Poisson deviance
  vec r = mu % wt;

  uvec p = find(y > 0);
  r(p) = wt(p) % (y(p) % log(y(p) / mu(p)) - (y(p) - mu(p)));

  return 2 * accu(r);
}

// Logit deviance matching R's base implementation exactly
double dev_resids_logit_(const vec &y, const vec &mu, const vec &wt) {
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

double dev_resids_gamma_(const vec &y, const vec &mu, const vec &wt) {
  // Standard Gamma deviance: -2 * sum(log(y/mu) - (y-mu)/mu)
  vec y_safe = clamp(y, 1e-15, arma::datum::inf);
  vec mu_safe = clamp(mu, 1e-15, arma::datum::inf);

  vec terms = log(y_safe / mu_safe) - (y - mu) / mu_safe;
  return -2.0 * dot(wt, terms);
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
      return is_finite(eta);
    case GAMMA:
      return is_finite(eta) &&
             all(eta != 0.0);  // reciprocal link can't have eta=0
    case INV_GAUSSIAN:
      return is_finite(eta) &&
             all(eta > 0.0);  // inverse squared link needs eta > 0
    default:
      stop("Unknown family");
  }
}

bool valid_mu_(const vec &mu, const FamilyType family_type) {
  switch (family_type) {
    case GAUSSIAN:
      return is_finite(mu);
    case POISSON:
    case NEG_BIN:
      return is_finite(mu) && all(mu > 0);
    case BINOMIAL:
      return is_finite(mu) && all(mu > 0 && mu < 1);
    case GAMMA:
      return is_finite(mu) && all(mu > 0.0);
    case INV_GAUSSIAN:
      return is_finite(mu) && all(mu > 0.0);
    default:
      stop("Unknown family");
  }
}

// d mu / d eta (derivative of inverse link function)
vec d_inv_link(const vec &eta, const FamilyType family_type) {
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
      result = -1.0 / arma::square(eta);
      break;
    case INV_GAUSSIAN:
      result = -1.0 / (2.0 * arma::pow(abs(eta), 1.5));
      break;
    default:
      stop("Unknown family");
  }

  return result;
}

// Variance function V(mu)
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

#endif  // CAPYBARA_EXPONENTIAL_FAMILY
