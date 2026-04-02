// Average Partial Effects (APEs) and Bias Correction for binary choice models
// Implements Fernández-Val and Weidner (2016, 2018) bias correction

#ifndef CAPYBARA_APES_BIAS_H
#define CAPYBARA_APES_BIAS_H

namespace capybara {

//////////////////////////////////////////////////////////////////////////////
// Result structures
//////////////////////////////////////////////////////////////////////////////

struct BiasResult {
  vec beta_corrected;       // Bias-corrected coefficients
  vec beta_uncorrected;     // Original coefficients
  vec bias_term;            // The bias term B
  mat hessian;              // Updated Hessian
  vec eta;                  // Updated linear predictor
  std::string panel_structure;
  uword bandwidth;
  bool success;
  
  BiasResult() : bandwidth(0), success(false) {}
};

struct APESResult {
  vec delta;                // Average partial effects
  mat vcov;                 // Variance-covariance matrix
  vec bias_term;            // Optional bias term (if bias_corr was applied)
  std::string panel_structure;
  std::string sampling_fe;
  bool weak_exo;
  uword bandwidth;
  bool success;
  
  APESResult() : weak_exo(false), bandwidth(0), success(false) {}
};

//////////////////////////////////////////////////////////////////////////////
// Partial derivatives of link functions
//////////////////////////////////////////////////////////////////////////////

// Second-order derivative of the inverse link function
// d^2 mu / d eta^2
inline double partial_mu_eta_2_scalar(double eta, double mu_eta, 
                                      const std::string& link) {
  if (link == "logit") {
    // For logit: mu = 1/(1+exp(-eta)), mu_eta = mu*(1-mu)
    // d^2 mu / d eta^2 = mu_eta * (1 - 2*mu)
    double mu = 1.0 / (1.0 + std::exp(-eta));
    return mu_eta * (1.0 - 2.0 * mu);
  } else if (link == "probit") {
    return -eta * mu_eta;
  } else if (link == "cloglog") {
    return mu_eta * (1.0 - std::exp(eta));
  } else {
    // cauchit
    return -2.0 * eta / (1.0 + eta * eta) * mu_eta;
  }
}

// Third-order derivative of the inverse link function
// d^3 mu / d eta^3
inline double partial_mu_eta_3_scalar(double eta, double mu_eta,
                                      const std::string& link) {
  if (link == "logit") {
    double mu = 1.0 / (1.0 + std::exp(-eta));
    double tmp = 1.0 - 2.0 * mu;
    return mu_eta * (tmp * tmp - 2.0 * mu_eta);
  } else if (link == "probit") {
    return (eta * eta - 1.0) * mu_eta;
  } else if (link == "cloglog") {
    double exp_eta = std::exp(eta);
    return mu_eta * (1.0 - exp_eta) * (2.0 - exp_eta) - mu_eta;
  } else {
    // cauchit
    double denom = 1.0 + eta * eta;
    return (6.0 * eta * eta - 2.0) / (denom * denom) * mu_eta;
  }
}

// Vectorized second-order partial derivative
inline vec partial_mu_eta_2(const vec& eta, const vec& mu_eta,
                            const std::string& link) {
  const uword n = eta.n_elem;
  vec result(n);
  
  if (link == "logit") {
    vec mu = 1.0 / (1.0 + exp(-eta));
    result = mu_eta % (1.0 - 2.0 * mu);
  } else if (link == "probit") {
    result = -eta % mu_eta;
  } else if (link == "cloglog") {
    result = mu_eta % (1.0 - exp(eta));
  } else {
    // cauchit
    result = -2.0 * eta / (1.0 + square(eta)) % mu_eta;
  }
  
  return result;
}

// Vectorized third-order partial derivative
inline vec partial_mu_eta_3(const vec& eta, const vec& mu_eta,
                            const std::string& link) {
  const uword n = eta.n_elem;
  vec result(n);
  
  if (link == "logit") {
    vec mu = 1.0 / (1.0 + exp(-eta));
    vec tmp = 1.0 - 2.0 * mu;
    result = mu_eta % (square(tmp) - 2.0 * mu_eta);
  } else if (link == "probit") {
    result = (square(eta) - 1.0) % mu_eta;
  } else if (link == "cloglog") {
    vec exp_eta = exp(eta);
    result = mu_eta % (1.0 - exp_eta) % (2.0 - exp_eta) - mu_eta;
  } else {
    // cauchit
    vec denom = 1.0 + square(eta);
    result = (6.0 * square(eta) - 2.0) / square(denom) % mu_eta;
  }
  
  return result;
}

//////////////////////////////////////////////////////////////////////////////
// Group index utilities
//////////////////////////////////////////////////////////////////////////////

// Build inverted index from FlatFEMap (group -> observation indices)
inline field<uvec> build_group_indices(const FlatFEMap& fe_map, uword k) {
  if (k >= fe_map.K) {
    return field<uvec>();
  }
  
  const uword n_groups = fe_map.n_groups[k];
  const uword n = fe_map.n_obs;
  
  // Count observations per group
  std::vector<uword> counts(n_groups, 0);
  for (uword i = 0; i < n; ++i) {
    counts[fe_map.fe_map[k][i]]++;
  }
  
  // Allocate result
  field<uvec> groups(n_groups);
  for (uword g = 0; g < n_groups; ++g) {
    groups(g).set_size(counts[g]);
    counts[g] = 0; // reuse as position counter
  }
  
  // Fill groups
  for (uword i = 0; i < n; ++i) {
    uword g = fe_map.fe_map[k][i];
    groups(g)(counts[g]++) = i;
  }
  
  return groups;
}

//////////////////////////////////////////////////////////////////////////////
// Bias correction implementation (Fernández-Val & Weidner 2016)
//////////////////////////////////////////////////////////////////////////////

inline BiasResult compute_bias_corr(
    const vec& y,
    const mat& X,
    const mat& TX,       // Centered regressor matrix
    const vec& eta,
    const vec& weights,
    const mat& hessian,
    const std::string& link,
    const FlatFEMap& fe_map,
    const std::string& panel_structure,
    uword bandwidth,
    double center_tol,
    uword iter_center_max,
    uword grand_acc_period) {
  
  BiasResult result;
  result.panel_structure = panel_structure;
  result.bandwidth = bandwidth;
  
  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const uword K = fe_map.K;
  
  // Validate panel structure vs number of FEs
  if (panel_structure == "classic" && !(K == 1 || K == 2)) {
    result.success = false;
    return result;
  }
  if (panel_structure == "network" && !(K == 2 || K == 3)) {
    result.success = false;
    return result;
  }
  
  // Compute derivatives
  vec mu = 1.0 / (1.0 + exp(-eta));  // For logit
  vec mu_eta = mu % (1.0 - mu);
  
  vec v = weights % (y - mu);
  vec w = weights % mu_eta;
  vec z = weights % partial_mu_eta_2(eta, mu_eta, link);
  
  // For non-logit links, apply h adjustment
  if (link != "logit") {
    vec variance = mu % (1.0 - mu);  // Binomial variance
    vec h = mu_eta / variance;
    v = h % v;
    w = h % w;
    z = h % z;
  }
  
  // Build group indices for each FE dimension
  std::vector<field<uvec>> k_groups(K);
  for (uword k = 0; k < K; ++k) {
    k_groups[k] = build_group_indices(fe_map, k);
  }
  
  // Compute bias terms
  vec b(p, fill::zeros);
  
  if (panel_structure == "classic") {
    // Compute B and D for classic panel
    b = group_sums(TX % z, w, k_groups[0]) / (2.0 * n);
    if (K > 1) {
      b = b + group_sums(TX % z, w, k_groups[1]) / (2.0 * n);
    }
    
    // Spectral density part
    if (bandwidth > 0) {
      b = (b + group_sums_spectral(TX % w, v, w, bandwidth, k_groups[0])) / n;
    }
  } else {
    // Network panel
    b = group_sums(TX % z, w, k_groups[0]) / (2.0 * n);
    b = (b + group_sums(TX % z, w, k_groups[1])) / (2.0 * n);
    if (K > 2) {
      b = (b + group_sums(TX % z, w, k_groups[2])) / (2.0 * n);
    }
    
    // Spectral density part
    if (K > 2 && bandwidth > 0) {
      b = (b + group_sums_spectral(TX % w, v, w, bandwidth, k_groups[2])) / n;
    }
  }
  
  // Compute bias-corrected coefficients
  result.beta_uncorrected = TX.t() * (w % eta) / n; // Extract from TX'Weta
  mat H_scaled = hessian / static_cast<double>(n);
  vec beta_corr;
  if (!solve(beta_corr, H_scaled, b)) {
    result.success = false;
    return result;
  }
  
  // For proper bias correction, we need the original beta
  // This will be done on R side since we need coef_table from the fit
  result.bias_term = b;
  result.hessian = hessian;
  result.success = true;
  
  return result;
}

//////////////////////////////////////////////////////////////////////////////
// APES implementation
//////////////////////////////////////////////////////////////////////////////

inline APESResult compute_apes(
    const vec& y,
    const mat& X,
    const mat& TX,       // Centered regressor matrix
    const vec& eta,
    const vec& weights,
    const vec& beta,
    const mat& hessian,
    const std::string& link,
    const FlatFEMap& fe_map,
    uword n_full,        // Full sample size (before any drops)
    const std::string& panel_structure,
    const std::string& sampling_fe,
    bool weak_exo,
    double adj,          // Finite population adjustment
    uword bandwidth,     // Only used if bias_corrected
    bool bias_corrected,
    double center_tol,
    uword iter_center_max,
    uword grand_acc_period) {
  
  APESResult result;
  result.panel_structure = panel_structure;
  result.sampling_fe = sampling_fe;
  result.weak_exo = weak_exo;
  result.bandwidth = bandwidth;
  
  const uword n = y.n_elem;
  const uword p = X.n_cols;
  const uword K = fe_map.K;
  const double nt_full = static_cast<double>(n_full);
  
  // Validate panel structure
  if (panel_structure == "classic" && !(K == 1 || K == 2)) {
    result.success = false;
    return result;
  }
  if (panel_structure == "network" && !(K == 2 || K == 3)) {
    result.success = false;
    return result;
  }
  
  // Compute derivatives
  vec mu = 1.0 / (1.0 + exp(-eta));
  vec mu_eta = mu % (1.0 - mu);
  
  vec v_score = weights % (y - mu);
  vec w = weights % mu_eta;
  vec z = weights % partial_mu_eta_2(eta, mu_eta, link);
  
  if (link != "logit") {
    vec variance = mu % (1.0 - mu);
    vec h = mu_eta / variance;
    v_score = h % v_score;
    w = h % w;
    z = h % z;
  }
  
  // Determine which regressors are binary
  uvec binary(p);
  for (uword j = 0; j < p; ++j) {
    bool is_binary = true;
    for (uword i = 0; i < n && is_binary; ++i) {
      double val = X(i, j);
      if (val != 0.0 && val != 1.0) {
        is_binary = false;
      }
    }
    binary(j) = is_binary ? 1 : 0;
  }
  
  // Compute average partial effects and derivatives
  mat delta(n, p);
  mat delta1(n, p);
  mat J(p, p, fill::zeros);
  
  vec delta1_nonbinary = partial_mu_eta_2(eta, mu_eta, link);
  
  for (uword j = 0; j < p; ++j) {
    if (binary(j)) {
      vec eta0 = eta - X.col(j) * beta(j);
      vec eta1 = eta0 + beta(j);
      vec mu0 = 1.0 / (1.0 + exp(-eta0));
      vec mu1 = 1.0 / (1.0 + exp(-eta1));
      vec f1 = mu1 % (1.0 - mu1);
      vec f0 = mu0 % (1.0 - mu0);
      
      delta.col(j) = mu1 - mu0;
      delta1.col(j) = f1 - f0;
      
      // Jacobian
      J.col(j) = -sum((X - TX) % delta1.col(j), 0).t() / nt_full;
      J(j, j) += accu(f1) / nt_full;
      
      for (uword i = 0; i < p; ++i) {
        if (i != j) {
          J(i, j) += accu(X.col(i) % delta1.col(j)) / nt_full;
        }
      }
    } else {
      delta.col(j) = beta(j) * mu_eta;
      delta1.col(j) = beta(j) * delta1_nonbinary;
      
      J.col(j) = sum(TX % delta1.col(j), 0).t() / nt_full;
      J(j, j) += accu(mu_eta) / nt_full;
    }
  }
  
  vec delta_avg = sum(delta, 0).t() / nt_full;
  delta.each_row() -= delta_avg.t();
  delta /= nt_full;
  
  // Center psi = -delta1 / w
  mat psi = -delta1;
  for (uword i = 0; i < n; ++i) {
    if (w(i) > 0) {
      psi.row(i) /= w(i);
    }
  }
  
  // Center psi to get mpsi, compute ppsi = psi - mpsi
  // Make a mutable copy of fe_map for centering (it caches group weights)
  FlatFEMap fe_map_copy = fe_map;
  fe_map_copy.update_weights(w);  // Update inverse weights for current w
  mat mpsi = psi;
  center_variables(mpsi, w, fe_map_copy, center_tol, iter_center_max, grand_acc_period);
  mat ppsi = psi - mpsi;
  
  // Build group indices if needed for bias correction or covariance adjustment
  std::vector<field<uvec>> k_groups;
  if (bias_corrected || adj > 0.0 || sampling_fe != "independence" || weak_exo) {
    k_groups.resize(K);
    for (uword k = 0; k < K; ++k) {
      k_groups[k] = build_group_indices(fe_map, k);
    }
  }
  
  // Bias correction for APEs
  if (bias_corrected && bandwidth > 0) {
    // Compute second-order partial derivatives for delta
    mat delta2(n, p);
    vec delta2_nonbinary = partial_mu_eta_3(eta, mu_eta, link);
    
    for (uword j = 0; j < p; ++j) {
      if (binary(j)) {
        vec eta0 = eta - X.col(j) * beta(j);
        vec eta1 = eta0 + beta(j);
        vec mu1 = 1.0 / (1.0 + exp(-eta1));
        vec mu0 = 1.0 / (1.0 + exp(-eta0));
        vec me1 = mu1 % (1.0 - mu1);
        vec me0 = mu0 % (1.0 - mu0);
        delta2.col(j) = partial_mu_eta_2(eta1, me1, link) - 
                        partial_mu_eta_2(eta0, me0, link);
      } else {
        delta2.col(j) = beta(j) * delta2_nonbinary;
      }
    }
    
    // Compute bias terms
    vec b(p, fill::zeros);
    
    if (panel_structure == "classic") {
      b = group_sums(delta2 + ppsi % z, w, k_groups[0]) / (2.0 * n);
      if (K > 1) {
        b = (b + group_sums(delta2 + ppsi % z, w, k_groups[1])) / (2.0 * n);
      }
      if (bandwidth > 0) {
        b = (b - group_sums_spectral(mpsi % w, v_score, w, bandwidth, k_groups[0])) / n;
      }
    } else {
      b = group_sums(delta2 + ppsi % z, w, k_groups[0]) / (2.0 * n);
      b = (b + group_sums(delta2 + ppsi % z, w, k_groups[1])) / (2.0 * n);
      if (K > 2) {
        b = (b + group_sums(delta2 + ppsi % z, w, k_groups[2])) / (2.0 * n);
      }
      if (K > 2 && bandwidth > 0) {
        b = (b - group_sums_spectral(mpsi % w, v_score, w, bandwidth, k_groups[2])) / n;
      }
    }
    
    delta_avg -= b;
    result.bias_term = b;
  }
  
  // Compute covariance matrix via delta method
  // gamma = (TX * inv(H/n) * J - ppsi) * v / n
  mat H_scaled = hessian / nt_full;
  mat H_inv_J;
  if (!solve(H_inv_J, H_scaled, J)) {
    result.success = false;
    return result;
  }
  
  mat gamma = (TX * H_inv_J - ppsi);
  for (uword i = 0; i < n; ++i) {
    gamma.row(i) *= v_score(i) / nt_full;
  }
  
  mat V = gamma.t() * gamma;
  
  // Finite population correction
  if (adj > 0.0) {
    if (sampling_fe == "independence") {
      V += adj * group_sums_var(delta, k_groups[0]);
      if (K > 1) {
        V += adj * (group_sums_var(delta, k_groups[1]) - delta.t() * delta);
      }
      if (panel_structure == "network" && K > 2) {
        V += adj * (group_sums_var(delta, k_groups[2]) - delta.t() * delta);
      }
    }
    
    // Weak exogeneity adjustment
    if (weak_exo) {
      if (panel_structure == "classic") {
        mat cl = group_sums_cov(delta, gamma, k_groups[0]);
        V += adj * (cl + cl.t());
      } else if (K > 2) {
        mat cl = group_sums_cov(delta, gamma, k_groups[2]);
        V += adj * (cl + cl.t());
      }
    }
  }
  
  result.delta = delta_avg;
  result.vcov = V;
  result.success = true;
  
  return result;
}

} // namespace capybara

#endif // CAPYBARA_APES_BIAS_H
