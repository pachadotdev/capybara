// GLM fitting with offset term and fixed effects using IRLS algorithm

#ifndef CAPYBARA_GLM_OFFSET_H
#define CAPYBARA_GLM_OFFSET_H

namespace capybara {
namespace glm_offset {

using demean::demean_variables;
using demean::DemeanResult;

// Use the functions from the existing glm namespace in 05_glm.h
using glm::d_inv_link;
using glm::dev_resids;
using glm::link_inv;
using glm::string_to_family;
using glm::tidy_family;
using glm::valid_eta;
using glm::valid_mu;
using glm::variance;

// Use convergence family types for consistency
using convergence::Family;
using glm::get_family_type;

//////////////////////////////////////////////////////////////////////////////
// RESULT STRUCTURES
//////////////////////////////////////////////////////////////////////////////

// GLM offset fitting result structure
struct InferenceGLMOffset {
  vec eta;
  bool conv;
  size_t iter;

  InferenceGLMOffset(size_t n) : eta(n, fill::none), conv(false), iter(0) {}

  cpp11::doubles to_doubles() const { return as_doubles(eta); }
};

//////////////////////////////////////////////////////////////////////////////
// GLM OFFSET FITTING
//////////////////////////////////////////////////////////////////////////////

inline InferenceGLMOffset feglm_offset_fit(
    vec eta, const vec &y, const vec &offset, const vec &wt,
    const field<uvec> &fe_indices, const uvec &nb_ids,
    const field<uvec> &fe_id_tables, double center_tol, double dev_tol,
    size_t iter_max, size_t iter_center_max, size_t iter_inner_max,
    size_t iter_interrupt, size_t iter_ssr, const std::string &family,
    double collin_tol,
    // Demean algorithm parameters
    size_t demean_extra_projections = 0, size_t demean_warmup_iterations = 15,
    size_t demean_projections_after_acc = 5,
    size_t demean_grand_acc_frequency = 20,
    size_t demean_ssr_check_frequency = 40, double safe_division_min = 1e-12) {

  const size_t n = y.n_elem;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceGLMOffset result(n);

  // Get family type
  std::string fam = tidy_family(family);
  Family family_type = string_to_family(fam);

  // Validate inputs
  if (!valid_mu(y, family_type)) {
    result.conv = false;
    return result;
  }

  // Initialize variables
  vec Myadj = vec(n, fill::zeros);
  vec mu = link_inv(eta, family_type);
  vec mu_eta(n, fill::none);
  vec yadj(n, fill::none);
  vec w(n, fill::none);
  vec eta_upd(n, fill::none);
  vec eta_old(n, fill::none);

  double dev = dev_resids(y, mu, 0.0, wt, family_type);
  double dev_old, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  // IRLS loop
  for (size_t iter = 0; iter < iter_max; ++iter) {
    result.iter = iter + 1;
    rho = 1.0;
    eta_old = eta;
    dev_old = dev;

    // Compute mu.eta and working weights
    mu_eta = d_inv_link(eta, family_type);
    vec var_mu = variance(mu, 0.0, family_type);

    // Check variance
    if (any(var_mu <= 0) || any(var_mu != var_mu)) {
      break;
    }

    w = (wt % square(mu_eta)) / var_mu;
    yadj = (y - mu) / mu_eta + eta - offset;

    // Center the adjusted response if we have fixed effects
    if (has_fixed_effects) {
      Myadj += yadj;

      // Prepare data for demeaning
      field<vec> variables_to_demean(1);
      variables_to_demean(0) = Myadj;

      // Demean the working response
      DemeanResult demean_result = demean_variables(
          variables_to_demean, w, fe_indices, nb_ids, fe_id_tables,
          iter_center_max, center_tol, demean_extra_projections,
          demean_warmup_iterations, demean_projections_after_acc,
          demean_grand_acc_frequency, demean_ssr_check_frequency, false,
          safe_division_min);

      Myadj = demean_result.demeaned_vars(0);
    } else {
      // No fixed effects - just update with current working response
      Myadj = yadj;
    }

    // Compute eta update
    eta_upd = yadj - Myadj + offset - eta;

    // Step halving loop
    for (size_t iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      mu = link_inv(eta, family_type);
      dev = dev_resids(y, mu, 0.0, wt, family_type);
      dev_ratio_inner = (dev - dev_old) / (0.1 + std::abs(dev_old));

      dev_crit = is_finite(dev);
      val_crit = valid_eta(eta, family_type) && valid_mu(mu, family_type);
      imp_crit = (dev_ratio_inner <= -dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= 0.5;
    }

    // Check if step-halving failed
    if (!dev_crit || !val_crit) {
      stop("Inner loop failed; cannot correct step size.");
    }

    // Check convergence
    dev_ratio = std::abs(dev - dev_old) / (0.1 + std::abs(dev));
    if (dev_ratio < dev_tol) {
      result.conv = true;
      break;
    }

    // Update starting guesses for acceleration
    if (has_fixed_effects) {
      Myadj = Myadj - yadj;
    }
  }

  result.eta = eta;
  return result;
}

} // namespace glm_offset
} // namespace capybara

#endif // CAPYBARA_GLM_OFFSET_H
