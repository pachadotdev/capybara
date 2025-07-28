// GLM fitting with offset term and fixed effects using IRLS algorithm

#ifndef CAPYBARA_GLM_OFFSET_H
#define CAPYBARA_GLM_OFFSET_H

namespace capybara {
namespace glm_offset {

using convergence::Family;
using demean::demean_variables;
using demean::DemeanResult;
using glm::d_inv_link;
using glm::dev_resids;
using glm::get_family_type;
using glm::link_inv;
using glm::string_to_family;
using glm::tidy_family;
using glm::valid_eta;
using glm::valid_mu;
using glm::variance;

struct InferenceGLMOffset {
  vec eta;
  bool conv;
  size_t iter;

  InferenceGLMOffset(size_t n) : eta(n, fill::none), conv(false), iter(0) {}

  cpp11::doubles to_doubles() const { return as_doubles(eta); }
};

inline InferenceGLMOffset
feglm_offset_fit(vec eta, const vec &y, const vec &offset, const vec &wt,
                 const field<uvec> &fe_indices, const uvec &nb_ids,
                 const field<uvec> &fe_id_tables, const std::string &family,
                 const CapybaraParameters &params) {

  const size_t n = y.n_elem;
  const bool has_fixed_effects =
      fe_indices.n_elem > 0 && fe_indices(0).n_elem > 0;

  InferenceGLMOffset result(n);

  std::string fam = tidy_family(family);
  Family family_type = string_to_family(fam);

  if (!valid_mu(y, family_type)) {
    result.conv = false;
    return result;
  }

  vec Myadj = vec(n, fill::zeros);
  vec mu(n);
  link_inv(eta, mu, family_type);
  vec mu_eta(n, fill::none);
  vec yadj(n, fill::none);
  vec w(n, fill::none);
  vec eta_upd(n, fill::none);
  vec eta_old(n, fill::none);

  double dev = dev_resids(y, mu, 0.0, wt, family_type, params.safe_clamp_min);
  double dev_old, dev_ratio, dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;

  for (size_t iter = 0; iter < params.iter_max; ++iter) {
    result.iter = iter + 1;
    rho = 1.0;
    eta_old = eta;
    dev_old = dev;

    d_inv_link(eta, mu_eta, family_type);
    vec var_mu(n);
    variance(mu, 0.0, var_mu, family_type);

    if (any(var_mu <= 0) || any(var_mu != var_mu)) {
      break;
    }

    w = (wt % square(mu_eta)) / var_mu;
    yadj = (y - mu) / mu_eta + eta - offset;

    if (has_fixed_effects) {
      Myadj += yadj;

      field<vec> variables_to_demean(1);
      variables_to_demean(0) = Myadj;

      DemeanResult demean_result =
          demean_variables(variables_to_demean, w, fe_indices, nb_ids,
                           fe_id_tables, false, params);

      Myadj = demean_result.demeaned_vars(0);
    } else {

      Myadj = yadj;
    }

    eta_upd = yadj - Myadj + offset - eta;

    for (size_t iter_inner = 0; iter_inner < params.iter_inner_max;
         ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      link_inv(eta, mu, family_type);
      dev = dev_resids(y, mu, 0.0, wt, family_type, params.safe_clamp_min);
      dev_ratio_inner =
          (dev - dev_old) / (params.rel_tol_denom + std::abs(dev_old));

      dev_crit = is_finite(dev);
      val_crit = valid_eta(eta, family_type) && valid_mu(mu, family_type);
      imp_crit = (dev_ratio_inner <= -params.dev_tol);

      if (dev_crit && val_crit && imp_crit) {
        break;
      }

      rho *= params.step_halving_factor;
    }

    if (!dev_crit || !val_crit) {
      stop("Inner loop failed; cannot correct step size.");
    }

    dev_ratio =
        std::abs(dev - dev_old) / (params.rel_tol_denom + std::abs(dev));
    if (dev_ratio < params.dev_tol) {
      result.conv = true;
      break;
    }

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
