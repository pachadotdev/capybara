#ifndef CAPYBARA_GLM_OFFSET
#define CAPYBARA_GLM_OFFSET

struct FeglmOffsetFitResult {
  vec eta;

  cpp11::doubles to_doubles() const { return as_doubles(eta); }
};

// Core function: pure Armadillo types
inline FeglmOffsetFitResult
feglm_offset_fit(vec eta, const vec &y, const vec &offset, const vec &wt,
                 const field<field<uvec>> &group_indices, double center_tol,
                 double dev_tol, size_t iter_max, size_t iter_center_max,
                 size_t iter_inner_max, size_t iter_interrupt, size_t iter_ssr,
                 const std::string &fam, FamilyType family_type,
                 double collin_tol) {
  FeglmOffsetFitResult res;
  size_t n = y.n_elem;
  vec Myadj = vec(n, fill::zeros);
  vec mu = link_inv_(eta, family_type), mu_eta(n, fill::none),
      yadj(n, fill::none), w(n, fill::none), eta_upd(n, fill::none),
      eta_old(n, fill::none);
  double dev = dev_resids_(y, mu, 0.0, wt, family_type), dev_old, dev_ratio,
         dev_ratio_inner, rho;
  bool dev_crit, val_crit, imp_crit;
  size_t iter, iter_inner;

  for (iter = 0; iter < iter_max; ++iter) {
    rho = 1.0;
    eta_old = eta;
    dev_old = dev;
    mu_eta = mu_eta_(eta, family_type);
    w = (wt % square(mu_eta)) / variance_(mu, 0.0, family_type);
    yadj = (y - mu) / mu_eta + eta - offset;
    Myadj += yadj;
    mat Myadj_mat = Myadj;
    
    // Convert field<field<uvec>> to umat format for new demean_variables
    umat fe_matrix;
    if (group_indices.n_elem > 0) {
      size_t n_obs = y.n_elem;
      fe_matrix.set_size(n_obs, group_indices.n_elem);
      
      for (size_t k = 0; k < group_indices.n_elem; k++) {
        // Set FE levels based on group indices
        for (size_t g = 0; g < group_indices(k).n_elem; g++) {
          const uvec &group_obs = group_indices(k)(g);
          if (group_obs.n_elem > 0) {
            fe_matrix.submat(group_obs, uvec{k}).fill(g);
          }
        }
      }
    }
    
    WeightedDemeanResult myadj_result = demean_variables(Myadj_mat, fe_matrix, w, center_tol, iter_center_max, "gaussian");
    Myadj = myadj_result.demeaned_data.col(0);
    eta_upd = yadj - Myadj + offset - eta;
    for (iter_inner = 0; iter_inner < iter_inner_max; ++iter_inner) {
      eta = eta_old + (rho * eta_upd);
      mu = link_inv_(eta, family_type);
      dev = dev_resids_(y, mu, 0.0, wt, family_type);
      dev_ratio_inner = (dev - dev_old) / (0.1 + fabs(dev_old));
      dev_crit = is_finite(dev);
      val_crit = (valid_eta_(eta, family_type) && valid_mu_(mu, family_type));
      imp_crit = (dev_ratio_inner <= -dev_tol);
      if (dev_crit && val_crit && imp_crit)
        break;
      rho *= 0.5;
    }
    if (!dev_crit || !val_crit)
      stop("Inner loop failed; cannot correct step size.");
    dev_ratio = fabs(dev - dev_old) / (0.1 + fabs(dev));
    if (dev_ratio < dev_tol)
      break;
    Myadj = Myadj - yadj;
  }
  res.eta = eta;
  return res;
}

#endif // CAPYBARA_GLM_OFFSET
