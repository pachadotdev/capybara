// Computing alpha in a model with fixed effects Y = alpha + X beta given beta
#ifndef CAPYBARA_ALPHA_H
#define CAPYBARA_ALPHA_H

namespace capybara {

struct InferenceAlpha {
  field<vec> coefficients;
  uvec nb_references;
  bool is_regular;
  bool success;
  field<std::string> fe_names;
  field<field<std::string>> fe_levels;

  InferenceAlpha() : is_regular(true), success(false) {}
};

// get_alpha now takes a const FlatFEMap& directly (no duplicate struct)
inline field<vec> get_alpha(const vec &pi, const FlatFEMap &map,
                            double tol = 1e-8, uword iter_max = 10000,
                            const vec *weights = nullptr) {
  const uword K = map.K;
  const uword N = pi.n_elem;
  field<vec> coefficients(K);

  if (K == 0 || N == 0) {
    return coefficients;
  }

  // Initialize coefficients to zero
  for (uword k = 0; k < K; ++k) {
    coefficients(k).zeros(map.n_groups[k]);
  }

  // Build inverse weights for alpha recovery
  // (may differ from centering weights if called with different w)
  std::vector<vec> alpha_inv_weights(K);
  const bool use_w = (weights != nullptr && weights->n_elem == N);
  const double *w_ptr = use_w ? weights->memptr() : nullptr;

  for (uword k = 0; k < K; ++k) {
    alpha_inv_weights[k].zeros(map.n_groups[k]);
    double *inv_w_ptr = alpha_inv_weights[k].memptr();
    const uword *map_k = map.fe_map[k].data();

    if (use_w) {
      for (uword i = 0; i < N; ++i) {
        inv_w_ptr[map_k[i]] += w_ptr[i];
      }
    } else {
      for (uword i = 0; i < N; ++i) {
        inv_w_ptr[map_k[i]] += 1.0;
      }
    }

    // Invert
    for (uword g = 0; g < map.n_groups[k]; ++g) {
      inv_w_ptr[g] = (inv_w_ptr[g] > 1e-12) ? (1.0 / inv_w_ptr[g]) : 0.0;
    }
  }

  // Residual: r = pi - sum_k(alpha_k expanded)
  vec r = pi;
  double *r_ptr_w = r.memptr();

  double crit = 1.0;
  uword iter = 0;

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (uword k = 0; k < K; ++k) {
      const uword *gk = map.fe_map[k].data();
      const uword J = map.n_groups[k];
      const double *inv_wk = alpha_inv_weights[k].memptr();
      vec &alpha_k = coefficients(k);
      double *ak_ptr = alpha_k.memptr();

      sum_sq0 += dot(alpha_k, alpha_k);

      // Compute new alpha using scatter-gather with raw pointers
      vec new_alpha(J, fill::zeros);
      double *new_ak_ptr = new_alpha.memptr();

      // Accumulate: new_alpha[g] += w[i] * (r[i] + alpha_k[g[i]])
      if (use_w) {
        for (uword i = 0; i < N; ++i) {
          uword g = gk[i];
          new_ak_ptr[g] += w_ptr[i] * (r_ptr_w[i] + ak_ptr[g]);
        }
      } else {
        for (uword i = 0; i < N; ++i) {
          uword g = gk[i];
          new_ak_ptr[g] += r_ptr_w[i] + ak_ptr[g];
        }
      }

      // Apply inverse weights
      for (uword j = 0; j < J; ++j) {
        new_ak_ptr[j] *= inv_wk[j];
      }

      // Compute diff and update criterion
      double local_diff_sq = 0.0;
      for (uword j = 0; j < J; ++j) {
        double d = new_ak_ptr[j] - ak_ptr[j];
        local_diff_sq += d * d;
      }
      sum_sq_diff += local_diff_sq;

      // Update residual: r[i] -= (new_alpha[g[i]] - alpha_k[g[i]])
      for (uword i = 0; i < N; ++i) {
        uword g = gk[i];
        r_ptr_w[i] -= new_ak_ptr[g] - ak_ptr[g];
      }

      // Update coefficients
      alpha_k = std::move(new_alpha);
    }

    // Convergence criterion
    crit = (sum_sq0 > 0.0) ? std::sqrt(sum_sq_diff / sum_sq0)
                           : std::sqrt(sum_sq_diff);

    ++iter;
  }

  // Normalize: shift so first level of last FE is zero (fixest/Stata
  // convention)
  if (K > 0 && coefficients(K - 1).n_elem > 0) {
    const double shift = coefficients(K - 1)(0);
    coefficients(K - 1) -= shift;
    coefficients(0) += shift;
  }

  return coefficients;
}

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
