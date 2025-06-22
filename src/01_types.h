#ifndef CAPYBARA_TYPES_H
#define CAPYBARA_TYPES_H

enum family_type {
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN,
  UNKNOWN
};

struct single_fe_indices {
  uvec all_indices;
  uvec group_offsets;
  uvec group_sizes;

  inline subview_col<uword> get_group(size_t j) const {
    if (group_sizes(j) == 0) {
      return all_indices.head(0); // empty subview
    }
    size_t start = group_offsets(j);
    size_t count = group_sizes(j);
    return all_indices.subvec(start, start + count - 1);
  }
};

struct indices_info {
  uvec all_indices;
  uvec group_offsets;
  uvec group_sizes;
  uvec fe_offsets;
  uvec fe_sizes;
  field<uvec> nonempty_groups;
  field<field<uvec>> precomputed_groups;

  // Cache-optimized sorted indices
  field<field<uvec>> sorted_groups;
  field<uvec> group_order;
  bool cache_optimized = false;

  inline const uvec &get_group(size_t k, size_t j) const {
    if (cache_optimized) {
      return sorted_groups(k)(j);
    }
    return precomputed_groups(k)(j);
  }

  // Cache-optimized group access methods
  inline const uvec &get_sorted_group(size_t k, size_t j) const {
    // Check bounds
    if (cache_optimized && k < sorted_groups.n_elem &&
        j < sorted_groups(k).n_elem) {
      return sorted_groups(k)(j);
    }
    // Fallback to regular precomputed groups
    return precomputed_groups(k)(j);
  }

  inline const uvec &get_group_processing_order(size_t k) const {
    // Check bounds before accessing cache-optimized fields
    if (cache_optimized && k < group_order.n_elem &&
        !group_order(k).is_empty()) {
      return group_order(k);
    }

    // Fallback: create sequential order
    static thread_local uvec fallback_order;
    if (k < fe_sizes.n_elem && fe_sizes(k) > 0) {
      fallback_order = regspace<uvec>(0, fe_sizes(k) - 1);
      return fallback_order;
    }

    // Return empty if no groups
    static const uvec empty_vec;
    return empty_vec;
  }

  void compute_nonempty_groups() {
    nonempty_groups = field<uvec>(fe_sizes.n_elem);
    for (size_t k = 0; k < fe_sizes.n_elem; ++k) {
      const auto span_k = span(fe_offsets(k), fe_offsets(k) + fe_sizes(k) - 1);
      nonempty_groups(k) =
          (fe_sizes(k) > 0) ? find(group_sizes(span_k) > 0) : uvec{};
    }
  }

  void precompute_all_groups() {
    precomputed_groups = field<field<uvec>>(fe_sizes.n_elem);
    for (size_t k = 0; k < fe_sizes.n_elem; ++k) {
      precomputed_groups(k) = field<uvec>(fe_sizes(k));
      for (size_t j = 0; j < fe_sizes(k); ++j) {
        size_t fe_start = fe_offsets(k);
        size_t group_idx = fe_start + j;
        size_t start = group_offsets(group_idx);
        size_t count = group_sizes(group_idx);
        if (count > 0) {
          precomputed_groups(k)(j) =
              all_indices.subvec(start, start + count - 1);
        } else {
          precomputed_groups(k)(j) = uvec(); // empty
        }
      }
    }
  }

  void optimize_cache_access() {
    if (cache_optimized)
      return; // Already optimized

    const uword K = fe_sizes.n_elem;
    sorted_groups = field<field<uvec>>(K);
    group_order = field<uvec>(K);

    for (uword k = 0; k < K; ++k) {
      const uword J = fe_sizes(k);
      sorted_groups(k) = field<uvec>(J);

      // Create (first_index, group_id) pairs for sorting
      std::vector<std::pair<uword, uword>> group_starts;
      group_starts.reserve(J);

      for (uword j = 0; j < J; ++j) {
        uvec grp = precomputed_groups(k)(j);
        if (!grp.is_empty()) {
          // Sort indices within group for sequential access
          sorted_groups(k)(j) = sort(grp);
          group_starts.emplace_back(grp.min(), j);
        } else {
          sorted_groups(k)(j) = uvec();
        }
      }

      // Sort groups by their first memory location
      std::sort(group_starts.begin(), group_starts.end());

      // Create processing order
      group_order(k).set_size(group_starts.size());
      for (size_t i = 0; i < group_starts.size(); ++i) {
        group_order(k)(i) = group_starts[i].second;
      }
    }

    cache_optimized = true;
  }
};

struct crossproduct_results {
  mat XW;
  crossproduct_results(size_t n, size_t p) : XW(n, p, fill::none) {}
};

struct beta_results {
  mat XtX;
  vec XtY;
  mat decomp; // either L (Cholesky) or R (QR)
  vec work;   // either z (Cholesky) or QtY (QR)
  mat Xt;     // avoid repeated transposes
  mat Q;      // for QR
  mat XW;     // weighted X
  vec coefficients;
  uvec valid_coefficients;

  beta_results(size_t n, size_t p)
      : XtX(p, p, fill::none), XtY(p, fill::none), decomp(p, p, fill::none),
        work(p, fill::none), Xt(p, n, fill::none), Q(p, 0, fill::none),
        XW(n, 0, fill::none), coefficients(p, fill::zeros),
        valid_coefficients(p, fill::ones) {}
};

struct felm_results {
  vec coefficients;
  uvec valid_coefficients;
  vec fitted_values;
  vec weights;
  mat hessian;

  template <typename V1, typename V2, typename V3, typename V4, typename M>
  felm_results(V1 &&coef, V2 &&valid_coef, V3 &&fitted, V4 &&w, M &&H)
      : coefficients(std::forward<V1>(coef)),
        valid_coefficients(std::forward<V2>(valid_coef)),
        fitted_values(std::forward<V3>(fitted)), weights(std::forward<V4>(w)),
        hessian(std::forward<M>(H)) {}

  list to_list() const {
    return writable::list({"coefficients"_nm = as_doubles(coefficients),
                           "fitted.values"_nm = as_doubles(fitted_values),
                           "weights"_nm = as_doubles(weights),
                           "hessian"_nm = as_doubles_matrix(hessian)});
  }
};

struct glm_workspace {
  // Work vectors
  vec exp_eta;
  vec w;
  vec nu;
  vec nu_old;
  vec mu;
  vec xi;
  vec var_mu;
  vec beta_upd;
  vec eta_upd;
  vec eta_full;
  vec beta_full;
  vec mu_full;
  vec yadj;

  // Work matrices
  mat MNU_accum;
  mat MNU;
  mat H;
  mat MX_work; // avoid copies

  // Sub-workspaces
  beta_results beta_ws;
  crossproduct_results cross_ws;

  // Line search working vectors
  vec eta_candidate;
  vec beta_candidate;
  vec mu_candidate;
  vec exp_eta_candidate;
  vec dev_vec_work;
  vec ratio_work;

  // Reusable index vectors
  uvec valid_idx;
  uvec invalid_idx;

  glm_workspace(size_t n, size_t p)
      : exp_eta(n, fill::none), w(n, fill::none), nu(n, fill::none),
        nu_old(n, fill::zeros), mu(n, fill::none), xi(n, fill::none),
        var_mu(n, fill::none), beta_upd(p, fill::none), eta_upd(n, fill::none),
        eta_full(n, fill::none), beta_full(p, fill::none),
        mu_full(n, fill::none), yadj(n, fill::none),
        MNU_accum(n, 1, fill::zeros), MNU(n, 1, fill::none),
        H(p, p, fill::none), MX_work(n, p, fill::none), beta_ws(n, p),
        cross_ws(n, p), eta_candidate(n, fill::none),
        beta_candidate(p, fill::none), mu_candidate(n, fill::none),
        exp_eta_candidate(n, fill::none), dev_vec_work(n, fill::none),
        ratio_work(n, fill::none) {}
};

static inline void reserve_glm_workspace(glm_workspace &ws, uword N, uword P) {
  // Pre-allocate workspace once
  ws.xi.set_size(N);
  ws.var_mu.set_size(N);
  ws.w.set_size(N);
  ws.nu.set_size(N);
  ws.nu_old.set_size(N);
  ws.MNU.set_size(N);
  ws.MNU_accum.set_size(N);
  ws.eta_upd.set_size(N);
  ws.eta_full.set_size(N);
  ws.beta_upd.set_size(P);
  ws.beta_full.set_size(P);
  ws.mu.set_size(N);
  ws.mu_full.set_size(N);
  ws.MX_work.set_size(N, P);
}

struct feglm_results {
  vec coefficients;
  uvec valid_coefficients;
  vec linear_predictors;
  vec weights;
  mat hessian;
  double deviance;
  double null_deviance;
  bool convergence;
  size_t iterations;
  mat centered_matrix;

  template <typename V1, typename V2, typename V3, typename V4, typename M,
            typename D1, typename D2, typename B, typename I>
  feglm_results(V1 &&coef, V2 &&valid_coef, V3 &&eta_pred, V4 &&wt, M &&H,
                D1 &&dev, D2 &&null_dev, B &&conv, I &&iter)
      : coefficients(std::forward<V1>(coef)),
        valid_coefficients(std::forward<V2>(valid_coef)),
        linear_predictors(std::forward<V3>(eta_pred)),
        weights(std::forward<V4>(wt)), hessian(std::forward<M>(H)),
        deviance(std::forward<D1>(dev)),
        null_deviance(std::forward<D2>(null_dev)),
        convergence(std::forward<B>(conv)), iterations(std::forward<I>(iter)) {}

  list to_list(const bool &keep_mx = false) const {
    writable::list out = writable::list({
        "coefficients"_nm = as_doubles(coefficients),
        "linear.predictors"_nm = as_doubles(linear_predictors),
        "weights"_nm = as_doubles(weights),
        "hessian"_nm = as_doubles_matrix(hessian),
        "deviance"_nm = writable::doubles({deviance}),
        "null_deviance"_nm = writable::doubles({null_deviance}),
        "conv"_nm = writable::logicals({convergence}),
        "iter"_nm = writable::integers({static_cast<int>(iterations)}),
    });

    if (keep_mx && !centered_matrix.is_empty()) {
      out.push_back({"MX"_nm = as_doubles_matrix(centered_matrix)});
    }

    return out;
  }
};

struct feglm_offset_results {
  vec coefficients;
  uvec valid_coefficients;

  template <typename V1, typename V2>
  feglm_offset_results(V1 &&coefs, V2 &&valid_coefs)
      : coefficients(std::forward<V1>(coefs)),
        valid_coefficients(std::forward<V2>(valid_coefs)) {}
};

struct solve_alpha_results {
  field<vec> alpha;

  solve_alpha_results(field<vec> &&a) : alpha(std::move(a)) {}

  list to_list() const {
    const size_t K = alpha.n_elem;
    writable::list alpha_r(K);
    for (size_t k = 0; k < K; ++k) {
      alpha_r[k] = as_doubles_matrix(alpha(k).eval());
    }
    return alpha_r;
  }
};

#endif // CAPYBARA_TYPES_H
