#ifndef CAPYBARA_TYPES_H
#define CAPYBARA_TYPES_H

// Enum for supported GLM family types
enum family_type {
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN,
  UNKNOWN
};

// Structure for single fixed effect group indices
struct single_fe_indices {
  uvec all_indices;   // All observation indices
  uvec group_offsets; // Start of each group in all_indices
  uvec group_sizes;   // Size of each group

  // Get indices for group j
  inline subview_col<uword> get_group(size_t j) const {
    if (group_sizes(j) == 0) {
      return all_indices.head(0);
    }
    const size_t start = group_offsets(j);
    const size_t count = group_sizes(j);
    return all_indices.subvec(start, start + count - 1);
  }
};

// Structure for multiple fixed effects group indices and cache optimization
struct indices_info {
  uvec all_indices;                      // All observation indices
  uvec group_offsets;                    // Start of each group in all_indices
  uvec group_sizes;                      // Size of each group
  uvec fe_offsets;                       // Start of each FE in group arrays
  uvec fe_sizes;                         // Number of groups per FE
  field<uvec> nonempty_groups;           // Indices of nonempty groups per FE
  field<field<uvec>> precomputed_groups; // Precomputed group indices

  field<field<uvec>> sorted_groups; // Cache-optimized group indices
  field<uvec> group_order;          // Cache-optimized group order
  bool cache_optimized = false;     // Whether cache optimization is enabled

  static constexpr size_t cache_line_size = 64;
  static constexpr size_t optimal_chunk = cache_line_size / sizeof(uword);

  // Get group indices for FE k, group j
  inline const uvec &get_group(size_t k, size_t j) const {
    if (cache_optimized) {
      if (j + 1 < sorted_groups(k).n_elem) {
        __builtin_prefetch(&sorted_groups(k)(j + 1), 0, 1);
      }
      return sorted_groups(k)(j);
    }
    return precomputed_groups(k)(j);
  }

  // Get sorted group indices for FE k, group j
  inline const uvec &get_sorted_group(size_t k, size_t j) const {
    if (cache_optimized && k < sorted_groups.n_elem &&
        j < sorted_groups(k).n_elem) {
      return sorted_groups(k)(j);
    }
    return precomputed_groups(k)(j);
  }

  // Get group processing order for FE k
  inline const uvec &get_group_processing_order(size_t k) const {
    if (cache_optimized && k < group_order.n_elem &&
        !group_order(k).is_empty()) {
      return group_order(k);
    }

    static thread_local uvec fallback_order;
    if (k < fe_sizes.n_elem && fe_sizes(k) > 0) {
      fallback_order = regspace<uvec>(0, fe_sizes(k) - 1);
      return fallback_order;
    }

    static const uvec empty_vec;
    return empty_vec;
  }

  // Compute nonempty groups for each FE
  void compute_nonempty_groups() {
    nonempty_groups = field<uvec>(fe_sizes.n_elem);
    for (size_t k = 0; k < fe_sizes.n_elem; ++k) {
      const auto span_k = span(fe_offsets(k), fe_offsets(k) + fe_sizes(k) - 1);
      nonempty_groups(k) =
          (fe_sizes(k) > 0) ? find(group_sizes(span_k) > 0) : uvec{};
    }
  }

  // Precompute all group indices for each FE
  void precompute_all_groups() {
    precomputed_groups = field<field<uvec>>(fe_sizes.n_elem);

    for (size_t k = 0; k < fe_sizes.n_elem; ++k) {
      precomputed_groups(k) = field<uvec>(fe_sizes(k));
      const size_t fe_start = fe_offsets(k);

      for (size_t j = 0; j < fe_sizes(k); ++j) {
        const size_t group_idx = fe_start + j;
        const size_t start = group_offsets(group_idx);
        const size_t count = group_sizes(group_idx);
        if (count > 0) {
          precomputed_groups(k)(j) =
              all_indices.subvec(start, start + count - 1);
        } else {
          precomputed_groups(k)(j) = uvec();
        }
      }
    }

    optimize_cache_access_internal();
  }

  // Enable cache optimization for group access
  void optimize_cache_access() { optimize_cache_access_internal(); }

  // Iterate over groups in cache-optimized order
  template <typename Func>
  void iterate_groups_cached(size_t k, Func &&func) const {
    if (cache_optimized && k < group_order.n_elem) {
      const uvec &order = group_order(k);
      for (uword i = 0; i < order.n_elem; ++i) {
        const uword j = order(i);
        func(j, sorted_groups(k)(j));
      }
    } else {
      for (size_t j = 0; j < fe_sizes(k); ++j) {
        if (!precomputed_groups(k)(j).is_empty()) {
          func(j, precomputed_groups(k)(j));
        }
      }
    }
  }

private:
  // Helper struct for cache optimization
  struct GroupInfo {
    uword idx;
    uword min_val;
    uword max_val;
    uword size;
    double density;
  };

  // Internal: optimize group access for cache
  void optimize_cache_access_internal() {
    if (cache_optimized)
      return;

    const uword K = fe_sizes.n_elem;
    sorted_groups = field<field<uvec>>(K);
    group_order = field<uvec>(K);

    for (uword k = 0; k < K; ++k) {
      const uword J = fe_sizes(k);
      sorted_groups(k) = field<uvec>(J);

      std::vector<GroupInfo> group_infos;
      group_infos.reserve(J);

      for (uword j = 0; j < J; ++j) {
        const uvec &grp = precomputed_groups(k)(j);
        if (!grp.is_empty()) {
          const uword min_idx = grp.min();
          const uword max_idx = grp.max();
          const uword range = max_idx - min_idx + 1;

          sorted_groups(k)(j) = sort(grp);

          group_infos.push_back({j, min_idx, max_idx, grp.n_elem,
                                 static_cast<double>(grp.n_elem) / range});
        } else {
          sorted_groups(k)(j) = uvec();
        }
      }

      std::sort(group_infos.begin(), group_infos.end(),
                [](const GroupInfo &a, const GroupInfo &b) {
                  const uword block_a = a.min_val / optimal_chunk;
                  const uword block_b = b.min_val / optimal_chunk;
                  if (block_a != block_b)
                    return block_a < block_b;

                  if (std::abs(a.density - b.density) > 0.1) {
                    return a.density > b.density;
                  }

                  return a.size < b.size;
                });

      group_order(k).set_size(group_infos.size());
      for (size_t i = 0; i < group_infos.size(); ++i) {
        group_order(k)(i) = group_infos[i].idx;
      }
    }

    cache_optimized = true;
  }
};

// M is either XtX, (XW)t(XW), H
struct crossproduct_results {
  mat M;
  crossproduct_results(size_t n, size_t p) : M(n, p, fill::none) {}
};

struct beta_results {
  mat XtX;
  vec XtY;
  mat decomp;
  vec work;
  mat Xt;
  mat Q;
  mat XW;
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

  mat MNU_accum;
  mat MNU;
  mat H;
  mat MX_work;

  beta_results beta_ws;
  crossproduct_results cross_ws;

  vec eta_candidate;
  vec beta_candidate;
  vec mu_candidate;
  vec exp_eta_candidate;
  vec dev_vec_work;
  vec ratio_work;

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
