#ifndef CAPYBARA_CENTER_PROJECTIONS_H
#define CAPYBARA_CENTER_PROJECTIONS_H

// #include "timing.h" // development only, for profiling

// Adaptive acceleration strategy selection based on system size and complexity
inline void select_acceleration_strategy(center_workspace &ws,
                                         const indices_info &indices, size_t N,
                                         size_t K) {
  // Use enhanced acceleration for large systems with many fixed effects
  if (K >= 3 && N >= 2000) {
    ws.use_enhanced_acceleration = true;
    ws.acceleration_damping = 0.7; // More conservative for complex systems
    ws.history_size = 2;           // Smaller history for memory efficiency
  } else if (K >= 2 && N >= 1000) {
    ws.use_enhanced_acceleration = true;
    ws.acceleration_damping = 0.8;
    ws.history_size = 3;
  } else {
    ws.use_enhanced_acceleration = false; // Use memory-efficient version
  }

  // Adjust damping based on cache optimization
  if (indices.cache_optimized) {
    ws.acceleration_damping *=
        1.1; // Slightly more aggressive with optimized access
  }
}

inline void init_center_workspace(center_workspace &ws,
                                  const indices_info &indices, const vec &w,
                                  size_t N) {
  const size_t K = indices.fe_sizes.n_elem;
  ws.x.set_size(N);
  ws.x0.set_size(N);
  ws.Gx.set_size(N);
  ws.G2x.set_size(N);
  ws.deltaG.set_size(N);
  ws.delta2.set_size(N);

  ws.group_indices.set_size(K);
  ws.group_inv_w.set_size(K);
  ws.max_groups = 0;
  const bool use_weights = (w.n_elem > 1);

  // Enhanced acceleration setup for K>=2 and large N
  select_acceleration_strategy(ws, indices, N, K);
  if (ws.use_enhanced_acceleration) {
    ws.acceleration_history.set_size(N, ws.history_size);
    ws.acceleration_weights.set_size(ws.history_size);
    ws.acceleration_weights.fill(1.0 / ws.history_size);
    ws.history_pos = 0;
  }

  for (size_t k = 0; k < K; ++k) {
    const size_t J = indices.fe_sizes(k);
    ws.max_groups = std::max(ws.max_groups, J);
    field<uvec> idxs(J);
    vec invs(J);

    for (size_t j = 0; j < J; ++j) {
      idxs(j) = indices.get_group(k, j);
      if (!idxs(j).is_empty()) {
        if (use_weights) {
          const double sum_w = accu(w(idxs(j)));
          invs(j) = (sum_w > 0) ? (1.0 / sum_w) : 0.0;
        } else {
          invs(j) = (idxs(j).n_elem > 0) ? (1.0 / idxs(j).n_elem) : 0.0;
        }
      } else {
        invs(j) = 0.0;
      }
    }
    ws.group_indices(k) = std::move(idxs);
    ws.group_inv_w(k) = std::move(invs);
  }
  ws.group_means.set_size(ws.max_groups);
}

inline void project_1fe(vec &v, const vec &w, const field<uvec> &groups,
                        const vec &group_inv_w, bool use_weights) {
  const size_t L = groups.n_elem;
  if (use_weights) {
    for (size_t l = 0; l < L; ++l) {
      const uvec &coords = groups(l);
      if (coords.is_empty())
        continue;
      const double mean_val =
          dot(w.elem(coords), v.elem(coords)) * group_inv_w(l);
      v.elem(coords) -= mean_val;
    }
  } else {
    for (size_t l = 0; l < L; ++l) {
      const uvec &coords = groups(l);
      if (coords.is_empty())
        continue;
      const double mean_val = mean(v.elem(coords));
      v.elem(coords) -= mean_val;
    }
  }
}

// Cache-optimized projection using indices_info structure
inline void project_1_to_K_fe(vec &v, const vec &w, const indices_info &indices,
                              size_t k, const vec &group_inv_w,
                              bool use_weights) {
  if (use_weights) {
    indices.iterate_groups_cached(k, [&](size_t j, const uvec &coords) {
      if (!coords.is_empty()) {
        const double mean_val =
            dot(w.elem(coords), v.elem(coords)) * group_inv_w(j);
        v.elem(coords) -= mean_val;
      }
    });
  } else {
    indices.iterate_groups_cached(k, [&](size_t j, const uvec &coords) {
      if (!coords.is_empty()) {
        const double mean_val = mean(v.elem(coords));
        v.elem(coords) -= mean_val;
      }
    });
  }
}

inline void absorb_2fe(arma::vec &y, const arma::uvec &fe1,
                       const arma::uvec &fe2, const arma::vec &w) {
  const size_t N = y.n_elem;
  const size_t G1 = fe1.max() + 1;
  const size_t G2 = fe2.max() + 1;
  const bool weighted = (w.n_elem == N);

  arma::vec mean1 = arma::zeros<arma::vec>(G1);
  arma::vec mean2 = arma::zeros<arma::vec>(G2);
  arma::vec wsum1 = arma::zeros<arma::vec>(G1);
  arma::vec wsum2 = arma::zeros<arma::vec>(G2);

  double grand_sum = 0.0, grand_wsum = 0.0;
  for (size_t i = 0; i < N; ++i) {
    double wi = weighted ? w(i) : 1.0;
    mean1(fe1(i)) += wi * y(i);
    mean2(fe2(i)) += wi * y(i);
    wsum1(fe1(i)) += wi;
    wsum2(fe2(i)) += wi;
    grand_sum += wi * y(i);
    grand_wsum += wi;
  }
  for (size_t g = 0; g < G1; ++g)
    if (wsum1(g) > 0)
      mean1(g) /= wsum1(g);
  for (size_t g = 0; g < G2; ++g)
    if (wsum2(g) > 0)
      mean2(g) /= wsum2(g);
  double grand_mean = grand_sum / grand_wsum;

  for (size_t i = 0; i < N; ++i)
    y(i) = y(i) - mean1(fe1(i)) - mean2(fe2(i)) + grand_mean;
}

inline void project_2fe(vec &v, const vec &w, const field<uvec> &groups1,
                        const vec &group_inv_w1, const field<uvec> &groups2,
                        const vec &group_inv_w2, bool use_weights) {
  // Build group id vectors for each FE
  size_t N = v.n_elem;
  arma::uvec fe1(N), fe2(N);
  for (size_t g = 0; g < groups1.n_elem; ++g) {
    const arma::uvec &idx = groups1(g);
    for (size_t i = 0; i < idx.n_elem; ++i)
      fe1(idx(i)) = g;
  }
  for (size_t g = 0; g < groups2.n_elem; ++g) {
    const arma::uvec &idx = groups2(g);
    for (size_t i = 0; i < idx.n_elem; ++i)
      fe2(idx(i)) = g;
  }
  absorb_2fe(v, fe1, fe2, w);
}

// Optimized projection for K fixed effects using cache structure
inline void project_Kfe_optimized(vec &v, const vec &w,
                                  const indices_info &indices,
                                  const field<vec> &group_inv_w,
                                  bool use_weights) {
  // TIME_FUNCTION;
  const size_t K = indices.fe_sizes.n_elem;
  for (size_t k = 0; k < K; ++k) {
    project_1_to_K_fe(v, w, indices, k, group_inv_w(k), use_weights);
  }
}

inline void project_Kfe(vec &v, const vec &w,
                        const field<field<uvec>> &group_indices,
                        const field<vec> &group_inv_w, bool use_weights) {
  const size_t K = group_indices.n_elem;
  for (size_t k = 0; k < K; ++k) {
    project_1fe(v, w, group_indices(k), group_inv_w(k), use_weights);
  }
}

#endif // CAPYBARA_CENTER_PROJECTIONS_H
