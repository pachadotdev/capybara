#ifndef CAPYBARA_CENTER_PROJECTIONS_H
#define CAPYBARA_CENTER_PROJECTIONS_H

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

inline void project_2fe(vec &v, const vec &w, const field<uvec> &groups1,
                        const vec &group_inv_w1, const field<uvec> &groups2,
                        const vec &group_inv_w2, bool use_weights) {
  project_1fe(v, w, groups1, group_inv_w1, use_weights);
  project_1fe(v, w, groups2, group_inv_w2, use_weights);
}

// Optimized projection for K fixed effects using cache structure
inline void project_Kfe_optimized(vec &v, const vec &w,
                                  const indices_info &indices,
                                  const field<vec> &group_inv_w,
                                  bool use_weights) {
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
