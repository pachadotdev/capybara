#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

struct center_workspace {
  // Basic projection workspace vectors
  vec x;      // Current iterate
  vec x0;     // Previous iterate
  vec Gx;     // Single projection step G(x)
  vec G2x;    // Double projection step G(G(x))
  vec deltaG; // Difference for acceleration
  vec delta2; // Second difference for acceleration

  // Group structure storage
  field<field<uvec>> group_indices; // Indices for each group in each FE
  field<vec> group_inv_w;           // Inverse weights for each group
  vec group_means;                  // Temporary storage for group means
  double ratio0;                    // Convergence ratio tracking
  double ssr0;                      // Sum of squares tracking
  size_t max_groups;                // Maximum number of groups across FEs

  // Irons-Tuck acceleration workspace for K>=2 systems
  mat acceleration_history;       // Store last few iterations for enhanced
                                  // acceleration
  vec acceleration_weights;       // Adaptive weights for combining history
  size_t history_size;            // Number of previous iterates to store
  size_t history_pos;             // Current position in circular buffer
  bool use_enhanced_acceleration; // Whether to use enhanced vs memory-efficient
                                  // mode
  double acceleration_damping;    // Damping factor for stability (0.7-0.8)
  double min_acceleration_norm;   // Minimum norm threshold for acceleration

  center_workspace()
      : ratio0(datum::inf), ssr0(datum::inf), max_groups(0), history_size(3),
        history_pos(0), use_enhanced_acceleration(false),
        acceleration_damping(0.8), min_acceleration_norm(1e-12) {}
};

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

inline void project_2fe(vec &v, const vec &w, const field<uvec> &groups1,
                        const vec &group_inv_w1, const field<uvec> &groups2,
                        const vec &group_inv_w2, bool use_weights) {
  project_1fe(v, w, groups1, group_inv_w1, use_weights);
  project_1fe(v, w, groups2, group_inv_w2, use_weights);
}

inline void project_Kfe(vec &v, const vec &w,
                        const field<field<uvec>> &group_indices,
                        const field<vec> &group_inv_w, bool use_weights) {
  const size_t K = group_indices.n_elem;
  for (size_t k = 0; k < K; ++k) {
    project_1fe(v, w, group_indices(k), group_inv_w(k), use_weights);
  }
}

// Enhanced Irons-Tuck acceleration for K>=2 systems with stability controls
inline bool apply_enhanced_acceleration(vec &v, center_workspace &ws,
                                        const vec &v_old, size_t iter,
                                        const vec &w,
                                        const indices_info &indices,
                                        bool use_weights) {
  if (!ws.use_enhanced_acceleration || iter < 3) {
    return false;
  }

  // Store current iterate in history
  ws.acceleration_history.col(ws.history_pos) = v;

  // Only accelerate every few iterations to avoid instability
  if ((iter % 3) != 0) {
    ws.history_pos = (ws.history_pos + 1) % ws.history_size;
    return false;
  }

  // Compute differences for acceleration
  const vec delta1 = v - v_old;
  const double delta1_norm = norm(delta1, 2);

  if (delta1_norm < ws.min_acceleration_norm) {
    ws.history_pos = (ws.history_pos + 1) % ws.history_size;
    return false;
  }

  // Apply one more projection step using optimized projection
  ws.Gx = v;
  if (indices.cache_optimized) {
    project_Kfe_optimized(ws.Gx, w, indices, ws.group_inv_w, use_weights);
  } else {
    project_Kfe(ws.Gx, w, ws.group_indices, ws.group_inv_w, use_weights);
  }

  const vec delta2 = ws.Gx - v;
  const double delta2_norm = norm(delta2, 2);

  if (delta2_norm < ws.min_acceleration_norm) {
    ws.history_pos = (ws.history_pos + 1) % ws.history_size;
    return false;
  }

  // Compute acceleration coefficient with damping for stability
  const vec ddelta = delta2 - delta1;
  const double ddelta_sq = dot(ddelta, ddelta);

  if (ddelta_sq > ws.min_acceleration_norm) {
    const double alpha = dot(delta1, ddelta) / ddelta_sq;
    const double damped_alpha = ws.acceleration_damping * alpha;

    // Apply bounds for stability
    if (damped_alpha > 0.1 && damped_alpha < 1.5) {
      v = ws.Gx - damped_alpha * delta2;
      ws.history_pos = (ws.history_pos + 1) % ws.history_size;
      return true;
    }
  }

  // Fallback to regular step
  v = ws.Gx;
  ws.history_pos = (ws.history_pos + 1) % ws.history_size;
  return false;
}

// Memory-efficient acceleration for large systems (Anderson acceleration
// variant)
inline bool apply_memory_efficient_acceleration(vec &v, center_workspace &ws,
                                                const vec &v_old, size_t iter,
                                                const vec &w,
                                                const indices_info &indices,
                                                bool use_weights) {
  if (iter < 5 || (iter % 5) != 0) {
    return false;
  }

  // Single-step Anderson acceleration variant
  ws.x = v;
  ws.Gx = ws.x;
  if (indices.cache_optimized) {
    project_Kfe_optimized(ws.Gx, w, indices, ws.group_inv_w, use_weights);
  } else {
    project_Kfe(ws.Gx, w, ws.group_indices, ws.group_inv_w, use_weights);
  }

  ws.G2x = ws.Gx;
  if (indices.cache_optimized) {
    project_Kfe_optimized(ws.G2x, w, indices, ws.group_inv_w, use_weights);
  } else {
    project_Kfe(ws.G2x, w, ws.group_indices, ws.group_inv_w, use_weights);
  }

  ws.deltaG = ws.G2x - ws.x;
  ws.delta2 = ws.G2x - 2.0 * ws.x + v_old;

  const double ssq = dot(ws.delta2, ws.delta2);

  if (ssq > 1e-10) {
    const double coef = dot(ws.deltaG, ws.delta2) / ssq;
    if (coef > 0.0 && coef < 2.0) {
      v = ws.G2x - coef * ws.deltaG;
      return true;
    } else {
      v = ws.G2x;
      return true;
    }
  }

  return false;
}

inline void center_mat_or_vec(mat *X, vec *y, const vec &w,
                              const indices_info &indices, double tol,
                              size_t max_iter, size_t iter_interrupt,
                              size_t iter_ssr, bool use_acceleration) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  const size_t N = (X != nullptr) ? X->n_rows : y->n_elem;
  const size_t P = (X != nullptr) ? X->n_cols : 0;
  const bool has_X = (X != nullptr && P > 0);
  const bool has_y = (y != nullptr && y->n_elem > 0);

  if (!has_X && !has_y)
    return;

  center_workspace ws;
  init_center_workspace(ws, indices, w, N);
  const bool use_weights = (w.n_elem > 1);
  const double inv_sw = use_weights ? 1.0 / accu(w) : 1.0 / N;

  bool y_converged = !has_y;
  uvec x_converged(P, fill::ones);
  if (has_X)
    x_converged.fill(0);

  vec y_old;
  mat X_old;
  if (has_y)
    y_old.set_size(N);
  if (has_X)
    X_old.set_size(N, P);

  double y_ssr0 = datum::inf;
  vec x_ssr0(P);
  if (has_X)
    x_ssr0.fill(datum::inf);

  size_t iint = iter_interrupt;
  const size_t isr0 = iter_ssr;

  size_t y_isr = isr0;
  vec x_isr(P);
  if (has_X)
    x_isr.fill(isr0);

  for (size_t iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    bool all_x_converged = true;
    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (!x_converged(p)) {
          all_x_converged = false;
          break;
        }
      }
    }
    if (y_converged && all_x_converged)
      break;

    if (has_y && !y_converged) {
      y_old = *y;
    }
    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (!x_converged(p)) {
          X_old.col(p) = X->col(p);
        }
      }
    }

    if (has_y && !y_converged) {
      if (K == 1) {
        project_1fe(*y, w, ws.group_indices(0), ws.group_inv_w(0), use_weights);
      } else if (K == 2) {
        project_2fe(*y, w, ws.group_indices(0), ws.group_inv_w(0),
                    ws.group_indices(1), ws.group_inv_w(1), use_weights);
      } else {
        if (indices.cache_optimized) {
          project_Kfe_optimized(*y, w, indices, ws.group_inv_w, use_weights);
        } else {
          project_Kfe(*y, w, ws.group_indices, ws.group_inv_w, use_weights);
        }
      }
    }

    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (x_converged(p))
          continue;
        vec col_p = X->col(p);

        if (K == 1) {
          project_1fe(col_p, w, ws.group_indices(0), ws.group_inv_w(0),
                      use_weights);
        } else if (K == 2) {
          project_2fe(col_p, w, ws.group_indices(0), ws.group_inv_w(0),
                      ws.group_indices(1), ws.group_inv_w(1), use_weights);
        } else {
          if (indices.cache_optimized) {
            project_Kfe_optimized(col_p, w, indices, ws.group_inv_w,
                                  use_weights);
          } else {
            project_Kfe(col_p, w, ws.group_indices, ws.group_inv_w,
                        use_weights);
          }
        }

        X->col(p) = col_p;
      }
    }

    // Enhanced acceleration for K>=2 and memory-efficient acceleration
    if (use_acceleration) {
      if (has_y && !y_converged) {
        if (ws.use_enhanced_acceleration) {
          apply_enhanced_acceleration(*y, ws, y_old, iter, w, indices,
                                      use_weights);
        } else {
          apply_memory_efficient_acceleration(*y, ws, y_old, iter, w, indices,
                                              use_weights);
        }
      }

      if (has_X) {
        for (size_t p = 0; p < P; ++p) {
          if (x_converged(p))
            continue;

          if (ws.use_enhanced_acceleration) {
            vec col_p = X->col(p);
            if (apply_enhanced_acceleration(col_p, ws, X_old.col(p), iter, w,
                                            indices, use_weights)) {
              X->col(p) = col_p;
            }
          } else {
            vec col_p = X->col(p);
            if (apply_memory_efficient_acceleration(
                    col_p, ws, X_old.col(p), iter, w, indices, use_weights)) {
              X->col(p) = col_p;
            }
          }
        }
      }
    }

    if (has_y && !y_converged) {
      const double dy_norm = norm(*y - y_old, 2);
      const double y0_norm = norm(y_old, 2);
      const double y_ratio = dy_norm / (1.0 + y0_norm);
      if (y_ratio < tol) {
        y_converged = true;
      }
    }

    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (!x_converged(p)) {
          const double dx_norm = norm(X->col(p) - X_old.col(p), 2);
          const double x0_norm = norm(X_old.col(p), 2);
          const double x_ratio = dx_norm / (1.0 + x0_norm);
          if (x_ratio < tol) {
            x_converged(p) = 1;
          }
        }
      }
    }

    if (has_y && !y_converged && iter == y_isr && iter > 0) {
      y_isr += isr0;
      const double y_ssr =
          use_weights ? dot(*y % *y, w) * inv_sw : dot(*y, *y) / N;
      if (y_ssr0 != datum::inf) {
        if (std::fabs(y_ssr - y_ssr0) / (1.0 + std::fabs(y_ssr0)) < tol) {
          y_converged = true;
        }
      }
      y_ssr0 = y_ssr;
    }

    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (!x_converged(p) && iter == x_isr(p) && iter > 0) {
          x_isr(p) += isr0;
          const vec col_p = X->col(p);
          const double x_ssr = use_weights ? dot(col_p % col_p, w) * inv_sw
                                           : dot(col_p, col_p) / N;
          if (x_ssr0(p) != datum::inf) {
            if (std::fabs(x_ssr - x_ssr0(p)) / (1.0 + std::fabs(x_ssr0(p))) <
                tol) {
              x_converged(p) = 1;
            }
          }
          x_ssr0(p) = x_ssr;
        }
      }
    }
  }
}

inline void center_variables(mat &V, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter, size_t iter_interrupt,
                             size_t iter_ssr = 40,
                             bool use_acceleration = true) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  center_mat_or_vec(&V, nullptr, w, indices, tol, max_iter, iter_interrupt,
                    iter_ssr, use_acceleration);
}

inline void center_variables(vec &y, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter, size_t iter_interrupt,
                             size_t iter_ssr = 40,
                             bool use_acceleration = true) {
  center_mat_or_vec(nullptr, &y, w, indices, tol, max_iter, iter_interrupt,
                    iter_ssr, use_acceleration);
}

inline void center_variables(mat &X_work, vec &y, const vec &w,
                             const mat &X_orig, const indices_info &indices,
                             double tol, size_t max_iter, size_t iter_interrupt,
                             size_t iter_ssr, bool use_acceleration) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  if (&X_work != &X_orig) {
    X_work = X_orig;
  }

  center_mat_or_vec(&X_work, &y, w, indices, tol, max_iter, iter_interrupt,
                    iter_ssr, use_acceleration);
}

#endif // CAPYBARA_CENTER_H
