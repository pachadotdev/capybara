#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

struct center_workspace {
  // Pre-allocated vectors for Irons-Tuck acceleration
  vec x;
  vec x0;
  vec Gx;
  vec G2x;
  vec deltaG;
  vec delta2;
  // Group information
  field<field<uvec>> group_indices;
  field<vec> group_inv_w;
  vec group_means;
  // Convergence tracking
  double ratio0;
  double ssr0;
  size_t max_groups;
  center_workspace() : ratio0(datum::inf), ssr0(datum::inf), max_groups(0) {}
};

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
  bool use_weights = (w.n_elem > 1);

  for (size_t k = 0; k < K; ++k) {
    const size_t J = indices.fe_sizes(k);
    ws.max_groups = std::max(ws.max_groups, J);
    field<uvec> idxs(J);
    vec invs(J);

    for (size_t j = 0; j < J; ++j) {
      idxs(j) = indices.get_group(k, j);
      if (!idxs(j).is_empty()) {
        if (use_weights) {
          double sum_w = accu(w(idxs(j)));
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

// K = 1 fixed effect projection
inline void project_1fe(vec &v, const vec &w, const field<uvec> &groups,
                        const vec &group_inv_w, bool use_weights) {
  const size_t L = groups.n_elem;
  if (use_weights) {
    for (size_t l = 0; l < L; ++l) {
      const uvec &coords = groups(l);
      if (coords.is_empty()) continue;
      double mean_val = dot(w(coords), v(coords)) * group_inv_w(l);
      v(coords) -= mean_val;
    }
  } else {
    for (size_t l = 0; l < L; ++l) {
      const uvec &coords = groups(l);
      if (coords.is_empty()) continue;
      double mean_val = mean(v(coords));
      v(coords) -= mean_val;
    }
  }
}

// K = 2 fixed effect projection
inline void project_2fe(vec &v, const vec &w, const field<uvec> &groups1,
                        const vec &group_inv_w1, const field<uvec> &groups2,
                        const vec &group_inv_w2, bool use_weights) {
  project_1fe(v, w, groups1, group_inv_w1, use_weights);
  project_1fe(v, w, groups2, group_inv_w2, use_weights);
}

// K >= 3 fixed effect projection
inline void project_Kfe(vec &v, const vec &w,
                        const field<field<uvec>> &group_indices,
                        const field<vec> &group_inv_w, bool use_weights) {
  const size_t K = group_indices.n_elem;
  for (size_t k = 0; k < K; ++k) {
    project_1fe(v, w, group_indices(k), group_inv_w(k), use_weights);
  }
}

// Unified centering function that handles y vector, X matrix, or both
inline void center_mat_or_vec(mat *X, vec *y, const vec &w,
                                     const indices_info &indices, double tol,
                                     size_t max_iter, size_t iter_interrupt,
                                     size_t iter_ssr, bool use_acceleration) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;

  const size_t N = w.n_elem;
  const size_t P = (X != nullptr) ? X->n_cols : 0;
  const bool has_X = (X != nullptr && P > 0);
  const bool has_y = (y != nullptr && y->n_elem > 0);

  if (!has_X && !has_y) return;

  center_workspace ws;
  init_center_workspace(ws, indices, w, N);
  const bool use_weights = (w.n_elem > 1);
  const double inv_sw = 1.0 / accu(w);

  // Convergence tracking
  bool y_converged = !has_y;
  uvec x_converged(P, fill::ones);
  if (has_X) x_converged.fill(0);

  // Storage for old values
  vec y_old;
  mat X_old;
  if (has_y) y_old.set_size(N);
  if (has_X) X_old.set_size(N, P);

  // SSR tracking per column
  double y_ssr0 = datum::inf;
  vec x_ssr0(P);
  if (has_X) x_ssr0.fill(datum::inf);

  // Iteration control
  size_t iint = iter_interrupt;
  const size_t isr0 = iter_ssr;

  // Column-wise SSR tracking and iteration control
  size_t y_isr = isr0;
  vec x_isr(P);
  if (has_X) x_isr.fill(isr0);

  for (size_t iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Check if everything has converged
    bool all_x_converged = true;
    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (!x_converged(p)) {
          all_x_converged = false;
          break;
        }
      }
    }
    if (y_converged && all_x_converged) break;

    // Store old values
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

    // Project y
    if (has_y && !y_converged) {
      if (K == 1) {
        project_1fe(*y, w, ws.group_indices(0), ws.group_inv_w(0), use_weights);
      } else if (K == 2) {
        project_2fe(*y, w, ws.group_indices(0), ws.group_inv_w(0),
                    ws.group_indices(1), ws.group_inv_w(1), use_weights);
      } else {
        project_Kfe(*y, w, ws.group_indices, ws.group_inv_w, use_weights);
      }
    }

    // Project X columns
    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (x_converged(p)) continue;
        vec col_p = X->col(p);

        if (K == 1) {
          project_1fe(col_p, w, ws.group_indices(0), ws.group_inv_w(0),
                      use_weights);
        } else if (K == 2) {
          project_2fe(col_p, w, ws.group_indices(0), ws.group_inv_w(0),
                      ws.group_indices(1), ws.group_inv_w(1), use_weights);
        } else {
          project_Kfe(col_p, w, ws.group_indices, ws.group_inv_w, use_weights);
        }

        X->col(p) = col_p;
      }
    }

    // Apply Irons-Tuck acceleration if requested
    if (use_acceleration && iter >= 5 && (iter % 5) == 0) {
      // Accelerate y
      if (has_y && !y_converged) {
        ws.x = *y;
        ws.Gx = ws.x;
        if (K == 1) {
          project_1fe(ws.Gx, w, ws.group_indices(0), ws.group_inv_w(0),
                      use_weights);
        } else if (K == 2) {
          project_2fe(ws.Gx, w, ws.group_indices(0), ws.group_inv_w(0),
                      ws.group_indices(1), ws.group_inv_w(1), use_weights);
        } else {
          project_Kfe(ws.Gx, w, ws.group_indices, ws.group_inv_w, use_weights);
        }

        ws.G2x = ws.Gx;
        if (K == 1) {
          project_1fe(ws.G2x, w, ws.group_indices(0), ws.group_inv_w(0),
                      use_weights);
        } else if (K == 2) {
          project_2fe(ws.G2x, w, ws.group_indices(0), ws.group_inv_w(0),
                      ws.group_indices(1), ws.group_inv_w(1), use_weights);
        } else {
          project_Kfe(ws.G2x, w, ws.group_indices, ws.group_inv_w, use_weights);
        }

        ws.deltaG = ws.G2x - ws.x;
        ws.delta2 = ws.G2x - 2.0 * ws.x + y_old;
        double ssq = dot(ws.delta2, ws.delta2);

        if (ssq > 1e-10) {
          double coef = dot(ws.deltaG, ws.delta2) / ssq;
          if (coef > 0.0 && coef < 2.0) {
            *y = ws.G2x - coef * ws.deltaG;
          } else {
            *y = ws.G2x;
          }
        }
      }

      // Accelerate X columns
      if (has_X) {
        for (size_t p = 0; p < P; ++p) {
          if (x_converged(p)) continue;

          ws.x = X->col(p);
          ws.Gx = ws.x;
          if (K == 1) {
            project_1fe(ws.Gx, w, ws.group_indices(0), ws.group_inv_w(0),
                        use_weights);
          } else if (K == 2) {
            project_2fe(ws.Gx, w, ws.group_indices(0), ws.group_inv_w(0),
                        ws.group_indices(1), ws.group_inv_w(1), use_weights);
          } else {
            project_Kfe(ws.Gx, w, ws.group_indices, ws.group_inv_w,
                        use_weights);
          }

          ws.G2x = ws.Gx;
          if (K == 1) {
            project_1fe(ws.G2x, w, ws.group_indices(0), ws.group_inv_w(0),
                        use_weights);
          } else if (K == 2) {
            project_2fe(ws.G2x, w, ws.group_indices(0), ws.group_inv_w(0),
                        ws.group_indices(1), ws.group_inv_w(1), use_weights);
          } else {
            project_Kfe(ws.G2x, w, ws.group_indices, ws.group_inv_w,
                        use_weights);
          }

          ws.deltaG = ws.G2x - ws.x;
          ws.delta2 = ws.G2x - 2.0 * ws.x + X_old.col(p);
          double ssq = dot(ws.delta2, ws.delta2);

          if (ssq > 1e-10) {
            double coef = dot(ws.deltaG, ws.delta2) / ssq;
            if (coef > 0.0 && coef < 2.0) {
              X->col(p) = ws.G2x - coef * ws.deltaG;
            } else {
              X->col(p) = ws.G2x;
            }
          }
        }
      }
    }

    // Check convergence for y
    if (has_y && !y_converged) {
      double dy_norm = norm(*y - y_old, 2);
      double y0_norm = norm(y_old, 2);
      double y_ratio = dy_norm / (1.0 + y0_norm);
      if (y_ratio < tol) {
        y_converged = true;
      }
    }

    // Check convergence for X columns
    if (has_X) {
      for (size_t p = 0; p < P; ++p) {
        if (!x_converged(p)) {
          double dx_norm = norm(X->col(p) - X_old.col(p), 2);
          double x0_norm = norm(X_old.col(p), 2);
          double x_ratio = dx_norm / (1.0 + x0_norm);
          if (x_ratio < tol) {
            x_converged(p) = 1;
          }
        }
      }
    }

    // SSR-based convergence check (like the old code)
    if (has_y && !y_converged && iter == y_isr && iter > 0) {
      y_isr += isr0;
      double y_ssr = use_weights ? dot(*y % *y, w) * inv_sw : dot(*y, *y) / N;
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
          vec col_p = X->col(p);
          double x_ssr = use_weights ? dot(col_p % col_p, w) * inv_sw
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

// Specialized function for single fixed effect (K=1)
inline void center_variables_1fe(mat &V, const vec &w,
                                 const indices_info &indices, double tol,
                                 size_t max_iter, size_t iter_interrupt,
                                 size_t iter_ssr,
                                 bool use_acceleration = false) {
  center_mat_or_vec(&V, nullptr, w, indices, tol, max_iter,
                           iter_interrupt, iter_ssr, use_acceleration);
}

// Specialized function for two fixed effects (K=2)
inline void center_variables_2fe(mat &V, const vec &w,
                                 const indices_info &indices, double tol,
                                 size_t max_iter, size_t iter_interrupt,
                                 size_t iter_ssr,
                                 bool use_acceleration = false) {
  center_mat_or_vec(&V, nullptr, w, indices, tol, max_iter,
                           iter_interrupt, iter_ssr, use_acceleration);
}

// General function for K>=3 fixed effects with Irons-Tuck acceleration
inline void center_variables_Kfe(mat &V, const vec &w,
                                 const indices_info &indices, double tol,
                                 size_t max_iter, size_t iter_interrupt,
                                 size_t iter_ssr,
                                 bool use_acceleration = true) {
  center_mat_or_vec(&V, nullptr, w, indices, tol, max_iter,
                           iter_interrupt, iter_ssr, use_acceleration);
}

// Main dispatcher for matrix centering only
inline void center_variables(mat &V, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter, size_t iter_interrupt,
                             size_t iter_ssr = 40,
                             bool use_acceleration = true) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;

  if (K == 1) {
    center_variables_1fe(V, w, indices, tol, max_iter, iter_interrupt, iter_ssr,
                         use_acceleration);
  } else if (K == 2) {
    center_variables_2fe(V, w, indices, tol, max_iter, iter_interrupt, iter_ssr,
                         use_acceleration);
  } else {
    center_variables_Kfe(V, w, indices, tol, max_iter, iter_interrupt, iter_ssr,
                         use_acceleration);
  }
}

// Vector centering only
inline void center_variables(vec &y, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter, size_t iter_interrupt,
                             size_t iter_ssr = 40,
                             bool use_acceleration = true) {
  center_mat_or_vec(nullptr, &y, w, indices, tol, max_iter,
                           iter_interrupt, iter_ssr, use_acceleration);
}

// Main function: centers both X and y together (was center_variables_batch)
inline void center_variables(mat &X_work, vec &y, const vec &w,
                             const mat &X_orig, const indices_info &indices,
                             double tol, size_t max_iter, size_t iter_interrupt,
                             size_t iter_ssr, bool use_acceleration) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;

  // Copy from original to avoid external matrix copying
  if (&X_work != &X_orig) {
    X_work = X_orig;
  }

  // Center both X and y simultaneously for better convergence
  center_mat_or_vec(&X_work, &y, w, indices, tol, max_iter,
                           iter_interrupt, iter_ssr, use_acceleration);
}

#endif  // CAPYBARA_CENTER_H
