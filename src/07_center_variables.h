#ifndef CAPYBARA_CENTER_VARIABLES_H
#define CAPYBARA_CENTER_VARIABLES_H

// This implements the fastest MAP from reghdfe:
// symmetric Kaczmarz + conjugate gradient

// Projection for a single fixed effect
inline void project_fe(mat *X, vec *y, size_t k, const indices_info &indices,
                       const vec &w, bool use_weights) {
  const size_t J = indices.fe_sizes(k);

  for (size_t j = 0; j < J; ++j) {
    const uvec &idx = indices.get_group(k, j);
    if (idx.is_empty())
      continue;

    const size_t n_idx = idx.n_elem;
    double inv_sumw;

    if (use_weights) {
      double sumw = accu(w(idx));
      inv_sumw = (sumw > 0) ? (1.0 / sumw) : 0.0;
    } else {
      inv_sumw = 1.0 / n_idx;
    }

    // Project matrix columns if present
    if (X != nullptr) {
      const size_t P = X->n_cols;
      for (size_t p = 0; p < P; ++p) {
        double xbar = 0.0;
        if (use_weights) {
          for (size_t i = 0; i < n_idx; ++i) {
            xbar += (*X)(idx(i), p) * w(idx(i));
          }
        } else {
          for (size_t i = 0; i < n_idx; ++i) {
            xbar += (*X)(idx(i), p);
          }
        }
        xbar *= inv_sumw;

        for (size_t i = 0; i < n_idx; ++i) {
          (*X)(idx(i), p) -= xbar;
        }
      }
    }

    // Project vector if present
    if (y != nullptr) {
      double ybar = 0.0;
      if (use_weights) {
        for (size_t i = 0; i < n_idx; ++i) {
          ybar += (*y)(idx(i)) * w(idx(i));
        }
      } else {
        for (size_t i = 0; i < n_idx; ++i) {
          ybar += (*y)(idx(i));
        }
      }
      ybar *= inv_sumw;

      for (size_t i = 0; i < n_idx; ++i) {
        (*y)(idx(i)) -= ybar;
      }
    }
  }
}

// Symmetric Kaczmarz transformation (forward + backward pass)
inline void transform_symmetric_kaczmarz(mat *X, vec *y, const vec &w,
                                         const indices_info &indices,
                                         bool use_weights) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  // Forward pass
  for (size_t k = 0; k < K; ++k) {
    project_fe(X, y, k, indices, w, use_weights);
  }

  // Backward pass
  for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
    project_fe(X, y, static_cast<size_t>(k), indices, w, use_weights);
  }
}

// Weighted sum of squares for convergence check
inline double weighted_ssr(const mat *X, const vec *y, const vec &w) {
  double ssr = 0.0;

  if (X != nullptr) {
    const size_t N = X->n_rows;
    const size_t P = X->n_cols;
    for (size_t p = 0; p < P; ++p) {
      for (size_t i = 0; i < N; ++i) {
        double val = (*X)(i, p);
        ssr += w.n_elem > 1 ? w(i) * val * val : val * val;
      }
    }
  }

  if (y != nullptr) {
    const size_t N = y->n_elem;
    for (size_t i = 0; i < N; ++i) {
      double val = (*y)(i);
      ssr += w.n_elem > 1 ? w(i) * val * val : val * val;
    }
  }

  return ssr;
}

// Helper to compute weighted quadratic sum for vectors
inline double weighted_quadsum_vec(const vec &x, const vec &y, const vec &w) {
  if (w.n_elem > 1) {
    return dot(x % y, w);
  } else {
    return dot(x, y);
  }
}

// Helper to compute weighted quadratic sum for matrices (column-wise)
inline rowvec weighted_quadsum_mat(const mat &x, const mat &y, const vec &w) {
  const size_t P = x.n_cols;
  rowvec result(P);

  for (size_t p = 0; p < P; ++p) {
    if (w.n_elem > 1) {
      result(p) = dot(x.col(p) % y.col(p), w);
    } else {
      result(p) = dot(x.col(p), y.col(p));
    }
  }
  return result;
}

// Conjugate gradient acceleration
inline void center_variables_cg(mat *X, vec *y, const vec &w,
                                const indices_info &indices, double tol,
                                size_t max_iter, size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;

  const bool has_X = (X != nullptr);
  const bool has_y = (y != nullptr);
  const bool use_weights = (w.n_elem > 1);
  size_t iint = iter_interrupt;

  // Workspace for CG
  mat r_X, u_X, v_X;
  vec r_y, u_y, v_y;
  rowvec ssr_X, ssr_old_X, alpha_X, beta_X, improvement_potential_X;
  double ssr_y, ssr_old_y, alpha_y, beta_y, improvement_potential_y;

  const size_t d = 1; // Number of recent SSR values for convergence
  mat recent_ssr_X;
  vec recent_ssr_y;

  if (has_X) {
    const size_t N = X->n_rows;
    const size_t P = X->n_cols;
    r_X.set_size(N, P);
    u_X.set_size(N, P);
    v_X.set_size(N, P);
    ssr_X.set_size(P);
    ssr_old_X.set_size(P);
    alpha_X.set_size(P);
    beta_X.set_size(P);
    improvement_potential_X.set_size(P);
    recent_ssr_X.set_size(d, P);

    // Initialize improvement potential
    improvement_potential_X = weighted_quadsum_mat(*X, *X, w);
  }

  if (has_y) {
    const size_t N = y->n_elem;
    r_y.set_size(N);
    u_y.set_size(N);
    v_y.set_size(N);
    recent_ssr_y.set_size(d);

    // Initialize improvement potential
    improvement_potential_y = weighted_quadsum_vec(*y, *y, w);
  }

  // Initialize: r = T(X,y) (first transformation)
  if (has_X) {
    r_X = *X;
    transform_symmetric_kaczmarz(&r_X, nullptr, w, indices, use_weights);
    r_X = *X - r_X; // Get residual
    ssr_X = weighted_quadsum_mat(r_X, r_X, w);
    u_X = r_X;
  }

  if (has_y) {
    r_y = *y;
    transform_symmetric_kaczmarz(nullptr, &r_y, w, indices, use_weights);
    r_y = *y - r_y; // Get residual
    ssr_y = weighted_quadsum_vec(r_y, r_y, w);
    u_y = r_y;
  }

  // Main CG loop
  for (size_t iter = 1; iter <= max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Apply transformation to u: v = T(u)
    if (has_X) {
      v_X = u_X;
      transform_symmetric_kaczmarz(&v_X, nullptr, w, indices, use_weights);
      v_X = u_X - v_X; // Get residual

      // Compute alpha = ssr / (u'v)
      rowvec uv = weighted_quadsum_mat(u_X, v_X, w);
      for (size_t p = 0; p < alpha_X.n_elem; ++p) {
        alpha_X(p) = (uv(p) > 0) ? ssr_X(p) / uv(p) : 0.0;
      }

      // Update: y = y - alpha * u, r = r - alpha * v
      for (size_t p = 0; p < X->n_cols; ++p) {
        X->col(p) -= alpha_X(p) * u_X.col(p);
        r_X.col(p) -= alpha_X(p) * v_X.col(p);
      }

      // Store recent SSR for convergence check
      for (size_t p = 0; p < alpha_X.n_elem; ++p) {
        recent_ssr_X(iter % d, p) = alpha_X(p) * ssr_X(p);
        improvement_potential_X(p) -= alpha_X(p) * ssr_X(p);
      }
    }

    if (has_y) {
      v_y = u_y;
      transform_symmetric_kaczmarz(nullptr, &v_y, w, indices, use_weights);
      v_y = u_y - v_y; // Get residual

      // Compute alpha = ssr / (u'v)
      double uv = weighted_quadsum_vec(u_y, v_y, w);
      alpha_y = (uv > 0) ? ssr_y / uv : 0.0;

      // Update: y = y - alpha * u, r = r - alpha * v
      *y -= alpha_y * u_y;
      r_y -= alpha_y * v_y;

      // Store recent SSR for convergence check
      recent_ssr_y(iter % d) = alpha_y * ssr_y;
      improvement_potential_y -= alpha_y * ssr_y;
    }

    // Update SSR
    if (has_X) {
      ssr_old_X = ssr_X;
      ssr_X = weighted_quadsum_mat(r_X, r_X, w);

      // Compute beta = ssr / ssr_old (Fletcher-Reeves)
      for (size_t p = 0; p < beta_X.n_elem; ++p) {
        beta_X(p) = (ssr_old_X(p) > 0) ? ssr_X(p) / ssr_old_X(p) : 0.0;
      }

      // Update u = r + beta * u
      for (size_t p = 0; p < u_X.n_cols; ++p) {
        u_X.col(p) = r_X.col(p) + beta_X(p) * u_X.col(p);
      }
    }

    if (has_y) {
      ssr_old_y = ssr_y;
      ssr_y = weighted_quadsum_vec(r_y, r_y, w);

      // Compute beta = ssr / ssr_old (Fletcher-Reeves)
      beta_y = (ssr_old_y > 0) ? ssr_y / ssr_old_y : 0.0;

      // Update u = r + beta * u
      u_y = r_y + beta_y * u_y;
    }

    // Check convergence (Hestenes-Stiefel)
    bool converged = false;
    if (has_X) {
      rowvec recent_sum = sum(recent_ssr_X.rows(0, std::min(iter, d) - 1), 0);
      for (size_t p = 0; p < recent_sum.n_elem; ++p) {
        double eps_threshold = 1e-15;
        double ratio =
            recent_sum(p) / (improvement_potential_X(p) + eps_threshold);
        if (sqrt(ratio) <= tol) {
          converged = true;
          break;
        }
      }
    }

    if (has_y && !converged) {
      double recent_sum = sum(recent_ssr_y.rows(0, std::min(iter, d) - 1));
      double eps_threshold = 1e-15;
      double ratio = recent_sum / (improvement_potential_y + eps_threshold);
      if (sqrt(ratio) <= tol) {
        converged = true;
      }
    }

    if (converged)
      break;
  }
}

inline void center_variables(mat &V, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter,
                             size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;
  center_variables_cg(&V, nullptr, w, indices, tol, max_iter, iter_interrupt);
}

inline void center_variables(vec &y, const vec &w, const indices_info &indices,
                             double tol, size_t max_iter,
                             size_t iter_interrupt) {
  center_variables_cg(nullptr, &y, w, indices, tol, max_iter, iter_interrupt);
}

inline void center_variables(mat &X_work, vec &y, const vec &w,
                             const mat &X_orig, const indices_info &indices,
                             double tol, size_t max_iter,
                             size_t iter_interrupt) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;
  if (&X_work != &X_orig) {
    X_work = X_orig;
  }
  center_variables_cg(&X_work, &y, w, indices, tol, max_iter, iter_interrupt);
}

#endif // CAPYBARA_CENTER_VARIABLES_H
