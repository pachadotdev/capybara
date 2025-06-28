#ifndef CAPYBARA_CENTER_VARIABLES_H
#define CAPYBARA_CENTER_VARIABLES_H

#include <iostream> // for std::cerr

// #include "timing.h" // development only, for profiling

// MAP (Method of Alternating Projections) absorption for fixed effects
// Inspired by reghdfe: iteratively partial out each FE, looping over FEs and
// groups
inline void map_partial_out(mat *X, vec *y, const vec &w,
                            const indices_info &indices, double tol,
                            size_t max_iter, size_t iter_interrupt,
                            bool use_weights) {
  // TIME_FUNCTION;
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0)
    return;
  const size_t N = (X != nullptr) ? X->n_rows : y->n_elem;
  const size_t P = (X != nullptr) ? X->n_cols : 0;
  const bool has_X = (X != nullptr && P > 0);
  const bool has_y = (y != nullptr && y->n_elem > 0);
  if (!has_X && !has_y)
    return;

  // Workspace for convergence check
  mat X_old;
  vec y_old;
  if (has_X)
    X_old.set_size(N, P);
  if (has_y)
    y_old.set_size(N);

  size_t iint = iter_interrupt;
  size_t max_iter_debug = 20; // TEMP: limit for debugging

  for (size_t iter = 0; iter < max_iter_debug; ++iter) {
    // std::cerr << "MAP iteration " << iter << std::endl;
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }
    if (has_X)
      X_old = *X;
    if (has_y)
      y_old = *y;

    // For each FE, partial out group means
    for (size_t k = 0; k < K; ++k) {
      const size_t J = indices.fe_sizes(k);
      for (size_t j = 0; j < J; ++j) {
        const uvec &idx = indices.get_group(k, j);
        if (idx.is_empty())
          continue;
        double inv_sumw = 0.0;
        if (use_weights) {
          double sumw = accu(w(idx));
          inv_sumw = (sumw > 0) ? (1.0 / sumw) : 0.0;
        } else {
          inv_sumw = (idx.n_elem > 0) ? (1.0 / idx.n_elem) : 0.0;
        }
        if (has_X) {
          for (size_t p = 0; p < P; ++p) {
            double mean = 0.0;
            if (use_weights) {
              double sum = 0.0;
              for (size_t ii = 0; ii < idx.n_elem; ++ii)
                sum += w(idx[ii]) * (*X)(idx[ii], p);
              mean = sum * inv_sumw;
            } else {
              double sum = 0.0;
              for (size_t ii = 0; ii < idx.n_elem; ++ii)
                sum += (*X)(idx[ii], p);
              mean = (idx.n_elem > 0) ? sum / idx.n_elem : 0.0;
            }
            for (size_t ii = 0; ii < idx.n_elem; ++ii)
              (*X)(idx[ii], p) -= mean;
          }
        }
        if (has_y) {
          double mean = 0.0;
          if (use_weights) {
            double sum = 0.0;
            for (size_t ii = 0; ii < idx.n_elem; ++ii)
              sum += w(idx[ii]) * (*y)(idx[ii]);
            mean = sum * inv_sumw;
          } else {
            double sum = 0.0;
            for (size_t ii = 0; ii < idx.n_elem; ++ii)
              sum += (*y)(idx[ii]);
            mean = (idx.n_elem > 0) ? sum / idx.n_elem : 0.0;
          }
          for (size_t ii = 0; ii < idx.n_elem; ++ii)
            (*y)(idx[ii]) -= mean;
        }
      }
    }
    // Convergence check
    bool x_converged = true, y_converged = true;
    if (has_X) {
      double norm_diff = norm(*X - X_old, "fro");
      double norm_X = norm(X_old, "fro");
      x_converged = (norm_diff / (1.0 + norm_X) < tol);
    }
    if (has_y) {
      double norm_diff = norm(*y - y_old, 2);
      double norm_y = norm(y_old, 2);
      y_converged = (norm_diff / (1.0 + norm_y) < tol);
    }
    if ((has_X && x_converged) && (has_y && y_converged))
      break;
    if (has_X && !has_y && x_converged)
      break;
    if (has_y && !has_X && y_converged)
      break;
  }
}

inline void center_mat_or_vec(mat *X, vec *y, const vec &w,
                              const indices_info &indices, double tol,
                              size_t max_iter, size_t iter_interrupt,
                              size_t /*iter_ssr*/, bool /*use_acceleration*/) {
  // Use MAP absorption for all K >= 1
  const bool use_weights = (w.n_elem > 1);
  map_partial_out(X, y, w, indices, tol, max_iter, iter_interrupt, use_weights);
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

#endif // CAPYBARA_CENTER_VARIABLES_H
