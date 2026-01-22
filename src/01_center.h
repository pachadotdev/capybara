// Symmetric Kaczmarz centering with Irons-Tuck acceleration (coefficient space)
#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Store group information with precomputed weight vector for vectorization
struct GroupInfo {
  const uvec *coords;
  vec norm_weights;
  double inv_weight;
  uword n_elem;
  bool is_singleton;
};

struct CenteringWorkspace {
  cube Alpha_hist; // History cube: (total_groups, P, 3) for acceleration
  mat Alpha;       // Current coefficients (total_groups x P)
  mat Scratch;     // Scratch space for intermediate calculations

  void ensure_size(uword m, uword p) {
    if (Alpha.n_rows != m || Alpha.n_cols != p) {
      Alpha.set_size(m, p);
      Alpha_hist.set_size(m, p, 3);
      Scratch.set_size(m, p);
    }
  }

  void clear() {
    Alpha.zeros();
    Alpha_hist.zeros();
    Scratch.zeros();
  }
};

// Precompute group information with normalized weights
inline field<field<GroupInfo>>
precompute_group_info(const field<field<uvec>> &group_indices, const vec &w) {
  const uword K = group_indices.n_elem;
  field<field<GroupInfo>> group_info(K);

  for (uword k = 0; k < K; ++k) {
    const field<uvec> &fe_groups = group_indices(k);
    const uword n_groups = fe_groups.n_elem;
    group_info(k).set_size(n_groups);

    for (uword l = 0; l < n_groups; ++l) {
      const uvec &coords = fe_groups(l);
      GroupInfo &info = group_info(k)(l);
      info.coords = &coords;
      info.n_elem = coords.n_elem;

      vec w_group = w.elem(coords);

      uword n_nonzero = accu(w_group > 0.0);
      info.is_singleton = (n_nonzero <= 1);

      double sum_w = accu(w_group);

      if (!info.is_singleton && sum_w > 0.0) {
        info.inv_weight = 1.0 / sum_w;
        // Precompute normalized weights for weighted mean calculation
        info.norm_weights = w_group * info.inv_weight;
      } else {
        info.inv_weight = 0.0;
        info.norm_weights.reset();
      }
    }
  }

  return group_info;
}

inline double weighted_quadsum(const mat &X, const vec &w) {
  return accu(X.each_col() % w % X);
}

inline double safe_divide(double num, double denom) {
  return (std::abs(denom) > 1e-14) ? (num / denom) : 0.0;
}

// Update coefficients for one group
inline void update_group(mat &r, mat &Alpha, const GroupInfo &info,
                         uword idx_alpha) {
  if (info.is_singleton)
    return;

  const uvec &coords = *info.coords;
  const uword n = coords.n_elem;

  rowvec mean_row = info.norm_weights.t() * r.rows(coords);

  Alpha.row(idx_alpha) += mean_row;

  for (uword i = 0; i < n; ++i) {
    r.row(coords(i)) -= mean_row;
  }
}

// Symmetric Kaczmarz projection
inline void apply_projection(mat &r, mat &Alpha, const vec &w,
                             const field<field<GroupInfo>> &all_group_info,
                             const uvec &group_offsets) {
  const uword K = all_group_info.n_elem;

  // Forward pass: FE1, FE2, ..., FEK
  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
    const uword offset = group_offsets(k);
    const uword n_groups = fe_info.n_elem;

    for (uword l = 0; l < n_groups; ++l) {
      update_group(r, Alpha, fe_info(l), offset + l);
    }
  }

  // Backward pass: FE(K-1), ..., FE1
  for (uword k = K - 1; k > 0; --k) {
    const field<GroupInfo> &fe_info = all_group_info(k - 1);
    const uword offset = group_offsets(k - 1);
    const uword n_groups = fe_info.n_elem;

    for (uword l = 0; l < n_groups; ++l) {
      update_group(r, Alpha, fe_info(l), offset + l);
    }
  }
}

// Update residuals based on diff in Alpha
inline void
update_residuals_from_diff(mat &r, const mat &diff_Alpha,
                           const field<field<GroupInfo>> &all_group_info,
                           const uvec &group_offsets) {
  const uword K = all_group_info.n_elem;

  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
    const uword offset = group_offsets(k);
    const uword n_groups = fe_info.n_elem;

    for (uword l = 0; l < n_groups; ++l) {
      const GroupInfo &info = fe_info(l);
      const rowvec d_row = diff_Alpha.row(offset + l);
      const uvec &coords = *info.coords;
      const uword n = coords.n_elem;

      for (uword i = 0; i < n; ++i) {
        r.row(coords(i)) -= d_row;
      }
    }
  }
}

// Irons-Tuck acceleration
// Alpha_hist slices:
// 0 = X (current)
// 1 = GX (one projection)
// 2 = GGX (two projections)
inline bool irons_tuck_accel(cube &Alpha_hist, mat &Alpha, double tol) {
  const mat &X = Alpha_hist.slice(0);
  const mat &GX = Alpha_hist.slice(1);
  const mat &GGX = Alpha_hist.slice(2);

  mat Delta_G = GGX - GX;
  mat Delta2 = Delta_G - (GX - X);

  double ssq = accu(square(Delta2));

  if (ssq < tol * tol) {
    Alpha = GGX;
    return true;
  }

  double vprod = accu(Delta_G % Delta2);
  double coef = safe_divide(vprod, ssq);

  // More aggressive extrapolation bounds
  if (coef > 0.0 && coef < 5.0) {
    Alpha = GGX - coef * Delta_G;
  } else {
    Alpha = GGX;
  }

  return false;
}

// Grand acceleration
inline bool grand_accel(mat &Alpha, cube &hist, int &state, double tol) {
  hist.slice(state) = Alpha;
  ++state;

  if (state < 3) {
    return false;
  }

  const mat &Y = hist.slice(0);
  const mat &GY = hist.slice(1);
  const mat &GGY = hist.slice(2);

  mat Delta_G = GGY - GY;
  mat Delta2 = Delta_G - (GY - Y);

  double ssq = accu(square(Delta2));

  if (ssq < tol * tol) {
    state = 0;
    Alpha = GGY;
    return true;
  }

  double vprod = accu(Delta_G % Delta2);
  double coef = safe_divide(vprod, ssq);

  // More aggressive extrapolation bounds
  if (coef > 0.0 && coef < 5.0) {
    Alpha = GGY - coef * Delta_G;
  } else {
    Alpha = GGY;
  }

  state = 0;
  return false;
}

// Center Matrix
inline void center_accel(mat &V, const vec &w,
                         const field<field<GroupInfo>> &all_group_info,
                         const uvec &group_offsets, double tol, uword max_iter,
                         uword iter_interrupt, CenteringWorkspace &ws) {
  ws.Alpha.zeros();

  const uword irons_tuck_interval = 2;
  const uword grand_accel_interval = 3;
  const uword accel_start = 1;

  // Grand acceleration state and history cube
  int grand_state = 0;
  cube grand_hist(ws.Alpha.n_rows, ws.Alpha.n_cols, 3);

  uword iint = iter_interrupt;
  double ssr_old = datum::inf;
  uword stall_count = 0;

  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Apply projection (updates V and Alpha)
    apply_projection(V, ws.Alpha, w, all_group_info, group_offsets);

    // Grand acceleration - apply more frequently
    if (iter >= accel_start && iter % grand_accel_interval == 0) {
      mat Alpha_before = ws.Alpha;

      if (grand_accel(ws.Alpha, grand_hist, grand_state, tol)) {
        break;
      }

      // If acceleration changed Alpha, update residuals
      if (grand_state == 0) {
        mat diff = ws.Alpha - grand_hist.slice(2);
        update_residuals_from_diff(V, diff, all_group_info, group_offsets);
        apply_projection(V, ws.Alpha, w, all_group_info, group_offsets);
      }
    }

    // Irons-Tuck acceleration
    if (iter >= accel_start && iter % irons_tuck_interval == 0 &&
        iter % grand_accel_interval != 0) {

      ws.Alpha_hist.slice(0) = ws.Alpha;

      // Apply projection to get G(X), then G^2(X)
      apply_projection(V, ws.Alpha, w, all_group_info, group_offsets);
      ws.Alpha_hist.slice(1) = ws.Alpha;

      apply_projection(V, ws.Alpha, w, all_group_info, group_offsets);
      ws.Alpha_hist.slice(2) = ws.Alpha;

      mat Alpha_before = ws.Alpha;

      if (irons_tuck_accel(ws.Alpha_hist, ws.Alpha, tol)) {
        break;
      }

      mat diff = ws.Alpha - ws.Alpha_hist.slice(2);
      update_residuals_from_diff(V, diff, all_group_info, group_offsets);
    }

    // Check convergence based on weighted SSR
    double ssr = weighted_quadsum(V, w);

    if (ssr < tol * tol)
      break;

    if (ssr_old < datum::inf) {
      double rel_change = std::abs(ssr - ssr_old) / (1.0 + ssr_old);
      if (rel_change < tol)
        break;

      // More lenient stall detection to avoid premature termination
      if (rel_change < tol * 10.0) {
        ++stall_count;
        if (stall_count > 5)
          break;
      } else {
        stall_count = 0;
      }
    }
    ssr_old = ssr;
  }
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const uword &max_iter, const uword &iter_interrupt,
    const field<field<GroupInfo>> *precomputed_group_info = nullptr,
    CenteringWorkspace *workspace = nullptr) {
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem)
    return;

  const uword K = group_indices.n_elem;
  if (K == 0)
    return;

  const uword P = V.n_cols;

  // Reuse precomputed group metadata when available
  const field<field<GroupInfo>> *group_info_ptr = precomputed_group_info;
  field<field<GroupInfo>> local_group_info;
  if (!group_info_ptr) {
    local_group_info = precompute_group_info(group_indices, w);
    group_info_ptr = &local_group_info;
  }
  const field<field<GroupInfo>> &all_group_info = *group_info_ptr;

  // Calculate offsets and total number of groups
  uvec group_sizes(K);
  for (uword k = 0; k < K; ++k) {
    group_sizes(k) = all_group_info(k).n_elem;
  }
  uvec group_offsets = cumsum(group_sizes) - group_sizes;
  uword total_groups = accu(group_sizes);

  CenteringWorkspace local_workspace;
  CenteringWorkspace *ws = workspace ? workspace : &local_workspace;

  ws->ensure_size(total_groups, P);
  ws->clear();

  center_accel(V, w, all_group_info, group_offsets, tol, max_iter,
               iter_interrupt, *ws);
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
