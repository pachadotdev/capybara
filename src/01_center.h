// Centering using observation-to-group mapping
#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Observation-to-group mapping
struct ObsToGroupMapping {
  field<uvec> obs_to_group;     // obs_to_group[k](i) = group index for obs i in FE k
  field<vec> group_inv_weights; // Precomputed 1/sum(weights) per group
  uword n_obs;
  uword K;
  
  void build(const field<field<uvec>> &group_indices, const vec &w) {
    K = group_indices.n_elem;
    n_obs = w.n_elem;
    
    obs_to_group.set_size(K);
    group_inv_weights.set_size(K);
    
    for (uword k = 0; k < K; ++k) {
      const field<uvec> &fe_groups = group_indices(k);
      const uword n_groups = fe_groups.n_elem;
      
      obs_to_group(k).set_size(n_obs);
      group_inv_weights(k).set_size(n_groups);
      
      // Build observation -> group mapping
      for (uword g = 0; g < n_groups; ++g) {
        const uvec &group_obs = fe_groups(g);
        for (uword j = 0; j < group_obs.n_elem; ++j) {
          obs_to_group(k)(group_obs(j)) = g;
        }
        
        // Precompute inverse weights
        double sum_w = accu(w.elem(group_obs));
        group_inv_weights(k)(g) = (sum_w > 0.0) ? (1.0 / sum_w) : 0.0;
      }
    }
  }
};

struct GroupInfo {
  const uvec *coords;
  double inv_weight;
  uword n_elem;
  bool is_singleton;
};

struct CenteringWorkspace {
  mat Alpha;       // Current coefficients (total_groups x P)
  mat Alpha_prev;  // Previous iteration
  
  // Acceleration history
  mat IT_X, IT_GX, IT_GGX;
  
  void ensure_size(uword m, uword p) {
    if (Alpha.n_rows != m || Alpha.n_cols != p) {
      Alpha.set_size(m, p);
      Alpha_prev.set_size(m, p);
      IT_X.set_size(m, p);
      IT_GX.set_size(m, p);
      IT_GGX.set_size(m, p);
    }
  }

  void clear() {
    Alpha.zeros();
    Alpha_prev.zeros();
  }
};

// Precompute group information
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
      info.inv_weight = (info.is_singleton || sum_w <= 0.0) ? 0.0 : (1.0 / sum_w);
    }
  }

  return group_info;
}

// Update one fixed effect given all others - O(n)
inline void project_one_fe(uword fe_idx, mat &alpha_all, const mat &V_residual,
                          const vec &w, const ObsToGroupMapping &mapping,
                          const field<uword> &group_offsets) {
  const uword n_obs = mapping.n_obs;
  const uword K = mapping.K;
  const uword P = V_residual.n_cols;
  
  const uvec &obs_to_g = mapping.obs_to_group(fe_idx);
  const vec &inv_weights = mapping.group_inv_weights(fe_idx);
  const uword offset = group_offsets(fe_idx);
  const uword n_groups = inv_weights.n_elem;
  
  // Initialize this FE's coefficients to zero
  alpha_all.rows(offset, offset + n_groups - 1).zeros();
  
  // Compute residuals by subtracting other FE contributions
  mat residual = V_residual;
  for (uword k = 0; k < K; ++k) {
    if (k != fe_idx) {
      const uvec &other_obs_to_g = mapping.obs_to_group(k);
      const uword other_offset = group_offsets(k);
      
      // Subtract contribution from FE k
      for (uword i = 0; i < n_obs; ++i) {
        residual.row(i) -= alpha_all.row(other_offset + other_obs_to_g(i));
      }
    }
  }
  
  // Accumulate weighted residuals into groups
  const uword *p_obs_to_g = obs_to_g.memptr();
  const double *p_w = w.memptr();
  
  for (uword p = 0; p < P; ++p) {
    const double *resid_col = residual.colptr(p);
    double *alpha_col = alpha_all.colptr(p) + offset;
    
    for (uword i = 0; i < n_obs; ++i) {
      const double wi = p_w[i];
      if (wi > 0.0) {
        alpha_col[p_obs_to_g[i]] += resid_col[i] * wi;
      }
    }
  }
  
  // Vectorized normalization by inverse weights
  alpha_all.rows(offset, offset + n_groups - 1).each_col() %= inv_weights;
}

// K-FE centering
inline void center_fixest_kfe(mat &V, const vec &w,
                             const field<field<uvec>> &group_indices,
                             double tol, uword max_iter) {
  const uword n_obs = V.n_rows;
  const uword K = group_indices.n_elem;
  const uword P = V.n_cols;
  
  // Build observation-to-group mapping (once)
  ObsToGroupMapping mapping;
  mapping.build(group_indices, w);
  
  // Group offsets
  field<uword> group_offsets(K);
  uword total_groups = 0;
  for (uword k = 0; k < K; ++k) {
    group_offsets(k) = total_groups;
    total_groups += group_indices(k).n_elem;
  }
  
  // All FE coefficients stacked: [alpha_0; alpha_1; ...; alpha_{K-1}]
  mat alpha_all(total_groups, P, fill::zeros);
  
  // Cycle through FEs, updating each given the others
  for (uword iter = 0; iter < max_iter; ++iter) {
    mat alpha_old = alpha_all;
    
    // Update each FE
    for (int k = K - 1; k >= 0; --k) {
      project_one_fe(k, alpha_all, V, w, mapping, group_offsets);
    }
    
    double diff = norm(alpha_all - alpha_old, "fro");
    if (diff < tol) {
      break;
    }
  }
  
  // Apply final centering
  for (uword i = 0; i < n_obs; ++i) {
    for (uword k = 0; k < K; ++k) {
      const uword g = mapping.obs_to_group(k)(i);
      const uword offset = group_offsets(k);
      V.row(i) -= alpha_all.row(offset + g);
    }
  }
}

// Coefficient update
inline void update_group_inplace(mat &r, rowvec &alpha_row, 
                                 const GroupInfo &info,
                                 const vec &w) {
  if (info.is_singleton)
    return;

  const uvec &coords = *info.coords;
  
  if (info.inv_weight > 0.0) {
    // Vectorized: alpha_row = sum(r[coords] .* w[coords]) * inv_weight
    mat r_group = r.rows(coords);
    vec w_group = w.elem(coords);
    
    alpha_row = sum(r_group.each_col() % w_group, 0) * info.inv_weight;
    
    r.rows(coords) -= repmat(alpha_row, coords.n_elem, 1);
  } else {
    alpha_row.zeros();
  }
}

// Single projection step
inline void apply_projection(mat &r, mat &Alpha, const vec &w,
                             const field<field<GroupInfo>> &all_group_info,
                             const uvec &group_offsets,
                             rowvec &work_row) {
  const uword K = all_group_info.n_elem;

  // Forward pass
  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
    const uword offset = group_offsets(k);
    const uword n_groups = fe_info.n_elem;

    for (uword l = 0; l < n_groups; ++l) {
      update_group_inplace(r, work_row, fe_info(l), w);
      Alpha.row(offset + l) += work_row;
    }
  }

  // Backward pass
  for (uword k = K - 1; k > 0; --k) {
    const field<GroupInfo> &fe_info = all_group_info(k - 1);
    const uword offset = group_offsets(k - 1);
    const uword n_groups = fe_info.n_elem;

    for (uword l = 0; l < n_groups; ++l) {
      update_group_inplace(r, work_row, fe_info(l), w);
      Alpha.row(offset + l) += work_row;
    }
  }
}

// Update residuals from coefficient difference
inline void apply_coef_diff(mat &r, const mat &diff_Alpha,
                           const field<field<GroupInfo>> &all_group_info,
                           const uvec &group_offsets) {
  const uword K = all_group_info.n_elem;

  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
    const uword offset = group_offsets(k);
    const uword n_groups = fe_info.n_elem;

    for (uword l = 0; l < n_groups; ++l) {
      const GroupInfo &info = fe_info(l);
      if (info.is_singleton) continue;
      
      const uvec &coords = *info.coords;
      const rowvec &d_row = diff_Alpha.row(offset + l);
      
      r.rows(coords) -= repmat(d_row, coords.n_elem, 1);
    }
  }
}

// Irons-Tuck acceleration
inline bool irons_tuck_accel(const mat &X, const mat &GX, const mat &GGX,
                             mat &Alpha_out, double tol) {
  mat Delta_G = GGX - GX;
  mat Delta2 = Delta_G - (GX - X);

  double ssq = accu(square(Delta2));
  
  if (ssq < tol * tol) {
    Alpha_out = GGX;
    return true;
  }

  double vprod = accu(Delta_G % Delta2);
  double coef = (std::abs(ssq) > 1e-14) ? (vprod / ssq) : 0.0;

  if (coef > 0.0 && coef < 2.0) {
    Alpha_out = GGX - coef * Delta_G;
  } else {
    Alpha_out = GGX;
  }

  return false;
}

// Main centering
inline void center_accel(mat &V, const vec &w,
                        const field<field<GroupInfo>> &all_group_info,
                        const uvec &group_offsets, double tol, 
                        uword max_iter, uword iter_interrupt, 
                        CenteringWorkspace &ws) {
  ws.Alpha.zeros();
  
  const uword P = V.n_cols;
  rowvec work_row(P);

  const uword accel_start = 3;
  const uword accel_interval = 3;
  const uword grand_accel_interval = 5;
  
  double ssr_old = datum::inf;
  uword stall_count = 0;
  uword iint = iter_interrupt;
  uword grand_acc_count = 0;
  
  // Grand acceleration storage
  mat Y, GY;
  
  for (uword iter = 0; iter < max_iter; ++iter) {
    if (iter == iint) {
      check_user_interrupt();
      iint += iter_interrupt;
    }

    // Single projection step (G(X) operation)
    apply_projection(V, ws.Alpha, w, all_group_info, group_offsets, work_row);

    // Compute SSR for convergence check
    double ssr = accu(square(V));
    
    if (ssr < tol * tol) {
      break;
    }
    
    // Relative change convergence
    if (ssr_old < datum::inf) {
      double rel_change = std::abs(ssr - ssr_old) / (1.0 + ssr_old);
      if (rel_change < tol) {
        break;
      }
      
      // Track stalling
      if (rel_change < tol * 5.0) {
        ++stall_count;
      } else {
        stall_count = 0;
      }
    }
    ssr_old = ssr;

    // Standard Irons-Tuck acceleration (every accel_interval iterations or when stalling)
    if (iter >= accel_start && (iter % accel_interval == 0 || stall_count > 3)) {
      ws.IT_X = ws.Alpha;

      // G(X)
      apply_projection(V, ws.Alpha, w, all_group_info, group_offsets, work_row);
      ws.IT_GX = ws.Alpha;

      // G^2(X)
      apply_projection(V, ws.Alpha, w, all_group_info, group_offsets, work_row);
      ws.IT_GGX = ws.Alpha;

      if (irons_tuck_accel(ws.IT_X, ws.IT_GX, ws.IT_GGX, ws.Alpha, tol)) {
        break;
      }

      // Apply coefficient difference to residuals
      mat diff = ws.Alpha - ws.IT_GGX;
      apply_coef_diff(V, diff, all_group_info, group_offsets);
      
      stall_count = 0;
    }
    
    // Grand acceleration
    // This is a second-level acceleration on top of regular IT
    if (iter > 0 && iter % grand_accel_interval == 0) {
      ++grand_acc_count;
      
      if (grand_acc_count == 1) {
        Y = ws.Alpha;
      } else if (grand_acc_count == 2) {
        GY = ws.Alpha;
      } else {
        mat GGY = ws.Alpha;
        mat Y_accel;
        
        if (irons_tuck_accel(Y, GY, GGY, Y_accel, tol)) {
          ws.Alpha = Y_accel;
          mat diff = ws.Alpha - GGY;
          apply_coef_diff(V, diff, all_group_info, group_offsets);
          break;
        }
        
        ws.Alpha = Y_accel;
        mat diff = ws.Alpha - GGY;
        apply_coef_diff(V, diff, all_group_info, group_offsets);
        
        grand_acc_count = 0;
        stall_count = 0;
      }
    }
    
    if (stall_count > 15) break;
  }
}

// Optimized 2-FE centering
inline void center_accel_2fe(mat &V, const vec &w,
                           const field<uvec> &fe1_groups,
                           const field<uvec> &fe2_groups,
                           double tol, uword max_iter) {
  const uword n_obs = V.n_rows;
  const uword n1 = fe1_groups.n_elem;
  const uword n2 = fe2_groups.n_elem;
  const uword P = V.n_cols;
  
  // Build observation-to-group mapping
  uvec obs_to_g1(n_obs);
  uvec obs_to_g2(n_obs);
  
  for (uword g = 0; g < n1; ++g) {
    const uvec &idx = fe1_groups(g);
    const uword n_idx = idx.n_elem;
    for (uword i = 0; i < n_idx; ++i) {
      obs_to_g1(idx(i)) = g;
    }
  }
  for (uword g = 0; g < n2; ++g) {
    const uvec &idx = fe2_groups(g);
    const uword n_idx = idx.n_elem;
    for (uword i = 0; i < n_idx; ++i) {
      obs_to_g2(idx(i)) = g;
    }
  }
  
  mat alpha1(n1, P, fill::zeros);
  mat alpha2(n2, P, fill::zeros);
  
  vec sum_w1(n1, fill::zeros);
  vec sum_w2(n2, fill::zeros);
  
  for (uword g = 0; g < n1; ++g) {
    sum_w1(g) = accu(w.elem(fe1_groups(g)));
  }
  for (uword g = 0; g < n2; ++g) {
    sum_w2(g) = accu(w.elem(fe2_groups(g)));
  }
  
  vec inv_w1 = 1.0 / sum_w1;
  vec inv_w2 = 1.0 / sum_w2;
  inv_w1.replace(datum::inf, 0.0);
  inv_w2.replace(datum::inf, 0.0);
  
  const uword *p_g1 = obs_to_g1.memptr();
  const uword *p_g2 = obs_to_g2.memptr();
  const double *p_w = w.memptr();
  
  // Irons-Tuck acceleration variables
  mat alpha1_prev(n1, P);
  mat alpha1_prev2(n1, P);
  bool use_accel = false;
  const uword accel_start = 3;
  
  // Alternating projection
  for (uword iter = 0; iter < max_iter; ++iter) {
    mat alpha1_old = alpha1;
    
    // Update alpha1 given alpha2
    alpha1.zeros();
    
    for (uword p = 0; p < P; ++p) {
      const double *V_col = V.colptr(p);
      const double *alpha2_col = alpha2.colptr(p);
      double *alpha1_col = alpha1.colptr(p);
      
      for (uword obs = 0; obs < n_obs; ++obs) {
        const uword g1 = p_g1[obs];
        const uword g2 = p_g2[obs];
        const double wi = p_w[obs];
        
        if (wi > 0.0) {
          alpha1_col[g1] += (V_col[obs] - alpha2_col[g2]) * wi;
        }
      }
    }
    
    alpha1.each_col() %= inv_w1;
    
    // Update alpha2 given new alpha1
    alpha2.zeros();
    
    for (uword p = 0; p < P; ++p) {
      const double *V_col = V.colptr(p);
      const double *alpha1_col = alpha1.colptr(p);
      double *alpha2_col = alpha2.colptr(p);
      
      for (uword obs = 0; obs < n_obs; ++obs) {
        const uword g1 = p_g1[obs];
        const uword g2 = p_g2[obs];
        const double wi = p_w[obs];
        
        if (wi > 0.0) {
          alpha2_col[g2] += (V_col[obs] - alpha1_col[g1]) * wi;
        }
      }
    }
    
    alpha2.each_col() %= inv_w2;
    
    // Irons-Tuck acceleration
    if (iter >= accel_start) {
      if (use_accel) {
        mat Delta_G = alpha1 - alpha1_prev;
        mat Delta2 = Delta_G - (alpha1_prev - alpha1_prev2);
        
        double ssq = accu(square(Delta2));
        if (ssq > 1e-14) {
          double vprod = accu(Delta_G % Delta2);
          double coef = vprod / ssq;
          
          if (coef > 0.0 && coef < 2.0) {
            alpha1 = alpha1 - coef * Delta_G;
          }
        }
      }
      
      alpha1_prev2 = alpha1_prev;
      alpha1_prev = alpha1_old;
      use_accel = true;
    }
    
    double diff = norm(alpha1 - alpha1_old, "fro");
    if (diff < tol) {
      break;
    }
  }
  
  // Final centering: V[obs] -= alpha1[g1] + alpha2[g2]
  for (uword p = 0; p < P; ++p) {
    double *V_col = V.colptr(p);
    const double *alpha1_col = alpha1.colptr(p);
    const double *alpha2_col = alpha2.colptr(p);
    
    for (uword obs = 0; obs < n_obs; ++obs) {
      V_col[obs] -= alpha1_col[p_g1[obs]] + alpha2_col[p_g2[obs]];
    }
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

  if (K == 2) {
    center_accel_2fe(V, w, group_indices(0), group_indices(1), tol, max_iter);
    return;
  }
  
  if (K >= 3) {
    center_fixest_kfe(V, w, group_indices, tol, max_iter);
    return;
  }

  // K == 1
  
  const field<field<GroupInfo>> *group_info_ptr = precomputed_group_info;
  field<field<GroupInfo>> local_group_info;
  if (!group_info_ptr) {
    local_group_info = precompute_group_info(group_indices, w);
    group_info_ptr = &local_group_info;
  }
  const field<field<GroupInfo>> &all_group_info = *group_info_ptr;

  uvec group_sizes(K);
  for (uword k = 0; k < K; ++k) {
    group_sizes(k) = all_group_info(k).n_elem;
  }
  uvec group_offsets = cumsum(group_sizes) - group_sizes;
  uword total_groups = accu(group_sizes);

  CenteringWorkspace local_workspace;
  CenteringWorkspace *ws = workspace ? workspace : &local_workspace;

  ws->ensure_size(total_groups, V.n_cols);
  ws->clear();

  center_accel(V, w, all_group_info, group_offsets, tol, max_iter,
               iter_interrupt, *ws);
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
