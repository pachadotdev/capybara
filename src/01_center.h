// Symmetric Kaczmarz with Conjugate Gradient acceleration

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Helper function for conjugate gradient acceleration
// Based on reghdfe's accelerate_cg implementation
template <typename ProjectFunc>
void conjugate_gradient_accel(vec &x, vec &g, vec &g_old, vec &p, 
                              const vec &w, const double &inv_sw,
                              ProjectFunc project, 
                              const size_t iter,
                              const size_t accel_start) {
  // g = gradient = residual = x_old - x_new (after projection)
  // p = search direction
  
  if (iter == accel_start) {
    // First CG iteration: p = g
    p = g;
  } else if (iter > accel_start) {
    // Hestenes-Stiefel formula (more stable than Fletcher-Reeves for MAP)
    // beta = <g, g - g_old> / <p, g - g_old>
    vec diff_g = g - g_old;
    double num = dot(g % diff_g, w) * inv_sw;
    double denom = dot(p % diff_g, w) * inv_sw;
    
    if (std::abs(denom) > 1e-10) {
      double beta = num / denom;
      // Restart if beta is negative or too large
      if (beta < 0.0 || beta > 10.0) {
        p = g; // Restart CG
      } else {
        p = g + beta * p;
      }
    } else {
      p = g; // Restart if denominator too small
    }
  }
  
  // Line search along direction p
  vec Ap = p;
  project(Ap); // Apply projection to get Ap
  Ap = p - Ap;  // Now Ap = (I - P)*p
  
  double pAp = dot(p % Ap, w) * inv_sw;
  if (pAp > 1e-10) {
    double gg = dot(g % g, w) * inv_sw;
    double alpha = gg / pAp;
    
    // Update x: x = x - alpha * p
    // Note: in reghdfe, they use x = x - alpha * p because g = x_old - x_new
    x = x - alpha * p;
  }
}

void center_variables_2fe(mat &V, const vec &w,
                          const field<field<uvec>> &group_indices,
                          const double &tol, const size_t &max_iter,
                          const size_t &iter_interrupt,
                          const size_t &iter_ssr,
                          const size_t &accel_start,
                          const bool use_cg) {
  // Early termination check
  if (V.is_empty() || w.is_empty()) {
    return;
  }
  
  // Dimensions
  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);

  // Extract the two FE groups
  const field<uvec> &fe1_groups = group_indices(0);
  const field<uvec> &fe2_groups = group_indices(1);
  const size_t L1 = fe1_groups.n_elem;
  const size_t L2 = fe2_groups.n_elem;

  // Precompute group weights and identify non-empty groups
  uvec non_empty_fe1, non_empty_fe2;
  vec group1_inv_w, group2_inv_w;
  
  // Count non-empty groups first
  size_t n_non_empty_fe1 = 0, n_non_empty_fe2 = 0;
  for (size_t l = 0; l < L1; ++l) {
    if (fe1_groups(l).n_elem > 1) {
      double sum_w = accu(w.elem(fe1_groups(l)));
      if (sum_w > 0.0) {
        n_non_empty_fe1++;
      }
    }
  }
  for (size_t l = 0; l < L2; ++l) {
    if (fe2_groups(l).n_elem > 1) {
      double sum_w = accu(w.elem(fe2_groups(l)));
      if (sum_w > 0.0) {
        n_non_empty_fe2++;
      }
    }
  }
  
  // Allocate for non-empty groups only
  non_empty_fe1.set_size(n_non_empty_fe1);
  non_empty_fe2.set_size(n_non_empty_fe2);
  group1_inv_w.set_size(n_non_empty_fe1);
  group2_inv_w.set_size(n_non_empty_fe2);
  
  // Fill non-empty group indices and weights
  size_t idx1 = 0, idx2 = 0;
  for (size_t l = 0; l < L1; ++l) {
    if (fe1_groups(l).n_elem > 1) {
      double sum_w = accu(w.elem(fe1_groups(l)));
      if (sum_w > 0.0) {
        non_empty_fe1(idx1) = l;
        group1_inv_w(idx1) = 1.0 / sum_w;
        idx1++;
      }
    }
  }
  for (size_t l = 0; l < L2; ++l) {
    if (fe2_groups(l).n_elem > 1) {
      double sum_w = accu(w.elem(fe2_groups(l)));
      if (sum_w > 0.0) {
        non_empty_fe2(idx2) = l;
        group2_inv_w(idx2) = 1.0 / sum_w;
        idx2++;
      }
    }
  }

  // Working vectors
  vec x(N, fill::none), x0(N, fill::none);
  vec diff(N, fill::none);
  
  // CG-specific vectors
  vec g(N, fill::none), g_old(N, fill::none), p(N, fill::none);
  
  // Fallback acceleration vectors
  vec Gx(N, fill::none), G2x(N, fill::none);
  vec deltaG(N, fill::none), delta2(N, fill::none);

  // Define optimized Symmetric Kaczmarz projection using only non-empty groups
  auto project_symmetric_kaczmarz_2fe_fast = [&](vec &v) {
    // Forward pass: FE1 then FE2
    for (size_t idx = 0; idx < n_non_empty_fe1; ++idx) {
      const uvec &coords = fe1_groups(non_empty_fe1(idx));
      double xbar = dot(w.elem(coords), v.elem(coords)) * group1_inv_w(idx);
      v.elem(coords) -= xbar;
    }
    
    for (size_t idx = 0; idx < n_non_empty_fe2; ++idx) {
      const uvec &coords = fe2_groups(non_empty_fe2(idx));
      double xbar = dot(w.elem(coords), v.elem(coords)) * group2_inv_w(idx);
      v.elem(coords) -= xbar;
    }
    
    // Backward pass: FE2 then FE1
    for (size_t idx = n_non_empty_fe2; idx-- > 0; ) {
      const uvec &coords = fe2_groups(non_empty_fe2(idx));
      double xbar = dot(w.elem(coords), v.elem(coords)) * group2_inv_w(idx);
      v.elem(coords) -= xbar;
    }
    
    for (size_t idx = n_non_empty_fe1; idx-- > 0; ) {
      const uvec &coords = fe1_groups(non_empty_fe1(idx));
      double xbar = dot(w.elem(coords), v.elem(coords)) * group1_inv_w(idx);
      v.elem(coords) -= xbar;
    }
  };

  // Process columns in blocks for better cache usage
  const size_t block_size = 4;
  for (size_t col_start = 0; col_start < P; col_start += block_size) {
    size_t col_end = std::min(col_start + block_size, P);
    
    // Process each column in the block
    for (size_t col = col_start; col < col_end; ++col) {
      x = V.col(col);
      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      size_t iint = iter_interrupt;
      size_t isr = iter_ssr;
      
      // Adaptive tolerance for large models
      double adaptive_tol = tol;
      if (N > 100000 || P > 1000) {
        adaptive_tol = std::max(tol, 1e-4);
      }

      for (size_t iter = 0; iter < max_iter; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iter_interrupt;
        }

        x0 = x;
        project_symmetric_kaczmarz_2fe_fast(x);

        // Compute gradient (residual before projection - residual after)
        g = x0 - x;

        // 1) convergence via weighted diff
        diff = abs(x - x0) / (1.0 + abs(x0));
        double ratio = dot(diff, w) * inv_sw;
        
        // Tighten tolerance as we converge for large models
        if (N > 100000 || P > 1000) {
          if (iter > 5 && ratio < 0.1) {
            adaptive_tol = tol;
          }
        }
        
        if (ratio < adaptive_tol) {
          // Early termination optimization: check if next columns might converge quickly
          if (col > col_start && ratio < adaptive_tol * 0.1) {
            size_t skip_ahead = std::min(size_t(2), col_end - col - 1);
            for (size_t s = 1; s <= skip_ahead; ++s) {
              if (col + s < col_end) {
                vec x_next = V.col(col + s);
                vec x_next_old = x_next;
                project_symmetric_kaczmarz_2fe_fast(x_next);
                vec diff_next = abs(x_next - x_next_old) / (1.0 + abs(x_next_old));
                double ratio_next = dot(diff_next, w) * inv_sw;
                if (ratio_next < adaptive_tol) {
                  V.unsafe_col(col + s) = x_next;
                  col++; // Skip this column in outer loop
                } else {
                  break; // Stop skipping if convergence slows
                }
              }
            }
          }
          break;
        }

        // 2) Acceleration
        if (use_cg && iter >= accel_start) {
          // Conjugate gradient acceleration
          conjugate_gradient_accel(x, g, g_old, p, w, inv_sw,
                                   project_symmetric_kaczmarz_2fe_fast, 
                                   iter, accel_start);
          g_old = g;
        } else if (!use_cg && iter >= 5 && (iter % 5) == 0) {
          // Fallback: Irons-Tuck acceleration
          Gx = x;
          project_symmetric_kaczmarz_2fe_fast(Gx);
          G2x = Gx;
          deltaG = G2x - x;
          delta2 = G2x - 2.0 * x + x0;
          double ssq = dot(delta2, delta2);
          if (ssq > 1e-10) {
            double coef = dot(deltaG, delta2) / ssq;
            x = (coef > 0.0 && coef < 2.0) ? (G2x - coef * deltaG) : G2x;
          }
        }

        // 3) SSR-based early exit
        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += iter_ssr;
          double ssr = dot(x % x, w) * inv_sw;
          if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < adaptive_tol)
            break;
          ssr0 = ssr;
        }

        // 4) heuristic early exit
        if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < adaptive_tol * 20)
          break;
        ratio0 = ratio;
      }

      V.unsafe_col(col) = x;
    }
  }
}

void center_variables(mat &V, const vec &w,
                      const field<field<uvec>> &group_indices,
                      const double &tol, const size_t &max_iter,
                      const size_t &iter_interrupt, const size_t &iter_ssr,
                      const size_t &accel_start,
                      const bool use_cg) {
  // Early termination check
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem) {
    return;
  }

  if (group_indices.n_elem == 2) {
    center_variables_2fe(V, w, group_indices, tol, max_iter, iter_interrupt,
                         iter_ssr, accel_start, use_cg);
    return;
  }

  // If no groups, just return
  if (group_indices.n_elem == 0) {
    return;
  }

  // Auxiliary variables (fixed)
  const size_t I = max_iter, N = V.n_rows, P = V.n_cols,
               K = group_indices.n_elem, iint0 = iter_interrupt,
               isr0 = iter_ssr;
  const double inv_sw = 1.0 / accu(w);

  // Pre-allocate and identify non-empty groups
  size_t total_groups = 0;
  for (size_t k = 0; k < K; ++k) {
    total_groups += group_indices(k).n_elem;
  }
  
  // Structures for non-empty groups
  struct GroupInfo {
    size_t k; // which fixed effect
    size_t l; // which level within that FE
    double inv_weight;
  };
  std::vector<GroupInfo> non_empty_groups;
  non_empty_groups.reserve(total_groups);
  
  // Precompute group weights and identify non-empty groups
  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &idxs = group_indices(k);
    const size_t L = idxs.n_elem;

    for (size_t l = 0; l < L; ++l) {
      const uvec &coords = idxs(l);
      if (coords.n_elem <= 1) continue;
      
      // Check validity
      bool all_valid = true;
      for (uword i = 0; i < coords.n_elem; ++i) {
        if (coords(i) >= w.n_elem) {
          all_valid = false;
          break;
        }
      }
      if (!all_valid) continue;
      
      double sum_w = accu(w.elem(coords));
      if (sum_w > 0.0) {
        non_empty_groups.push_back({k, l, 1.0 / sum_w});
      }
    }
  }
  
  const size_t n_non_empty = non_empty_groups.size();

  // Auxiliary variables (storage)
  vec x(N, fill::none), x0(N, fill::none);
  vec diff(N, fill::none);
  
  // CG-specific vectors
  vec g(N, fill::none), g_old(N, fill::none), p(N, fill::none);
  
  // Fallback acceleration vectors
  vec Gx(N, fill::none), G2x(N, fill::none);
  vec deltaG(N, fill::none), delta2(N, fill::none);

  // Optimized Symmetric Kaczmarz projection
  auto project_symmetric_kaczmarz_fast = [&](vec &v) {
    // Forward pass - use pre-filtered non-empty groups
    for (size_t idx = 0; idx < n_non_empty; ++idx) {
      const GroupInfo &gi = non_empty_groups[idx];
      const uvec &coords = group_indices(gi.k)(gi.l);
      
      double xbar = dot(w.elem(coords), v.elem(coords)) * gi.inv_weight;
      v.elem(coords) -= xbar;
    }
    
    // Backward pass - process in reverse order
    for (size_t idx = n_non_empty; idx-- > 0; ) {
      const GroupInfo &gi = non_empty_groups[idx];
      const uvec &coords = group_indices(gi.k)(gi.l);
      
      double xbar = dot(w.elem(coords), v.elem(coords)) * gi.inv_weight;
      v.elem(coords) -= xbar;
    }
  };

  // Process columns in blocks for better cache usage
  const size_t block_size = 4;
  for (size_t col_start = 0; col_start < P; col_start += block_size) {
    size_t col_end = std::min(col_start + block_size, P);
    
    for (size_t col = col_start; col < col_end; ++col) {
      x = V.col(col);
      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      size_t iint = iint0;
      size_t isr = isr0;
      
      // Adaptive tolerance for large models
      double adaptive_tol = tol;
      if (N > 100000 || P > 1000) {
        adaptive_tol = std::max(tol, 1e-4);
      }

      for (size_t iter = 0; iter < I; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iint0;
        }

        x0 = x;
        project_symmetric_kaczmarz_fast(x);

        // Compute gradient
        g = x0 - x;

        // 1) convergence via weighted diff
        diff = abs(x - x0) / (1.0 + abs(x0));
        double ratio = dot(diff, w) * inv_sw;
        
        // Tighten tolerance as we converge for large models
        if (N > 100000 || P > 1000) {
          if (iter > 5 && ratio < 0.1) {
            adaptive_tol = tol;
          }
        }
        
        if (ratio < adaptive_tol) {
          // Early termination optimization
          if (col > col_start && ratio < adaptive_tol * 0.1) {
            size_t skip_ahead = std::min(size_t(2), col_end - col - 1);
            for (size_t s = 1; s <= skip_ahead; ++s) {
              if (col + s < col_end) {
                vec x_next = V.col(col + s);
                vec x_next_old = x_next;
                project_symmetric_kaczmarz_fast(x_next);
                vec diff_next = abs(x_next - x_next_old) / (1.0 + abs(x_next_old));
                double ratio_next = dot(diff_next, w) * inv_sw;
                if (ratio_next < adaptive_tol) {
                  V.unsafe_col(col + s) = x_next;
                  col++; // Skip this column in outer loop
                } else {
                  break;
                }
              }
            }
          }
          break;
        }

        // 2) Acceleration
        if (use_cg && iter >= accel_start) {
          // Conjugate gradient acceleration
          conjugate_gradient_accel(x, g, g_old, p, w, inv_sw,
                                   project_symmetric_kaczmarz_fast, 
                                   iter, accel_start);
          g_old = g;
        } else if (!use_cg && iter >= 5 && (iter % 5) == 0) {
          // Fallback: Irons-Tuck acceleration
          Gx = x;
          project_symmetric_kaczmarz_fast(Gx);
          G2x = Gx;
          deltaG = G2x - x;
          delta2 = G2x - 2.0 * x + x0;
          double ssq = dot(delta2, delta2);
          if (ssq > 1e-10) {
            double coef = dot(deltaG, delta2) / ssq;
            x = (coef > 0.0 && coef < 2.0) ? (G2x - coef * deltaG) : G2x;
          }
        }

        // 3) SSR-based early exit
        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += isr0;
          double ssr = dot(x % x, w) * inv_sw;
          if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < adaptive_tol)
            break;
          ssr0 = ssr;
        }

        // 4) heuristic early exit  
        if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < adaptive_tol * 20)
          break;
        ratio0 = ratio;
      }

      V.unsafe_col(col) = x;
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H