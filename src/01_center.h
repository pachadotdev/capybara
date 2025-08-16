// Symmetric Kaczmarz with Conjugate Gradient acceleration

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

#include <vector>
#include <cstring>

namespace capybara {

// Helper functions for convergence criteria (matching fixest style)
inline bool continue_criterion(double a, double b, double tol) {
  double diff = std::abs(a - b);
  return (diff > tol) && (diff / (0.1 + std::abs(a)) > tol);
}

inline bool stopping_criterion(double a, double b, double tol) {
  double diff = std::abs(a - b);
  return (diff < tol) || (diff / (0.1 + std::abs(a)) < tol);
}

// Optimized group projection using raw pointers
inline void project_group(double* v, const double* w, const uvec& coords, 
                          double inv_group_weight) {
  double weighted_sum = 0.0;
  const uword n = coords.n_elem;
  const uword* coord_ptr = coords.memptr();
  
  // Compute weighted sum
  for (uword i = 0; i < n; ++i) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
  }
  
  double mean = weighted_sum * inv_group_weight;
  
  // Subtract mean
  for (uword i = 0; i < n; ++i) {
    v[coord_ptr[i]] -= mean;
  }
}

// Structure to hold precomputed group info (similar to fixest's approach)
struct GroupInfo {
  const uvec* coords;
  double inv_weight;
  uword n_elem;
};

// Optimized conjugate gradient acceleration
template <typename ProjectFunc>
void cg_acceleration(double* x, double* g, double* g_old, double* p,
                                        const double* w, double inv_sw,
                                        ProjectFunc project,
                                        size_t iter, size_t accel_start,
                                        size_t n) {
  if (iter == accel_start) {
    std::memcpy(p, g, n * sizeof(double));
  } else if (iter > accel_start) {
    // Hestenes-Stiefel formula with raw pointer arithmetic
    double num = 0.0, denom = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
      double diff_g = g[i] - g_old[i];
      double weighted_diff = w[i] * diff_g;
      num += g[i] * weighted_diff;
      denom += p[i] * weighted_diff;
    }
    
    num *= inv_sw;
    denom *= inv_sw;
    
    if (std::abs(denom) > 1e-10) {
      double beta = num / denom;
      if (beta < 0.0 || beta > 10.0) {
        std::memcpy(p, g, n * sizeof(double));
      } else {
        for (size_t i = 0; i < n; ++i) {
          p[i] = g[i] + beta * p[i];
        }
      }
    } else {
      std::memcpy(p, g, n * sizeof(double));
    }
  }
  
  // Line search - use pre-allocated workspace
  vec Ap(n, fill::none);
  double* Ap_ptr = Ap.memptr();
  std::memcpy(Ap_ptr, p, n * sizeof(double));
  
  project(Ap);
  
  double pAp = 0.0, gg = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double Ap_i = p[i] - Ap_ptr[i];
    pAp += w[i] * p[i] * Ap_i;
    gg += w[i] * g[i] * g[i];
  }
  
  pAp *= inv_sw;
  gg *= inv_sw;
  
  if (pAp > 1e-10) {
    double alpha = gg / pAp;
    for (size_t i = 0; i < n; ++i) {
      x[i] -= alpha * p[i];
    }
  }
}

// Specialized 2-FE implementation with optimizations
void center_variables_2fe(mat &V, const vec &w,
                          const field<field<uvec>> &group_indices,
                          const double &tol, const size_t &max_iter,
                          const size_t &iter_interrupt,
                          const size_t &iter_ssr,
                          const size_t &accel_start,
                          const bool use_cg) {
  if (V.is_empty() || w.is_empty()) return;
  
  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);
  const double* w_ptr = w.memptr();
  
  // Extract and precompute group information
  const field<uvec>& fe1_groups = group_indices(0);
  const field<uvec>& fe2_groups = group_indices(1);
  
  // Precompute non-empty groups with their weights
  std::vector<GroupInfo> groups1, groups2;
  groups1.reserve(fe1_groups.n_elem);
  groups2.reserve(fe2_groups.n_elem);
  
  for (size_t l = 0; l < fe1_groups.n_elem; ++l) {
    const uvec& coords = fe1_groups(l);
    if (coords.n_elem > 1) {
      double sum_w = 0.0;
      const uword* coord_ptr = coords.memptr();
      for (uword i = 0; i < coords.n_elem; ++i) {
        sum_w += w_ptr[coord_ptr[i]];
      }
      if (sum_w > 0.0) {
        groups1.push_back({&coords, 1.0 / sum_w, coords.n_elem});
      }
    }
  }
  
  for (size_t l = 0; l < fe2_groups.n_elem; ++l) {
    const uvec& coords = fe2_groups(l);
    if (coords.n_elem > 1) {
      double sum_w = 0.0;
      const uword* coord_ptr = coords.memptr();
      for (uword i = 0; i < coords.n_elem; ++i) {
        sum_w += w_ptr[coord_ptr[i]];
      }
      if (sum_w > 0.0) {
        groups2.push_back({&coords, 1.0 / sum_w, coords.n_elem});
      }
    }
  }
   
  // Pre-allocate all working vectors
  vec x(N, fill::none), x0(N, fill::none);
  vec g(N, fill::none), g_old(N, fill::none), p(N, fill::none);
  vec Gx(N, fill::none), G2x(N, fill::none);
  
  // Lambda for symmetric Kaczmarz projection
  auto project_symmetric_kaczmarz_2fe = [&](vec& v) {
    double* v_ptr = v.memptr();
    
    // Forward pass
    for (const auto& group : groups1) {
      project_group(v_ptr, w_ptr, *group.coords, group.inv_weight);
    }
    for (const auto& group : groups2) {
      project_group(v_ptr, w_ptr, *group.coords, group.inv_weight);
    }
    
    // Backward pass
    for (auto it = groups2.rbegin(); it != groups2.rend(); ++it) {
      project_group(v_ptr, w_ptr, *it->coords, it->inv_weight);
    }
    for (auto it = groups1.rbegin(); it != groups1.rend(); ++it) {
      project_group(v_ptr, w_ptr, *it->coords, it->inv_weight);
    }
  };
  
  // Process columns with improved memory access pattern
  const size_t block_size = 8; // Larger blocks for better cache usage
  
  for (size_t col_start = 0; col_start < P; col_start += block_size) {
    size_t col_end = std::min(col_start + block_size, P);
    
    for (size_t col = col_start; col < col_end; ++col) {
      double* col_ptr = V.colptr(col);
      double* x_ptr = x.memptr();
      double* x0_ptr = x0.memptr();
      double* g_ptr = g.memptr();
      double* g_old_ptr = g_old.memptr();
      double* p_ptr = p.memptr();
      
      // Copy column data
      std::memcpy(x_ptr, col_ptr, N * sizeof(double));
      
      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      size_t iint = iter_interrupt;
      size_t isr = iter_ssr;
      
      for (size_t iter = 0; iter < max_iter; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iter_interrupt;
        }
        
        // Save current state
        std::memcpy(x0_ptr, x_ptr, N * sizeof(double));
        
        // Apply projection
        project_symmetric_kaczmarz_2fe(x);
        
        // Compute gradient (g = x0 - x)
        for (size_t i = 0; i < N; ++i) {
          g_ptr[i] = x0_ptr[i] - x_ptr[i];
        }
        
        // Check convergence
        double weighted_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
          double rel_diff = std::abs(x_ptr[i] - x0_ptr[i]) / (1.0 + std::abs(x0_ptr[i]));
          weighted_diff += w_ptr[i] * rel_diff;
        }
        double ratio = weighted_diff * inv_sw;
        
        if (ratio < tol) break;
        
        // Acceleration
        if (use_cg && iter >= accel_start) {
          cg_acceleration(x_ptr, g_ptr, g_old_ptr, p_ptr,
                                             w_ptr, inv_sw,
                                             project_symmetric_kaczmarz_2fe,
                                             iter, accel_start, N);
          std::memcpy(g_old_ptr, g_ptr, N * sizeof(double));
        } else if (!use_cg && iter >= 5 && (iter % 5) == 0) {
          // Irons-Tuck acceleration
          double* Gx_ptr = Gx.memptr();
          double* G2x_ptr = G2x.memptr();
          
          std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));
          project_symmetric_kaczmarz_2fe(Gx);
          std::memcpy(G2x_ptr, Gx_ptr, N * sizeof(double));
          
          double vprod = 0.0, ssq = 0.0;
          for (size_t i = 0; i < N; ++i) {
            double deltaG = G2x_ptr[i] - x_ptr[i];
            double delta2 = G2x_ptr[i] - 2.0 * x_ptr[i] + x0_ptr[i];
            vprod += deltaG * delta2;
            ssq += delta2 * delta2;
          }
          
          if (ssq > 1e-10) {
            double coef = vprod / ssq;
            if (coef > 0.0 && coef < 2.0) {
              for (size_t i = 0; i < N; ++i) {
                x_ptr[i] = G2x_ptr[i] - coef * (G2x_ptr[i] - x_ptr[i]);
              }
            }
          }
        }
        
        // SSR-based early exit
        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += iter_ssr;
          double ssr = 0.0;
          for (size_t i = 0; i < N; ++i) {
            ssr += w_ptr[i] * x_ptr[i] * x_ptr[i];
          }
          ssr *= inv_sw;
          if (std::abs(ssr - ssr0) / (1.0 + std::abs(ssr0)) < tol) break;
          ssr0 = ssr;
        }
        
        // Heuristic early exit
        if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20) break;
        ratio0 = ratio;
      }
      
      // Copy result back
      std::memcpy(col_ptr, x_ptr, N * sizeof(double));
    }
  }
}

// General K-FE implementation with optimizations
void center_variables(mat &V, const vec &w,
                      const field<field<uvec>> &group_indices,
                      const double &tol, const size_t &max_iter,
                      const size_t &iter_interrupt, const size_t &iter_ssr,
                      const size_t &accel_start,
                      const bool use_cg) {
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem) return;
  
  const size_t K = group_indices.n_elem;
  
  // Use specialized 2-FE version when applicable
  if (K == 2) {
    center_variables_2fe(V, w, group_indices, tol, max_iter, iter_interrupt,
                         iter_ssr, accel_start, use_cg);
    return;
  }
  
  if (K == 0) return;
  
  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);
  const double* w_ptr = w.memptr();
  
  // Precompute all non-empty groups
  std::vector<GroupInfo> all_groups;
  std::vector<size_t> fe_boundaries(K + 1, 0);
  
  for (size_t k = 0; k < K; ++k) {
    const field<uvec>& fe_groups = group_indices(k);
    fe_boundaries[k] = all_groups.size();
    
    for (size_t l = 0; l < fe_groups.n_elem; ++l) {
      const uvec& coords = fe_groups(l);
      if (coords.n_elem > 1) {
        double sum_w = 0.0;
        const uword* coord_ptr = coords.memptr();
        for (uword i = 0; i < coords.n_elem; ++i) {
          if (coord_ptr[i] < w.n_elem) {
            sum_w += w_ptr[coord_ptr[i]];
          }
        }
        if (sum_w > 0.0) {
          all_groups.push_back({&coords, 1.0 / sum_w, coords.n_elem});
        }
      }
    }
  }
  fe_boundaries[K] = all_groups.size();
  
  const size_t n_groups_total = all_groups.size();
  
  // Pre-allocate working vectors
  vec x(N, fill::none), x0(N, fill::none);
  vec g(N, fill::none), g_old(N, fill::none), p(N, fill::none);
  vec Gx(N, fill::none), G2x(N, fill::none);
  
  // Lambda for symmetric Kaczmarz projection
  auto project_symmetric_kaczmarz = [&](vec& v) {
    double* v_ptr = v.memptr();
    
    // Forward pass
    for (size_t i = 0; i < n_groups_total; ++i) {
      const auto& group = all_groups[i];
      project_group(v_ptr, w_ptr, *group.coords, group.inv_weight);
    }
    
    // Backward pass
    for (size_t i = n_groups_total; i-- > 0;) {
      const auto& group = all_groups[i];
      project_group(v_ptr, w_ptr, *group.coords, group.inv_weight);
    }
  };
  
  // Process columns in blocks
  const size_t block_size = 8;
  
  for (size_t col_start = 0; col_start < P; col_start += block_size) {
    size_t col_end = std::min(col_start + block_size, P);
    
    for (size_t col = col_start; col < col_end; ++col) {
      double* col_ptr = V.colptr(col);
      double* x_ptr = x.memptr();
      double* x0_ptr = x0.memptr();
      double* g_ptr = g.memptr();
      double* g_old_ptr = g_old.memptr();
      double* p_ptr = p.memptr();
      
      // Copy column data
      std::memcpy(x_ptr, col_ptr, N * sizeof(double));
      
      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      size_t iint = iter_interrupt;
      size_t isr = iter_ssr;
      
      for (size_t iter = 0; iter < max_iter; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iter_interrupt;
        }
        
        // Save current state
        std::memcpy(x0_ptr, x_ptr, N * sizeof(double));
        
        // Apply projection
        project_symmetric_kaczmarz(x);
        
        // Compute gradient
        for (size_t i = 0; i < N; ++i) {
          g_ptr[i] = x0_ptr[i] - x_ptr[i];
        }
        
        // Check convergence
        double weighted_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
          double rel_diff = std::abs(x_ptr[i] - x0_ptr[i]) / (1.0 + std::abs(x0_ptr[i]));
          weighted_diff += w_ptr[i] * rel_diff;
        }
        double ratio = weighted_diff * inv_sw;
        
        if (ratio < tol) break;
        
        // Acceleration
        if (use_cg && iter >= accel_start) {
          cg_acceleration(x_ptr, g_ptr, g_old_ptr, p_ptr,
                                             w_ptr, inv_sw,
                                             project_symmetric_kaczmarz,
                                             iter, accel_start, N);
          std::memcpy(g_old_ptr, g_ptr, N * sizeof(double));
        } else if (!use_cg && iter >= 5 && (iter % 5) == 0) {
          // Irons-Tuck acceleration
          double* Gx_ptr = Gx.memptr();
          double* G2x_ptr = G2x.memptr();
          
          std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));
          project_symmetric_kaczmarz(Gx);
          std::memcpy(G2x_ptr, Gx_ptr, N * sizeof(double));
          
          double vprod = 0.0, ssq = 0.0;
          for (size_t i = 0; i < N; ++i) {
            double deltaG = G2x_ptr[i] - x_ptr[i];
            double delta2 = G2x_ptr[i] - 2.0 * x_ptr[i] + x0_ptr[i];
            vprod += deltaG * delta2;
            ssq += delta2 * delta2;
          }
          
          if (ssq > 1e-10) {
            double coef = vprod / ssq;
            if (coef > 0.0 && coef < 2.0) {
              for (size_t i = 0; i < N; ++i) {
                x_ptr[i] = G2x_ptr[i] - coef * (G2x_ptr[i] - x_ptr[i]);
              }
            }
          }
        }
        
        // SSR-based early exit
        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += iter_ssr;
          double ssr = 0.0;
          for (size_t i = 0; i < N; ++i) {
            ssr += w_ptr[i] * x_ptr[i] * x_ptr[i];
          }
          ssr *= inv_sw;
          if (std::abs(ssr - ssr0) / (1.0 + std::abs(ssr0)) < tol) break;
          ssr0 = ssr;
        }
        
        // Heuristic early exit
        if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20) break;
        ratio0 = ratio;
      }
      
      // Copy result back
      std::memcpy(col_ptr, x_ptr, N * sizeof(double));
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
