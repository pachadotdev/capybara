// Centering using observation-to-group mapping
#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Optimized mapping for Linear Scan
struct ObsToGroupMapping {
    // Flattened maps for direct access: maps[k][i] = group_id
    // Using std::vector for guaranteed contiguous memory and fast access
    std::vector<std::vector<uword>> maps;
    
    // Inverse weights: inv_weights[k][group_id]
    std::vector<vec> inv_weights;
    
    uword n_obs;
    uword K;
    
    // Check if we have initialized indices
    bool indices_ready = false;

    // Build only the index mapping (expensive, do once)
    void build_index(const field<field<uvec>> &group_indices) {
        K = group_indices.n_elem;
        if (K == 0) return;
        
        maps.resize(K);
        inv_weights.resize(K);
        
        // Infer n_obs from max index in first group system
        // (Assuming valid group structure covering all obs or 0-based indices)
        // Ideally we should have n_obs passed, but we can scan.
        uword max_idx = 0;
        const field<uvec>& g0 = group_indices(0);
        for(uword g=0; g<g0.n_elem; ++g) {
            if (!g0(g).is_empty()) max_idx = std::max(max_idx, arma::max(g0(g)));
        }
        n_obs = max_idx + 1;
        
        for(uword k=0; k<K; ++k) {
            maps[k].assign(n_obs, 0); // Default 0, but must be filled
            const field<uvec> &fe_groups = group_indices(k);
            uword n_groups = fe_groups.n_elem;
            inv_weights[k].set_size(n_groups);
            
            // Fill map
            for(uword g=0; g<n_groups; ++g) {
                const uvec& idx = fe_groups(g);
                const uword* ptr = idx.memptr();
                uword cnt = idx.n_elem;
                for(uword j=0; j<cnt; ++j) {
                    maps[k][ptr[j]] = g;
                }
            }
        }
        indices_ready = true;
    }

    // Update weights (cheaper, do per iter)
    void update_weights(const vec &w) {
         if (!indices_ready || K == 0) return;
         
         const double* w_ptr = w.memptr();
         bool use_w = (w.n_elem == n_obs);
         
         for(uword k=0; k<K; ++k) {
             vec& inv_w_vec = inv_weights[k];
             inv_w_vec.zeros();
             double* inv_w_ptr = inv_w_vec.memptr();
             const std::vector<uword>& map = maps[k];
             const uword* m_ptr = map.data();
             uword n_groups = inv_w_vec.n_elem;

             if (use_w) {
                 for(uword i=0; i<n_obs; ++i) {
                     inv_w_ptr[m_ptr[i]] += w_ptr[i];
                 }
             } else {
                 for(uword i=0; i<n_obs; ++i) {
                     inv_w_ptr[m_ptr[i]] += 1.0;
                 }
             }
             
             // Invert
             for(uword g=0; g<n_groups; ++g) {
                 if (inv_w_ptr[g] > 1e-12) inv_w_ptr[g] = 1.0 / inv_w_ptr[g];
                 else inv_w_ptr[g] = 0.0;
             }
         }
    }
};

struct CenteringWorkspace {
  mat Alpha;      
  mat Alpha_prev; 
  const double* ptrs_V[8];
  const double* ptrs_a_read[8];
  double* ptrs_a_write[8];
  
  // Buffers for 2FE speed
  const double* ptrs_a1[8];
  const double* ptrs_a2[8];
  double* ptrs_a1_w[8];
  double* ptrs_a2_w[8];

  void ensure_size(uword m, uword p) {
      if (Alpha.n_rows != m || Alpha.n_cols != p) {
           Alpha.zeros(m, p);
           Alpha_prev.zeros(m, p);
      }
  }
  void ensure_scratch(uword K) {} // Fixed size arrays
  void clear() { Alpha.zeros(); Alpha_prev.zeros(); }
};

// --------------------------------------------------------------------------
// Compat Shim
// --------------------------------------------------------------------------
// We define precompute_group_info to return ObsToGroupMapping
// This replaces the old function.
inline ObsToGroupMapping precompute_group_info(const field<field<uvec>> &group_indices, const vec &w) {
    ObsToGroupMapping map;
    map.build_index(group_indices);
    map.update_weights(w);
    return map;
}

// --------------------------------------------------------------------------
// Core Solvers
// --------------------------------------------------------------------------

template<int B_SZ>
inline void center_2fe_block_iter(
    const uword n_obs, const uword p_start, const uword real_b_sz,
    const double* w_ptr,
    const uword* g1, const uword* g2,
    mat& V, mat& alpha1, mat& alpha2) 
{
    const double* v_ptrs[B_SZ];
    double* a1_ptrs[B_SZ]; // Write
    const double* a2_ptrs[B_SZ]; // Read
    
    for(int j=0; j<B_SZ; ++j) {
        if(j < (int)real_b_sz) {
            v_ptrs[j] = V.colptr(p_start + j);
            a1_ptrs[j] = alpha1.colptr(p_start + j);
            a2_ptrs[j] = alpha2.colptr(p_start + j);
        }
    }
    
    for(uword i=0; i<n_obs; ++i) {
        double wi = w_ptr[i];
        if (wi > 1e-14) {
            uword u_g1 = g1[i];
            uword u_g2 = g2[i];
            
            #pragma GCC unroll 4
            for(int j=0; j<B_SZ; ++j) {
                if(j < (int)real_b_sz) {
                   a1_ptrs[j][u_g1] += (v_ptrs[j][i] - a2_ptrs[j][u_g2]) * wi;
                }
            }
        }
    }
}

template<int B_SZ>
inline void center_2fe_block_iter_a2(
    const uword n_obs, const uword p_start, const uword real_b_sz,
    const double* w_ptr,
    const uword* g1, const uword* g2,
    mat& V, mat& alpha1, mat& alpha2) 
{
    const double* v_ptrs[B_SZ];
    const double* a1_ptrs[B_SZ];
    double* a2_ptrs[B_SZ];
    
    for(int j=0; j<B_SZ; ++j) {
        if(j < (int)real_b_sz) {
            v_ptrs[j] = V.colptr(p_start + j);
            a1_ptrs[j] = alpha1.colptr(p_start + j);
            a2_ptrs[j] = alpha2.colptr(p_start + j);
        }
    }
    
    for(uword i=0; i<n_obs; ++i) {
        double wi = w_ptr[i];
        if (wi > 1e-14) {
            uword u_g1 = g1[i];
            uword u_g2 = g2[i];
            
            #pragma GCC unroll 4
            for(int j=0; j<B_SZ; ++j) {
                if(j < (int)real_b_sz) {
                   a2_ptrs[j][u_g2] += (v_ptrs[j][i] - a1_ptrs[j][u_g1]) * wi;
                }
            }
        }
    }
}

inline void center_accel_2fe(mat &V, const vec &w,
                             const ObsToGroupMapping &mapping,
                             double tol, uword max_iter,
                             CenteringWorkspace &ws) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword n1 = mapping.inv_weights[0].n_elem;
  const uword n2 = mapping.inv_weights[1].n_elem;

  mat alpha1(n1, P, fill::zeros);
  mat alpha2(n2, P, fill::zeros);
  
  const uword* g1 = mapping.maps[0].data();
  const uword* g2 = mapping.maps[1].data();
  const double* p_w = w.memptr();
  
  const uword block_size = 4;
  
  // Acceleration
  mat alpha1_prev = alpha1;
  mat alpha1_prev2 = alpha1;
  bool use_accel = false;
  
  for (uword iter = 0; iter < max_iter; ++iter) {
     mat alpha1_old = alpha1;
     
     // Update A1
     alpha1.zeros();
     for (uword p = 0; p < P; p += block_size) {
        uword b_sz = std::min(block_size, P - p);
        center_2fe_block_iter<4>(n_obs, p, b_sz, p_w, g1, g2, V, alpha1, alpha2);
     }
     alpha1.each_col() %= mapping.inv_weights[0];
     
     // Update A2
     alpha2.zeros();
     for (uword p = 0; p < P; p += block_size) {
        uword b_sz = std::min(block_size, P - p);
        center_2fe_block_iter_a2<4>(n_obs, p, b_sz, p_w, g1, g2, V, alpha1, alpha2);
     }
     alpha2.each_col() %= mapping.inv_weights[1];
     
     // Acceleration (Irons-Tuck)
     // Based on alpha1 sequence
     if (iter >= 3) {
        if (use_accel) {
             mat D1 = alpha1 - alpha1_prev;
             mat D2 = D1 - (alpha1_prev - alpha1_prev2);
             double ssq = accu(square(D2));
             if (ssq > 1e-14) {
                 double vprod = accu(D1 % D2);
                 double coef = vprod / ssq;
                 if (coef > 0.0 && coef < 2.0) {
                     alpha1 = alpha1 - coef * D1;
                     // Re-update A2 to be consistent with accelerated A1?
                     // Standard MAP accel usually accelerates one iterate.
                     // But we need A2 to match. 
                     // Or we just update A2 in next step.
                     // The original code accelerates alpha1 and proceeds.
                 }
             }
        }
        alpha1_prev2 = alpha1_prev;
        alpha1_prev = alpha1_old;
        use_accel = true;
     } else {
        alpha1_prev2 = alpha1_prev;
        alpha1_prev = alpha1_old; 
     }

     // Convergence
     double diff = norm(alpha1 - alpha1_old, "fro");
     if (diff < tol) break;
  }
  
  // Final Subtract
  for (uword p = 0; p < P; p += block_size) {
     uword b_sz = std::min(block_size, P - p);
     double* v_ptrs[4]; 
     const double* a1_ptrs[4];
     const double* a2_ptrs[4];
     for(uword j=0; j<b_sz; ++j) {
         v_ptrs[j] = V.colptr(p+j);
         a1_ptrs[j] = alpha1.colptr(p+j);
         a2_ptrs[j] = alpha2.colptr(p+j);
     }
     for(uword i=0; i<n_obs; ++i) {
         uword u_g1 = g1[i];
         uword u_g2 = g2[i];
         for(uword j=0; j<b_sz; ++j) { 
             v_ptrs[j][i] -= a1_ptrs[j][u_g1] + a2_ptrs[j][u_g2];
         }
     }
  }
}

// --------------------------------------------------------------------------
// K-FE (Generalized & Accelerated)
// --------------------------------------------------------------------------

// Project on one FE: alpha_k = weighted_mean(V - sum_{j!=k} alpha_j)
template<int B_SZ>
inline void project_kfe_block_iter(
    const uword n_obs, const uword p_start, const uword real_b_sz,
    const double* w_ptr,
    const uword* target_map,
    const std::vector<const uword*>& all_maps,
    const std::vector<const double*>& all_alphas_read, // [K][block_col]
    const uword k_idx,
    const uword K,
    mat& V,
    double* target_alpha_write[B_SZ]) // [block_col]
{
    // Local pointers for V
    const double* v_ptrs[B_SZ];
    for(int j=0; j<B_SZ; ++j) {
        if(j < (int)real_b_sz) v_ptrs[j] = V.colptr(p_start + j);
    }
    
    // We need to access alpha_j for j != k
    // Construct local array of pointers for this block for each K
    // This might be register pressure heavy if K is large.
    // But usually K is small (3-5).
    // If K is large, we iterate K in inner loop.
    
    // Optimization: Unroll K loop inside obs loop?
    // Or Precompute sum_alpha_others? No, expensive to store.
    
    // Inner loop
    for(uword i=0; i<n_obs; ++i) {
        double wi = w_ptr[i];
        if (wi > 1e-14) {
            uword g_target = target_map[i];
            
            // For each column in block
            #pragma GCC unroll 4
            for(int j=0; j<B_SZ; ++j) {
                if(j < (int)real_b_sz) {
                    double val = v_ptrs[j][i];
                    // Subtract others
                    for(uword o=0; o<K; ++o) {
                        if(o != k_idx) {
                             // all_alphas_read[o] points to start of block for FE 'o'
                             // But we need to index [j] within that block... 
                             // Wait, all_alphas_read structure is complex.
                             // Let's pass: array of pointers where ptrs[o] = alpha[o].colptr(p_start+j)
                             // But we have multiple j.
                             
                             // Better: all_alphas_read is [K * B_SZ].
                             // ptr = all_alphas_read[o * B_SZ + j]
                             val -= all_alphas_read[o * B_SZ + j][all_maps[o][i]];
                        }
                    }
                    target_alpha_write[j][g_target] += val * wi;
                }
            }
        }
    }
}

inline void center_kfe_accel(mat &V, const vec &w,
                       const ObsToGroupMapping &mapping,
                       double tol, uword max_iter,
                       CenteringWorkspace &ws) {
   uword n_obs = V.n_rows;
   uword P = V.n_cols;
   uword K = mapping.K;
   
   // Allocate Alpha: Total Groups
   uword total_groups = 0;
   std::vector<uword> offsets(K);
   for(uword k=0; k<K; ++k) {
       offsets[k] = total_groups;
       total_groups += mapping.inv_weights[k].n_elem;
   }
   
   ws.ensure_size(total_groups, P);
   mat& Alpha = ws.Alpha; // (total_groups x P)
   // Alpha.zeros(); // Usually start from 0 or warm start? For now 0.
   // But keep persistent if workspace is reused? The caller might expect 0 start.
   // center_variables doesn't promise warm start across calls usually.
   Alpha.zeros();

   const double* w_ptr = w.memptr();
   const uword block_size = 4;
   
   // Maps pointers
   std::vector<const uword*> maps_ptrs(K);
   for(uword k=0; k<K; ++k) maps_ptrs[k] = mapping.maps[k].data();

   // Acceleration
   mat Alpha_prev = Alpha;
   mat Alpha_prev2 = Alpha;
   bool use_accel = false;

   for(uword iter=0; iter<max_iter; ++iter) {
       mat Alpha_old = Alpha;
       
       // Cycle through K
       for(uword k=0; k<K; ++k) {
           uword off_k = offsets[k];
           uword ng_k = mapping.inv_weights[k].n_elem;
           
           // Zero out current FE block in Alpha
           Alpha.rows(off_k, off_k + ng_k - 1).zeros();
           
           // Blocked processing over columns
           for (uword p = 0; p < P; p += block_size) {
                uword b_sz = std::min(block_size, P - p);
                
                // Prepare pointers for other alphas
                // We need [K * block_size] pointers
                std::vector<const double*> all_alphas_ptrs(K * block_size); 
                double* write_ptrs[4];
                
                for(uword j=0; j<b_sz; ++j) {
                    write_ptrs[j] = Alpha.colptr(p+j) + off_k;
                    for(uword o=0; o<K; ++o) {
                        all_alphas_ptrs[o * block_size + j] = Alpha.colptr(p+j) + offsets[o];
                    }
                }
                
                project_kfe_block_iter<4>(
                    n_obs, p, b_sz, w_ptr, maps_ptrs[k],
                    maps_ptrs, all_alphas_ptrs, k, K, V, write_ptrs
                );
           }
           
           // Apply weights
           Alpha.rows(off_k, off_k + ng_k - 1).each_col() %= mapping.inv_weights[k];
       }
       
       // Acceleration
       if (iter >= 3) {
            if (use_accel) {
                 mat Delta_G = Alpha - Alpha_prev; // G(Alpha) - Alpha_prev
                 // Wait, MAP is G(x).
                 // Alpha is now G(Alpha_old).
                 // Alpha_prev is x_old.
                 // Correct logic:
                 // x_k = Alpha_prev (input to iter)
                 // x_{k+1} = Alpha (output of iter)
                 // Delta = x_{k+1} - x_k
                 // But we have sequence x_{k-2}, x_{k-1}, x_k.
                 
                 // Irons-Tuck on the sequence calls:
                 // Let x0 = Alpha_prev2
                 // Let x1 = Alpha_prev
                 // Let x2 = Alpha (current)
                 // D1 = x1 - x0
                 // D2 = x2 - x1
                 // We want accel based on these.
                 
                 mat D1 = Alpha_prev - Alpha_prev2;
                 mat D2 = Alpha - Alpha_prev; 
                 // Note: Standard notation is Delta^2 = (x2-x1) - (x1-x0).
                 mat DD = D2 - D1;
                 
                 double ssq = accu(square(DD));
                 if (ssq > 1e-14) {
                     // Verify IT form:
                     // x_inf = x_n - (D X_n . D^2 X_n / |D^2 X_n|^2) * D^2 X_n ?
                     // Usual form: x_accel = x_n - coef * (x_n - x_{n-1})
                     // Using code from original file:
                     // mat Delta_G = GGX - GX;
                     // mat Delta2 = Delta_G - (GX - X);
                     // Here X=prev2, GX=prev, GGX=Alpha.
                     // Delta_G = Alpha - Alpha_prev
                     // Delta2 = (Alpha - Alpha_prev) - (Alpha_prev - Alpha_prev2)
                     
                     mat Delta_G = Alpha - Alpha_prev; 
                     mat Delta2 = Delta_G - (Alpha_prev - Alpha_prev2);
                     double coef = accu(Delta_G % Delta2) / ssq; // Re-check original code logic
                     
                     if (std::abs(ssq) > 1e-14) {
                         if (coef > 0.0 && coef < 2.0) {
                             Alpha = Alpha - coef * Delta_G;
                         }
                     }
                 }
            }
            Alpha_prev2 = Alpha_prev;
            Alpha_prev = Alpha_old;
            use_accel = true;
       } else {
            Alpha_prev2 = Alpha_prev;
            Alpha_prev = Alpha_old;
       }
       
       if (norm(Alpha - Alpha_old, "fro") < tol) break;
   }
   
   // Final Subtract
   for (uword p = 0; p < P; p += block_size) {
      uword b_sz = std::min(block_size, P - p);
      double* v_ptrs[4]; 
      std::vector<const double*> all_alphas_ptrs(K * block_size);
      
      for(uword j=0; j<b_sz; ++j) {
          v_ptrs[j] = V.colptr(p+j);
          for(uword o=0; o<K; ++o) {
              all_alphas_ptrs[o * block_size + j] = Alpha.colptr(p+j) + offsets[o];
          }
      }
      
      for(uword i=0; i<n_obs; ++i) {
          for(uword j=0; j<b_sz; ++j) { 
              double sum_a = 0.0;
              for(uword o=0; o<K; ++o) {
                  sum_a += all_alphas_ptrs[o * block_size + j][maps_ptrs[o][i]];
              }
              v_ptrs[j][i] -= sum_a;
          }
      }
   }
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const uword &max_iter, const uword &iter_interrupt,
    const ObsToGroupMapping *precomputed_map = nullptr,
    CenteringWorkspace *workspace = nullptr) {

  if (V.is_empty()) return;
  const uword K = group_indices.n_elem;
  if (K == 0) return;

  CenteringWorkspace local_ws;
  CenteringWorkspace* ws = workspace ? workspace : &local_ws;
  
  ObsToGroupMapping local_map;
  const ObsToGroupMapping* map = precomputed_map;
  
  if (!map) {
      local_map.build_index(group_indices);
      local_map.update_weights(w);
      map = &local_map;
  }
  
  ws->ensure_size(1, 1); 
  
  if (K == 2) {
      center_accel_2fe(V, w, *map, tol, max_iter, *ws);
  } else {
      // General K (1, 3+)
      center_kfe_accel(V, w, *map, tol, max_iter, *ws);
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
