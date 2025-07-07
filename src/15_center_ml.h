#ifndef CAPYBARA_CENTER_ML_H
#define CAPYBARA_CENTER_ML_H

// Maximum likelihood centering approach inspired by FENmlm
// Optimized for performance with pure Armadillo operations

// Forward declarations
struct ml_center_workspace;

// Family-specific centering algorithms
inline void center_ml_gaussian(mat &X, vec &y, const vec &w, 
                               const indices_info &indices,
                               double tol, size_t max_iter);

inline void center_ml_poisson(mat &X, vec &nu, const vec &w, const vec &mu,
                              const indices_info &indices,
                              double tol, size_t max_iter);

inline void center_ml_general(mat &X, vec &y_adj, const vec &w,
                              const indices_info &indices,
                              double tol, size_t max_iter);

// Core ML centering workspace
struct ml_center_workspace {
  // Unified matrix for all variables (X columns + y column)
  mat Z;
  
  // Pre-computed group statistics for cache efficiency
  field<vec> group_means;
  field<vec> group_weights;
  field<uvec> active_groups;
  
  // Convergence tracking
  vec prev_residuals;
  double prev_ssr;
  
  // Family-specific temporary storage
  vec mu_temp;
  vec weights_temp;
  
  void reserve(size_t N, size_t P, size_t K) {
    Z.set_size(N, P + 1); // +1 for y column
    group_means.set_size(K);
    group_weights.set_size(K);
    active_groups.set_size(K);
    prev_residuals.set_size(N);
    mu_temp.set_size(N);
    weights_temp.set_size(N);
  }
  
  void clear() {
    Z.reset();
    group_means.reset();
    group_weights.reset();
    active_groups.reset();
    prev_residuals.reset();
    mu_temp.reset();
    weights_temp.reset();
  }
};

// Fast projection for single FE using ML approach
inline void project_ml_1fe(mat &Z, size_t fe_idx, const indices_info &indices,
                           const vec &w, ml_center_workspace &ws) {
  const size_t J = indices.fe_sizes(fe_idx);
  const bool weighted = (w.n_elem > 1);
  
  // Pre-compute group weights for efficiency
  vec &group_inv_w = ws.group_weights(fe_idx);
  if (group_inv_w.n_elem != J) {
    group_inv_w.set_size(J);
  }
  
  if (weighted) {
    for (size_t j = 0; j < J; ++j) {
      const uvec &idx = indices.get_group(fe_idx, j);
      if (idx.is_empty()) {
        group_inv_w(j) = 0.0;
      } else {
        double sumw = accu(w(idx));
        group_inv_w(j) = (sumw > 0) ? (1.0 / sumw) : 0.0;
      }
    }
  } else {
    for (size_t j = 0; j < J; ++j) {
      const uvec &idx = indices.get_group(fe_idx, j);
      group_inv_w(j) = idx.is_empty() ? 0.0 : (1.0 / idx.n_elem);
    }
  }
  
  // Project all columns simultaneously for cache efficiency
  for (size_t j = 0; j < J; ++j) {
    const uvec &idx = indices.get_group(fe_idx, j);
    if (idx.is_empty() || group_inv_w(j) == 0.0) continue;
    
    if (weighted) {
      const vec w_subset = w(idx);
      mat Z_subset = Z.rows(idx);
      // Compute means across all columns at once
      rowvec means = sum(Z_subset.each_col() % w_subset, 0) * group_inv_w(j);
      // Subtract means from all columns
      Z.rows(idx) -= ones<vec>(idx.n_elem) * means;
    } else {
      mat Z_subset = Z.rows(idx);
      rowvec means = mean(Z_subset, 0);
      Z.rows(idx) -= ones<vec>(idx.n_elem) * means;
    }
  }
}

// Specialized 2-FE absorption using analytical solution
inline void absorb_ml_2fe(mat &Z, const indices_info &indices, const vec &w,
                          ml_center_workspace &ws) {
  const size_t N = Z.n_rows;
  const size_t P = Z.n_cols;
  const bool weighted = (w.n_elem > 1);
  
  // Build group ID vectors efficiently
  uvec fe1(N), fe2(N);
  
  // Use pre-computed indices for cache efficiency
  for (size_t j = 0; j < indices.fe_sizes(0); ++j) {
    const uvec &idx = indices.get_group(0, j);
    fe1.elem(idx).fill(j);
  }
  
  for (size_t j = 0; j < indices.fe_sizes(1); ++j) {
    const uvec &idx = indices.get_group(1, j);
    fe2.elem(idx).fill(j);
  }
  
  // Process each column with optimized memory access
  for (size_t p = 0; p < P; ++p) {
    vec col = Z.col(p);  // Make a copy to avoid reference issues
    
    const size_t G1 = indices.fe_sizes(0);
    const size_t G2 = indices.fe_sizes(1);
    
    vec mean1(G1, fill::zeros);
    vec mean2(G2, fill::zeros);
    vec wsum1(G1, fill::zeros);
    vec wsum2(G2, fill::zeros);
    
    double grand_sum = 0.0, grand_wsum = 0.0;
    
    // First pass: compute group sums
    if (weighted) {
      for (size_t i = 0; i < N; ++i) {
        const double wi = w(i);
        const double zi = col(i);
        const size_t g1 = fe1(i), g2 = fe2(i);
        
        mean1(g1) += wi * zi;
        mean2(g2) += wi * zi;
        wsum1(g1) += wi;
        wsum2(g2) += wi;
        grand_sum += wi * zi;
        grand_wsum += wi;
      }
    } else {
      for (size_t i = 0; i < N; ++i) {
        const double zi = col(i);
        const size_t g1 = fe1(i), g2 = fe2(i);
        
        mean1(g1) += zi;
        mean2(g2) += zi;
        grand_sum += zi;
      }
      wsum1.fill(1.0);
      wsum2.fill(1.0);
      grand_wsum = N;
    }
    
    // Normalize means
    for (size_t g = 0; g < G1; ++g) {
      if (wsum1(g) > 0) mean1(g) /= wsum1(g);
    }
    for (size_t g = 0; g < G2; ++g) {
      if (wsum2(g) > 0) mean2(g) /= wsum2(g);
    }
    const double grand_mean = grand_sum / grand_wsum;
    
    // Second pass: apply absorption
    for (size_t i = 0; i < N; ++i) {
      col(i) = col(i) - mean1(fe1(i)) - mean2(fe2(i)) + grand_mean;
    }
    
    // Write back to Z
    Z.col(p) = col;
  }
}

// Main ML centering routine with family-specific optimizations
inline void center_variables_ml(mat &X, vec &y, const vec &w,
                                const indices_info &indices,
                                family_type family, double tol, size_t max_iter,
                                ml_center_workspace &ws) {
  const size_t N = X.n_rows;
  const size_t P = X.n_cols;
  const size_t K = indices.fe_sizes.n_elem;
  
  if (K == 0) return; // No fixed effects
  
  // Reserve workspace
  ws.reserve(N, P, K);
  
  // Combine X and y into unified matrix for cache efficiency
  ws.Z.cols(0, P - 1) = X;
  ws.Z.col(P) = y;
  
  // Family-specific centering
  if (K == 1) {
    // Single FE - use optimized single projection
    project_ml_1fe(ws.Z, 0, indices, w, ws);
  } else if (K == 2) {
    // Two-way FE - use analytical absorption
    absorb_ml_2fe(ws.Z, indices, w, ws);
  } else {
    // K-way FE - use iterative ML approach
    const bool weighted = (w.n_elem > 1);
    double prev_ssr = datum::inf;
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
      // Forward pass
      for (size_t k = 0; k < K; ++k) {
        project_ml_1fe(ws.Z, k, indices, w, ws);
      }
      
      // Backward pass for symmetric Kaczmarz
      for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
        project_ml_1fe(ws.Z, static_cast<size_t>(k), indices, w, ws);
      }
      
      // Check convergence using SSR
      double ssr;
      if (weighted) {
        ssr = accu(square(ws.Z) % w.each_row());
      } else {
        ssr = accu(square(ws.Z));
      }
      
      const double rel_change = std::fabs(ssr - prev_ssr) / (0.1 + std::fabs(ssr));
      if (rel_change < tol) break;
      
      prev_ssr = ssr;
      
      if ((iter % 100) == 0 && iter > 0) {
        check_user_interrupt();
      }
    }
  }
  
  // Extract results
  X = ws.Z.cols(0, P - 1);
  y = ws.Z.col(P);
}

// Overloaded version for matrix-only centering
inline void center_variables_ml(mat &X, const vec &w,
                                const indices_info &indices,
                                double tol, size_t max_iter,
                                ml_center_workspace &ws) {
  const size_t N = X.n_rows;
  const size_t P = X.n_cols;
  const size_t K = indices.fe_sizes.n_elem;
  
  if (K == 0) return;
  
  ws.reserve(N, P, K);
  ws.Z.cols(0, P - 1) = X;
  
  if (K == 1) {
    mat temp_Z = ws.Z.cols(0, P - 1);
    project_ml_1fe(temp_Z, 0, indices, w, ws);
    ws.Z.cols(0, P - 1) = temp_Z;
  } else if (K == 2) {
    mat temp_Z = ws.Z.cols(0, P - 1);
    absorb_ml_2fe(temp_Z, indices, w, ws);
    ws.Z.cols(0, P - 1) = temp_Z;
  } else {
    const bool weighted = (w.n_elem > 1);
    double prev_ssr = datum::inf;
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
      for (size_t k = 0; k < K; ++k) {
        mat temp_Z = ws.Z.cols(0, P - 1);
        project_ml_1fe(temp_Z, k, indices, w, ws);
        ws.Z.cols(0, P - 1) = temp_Z;
      }
      
      for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
        mat temp_Z = ws.Z.cols(0, P - 1);
        project_ml_1fe(temp_Z, static_cast<size_t>(k), indices, w, ws);
        ws.Z.cols(0, P - 1) = temp_Z;
      }
      
      double ssr;
      if (weighted) {
        mat temp_Z = ws.Z.cols(0, P - 1);
        ssr = accu(square(temp_Z) % w.each_row());
      } else {
        mat temp_Z = ws.Z.cols(0, P - 1);
        ssr = accu(square(temp_Z));
      }
      
      const double rel_change = std::fabs(ssr - prev_ssr) / (0.1 + std::fabs(ssr));
      if (rel_change < tol) break;
      
      prev_ssr = ssr;
      
      if ((iter % 100) == 0 && iter > 0) {
        check_user_interrupt();
      }
    }
  }
  
  X = ws.Z.cols(0, P - 1);
}

#endif // CAPYBARA_CENTER_ML_H
