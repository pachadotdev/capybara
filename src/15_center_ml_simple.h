#ifndef CAPYBARA_CENTER_ML_SIMPLE_H
#define CAPYBARA_CENTER_ML_SIMPLE_H

// Simplified ML centering that avoids reference binding issues
// This is a working version we can optimize later

// Simple workspace for ML centering
struct ml_center_simple_workspace {
  mat temp_matrix;
  vec temp_vec;
  field<vec> group_weights;
  
  void reserve(size_t N, size_t P, size_t K) {
    temp_matrix.set_size(N, P);
    temp_vec.set_size(N);
    group_weights.set_size(K);
  }
  
  void clear() {
    temp_matrix.reset();
    temp_vec.reset();
    group_weights.reset();
  }
};

// Simple projection for single FE
inline void project_simple_1fe(mat &X, vec &y, size_t fe_idx, 
                               const indices_info &indices, const vec &w,
                               ml_center_simple_workspace &ws) {
  const size_t J = indices.fe_sizes(fe_idx);
  const bool weighted = (w.n_elem > 1);
  
  // Ensure workspace is allocated
  if (ws.group_weights.n_elem <= fe_idx || ws.group_weights(fe_idx).n_elem != J) {
    if (ws.group_weights.n_elem <= fe_idx) {
      ws.group_weights.set_size(fe_idx + 1);
    }
    ws.group_weights(fe_idx).set_size(J);
  }
  
  vec &group_inv_w = ws.group_weights(fe_idx);
  
  // Pre-compute inverse weights
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
  
  // Project X columns
  for (size_t p = 0; p < X.n_cols; ++p) {
    for (size_t j = 0; j < J; ++j) {
      const uvec &idx = indices.get_group(fe_idx, j);
      if (idx.is_empty() || group_inv_w(j) == 0.0) continue;
      
      double mean_val;
      if (weighted) {
        mean_val = accu(X.col(p).elem(idx) % w.elem(idx)) * group_inv_w(j);
      } else {
        mean_val = mean(X.col(p).elem(idx));
      }
      
      X.col(p).elem(idx) -= mean_val;
    }
  }
  
  // Project y vector
  for (size_t j = 0; j < J; ++j) {
    const uvec &idx = indices.get_group(fe_idx, j);
    if (idx.is_empty() || group_inv_w(j) == 0.0) continue;
    
    double mean_val;
    if (weighted) {
      mean_val = accu(y.elem(idx) % w.elem(idx)) * group_inv_w(j);
    } else {
      mean_val = mean(y.elem(idx));
    }
    
    y.elem(idx) -= mean_val;
  }
}

// Simple centering for matrix only
inline void project_simple_1fe_matrix(mat &X, size_t fe_idx, 
                                      const indices_info &indices, const vec &w,
                                      ml_center_simple_workspace &ws) {
  const size_t J = indices.fe_sizes(fe_idx);
  const bool weighted = (w.n_elem > 1);
  
  // Ensure workspace is allocated
  if (ws.group_weights.n_elem <= fe_idx || ws.group_weights(fe_idx).n_elem != J) {
    if (ws.group_weights.n_elem <= fe_idx) {
      ws.group_weights.set_size(fe_idx + 1);
    }
    ws.group_weights(fe_idx).set_size(J);
  }
  
  vec &group_inv_w = ws.group_weights(fe_idx);
  
  // Pre-compute inverse weights
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
  
  // Project X columns
  for (size_t p = 0; p < X.n_cols; ++p) {
    for (size_t j = 0; j < J; ++j) {
      const uvec &idx = indices.get_group(fe_idx, j);
      if (idx.is_empty() || group_inv_w(j) == 0.0) continue;
      
      double mean_val;
      if (weighted) {
        mean_val = accu(X.col(p).elem(idx) % w.elem(idx)) * group_inv_w(j);
      } else {
        mean_val = mean(X.col(p).elem(idx));
      }
      
      X.col(p).elem(idx) -= mean_val;
    }
  }
}

// Main simple centering routine
inline void center_variables_simple(mat &X, vec &y, const vec &w,
                                    const indices_info &indices,
                                    double tol, size_t max_iter,
                                    ml_center_simple_workspace &ws) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;
  
  const size_t N = X.n_rows;
  const size_t P = X.n_cols;
  ws.reserve(N, P, K);
  
  if (K == 1) {
    // Single FE - direct projection
    project_simple_1fe(X, y, 0, indices, w, ws);
  } else {
    // Multiple FE - iterative approach
    const bool weighted = (w.n_elem > 1);
    double prev_ssr = datum::inf;
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
      // Forward pass
      for (size_t k = 0; k < K; ++k) {
        project_simple_1fe(X, y, k, indices, w, ws);
      }
      
      // Backward pass for symmetric Kaczmarz
      for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
        project_simple_1fe(X, y, static_cast<size_t>(k), indices, w, ws);
      }
      
      // Check convergence
      double ssr = 0.0;
      if (weighted) {
        for (size_t p = 0; p < P; ++p) {
          ssr += accu(square(X.col(p)) % w);
        }
        ssr += accu(square(y) % w);
      } else {
        ssr = accu(square(X)) + accu(square(y));
      }
      
      const double rel_change = std::fabs(ssr - prev_ssr) / (0.1 + std::fabs(ssr));
      if (rel_change < tol) break;
      
      prev_ssr = ssr;
      
      if ((iter % 100) == 0 && iter > 0) {
        check_user_interrupt();
      }
    }
  }
}

// Matrix-only version
inline void center_variables_simple(mat &X, const vec &w,
                                    const indices_info &indices,
                                    double tol, size_t max_iter,
                                    ml_center_simple_workspace &ws) {
  const size_t K = indices.fe_sizes.n_elem;
  if (K == 0) return;
  
  const size_t N = X.n_rows;
  const size_t P = X.n_cols;
  ws.reserve(N, P, K);
  
  if (K == 1) {
    // Single FE - direct projection
    project_simple_1fe_matrix(X, 0, indices, w, ws);
  } else {
    // Multiple FE - iterative approach
    const bool weighted = (w.n_elem > 1);
    double prev_ssr = datum::inf;
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
      // Forward pass
      for (size_t k = 0; k < K; ++k) {
        project_simple_1fe_matrix(X, k, indices, w, ws);
      }
      
      // Backward pass for symmetric Kaczmarz
      for (int k = static_cast<int>(K) - 1; k >= 0; --k) {
        project_simple_1fe_matrix(X, static_cast<size_t>(k), indices, w, ws);
      }
      
      // Check convergence
      double ssr;
      if (weighted) {
        ssr = 0.0;
        for (size_t p = 0; p < P; ++p) {
          ssr += accu(square(X.col(p)) % w);
        }
      } else {
        ssr = accu(square(X));
      }
      
      const double rel_change = std::fabs(ssr - prev_ssr) / (0.1 + std::fabs(ssr));
      if (rel_change < tol) break;
      
      prev_ssr = ssr;
      
      if ((iter % 100) == 0 && iter > 0) {
        check_user_interrupt();
      }
    }
  }
}

#endif // CAPYBARA_CENTER_ML_SIMPLE_H
