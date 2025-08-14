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
  // Dimensions
  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);

  // Extract the two FE groups
  const field<uvec> &fe1_groups = group_indices(0);
  const field<uvec> &fe2_groups = group_indices(1);
  const size_t L1 = fe1_groups.n_elem;
  const size_t L2 = fe2_groups.n_elem;

  // Precompute group weights
  vec group1_inv_w(L1, fill::none);
  vec group2_inv_w(L2, fill::none);

  // FE1 weights
  for (size_t l = 0; l < L1; ++l) {
    if (fe1_groups(l).n_elem == 0) {
      group1_inv_w(l) = 0.0;
    } else {
      double sum_w = accu(w.elem(fe1_groups(l)));
      group1_inv_w(l) = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
    }
  }

  // FE2 weights
  for (size_t l = 0; l < L2; ++l) {
    if (fe2_groups(l).n_elem == 0) {
      group2_inv_w(l) = 0.0;
    } else {
      double sum_w = accu(w.elem(fe2_groups(l)));
      group2_inv_w(l) = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
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

  // Define Symmetric Kaczmarz projection
  auto project_symmetric_kaczmarz_2fe = [&](vec &v) {
    // Forward pass: FE1 then FE2
    for (size_t l = 0; l < L1; ++l) {
      const uvec &coords = fe1_groups(l);
      if (coords.n_elem <= 1) continue;
      
      double xbar = dot(w.elem(coords), v.elem(coords)) * group1_inv_w(l);
      v.elem(coords) -= xbar;
    }
    
    for (size_t l = 0; l < L2; ++l) {
      const uvec &coords = fe2_groups(l);
      if (coords.n_elem <= 1) continue;
      
      double xbar = dot(w.elem(coords), v.elem(coords)) * group2_inv_w(l);
      v.elem(coords) -= xbar;
    }
    
    // Backward pass: FE2 then FE1
    for (size_t l = L2; l-- > 0; ) {
      const uvec &coords = fe2_groups(l);
      if (coords.n_elem <= 1) continue;
      
      double xbar = dot(w.elem(coords), v.elem(coords)) * group2_inv_w(l);
      v.elem(coords) -= xbar;
    }
    
    for (size_t l = L1; l-- > 0; ) {
      const uvec &coords = fe1_groups(l);
      if (coords.n_elem <= 1) continue;
      
      double xbar = dot(w.elem(coords), v.elem(coords)) * group1_inv_w(l);
      v.elem(coords) -= xbar;
    }
  };

  // Process each column
  for (size_t col = 0; col < P; ++col) {
    x = V.col(col);
    double ratio0 = std::numeric_limits<double>::infinity();
    double ssr0 = std::numeric_limits<double>::infinity();
    size_t iint = iter_interrupt;
    size_t isr = iter_ssr;

    for (size_t iter = 0; iter < max_iter; ++iter) {
      if (iter == iint) {
        check_user_interrupt();
        iint += iter_interrupt;
      }

      x0 = x;
      project_symmetric_kaczmarz_2fe(x);

      // Compute gradient (residual before projection - residual after)
      g = x0 - x;

      // 1) convergence via weighted diff
      diff = abs(x - x0) / (1.0 + abs(x0));
      double ratio = dot(diff, w) * inv_sw;
      if (ratio < tol)
        break;

      // 2) Acceleration
      if (use_cg && iter >= accel_start) {
        // Conjugate gradient acceleration
        conjugate_gradient_accel(x, g, g_old, p, w, inv_sw,
                                 project_symmetric_kaczmarz_2fe, 
                                 iter, accel_start);
        g_old = g;
      } else if (!use_cg && iter >= 5 && (iter % 5) == 0) {
        // Fallback: Irons-Tuck acceleration
        Gx = x;
        project_symmetric_kaczmarz_2fe(Gx);
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
        if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < tol)
          break;
        ssr0 = ssr;
      }

      // 4) heuristic early exit
      if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20)
        break;
      ratio0 = ratio;
    }

    V.unsafe_col(col) = x;
  }
}

void center_variables(mat &V, const vec &w,
                      const field<field<uvec>> &group_indices,
                      const double &tol, const size_t &max_iter,
                      const size_t &iter_interrupt, const size_t &iter_ssr,
                      const size_t &accel_start,
                      const bool use_cg) {
  // Safety check for dimensions
  if (V.n_rows != w.n_elem) {
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

  // Auxiliary variables (storage)
  size_t iter, iint, isr, k, l, col, L;
  double coef, xbar, ratio, ssr, ssq, ratio0, ssr0;
  vec x(N, fill::none), x0(N, fill::none);
  vec diff(N, fill::none);
  
  // CG-specific vectors
  vec g(N, fill::none), g_old(N, fill::none), p(N, fill::none);
  
  // Fallback acceleration vectors
  vec Gx(N, fill::none), G2x(N, fill::none);
  vec deltaG(N, fill::none), delta2(N, fill::none);

  // Precompute group weights
  field<vec> group_inv_w(K);
  for (k = 0; k < K; ++k) {
    const field<uvec> &idxs = group_indices(k);
    const size_t L = idxs.n_elem;

    vec invs(L);
    for (l = 0; l < L; ++l) {
      if (idxs(l).n_elem == 0) {
        invs(l) = 0.0;
        continue;
      }

      // Check if all indices are valid
      bool all_valid = true;
      for (uword i = 0; i < idxs(l).n_elem; ++i) {
        if (idxs(l)(i) >= w.n_elem) {
          all_valid = false;
          break;
        }
      }

      if (!all_valid) {
        invs(l) = 0.0;
        continue;
      }

      // Safely compute the inverse weight sum
      double sum_w = accu(w.elem(idxs(l)));
      invs(l) = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
    }
    group_inv_w(k) = std::move(invs);
  }

  // Symmetric Kaczmarz projection
  auto project_symmetric_kaczmarz = [&](vec &v) {
    // Forward pass
    for (k = 0; k < K; ++k) {
      const auto &idxs = group_indices(k);
      const auto &invs = group_inv_w(k);
      L = idxs.n_elem;

      for (l = 0; l < L; ++l) {
        const uvec &coords = idxs(l);
        const uword coord_size = coords.n_elem;

        if (coord_size <= 1) continue;

        xbar = dot(w.elem(coords), v.elem(coords)) * invs(l);
        v.elem(coords) -= xbar;
      }
    }
    
    // Backward pass
    for (k = K; k-- > 0; ) {
      const auto &idxs = group_indices(k);
      const auto &invs = group_inv_w(k);
      L = idxs.n_elem;

      for (l = L; l-- > 0; ) {
        const uvec &coords = idxs(l);
        const uword coord_size = coords.n_elem;

        if (coord_size <= 1) continue;

        xbar = dot(w.elem(coords), v.elem(coords)) * invs(l);
        v.elem(coords) -= xbar;
      }
    }
  };

  // Column-wise centering with acceleration and SSR checks
  for (col = 0; col < P; ++col) {
    x = V.col(col);
    ratio0 = std::numeric_limits<double>::infinity();
    ssr0 = std::numeric_limits<double>::infinity();
    iint = iint0;
    isr = isr0;

    for (iter = 0; iter < I; ++iter) {
      if (iter == iint) {
        check_user_interrupt();
        iint += iint0;
      }

      x0 = x;
      project_symmetric_kaczmarz(x);

      // Compute gradient
      g = x0 - x;

      // 1) convergence via weighted diff
      diff = abs(x - x0) / (1.0 + abs(x0));
      ratio = dot(diff, w) * inv_sw;
      if (ratio < tol)
        break;

      // 2) Acceleration
      if (use_cg && iter >= accel_start) {
        // Conjugate gradient acceleration
        conjugate_gradient_accel(x, g, g_old, p, w, inv_sw,
                                 project_symmetric_kaczmarz, 
                                 iter, accel_start);
        g_old = g;
      } else if (!use_cg && iter >= 5 && (iter % 5) == 0) {
        // Fallback: Irons-Tuck acceleration
        Gx = x;
        project_symmetric_kaczmarz(Gx);
        G2x = Gx;
        deltaG = G2x - x;
        delta2 = G2x - 2.0 * x + x0;
        ssq = dot(delta2, delta2);
        if (ssq > 1e-10) {
          coef = dot(deltaG, delta2) / ssq;
          x = (coef > 0.0 && coef < 2.0) ? (G2x - coef * deltaG) : G2x;
        }
      }

      // 3) SSR-based early exit
      if (iter == isr && iter > 0) {
        check_user_interrupt();
        isr += isr0;
        ssr = dot(x % x, w) * inv_sw;
        if (std::fabs(ssr - ssr0) / (1.0 + std::fabs(ssr0)) < tol)
          break;
        ssr0 = ssr;
      }

      // 4) heuristic early exit  
      if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20)
        break;
      ratio0 = ratio;
    }

    V.unsafe_col(col) = x;
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
