// Method of alternating projections (Halperin)

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

void center_variables_2fe(mat &V, const vec &w,
                          const field<field<uvec>> &group_indices,
                          const double &tol, const size_t &max_iter,
                          const size_t &iter_interrupt,
                          const size_t &iter_ssr) {
  // Dimensions
  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);

  // Extract the two FE groups
  const field<uvec> &fe1_groups = group_indices(0);
  const field<uvec> &fe2_groups = group_indices(1);
  const size_t L1 = fe1_groups.n_elem;
  const size_t L2 = fe2_groups.n_elem;

  // Precompute group weights exactly like the general version
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
  vec Gx(N, fill::none), G2x(N, fill::none);
  vec deltaG(N, fill::none), delta2(N, fill::none);

  // Define projection function
  auto project_2fe = [&](vec &v) {
    // Project FE1
    for (size_t l = 0; l < L1; ++l) {
      const uvec &coords = fe1_groups(l);
      const uword coord_size = coords.n_elem;

      if (coord_size <= 1)
        continue; // Skip singleton/empty groups

      double xbar = dot(w.elem(coords), v.elem(coords)) * group1_inv_w(l);
      v.elem(coords) -= xbar;
    }

    // Project FE2
    for (size_t l = 0; l < L2; ++l) {
      const uvec &coords = fe2_groups(l);
      const uword coord_size = coords.n_elem;

      if (coord_size <= 1)
        continue; // Skip singleton/empty groups

      double xbar = dot(w.elem(coords), v.elem(coords)) * group2_inv_w(l);
      v.elem(coords) -= xbar;
    }
  };

  // Process each column (simplified from general version)
  for (size_t p = 0; p < P; ++p) {
    x = V.col(p);
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
      project_2fe(x);

      // 1) convergence via weighted diff
      diff = abs(x - x0) / (1.0 + abs(x0));
      double ratio = dot(diff, w) * inv_sw;
      if (ratio < tol)
        break;

      // 2) Irons-Tuck acceleration every 5 iterations
      if (iter >= 5 && (iter % 5) == 0) {
        Gx = x;
        project_2fe(Gx);
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

    V.unsafe_col(p) = x;
  }
}

void center_variables(mat &V, const vec &w,
                      const field<field<uvec>> &group_indices,
                      const double &tol, const size_t &max_iter,
                      const size_t &iter_interrupt, const size_t &iter_ssr) {
  // Safety check for dimensions
  if (V.n_rows != w.n_elem) {
    return;
  }

  if (group_indices.n_elem == 2) {
    center_variables_2fe(V, w, group_indices, tol, max_iter, iter_interrupt,
                         iter_ssr);
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
  size_t iter, iint, isr, k, l, p, L;
  double coef, xbar, ratio, ssr, ssq, ratio0, ssr0;
  vec x(N, fill::none), x0(N, fill::none), Gx(N, fill::none),
      G2x(N, fill::none), deltaG(N, fill::none), delta2(N, fill::none),
      diff(N, fill::none);

  // Precompute group weights
  field<vec> group_inv_w(K);
  for (k = 0; k < K; ++k) {
    if (k >= group_indices.n_elem) {

      continue;
    }

    const field<uvec> &idxs = group_indices(k);
    const size_t L = idxs.n_elem;

    vec invs(L);
    for (l = 0; l < L; ++l) {
      if (l >= idxs.n_elem) {

        continue;
      }

      // Skip empty groups
      if (idxs(l).n_elem == 0) {

        invs(l) = 0.0; // Set to 0 for empty groups
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
        invs(l) = 0.0; // Set to 0 for groups with invalid indices
        continue;
      }

      // Safely compute the inverse weight sum
      try {
        double sum_w = accu(w.elem(idxs(l)));
        invs(l) = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
      } catch (const std::exception &e) {

        invs(l) = 0.0;
      }
    }
    group_inv_w(k) = std::move(invs);
  }

  // Single projection step (in-place)
  auto project = [&](vec &v) {
    for (k = 0; k < K; ++k) {
      // Check if we have a valid group
      if (k >= group_indices.n_elem) {

        continue;
      }

      const auto &idxs = group_indices(k);

      // Check if we have valid group weights
      if (k >= group_inv_w.n_elem) {

        continue;
      }

      const auto &invs = group_inv_w(k);
      L = idxs.n_elem;

      if (L == 0)
        continue;

      // Make sure we don't have more groups than weights
      if (L > invs.n_elem) {

        continue;
      }

      for (l = 0; l < L; ++l) {
        // Check if the group exists
        if (l >= idxs.n_elem) {

          continue;
        }

        const uvec &coords = idxs(l);
        const uword coord_size = coords.n_elem;

        if (coord_size <= 1)
          continue;

        // Check if all indices are within bounds
        bool valid_indices = true;
        for (uword i = 0; i < coord_size; ++i) {
          if (coords(i) >= N) {
            valid_indices = false;
            break;
          }
        }

        if (!valid_indices)
          continue;

        // Now safely compute the group mean and demean
        xbar = dot(w.elem(coords), v.elem(coords)) * invs(l);
        v.elem(coords) -= xbar;
      }
    }
  };

  // Column-wise centering with acceleration and SSR checks
  for (p = 0; p < P; ++p) {
    x = V.col(p);
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
      project(x);

      // 1) convergence via weighted diff
      diff = abs(x - x0) / (1.0 + abs(x0));
      ratio = dot(diff, w) * inv_sw;
      if (ratio < tol)
        break;

      // 2) Irons-Tuck acceleration every 5 iterations
      if (iter >= 5 && (iter % 5) == 0) {
        Gx = x;
        project(Gx);
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

    V.unsafe_col(p) = x;
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
