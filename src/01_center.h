// Symmetric Kaczmarz with Irons-Tuck acceleration

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {

// Store group information
struct GroupInfo {
  const uvec *coords;
  double inv_weight;
  uword n_elem;
  bool is_singleton;
};

// Cache-friendly block size calculation
inline size_t get_block_size(size_t n, size_t p) {
  const size_t L1_CACHE_SIZE = 32 * 1024;
  const size_t ELEMENT_SIZE = sizeof(double);
  size_t max_block_elements = L1_CACHE_SIZE / ELEMENT_SIZE;
  size_t max_block_size = max_block_elements / (p + 1);
  const size_t MIN_BLOCK_SIZE = 64;
  const size_t MAX_BLOCK_SIZE = 1024;
  size_t block_size =
      std::max(MIN_BLOCK_SIZE, std::min(max_block_size, MAX_BLOCK_SIZE));
  return std::min(block_size, n);
}

// Precompute group information for all variables
inline std::vector<std::vector<GroupInfo>>
precompute_group_info(const field<field<uvec>> &group_indices, const vec &w) {
  const size_t K = group_indices.n_elem;
  const double *w_ptr = w.memptr();

  std::vector<std::vector<GroupInfo>> group_info(K);

  for (size_t k = 0; k < K; ++k) {
    const field<uvec> &fe_groups = group_indices(k);
    group_info[k].reserve(fe_groups.n_elem);

    for (size_t l = 0; l < fe_groups.n_elem; ++l) {
      const uvec &coords = fe_groups(l);
      GroupInfo info;
      info.coords = &coords;
      info.n_elem = coords.n_elem;
      info.is_singleton = (coords.n_elem <= 1);

      if (!info.is_singleton) {
        double sum_w = 0.0;
        const uword *coord_ptr = coords.memptr();
        for (uword i = 0; i < coords.n_elem; ++i) {
          sum_w += w_ptr[coord_ptr[i]];
        }
        info.inv_weight = (sum_w > 0.0) ? 1.0 / sum_w : 0.0;
      } else {
        info.inv_weight = 0.0;
      }

      group_info[k].push_back(info);
    }
  }

  return group_info;
}

// Convergence checks
inline bool stopping_criterion(double ssr_old, double ssr_new, double tol) {
  if (ssr_old == 0.0)
    return false;
  return std::abs(ssr_new - ssr_old) / (1.0 + std::abs(ssr_old)) < tol;
}

inline bool continue_criterion(double x_old, double x_new, double tol) {
  return std::abs(x_new - x_old) / (1.0 + std::abs(x_old)) > tol;
}

// Grand acceleration (based on fixest)
inline bool grand_acceleration(double *x, double *Y, double *GY, double *GGY,
                               std::vector<double> &delta_GX,
                               std::vector<double> &delta2_X, size_t n,
                               int &grand_acc, double tol) {
  bool converged = false;

  if (grand_acc == 0) {
    std::memcpy(Y, x, n * sizeof(double));
  } else if (grand_acc == 1) {
    std::memcpy(GY, x, n * sizeof(double));
  } else {
    std::memcpy(GGY, x, n * sizeof(double));

    // Apply Irons-Tuck with the saved vectors
    double vprod = 0.0, ssq = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double delta_G = GGY[i] - GY[i];
      double delta2 = delta_G - GY[i] + Y[i];
      delta_GX[i] = delta_G;
      delta2_X[i] = delta2;
      vprod += delta_G * delta2;
      ssq += delta2 * delta2;
    }

    if (ssq < tol) {
      converged = true;
    } else {
      double coef = vprod / ssq;
      if (coef > 0.0 && coef < 2.0) {
        for (size_t i = 0; i < n; ++i) {
          x[i] = GGY[i] - coef * delta_GX[i];
        }
      } else {
        std::memcpy(x, GGY, n * sizeof(double));
      }
    }

    grand_acc = -1; // Reset for next cycle
  }

  ++grand_acc;
  return converged;
}

// Adaptive SSR checking
inline bool adaptive_ssr_check(double *x, const double *w, size_t n,
                               double &ssr_old, double inv_sw, double tol) {
  double ssr = 0.0;
  for (size_t i = 0; i < n; ++i) {
    ssr += w[i] * x[i] * x[i];
  }
  ssr *= inv_sw;

  bool converged = stopping_criterion(ssr_old, ssr, tol);
  ssr_old = ssr;
  return converged;
}

inline void project_group(double *v, const double *w, const GroupInfo &info,
                          bool &any_change, double tol) {
  if (info.is_singleton)
    return;

  double weighted_sum = 0.0;
  const uword *coord_ptr = info.coords->memptr();

  // Unroll loop for small groups for better cache performance
  if (info.n_elem <= 4) {
    for (uword i = 0; i < info.n_elem; ++i) {
      weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
    }
  } else {
    for (uword i = 0; i < info.n_elem; ++i) {
      weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
    }
  }

  double mean = weighted_sum * info.inv_weight;

  // Check if change is significant for early convergence detection
  if (std::abs(mean) > tol) {
    any_change = true;
    for (uword i = 0; i < info.n_elem; ++i) {
      v[coord_ptr[i]] -= mean;
    }
  }
}

inline bool irons_tuck_acceleration(double *x, const double *Gx,
                                    const double *GGx, double *x0, size_t n,
                                    double tol) {
  double vprod = 0.0, ssq = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double delta_G = GGx[i] - Gx[i];
    double delta2 = delta_G - Gx[i] + x[i];
    vprod += delta_G * delta2;
    ssq += delta2 * delta2;
  }

  if (ssq < tol) {
    return true; // Converged
  }

  double coef = vprod / ssq;

  // Apply acceleration if coefficient is reasonable
  if (coef > 0.0 && coef < 2.0) {
    for (size_t i = 0; i < n; ++i) {
      x[i] = GGx[i] - coef * (GGx[i] - Gx[i]);
    }
  } else {
    // Fall back to GGx
    std::memcpy(x, GGx, n * sizeof(double));
  }

  return false;
}

inline bool project_2fe(double *x, const double *w,
                        const std::vector<GroupInfo> &fe1_info,
                        const std::vector<GroupInfo> &fe2_info, double tol) {
  bool any_change = false;

  // Forward pass: FE1 then FE2
  for (const auto &info : fe1_info) {
    project_group(x, w, info, any_change, tol);
  }
  for (const auto &info : fe2_info) {
    project_group(x, w, info, any_change, tol);
  }

  // Backward pass: FE2 then FE1 (symmetric Kaczmarz)
  for (auto it = fe2_info.rbegin(); it != fe2_info.rend(); ++it) {
    project_group(x, w, *it, any_change, tol);
  }
  for (auto it = fe1_info.rbegin(); it != fe1_info.rend(); ++it) {
    project_group(x, w, *it, any_change, tol);
  }

  return !any_change; // Converged if no changes
}

inline bool
project_kfe(double *x, const double *w,
            const std::vector<std::vector<GroupInfo>> &all_group_info,
            double tol) {
  const size_t K = all_group_info.size();
  bool any_change = false;

  // Forward pass
  for (size_t k = 0; k < K; ++k) {
    for (const auto &info : all_group_info[k]) {
      project_group(x, w, info, any_change, tol);
    }
  }

  // Backward pass (symmetric Kaczmarz)
  for (size_t k = K; k-- > 0;) {
    for (auto it = all_group_info[k].rbegin(); it != all_group_info[k].rend();
         ++it) {
      project_group(x, w, *it, any_change, tol);
    }
  }

  return !any_change; // Converged if no changes
}

void center_variables(
    mat &V, const vec &w, const field<field<uvec>> &group_indices,
    const double &tol, const size_t &max_iter, const size_t &iter_interrupt,
    const size_t &iter_ssr, const size_t &accel_start,
    const double &project_tol_factor, const double &grand_accel_tol,
    const double &project_group_tol, const double &irons_tuck_tol,
    const size_t &grand_accel_interval, const size_t &irons_tuck_interval,
    const size_t &ssr_check_interval, const double &convergence_factor,
    const double &tol_multiplier) {
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem)
    return;

  const size_t K = group_indices.n_elem;
  if (K == 0)
    return;

  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);
  const double *w_ptr = w.memptr();

  // Precompute group information
  auto all_group_info = precompute_group_info(group_indices, w);

  const size_t col_block_size = get_block_size(N, P);

  for (size_t col_block = 0; col_block < P; col_block += col_block_size) {
    const size_t col_end = std::min(col_block + col_block_size, P);

    for (size_t col = col_block; col < col_end; ++col) {
      double *col_ptr = V.colptr(col);

      // Working vectors for acceleration
      vec x(N, fill::none), x0(N, fill::none), Gx(N, fill::none),
          GGx(N, fill::none);
      vec Y(N, fill::none), GY(N, fill::none), GGY(N, fill::none);
      double *x_ptr = x.memptr();
      double *x0_ptr = x0.memptr();
      double *Gx_ptr = Gx.memptr();
      double *GGx_ptr = GGx.memptr();
      double *Y_ptr = Y.memptr();
      double *GY_ptr = GY.memptr();
      double *GGY_ptr = GGY.memptr();

      std::vector<double> delta_GX(N), delta2_X(N);

      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      size_t iint = iter_interrupt;
      size_t isr = iter_ssr;
      int grand_acc = 0;

      std::memcpy(x_ptr, col_ptr, N * sizeof(double));

      // Projection with convergence checking
      auto project_with_check = [&]() -> bool {
        if (K == 1) {
          bool any_change = false;
          for (const auto &info : all_group_info[0]) {
            project_group(x_ptr, w_ptr, info, any_change,
                          tol * project_tol_factor);
          }
          return !any_change;
        } else if (K == 2) {
          return project_2fe(x_ptr, w_ptr, all_group_info[0], all_group_info[1],
                             tol * project_tol_factor);
        } else {
          return project_kfe(x_ptr, w_ptr, all_group_info,
                             tol * project_tol_factor);
        }
      };

      for (size_t iter = 0; iter < max_iter; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iter_interrupt;
        }

        std::memcpy(x0_ptr, x_ptr, N * sizeof(double));

        if (project_with_check()) {
          break; // Converged at projection level
        }

        if (iter >= accel_start && iter % grand_accel_interval == 0) {
          if (grand_acceleration(x_ptr, Y_ptr, GY_ptr, GGY_ptr, delta_GX,
                                 delta2_X, N, grand_acc, grand_accel_tol)) {
            break; // Converged via grand acceleration
          }
          // Apply projection after acceleration
          project_with_check();
        }

        if (iter >= accel_start && iter % irons_tuck_interval == 0 &&
            iter % grand_accel_interval != 0) {
          std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));
          project_with_check();
          std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));

          std::memcpy(GGx_ptr, Gx_ptr, N * sizeof(double));
          project_with_check();
          std::memcpy(GGx_ptr, x_ptr, N * sizeof(double));

          if (irons_tuck_acceleration(x_ptr, Gx_ptr, GGx_ptr, x0_ptr, N,
                                      irons_tuck_tol)) {
            break; // Converged via Irons-Tuck
          }
        }

        // Check convergence based on relative change
        double weighted_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
          double rel_diff =
              std::abs(x_ptr[i] - x0_ptr[i]) / (1.0 + std::abs(x0_ptr[i]));
          weighted_diff += w_ptr[i] * rel_diff;
        }
        double ratio = weighted_diff * inv_sw;

        if (ratio < tol)
          break;

        // Adaptive SSR-based convergence check (like fixest every
        // ssr_check_interval iterations)
        if (iter > 0 && iter % ssr_check_interval == 0) {
          if (adaptive_ssr_check(x_ptr, w_ptr, N, ssr0, inv_sw, tol)) {
            break; // Converged via SSR check
          }
        }

        // Standard SSR check at specified intervals
        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += iter_ssr;
          double ssr = 0.0;
          for (size_t i = 0; i < N; ++i) {
            ssr += w_ptr[i] * x_ptr[i] * x_ptr[i];
          }
          ssr *= inv_sw;
          if (stopping_criterion(ssr0, ssr, tol))
            break;
          ssr0 = ssr;
        }

        // Early termination if slow convergence
        if (iter > 3 && (ratio0 / ratio) < convergence_factor &&
            ratio < tol * tol_multiplier)
          break;
        ratio0 = ratio;
      }

      std::memcpy(col_ptr, x_ptr, N * sizeof(double));
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
