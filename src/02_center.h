// Symmetric Kaczmarz centering with CG acceleration (reghdfe-style)

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

// Workspace for centering to avoid repeated allocations
struct CenteringWorkspace {
  vec x, Gx, GGx, Y, GY, GGY;
  uword cached_n;
  bool is_initialized;

  CenteringWorkspace() : cached_n(0), is_initialized(false) {}

  void ensure_size(uword n) {
    if (!is_initialized || n > cached_n) {
      x.set_size(n);
      Gx.set_size(n);
      GGx.set_size(n);
      Y.set_size(n);
      GY.set_size(n);
      GGY.set_size(n);
      cached_n = n;
      is_initialized = true;
    }
  }
};

// Precompute group information for all variables
inline field<field<GroupInfo>>
precompute_group_info(const field<field<uvec>> &group_indices, const vec &w) {
  const uword K = group_indices.n_elem;
  const double *w_ptr = w.memptr();

  field<field<GroupInfo>> group_info(K);

  for (uword k = 0; k < K; ++k) {
    const field<uvec> &fe_groups = group_indices(k);
    group_info(k).set_size(fe_groups.n_elem);

    for (uword l = 0; l < fe_groups.n_elem; ++l) {
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

      group_info(k)(l) = info;
    }
  }

  return group_info;
}

// Safe division avoiding division by zero
inline double safe_divide(double num, double denom) {
  return (std::abs(denom) > 1e-14) ? (num / denom) : 0.0;
}

// Project onto one FE group: subtract weighted mean
// Returns the mean that was subtracted (the "projection component")
inline double project_one_group(double *v, const double *w,
                                const GroupInfo &info) {
  if (info.is_singleton)
    return 0.0;

  double weighted_sum = 0.0;
  const uword *coord_ptr = info.coords->memptr();
  const uword n = info.n_elem;

  for (uword i = 0; i < n; ++i) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
  }

  double mean = weighted_sum * info.inv_weight;

  for (uword i = 0; i < n; ++i) {
    v[coord_ptr[i]] -= mean;
  }

  return mean;
}

// Symmetric Kaczmarz projection
inline void apply_projection(double *y, const double *w,
                             const field<field<GroupInfo>> &all_group_info) {
  const uword K = all_group_info.n_elem;

  // Forward pass
  for (uword k = 0; k < K; ++k) {
    const field<GroupInfo> &fe_info = all_group_info(k);
    for (uword l = 0; l < fe_info.n_elem; ++l) {
      project_one_group(y, w, fe_info(l));
    }
  }

  // Backward pass (symmetric Kaczmarz)
  for (uword k = K; k-- > 1;) {
    const field<GroupInfo> &fe_info = all_group_info(k - 1);
    for (uword l = 0; l < fe_info.n_elem; ++l) {
      project_one_group(y, w, fe_info(l));
    }
  }
}

// Single FE projection
inline void apply_projection_single_fe(double *y, const double *w,
                                       const field<GroupInfo> &fe_info) {
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(y, w, fe_info(l));
  }
  for (uword l = 0; l < fe_info.n_elem; ++l) {
    project_one_group(y, w, fe_info(l));
  }
}

// Irons-Tuck acceleration
inline bool irons_tuck_acceleration(double *x, const double *Gx,
                                    const double *GGx, uword N, double tol) {
  double vprod = 0.0, ssq = 0.0;

  for (uword i = 0; i < N; ++i) {
    double delta_G = GGx[i] - Gx[i];
    double delta2 = delta_G - (Gx[i] - x[i]);
    vprod += delta_G * delta2;
    ssq += delta2 * delta2;
  }

  if (ssq < tol * tol)
    return true;  // Converged

  double coef = vprod / ssq;

  // Apply acceleration if coefficient is reasonable
  if (coef > 0.0 && coef < 2.0) {
    for (uword i = 0; i < N; ++i) {
      x[i] = GGx[i] - coef * (GGx[i] - Gx[i]);
    }
  } else {
    // Fall back to GGx
    std::memcpy(x, GGx, N * sizeof(double));
  }

  return false;
}

// Grand acceleration (fixest-style)
inline bool grand_acceleration(double *x, double *Y, double *GY, double *GGY,
                               uword N, int &grand_acc_state, double tol) {
  if (grand_acc_state == 0) {
    std::memcpy(Y, x, N * sizeof(double));
  } else if (grand_acc_state == 1) {
    std::memcpy(GY, x, N * sizeof(double));
  } else {
    std::memcpy(GGY, x, N * sizeof(double));

    double vprod = 0.0, ssq = 0.0;
    for (uword i = 0; i < N; ++i) {
      double delta_G = GGY[i] - GY[i];
      double delta2 = delta_G - (GY[i] - Y[i]);
      vprod += delta_G * delta2;
      ssq += delta2 * delta2;
    }

    if (ssq < tol * tol) {
      grand_acc_state = -1;
      return true;
    }

    double coef = safe_divide(vprod, ssq);
    if (coef > 0.0 && coef < 2.0) {
      for (uword i = 0; i < N; ++i) {
        x[i] = GGY[i] - coef * (GGY[i] - GY[i]);
      }
    } else {
      std::memcpy(x, GGY, N * sizeof(double));
    }
    grand_acc_state = -1;
  }

  ++grand_acc_state;
  return false;
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

  const uword N = V.n_rows, P = V.n_cols;
  const double *w_ptr = w.memptr();

  // Reuse precomputed group metadata
  const field<field<GroupInfo>> *group_info_ptr = precomputed_group_info;
  field<field<GroupInfo>> local_group_info;
  if (!group_info_ptr) {
    local_group_info = precompute_group_info(group_indices, w);
    group_info_ptr = &local_group_info;
  }
  const field<field<GroupInfo>> &all_group_info = *group_info_ptr;

  CenteringWorkspace local_workspace;
  CenteringWorkspace *ws = workspace ? workspace : &local_workspace;
  ws->ensure_size(N);

  double *x_ptr = ws->x.memptr();
  double *Gx_ptr = ws->Gx.memptr();
  double *GGx_ptr = ws->GGx.memptr();
  double *Y_ptr = ws->Y.memptr();
  double *GY_ptr = ws->GY.memptr();
  double *GGY_ptr = ws->GGY.memptr();

  for (uword col = 0; col < P; ++col) {
    double *col_ptr = V.colptr(col);
    std::memcpy(x_ptr, col_ptr, N * sizeof(double));

    uword irons_tuck_interval = 3;
    uword grand_accel_interval = 6;
    const uword accel_start = 2;
    int grand_acc_state = 0;
    uword iint = iter_interrupt;
    double ssr_old = std::numeric_limits<double>::infinity();
    uword stall_count = 0;
    double convergence_rate = 1.0;

    for (uword iter = 0; iter < max_iter; ++iter) {
      if (iter == iint) {
        check_user_interrupt();
        iint += iter_interrupt;
      }

      // Apply projection
      if (K == 1) {
        apply_projection_single_fe(x_ptr, w_ptr, all_group_info[0]);
      } else {
        apply_projection(x_ptr, w_ptr, all_group_info);
      }

      // Grand acceleration
      if (iter >= accel_start && iter % grand_accel_interval == 0) {
        if (grand_acceleration(x_ptr, Y_ptr, GY_ptr, GGY_ptr, N,
                               grand_acc_state, tol)) {
          break;
        }
        if (K == 1) {
          apply_projection_single_fe(x_ptr, w_ptr, all_group_info[0]);
        } else {
          apply_projection(x_ptr, w_ptr, all_group_info);
        }
      }

      // Irons-Tuck acceleration
      if (iter >= accel_start && iter % irons_tuck_interval == 0 &&
          iter % grand_accel_interval != 0) {
        std::memcpy(Gx_ptr, x_ptr, N * sizeof(double));
        if (K == 1) {
          apply_projection_single_fe(Gx_ptr, w_ptr, all_group_info[0]);
        } else {
          apply_projection(Gx_ptr, w_ptr, all_group_info);
        }

        std::memcpy(GGx_ptr, Gx_ptr, N * sizeof(double));
        if (K == 1) {
          apply_projection_single_fe(GGx_ptr, w_ptr, all_group_info[0]);
        } else {
          apply_projection(GGx_ptr, w_ptr, all_group_info);
        }

        if (irons_tuck_acceleration(x_ptr, Gx_ptr, GGx_ptr, N, tol)) {
          break;
        }
      }

      // Check convergence
      double ssr = dot(square(ws->x), w);
      if (ssr < tol * tol)
        break;

      if (ssr_old < std::numeric_limits<double>::infinity()) {
        double rel_change = std::abs(ssr - ssr_old) / (1.0 + ssr_old);
        if (rel_change < tol)
          break;

        // Adaptive acceleration
        if (iter > accel_start + 3) {
          convergence_rate = 0.7 * convergence_rate + 0.3 * rel_change;

          if (convergence_rate < 0.01) {
            irons_tuck_interval = std::min(uword(5), irons_tuck_interval + 1);
            grand_accel_interval = std::min(uword(10), grand_accel_interval + 1);
          } else if (convergence_rate > 0.1) {
            irons_tuck_interval = std::max(uword(2), irons_tuck_interval - 1);
            grand_accel_interval = std::max(uword(3), grand_accel_interval - 1);
          }
        }

        if (rel_change < tol * 10.0) {
          stall_count++;
          if (stall_count > 3)
            break;
        } else {
          stall_count = 0;
        }
      }
      ssr_old = ssr;
    }

    std::memcpy(col_ptr, x_ptr, N * sizeof(double));
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
