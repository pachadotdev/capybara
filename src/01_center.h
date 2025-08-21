// Symmetric Kaczmarz with Conjugate Gradient acceleration

#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

#include <cstring>
#include <vector>

namespace capybara {

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

inline void project_group(double *v, const double *w, const uvec &coords,
                          double inv_group_weight) {
  double weighted_sum = 0.0;
  const uword n = coords.n_elem;
  const uword *coord_ptr = coords.memptr();

  for (uword i = 0; i < n; ++i) {
    weighted_sum += w[coord_ptr[i]] * v[coord_ptr[i]];
  }

  double mean = weighted_sum * inv_group_weight;

  for (uword i = 0; i < n; ++i) {
    v[coord_ptr[i]] -= mean;
  }
}

struct GroupInfo {
  const uvec *coords;
  double inv_weight;
  uword n_elem;
};

template <typename ProjectFunc>
void cg_acceleration(double *x, double *g, double *g0, double *p,
                     const double *w, double inv_sw, ProjectFunc project,
                     size_t iter, size_t accel_start, size_t n) {
  if (iter == accel_start) {
    std::memcpy(p, g, n * sizeof(double));
  } else if (iter > accel_start) {
    double num = 0.0, denom = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double diff_g = g[i] - g0[i];
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

  vec Ap(n, fill::none);
  double *Ap_ptr = Ap.memptr();
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

template <typename ProjectFunc>
void it_acceleration(double *x, double *x0, double *Gx, double *G2x,
                     ProjectFunc project, size_t n) {
  std::memcpy(Gx, x, n * sizeof(double));
  project(Gx);
  std::memcpy(G2x, Gx, n * sizeof(double));
  project(G2x);

  double vprod = 0.0, ssq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double deltaG = G2x[i] - x[i];
    double delta2 = G2x[i] - 2.0 * x[i] + x0[i];
    vprod += deltaG * delta2;
    ssq += delta2 * delta2;
  }

  if (ssq > 1e-10) {
    double coef = vprod / ssq;
    if (coef > 0.0 && coef < 2.0) {
      for (size_t i = 0; i < n; ++i) {
        x[i] = G2x[i] - coef * (G2x[i] - x[i]);
      }
    }
  }
}

void center_variables(mat &V, const vec &w,
                      const field<field<uvec>> &group_indices,
                      const double &tol, const size_t &max_iter,
                      const size_t &iter_interrupt, const size_t &iter_ssr,
                      const size_t &accel_start, const bool use_cg) {
  if (V.is_empty() || w.is_empty() || V.n_rows != w.n_elem)
    return;

  const size_t K = group_indices.n_elem;
  if (K == 0)
    return;

  const size_t N = V.n_rows, P = V.n_cols;
  const double inv_sw = 1.0 / accu(w);
  const double *w_ptr = w.memptr();

  const size_t col_block_size = get_block_size(N, P);

  for (size_t col_block = 0; col_block < P; col_block += col_block_size) {
    const size_t col_end = std::min(col_block + col_block_size, P);

    for (size_t col = col_block; col < col_end; ++col) {
      double *col_ptr = V.colptr(col);

      vec x(N, fill::none), x0(N, fill::none);
      double *x_ptr = x.memptr();
      double *x0_ptr = x0.memptr();

      double ratio0 = std::numeric_limits<double>::infinity();
      double ssr0 = std::numeric_limits<double>::infinity();
      size_t iint = iter_interrupt;
      size_t isr = iter_ssr;

      std::memcpy(x_ptr, col_ptr, N * sizeof(double));

      for (size_t iter = 0; iter < max_iter; ++iter) {
        if (iter == iint) {
          check_user_interrupt();
          iint += iter_interrupt;
        }

        std::memcpy(x0_ptr, x_ptr, N * sizeof(double));

        if (K == 1) {
          const field<uvec> &fe_groups = group_indices(0);
          for (size_t l = 0; l < fe_groups.n_elem; ++l) {
            const uvec &coords = fe_groups(l);
            if (coords.n_elem > 1) {
              double sum_w = 0.0, weighted_sum = 0.0;
              const uword *coord_ptr = coords.memptr();
              for (uword i = 0; i < coords.n_elem; ++i) {
                sum_w += w_ptr[coord_ptr[i]];
                weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
              }
              if (sum_w > 0.0) {
                double mean = weighted_sum / sum_w;
                for (uword i = 0; i < coords.n_elem; ++i) {
                  x_ptr[coord_ptr[i]] -= mean;
                }
              }
            }
          }
        } else if (K == 2) {
          const field<uvec> &fe1_groups = group_indices(0);
          const field<uvec> &fe2_groups = group_indices(1);

          for (size_t l = 0; l < fe1_groups.n_elem; ++l) {
            const uvec &coords = fe1_groups(l);
            if (coords.n_elem > 1) {
              double sum_w = 0.0, weighted_sum = 0.0;
              const uword *coord_ptr = coords.memptr();
              for (uword i = 0; i < coords.n_elem; ++i) {
                sum_w += w_ptr[coord_ptr[i]];
                weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
              }
              if (sum_w > 0.0) {
                double mean = weighted_sum / sum_w;
                for (uword i = 0; i < coords.n_elem; ++i) {
                  x_ptr[coord_ptr[i]] -= mean;
                }
              }
            }
          }
          for (size_t l = 0; l < fe2_groups.n_elem; ++l) {
            const uvec &coords = fe2_groups(l);
            if (coords.n_elem > 1) {
              double sum_w = 0.0, weighted_sum = 0.0;
              const uword *coord_ptr = coords.memptr();
              for (uword i = 0; i < coords.n_elem; ++i) {
                sum_w += w_ptr[coord_ptr[i]];
                weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
              }
              if (sum_w > 0.0) {
                double mean = weighted_sum / sum_w;
                for (uword i = 0; i < coords.n_elem; ++i) {
                  x_ptr[coord_ptr[i]] -= mean;
                }
              }
            }
          }

          for (size_t l = fe2_groups.n_elem; l-- > 0;) {
            const uvec &coords = fe2_groups(l);
            if (coords.n_elem > 1) {
              double sum_w = 0.0, weighted_sum = 0.0;
              const uword *coord_ptr = coords.memptr();
              for (uword i = 0; i < coords.n_elem; ++i) {
                sum_w += w_ptr[coord_ptr[i]];
                weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
              }
              if (sum_w > 0.0) {
                double mean = weighted_sum / sum_w;
                for (uword i = 0; i < coords.n_elem; ++i) {
                  x_ptr[coord_ptr[i]] -= mean;
                }
              }
            }
          }
          for (size_t l = fe1_groups.n_elem; l-- > 0;) {
            const uvec &coords = fe1_groups(l);
            if (coords.n_elem > 1) {
              double sum_w = 0.0, weighted_sum = 0.0;
              const uword *coord_ptr = coords.memptr();
              for (uword i = 0; i < coords.n_elem; ++i) {
                sum_w += w_ptr[coord_ptr[i]];
                weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
              }
              if (sum_w > 0.0) {
                double mean = weighted_sum / sum_w;
                for (uword i = 0; i < coords.n_elem; ++i) {
                  x_ptr[coord_ptr[i]] -= mean;
                }
              }
            }
          }
        } else {
          for (size_t k = 0; k < K; ++k) {
            const field<uvec> &fe_groups = group_indices(k);
            for (size_t l = 0; l < fe_groups.n_elem; ++l) {
              const uvec &coords = fe_groups(l);
              if (coords.n_elem > 1) {
                double sum_w = 0.0, weighted_sum = 0.0;
                const uword *coord_ptr = coords.memptr();
                for (uword i = 0; i < coords.n_elem; ++i) {
                  if (coord_ptr[i] < w.n_elem) {
                    sum_w += w_ptr[coord_ptr[i]];
                    weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
                  }
                }
                if (sum_w > 0.0) {
                  double mean = weighted_sum / sum_w;
                  for (uword i = 0; i < coords.n_elem; ++i) {
                    if (coord_ptr[i] < w.n_elem) {
                      x_ptr[coord_ptr[i]] -= mean;
                    }
                  }
                }
              }
            }
          }

          for (size_t k = K; k-- > 0;) {
            const field<uvec> &fe_groups = group_indices(k);
            for (size_t l = fe_groups.n_elem; l-- > 0;) {
              const uvec &coords = fe_groups(l);
              if (coords.n_elem > 1) {
                double sum_w = 0.0, weighted_sum = 0.0;
                const uword *coord_ptr = coords.memptr();
                for (uword i = 0; i < coords.n_elem; ++i) {
                  if (coord_ptr[i] < w.n_elem) {
                    sum_w += w_ptr[coord_ptr[i]];
                    weighted_sum += w_ptr[coord_ptr[i]] * x_ptr[coord_ptr[i]];
                  }
                }
                if (sum_w > 0.0) {
                  double mean = weighted_sum / sum_w;
                  for (uword i = 0; i < coords.n_elem; ++i) {
                    if (coord_ptr[i] < w.n_elem) {
                      x_ptr[coord_ptr[i]] -= mean;
                    }
                  }
                }
              }
            }
          }
        }

        double weighted_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
          double rel_diff =
              std::abs(x_ptr[i] - x0_ptr[i]) / (1.0 + std::abs(x0_ptr[i]));
          weighted_diff += w_ptr[i] * rel_diff;
        }
        double ratio = weighted_diff * inv_sw;

        if (ratio < tol)
          break;

        if (iter == isr && iter > 0) {
          check_user_interrupt();
          isr += iter_ssr;
          double ssr = 0.0;
          for (size_t i = 0; i < N; ++i) {
            ssr += w_ptr[i] * x_ptr[i] * x_ptr[i];
          }
          ssr *= inv_sw;
          if (std::abs(ssr - ssr0) / (1.0 + std::abs(ssr0)) < tol)
            break;
          ssr0 = ssr;
        }

        if (iter > 3 && (ratio0 / ratio) < 1.1 && ratio < tol * 20)
          break;
        ratio0 = ratio;
      }

      std::memcpy(col_ptr, x_ptr, N * sizeof(double));
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
