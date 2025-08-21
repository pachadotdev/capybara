// Computing alpha a in a model with fixed effects Y = alpha + X beta given beta

#ifndef CAPYBARA_ALPHA_H
#define CAPYBARA_ALPHA_H

namespace capybara {

struct InferenceAlpha {
  field<vec> coefficients;
  uvec nb_references;
  bool is_regular;
  bool success;
  field<std::string> fe_names;
  field<field<std::string>> fe_levels;

  InferenceAlpha() : is_regular(true), success(false) {}
};

struct AlphaGroupInfo {
  std::vector<size_t> group_start;
  std::vector<size_t> group_size;
  std::vector<size_t> obs_to_group;
  size_t n_groups;
};

inline field<vec> get_alpha(const vec &pi,
                            const field<field<uvec>> &group_indices,
                            double tol = 1e-8, size_t iter_max = 10000) {
  const size_t K = group_indices.n_elem;
  const size_t N = pi.n_elem;
  field<vec> coefficients(K);

  if (K == 0 || N == 0) {
    return coefficients;
  }

  std::vector<AlphaGroupInfo> group_info(K);

  for (size_t k = 0; k < K; ++k) {
    const size_t J = group_indices(k).n_elem;
    coefficients(k).set_size(J);
    coefficients(k).zeros();

    group_info[k].n_groups = J;
    group_info[k].group_size.resize(J);
    group_info[k].obs_to_group.resize(N, J);

    for (size_t j = 0; j < J; ++j) {
      const uvec &indexes = group_indices(k)(j);
      group_info[k].group_size[j] = indexes.n_elem;

      for (size_t t = 0; t < indexes.n_elem; ++t) {
        group_info[k].obs_to_group[indexes(t)] = j;
      }
    }
  }

  vec y(N, fill::none);
  vec alpha0, group_sums;

  double crit = 1.0;
  size_t iter = 0;

  const double *pi_ptr = pi.memptr();

  while (crit > tol && iter < iter_max) {
    double sum_sq0 = 0.0, sum_sq_diff = 0.0;

    for (size_t k = 0; k < K; ++k) {
      const AlphaGroupInfo &info = group_info[k];
      const size_t J = info.n_groups;

      alpha0.set_size(J);
      group_sums.set_size(J);

      double *alpha0_ptr = alpha0.memptr();
      const double *coef_k_ptr = coefficients(k).memptr();

      for (size_t j = 0; j < J; ++j) {
        alpha0_ptr[j] = coef_k_ptr[j];
      }
      sum_sq0 += dot(alpha0, alpha0);

      double *y_ptr = y.memptr();

      std::memcpy(y_ptr, pi_ptr, N * sizeof(double));

      const size_t obs_block_size = get_block_size(N, K);

      for (size_t kk = 0; kk < K; ++kk) {
        if (kk != k) {
          const AlphaGroupInfo &info_kk = group_info[kk];
          const double *coef_kk_ptr = coefficients(kk).memptr();
          const std::vector<size_t> &obs_to_group_kk = info_kk.obs_to_group;

          for (size_t block_start = 0; block_start < N;
               block_start += obs_block_size) {
            const size_t block_end = std::min(block_start + obs_block_size, N);

            for (size_t i = block_start; i < block_end; ++i) {
              size_t group = obs_to_group_kk[i];
              if (group < info_kk.n_groups) {
                y_ptr[i] -= coef_kk_ptr[group];
              }
            }
          }
        }
      }

      group_sums.zeros();
      double *group_sums_ptr = group_sums.memptr();
      const std::vector<size_t> &obs_to_group = info.obs_to_group;

      for (size_t block_start = 0; block_start < N;
           block_start += obs_block_size) {
        const size_t block_end = std::min(block_start + obs_block_size, N);

        for (size_t i = block_start; i < block_end; ++i) {
          size_t group = obs_to_group[i];
          if (group < J) {
            group_sums_ptr[group] += y_ptr[i];
          }
        }
      }

      double *coef_k_ptr_new = coefficients(k).memptr();
      const std::vector<size_t> &group_size = info.group_size;

      for (size_t j = 0; j < J; ++j) {
        if (group_size[j] > 0) {
          coef_k_ptr_new[j] =
              group_sums_ptr[j] / static_cast<double>(group_size[j]);
        } else {
          coef_k_ptr_new[j] = 0.0;
        }
      }

      double local_sum_sq_diff = 0.0;
      for (size_t j = 0; j < J; ++j) {
        double diff = coef_k_ptr_new[j] - alpha0_ptr[j];
        local_sum_sq_diff += diff * diff;
      }
      sum_sq_diff += local_sum_sq_diff;
    }

    if (sum_sq0 > 0.0) {
      crit = std::sqrt(sum_sq_diff / sum_sq0);
    } else if (sum_sq_diff > 0.0) {
      crit = std::sqrt(sum_sq_diff);
    } else {
      crit = 0.0;
    }

    ++iter;
  }

  return coefficients;
}

} // namespace capybara

#endif // CAPYBARA_ALPHA_H
