#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// Group information structure
struct GroupInfo {
  field<uvec> indices;
  vec sum_weights;
  vec inv_weights;
  size_t n_groups;

  GroupInfo() = default;
  GroupInfo(const list &jlist, const vec &w) {
    n_groups = jlist.size();
    indices.set_size(n_groups);
    sum_weights.set_size(n_groups);
    inv_weights.set_size(n_groups);

    for (size_t j = 0; j < n_groups; ++j) {
      indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
      sum_weights(j) = accu(w.elem(indices(j)));
      inv_weights(j) = 1.0 / sum_weights(j);
    }
  }
};

inline void center_gaussian_2way(mat &V, const vec &w, const list &klist,
                                 const double &tol, const int &max_iter) {
  const size_t P = V.n_cols;

  // Use GroupInfo structure like the general algorithm
  GroupInfo gi0(klist[0], w);
  GroupInfo gi1(klist[1], w);

  for (size_t p = 0; p < P; ++p) {
    const vec x_orig = V.col(p);
    vec x = x_orig;
    vec alpha_i(gi0.n_groups, fill::zeros);
    vec alpha_j(gi1.n_groups, fill::zeros);

    // Alternating projections with closed-form updates
    for (int iter = 0; iter < max_iter; ++iter) {
      vec x_prev = x;

      // Update alpha_i given alpha_j
      alpha_i.zeros();
      for (size_t l = 0; l < gi0.n_groups; ++l) {
        const uvec &coords = gi0.indices(l);
        if (coords.n_elem == 0) continue;

        double sum_weighted = 0.0;
        for (size_t idx = 0; idx < coords.n_elem; ++idx) {
          size_t obs = coords(idx);
          // Subtract current alpha_j contributions
          double residual = x_orig(obs);
          for (size_t k = 0; k < gi1.n_groups; ++k) {
            const uvec &coords_j = gi1.indices(k);
            for (size_t idx_j = 0; idx_j < coords_j.n_elem; ++idx_j) {
              if (coords_j(idx_j) == obs) {
                residual -= alpha_j(k);
                break;
              }
            }
          }
          sum_weighted += w(obs) * residual;
        }
        alpha_i(l) = sum_weighted * gi0.inv_weights(l);
      }

      // Update alpha_j given alpha_i
      alpha_j.zeros();
      for (size_t l = 0; l < gi1.n_groups; ++l) {
        const uvec &coords = gi1.indices(l);
        if (coords.n_elem == 0) continue;

        double sum_weighted = 0.0;
        for (size_t idx = 0; idx < coords.n_elem; ++idx) {
          size_t obs = coords(idx);
          // Subtract current alpha_i contributions
          double residual = x_orig(obs);
          for (size_t k = 0; k < gi0.n_groups; ++k) {
            const uvec &coords_i = gi0.indices(k);
            for (size_t idx_i = 0; idx_i < coords_i.n_elem; ++idx_i) {
              if (coords_i(idx_i) == obs) {
                residual -= alpha_i(k);
                break;
              }
            }
          }
          sum_weighted += w(obs) * residual;
        }
        alpha_j(l) = sum_weighted * gi1.inv_weights(l);
      }

      // Compute final centered values
      x = x_orig;
      for (size_t l = 0; l < gi0.n_groups; ++l) {
        const uvec &coords = gi0.indices(l);
        if (coords.n_elem == 0) continue;
        x.elem(coords) -= alpha_i(l);
      }
      for (size_t l = 0; l < gi1.n_groups; ++l) {
        const uvec &coords = gi1.indices(l);
        if (coords.n_elem == 0) continue;
        x.elem(coords) -= alpha_j(l);
      }

      // Check convergence
      double diff = dot(abs(x - x_prev), w) / accu(w);
      if (diff < tol) break;
    }

    V.col(p) = x;
  }
}

inline void center_poisson_2way(mat &V, const vec &w, const list &klist,
                                const double &tol, const int &max_iter) {
  const size_t P = V.n_cols;

  // Use GroupInfo structure like the general algorithm
  GroupInfo gi0(klist[0], w);
  GroupInfo gi1(klist[1], w);

  for (size_t p = 0; p < P; ++p) {
    const vec x_orig = V.col(p);
    vec x = x_orig;
    vec alpha0(gi0.n_groups, fill::zeros);
    vec alpha1(gi1.n_groups, fill::zeros);

    for (int iter = 0; iter < max_iter; ++iter) {
      vec x0 = x;

      // Update FE 1: x = x_orig - alpha1, solve for alpha0
      x = x_orig;
      for (size_t l = 0; l < gi1.n_groups; ++l) {
        const uvec &coords = gi1.indices(l);
        if (coords.n_elem == 0) continue;
        x.elem(coords) -= alpha1(l);
      }

      // Solve for alpha0 given x
      for (size_t l = 0; l < gi0.n_groups; ++l) {
        const uvec &coords = gi0.indices(l);
        if (coords.n_elem == 0) continue;

        double maxval = x.elem(coords).max();
        double sum_exp = accu(w.elem(coords) % exp(x.elem(coords) - maxval));
        alpha0(l) = log(gi0.sum_weights(l)) - log(sum_exp) + maxval;
      }

      // Update FE 2: x = x_orig - alpha0, solve for alpha1
      x = x_orig;
      for (size_t l = 0; l < gi0.n_groups; ++l) {
        const uvec &coords = gi0.indices(l);
        if (coords.n_elem == 0) continue;
        x.elem(coords) -= alpha0(l);
      }

      // Solve for alpha1 given x
      for (size_t l = 0; l < gi1.n_groups; ++l) {
        const uvec &coords = gi1.indices(l);
        if (coords.n_elem == 0) continue;

        double maxval = x.elem(coords).max();
        double sum_exp = accu(w.elem(coords) % exp(x.elem(coords) - maxval));
        alpha1(l) = log(gi1.sum_weights(l)) - log(sum_exp) + maxval;
      }

      // Final residual computation
      x = x_orig;
      for (size_t l = 0; l < gi0.n_groups; ++l) {
        const uvec &coords = gi0.indices(l);
        if (coords.n_elem == 0) continue;
        x.elem(coords) -= alpha0(l);
      }
      for (size_t l = 0; l < gi1.n_groups; ++l) {
        const uvec &coords = gi1.indices(l);
        if (coords.n_elem == 0) continue;
        x.elem(coords) -= alpha1(l);
      }

      // Check convergence using SSR-based criterion
      double ssr = dot(w, square(x));
      double ssr_prev = dot(w, square(x0));
      if (std::abs(ssr - ssr_prev) / (0.1 + std::abs(ssr)) < tol) break;
    }

    V.col(p) = x;
  }
}

double update_negbin(const arma::vec &y, const arma::vec &mu,
                     const arma::vec &w, double theta, double prev_alpha,
                     double tol = 1e-8, int max_iter = 100) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double mui = mu(i) * exp(x1);
      f += w(i) * (y(i) - (mui + theta) * (y(i) / mui));
      df += w(i) * (-mui * (y(i) / (mui * mui)));
    }
    if (std::abs(df) < 1e-12)
      break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper)
      x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol)
      break;
    if (f > 0)
      lower = x1;
    else
      upper = x1;
  }
  return x1;
}

double update_logit(const arma::vec &y, const arma::vec &mu, const arma::vec &w,
                    double prev_alpha, double tol = 1e-8, int max_iter = 100) {
  double x1 = prev_alpha;
  double lower = x1 - 10, upper = x1 + 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    double f = 0, df = 0;
    for (size_t i = 0; i < y.n_elem; ++i) {
      double eta = x1 + mu(i);
      double p = 1.0 / (1.0 + std::exp(-eta));
      f += w(i) * (y(i) - p);
      df += w(i) * (-p * (1 - p));
    }
    if (std::abs(df) < 1e-12)
      break;
    double x0 = x1;
    x1 = x0 - f / df;
    if (x1 < lower || x1 > upper)
      x1 = 0.5 * (lower + upper);
    if (std::abs(x1 - x0) < tol)
      break;
    if (f > 0)
      lower = x1;
    else
      upper = x1;
  }
  return x1;
}

// Group mean subtraction
void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt, const int &iter_ssr,
                       const std::string &family = "gaussian") {
  const size_t N = V.n_rows, P = V.n_cols, K = klist.size();
  
  // Use specialized algorithms for 2-way models (disabled for debugging)
  if (false && K == 2) {
    if (family == "gaussian") {
      center_gaussian_2way(V, w, klist, tol, max_iter);
      return;
    } else if (family == "poisson") {
      center_poisson_2way(V, w, klist, tol, max_iter);
      return;
    }
  }
  
  // Fall back to general algorithm for other cases
  const double inv_sw = 1.0 / accu(w);

  // Precompute group information
  field<GroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info(k) = GroupInfo(klist[k], w);
  }

  auto convergence_check = [&](const vec &x, const vec &x0, const vec &w,
                               double tol) {
    if (family == "poisson") {
      double ssr = dot(w, square(x));
      double ssr0 = dot(w, square(x0));
      return std::abs(ssr - ssr0) / (0.1 + std::abs(ssr)) < tol;
    } else if (family == "gaussian") {
      return dot(abs(x - x0), w) * inv_sw < tol;
    } else {
      return dot(abs(x - x0), w) * inv_sw < tol;
    }
  };

  for (size_t p = 0; p < P; ++p) {
    const vec x_orig = V.col(p);
    vec x = x_orig;
    field<vec> alpha(K);
    for (size_t k = 0; k < K; ++k) {
      alpha(k).zeros(group_info(k).n_groups);
    }
    vec alpha_sum = zeros<vec>(N);
    vec x0(N, fill::none), x1(N, fill::none), x2(N, fill::none);
    if (K == 2) {
      const GroupInfo &gi0 = group_info(0);
      const GroupInfo &gi1 = group_info(1);

      for (int iter = 0; iter < max_iter; ++iter) {
        x0 = x;

        // Update FE 1: x = x_orig - alpha1, solve for alpha0
        x = x_orig;
        for (size_t l = 0; l < gi1.n_groups; ++l) {
          const uvec &coords = gi1.indices(l);
          if (coords.n_elem == 0)
            continue;
          x.elem(coords) -= alpha(1)(l);
        }

        // Solve for alpha0 given x
        for (size_t l = 0; l < gi0.n_groups; ++l) {
          const uvec &coords = gi0.indices(l);
          if (coords.n_elem == 0)
            continue;
          double new_alpha;
          if (family == "poisson") {
            double maxval = x.elem(coords).max();
            double sum_exp =
                accu(w.elem(coords) % exp(x.elem(coords) - maxval));
            new_alpha = log(gi0.sum_weights(l)) - log(sum_exp) + maxval;
          } else if (family == "negbin") {
            new_alpha =
                update_negbin(x.elem(coords), zeros<vec>(coords.n_elem),
                              w.elem(coords), /*theta*/ 1.0, alpha(0)(l));
          } else if (family == "logit" || family == "binomial") {
            new_alpha = update_logit(x.elem(coords), zeros<vec>(coords.n_elem),
                                     w.elem(coords), alpha(0)(l));
          } else {
            // Gaussian: direct closed form
            new_alpha =
                dot(w.elem(coords), x.elem(coords)) * gi0.inv_weights(l);
          }
          alpha(0)(l) = new_alpha;
        }

        // Update FE 2: x = x_orig - alpha0, solve for alpha1
        x = x_orig;
        for (size_t l = 0; l < gi0.n_groups; ++l) {
          const uvec &coords = gi0.indices(l);
          if (coords.n_elem == 0)
            continue;
          x.elem(coords) -= alpha(0)(l);
        }

        // Solve for alpha1 given x
        for (size_t l = 0; l < gi1.n_groups; ++l) {
          const uvec &coords = gi1.indices(l);
          if (coords.n_elem == 0)
            continue;
          double new_alpha;
          if (family == "poisson") {
            double maxval = x.elem(coords).max();
            double sum_exp =
                accu(w.elem(coords) % exp(x.elem(coords) - maxval));
            new_alpha = log(gi1.sum_weights(l)) - log(sum_exp) + maxval;
          } else if (family == "negbin") {
            new_alpha =
                update_negbin(x.elem(coords), zeros<vec>(coords.n_elem),
                              w.elem(coords), /*theta*/ 1.0, alpha(1)(l));
          } else if (family == "logit" || family == "binomial") {
            new_alpha = update_logit(x.elem(coords), zeros<vec>(coords.n_elem),
                                     w.elem(coords), alpha(1)(l));
          } else {
            // Gaussian: direct closed form
            new_alpha =
                dot(w.elem(coords), x.elem(coords)) * gi1.inv_weights(l);
          }
          alpha(1)(l) = new_alpha;
        }

        // Final residual computation
        x = x_orig;
        for (size_t l = 0; l < gi0.n_groups; ++l) {
          const uvec &coords = gi0.indices(l);
          if (coords.n_elem == 0)
            continue;
          x.elem(coords) -= alpha(0)(l);
        }
        for (size_t l = 0; l < gi1.n_groups; ++l) {
          const uvec &coords = gi1.indices(l);
          if (coords.n_elem == 0)
            continue;
          x.elem(coords) -= alpha(1)(l);
        }

        // Convergence check
        if (convergence_check(x, x0, w, tol))
          break;
      }
    } else {
      // K > 2 => use Irons-Tuck acceleration
      const int warmup = 15;
      const int grand_acc = 40;
      int acc_count = 0;
      int iter = 0;

      // Pre-compute sum of other FE contributions
      vec sum_other_coef(N, fill::zeros);

      // Warmup iterations
      for (; iter < std::min(max_iter, warmup); ++iter) {
        x0 = x;
        for (size_t k = 0; k < K; ++k) {
          const GroupInfo &gi = group_info(k);
          // Remove current FE from alpha_sum
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const uvec &coords = gi.indices(l);
            if (coords.n_elem == 0)
              continue;
            alpha_sum.elem(coords) -= alpha(k)(l);
          }
          x = x_orig - alpha_sum; // Residual
          // Update alpha and add back (using precomputed weights)
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const uvec &coords = gi.indices(l);
            if (coords.n_elem == 0)
              continue;
            double new_alpha;
            if (family == "poisson") {
              double maxval = x.elem(coords).max();
              double sum_exp =
                  accu(w.elem(coords) % exp(x.elem(coords) - maxval));
              new_alpha = log(gi.sum_weights(l)) - log(sum_exp) + maxval;
            } else if (family == "negbin") {
              new_alpha =
                  update_negbin(x.elem(coords), zeros<vec>(coords.n_elem),
                                w.elem(coords), /*theta*/ 1.0, alpha(k)(l));
            } else if (family == "logit" || family == "binomial") {
              new_alpha =
                  update_logit(x.elem(coords), zeros<vec>(coords.n_elem),
                               w.elem(coords), alpha(k)(l));
            } else {
              new_alpha =
                  dot(w.elem(coords), x.elem(coords)) * gi.inv_weights(l);
            }
            alpha(k)(l) = new_alpha;
            alpha_sum.elem(coords) += new_alpha;
          }
        }
        x = x_orig - alpha_sum;
        if (convergence_check(x, x0, w, tol))
          break;
      }

      // Main iterations with acceleration
      x1 = x0;
      x2 = x0;
      for (; iter < max_iter; ++iter) {
        x2 = x1;
        x1 = x0;
        x0 = x;
        for (size_t k = 0; k < K; ++k) {
          const GroupInfo &gi = group_info(k);
          // Remove current FE from alpha_sum
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const uvec &coords = gi.indices(l);
            if (coords.n_elem == 0)
              continue;
            alpha_sum.elem(coords) -= alpha(k)(l);
          }
          x = x_orig - alpha_sum; // Residual
          // Update alpha and add back
          for (size_t l = 0; l < gi.n_groups; ++l) {
            const uvec &coords = gi.indices(l);
            if (coords.n_elem == 0)
              continue;
            double new_alpha;
            if (family == "poisson") {
              double maxval = x.elem(coords).max();
              double sum_exp =
                  accu(w.elem(coords) % exp(x.elem(coords) - maxval));
              new_alpha = log(gi.sum_weights(l)) - log(sum_exp) + maxval;
            } else if (family == "negbin") {
              new_alpha =
                  update_negbin(x.elem(coords), zeros<vec>(coords.n_elem),
                                w.elem(coords), /*theta*/ 1.0, alpha(k)(l));
            } else if (family == "logit" || family == "binomial") {
              new_alpha =
                  update_logit(x.elem(coords), zeros<vec>(coords.n_elem),
                               w.elem(coords), alpha(k)(l));
            } else {
              new_alpha =
                  dot(w.elem(coords), x.elem(coords)) * gi.inv_weights(l);
            }
            alpha(k)(l) = new_alpha;
            alpha_sum.elem(coords) += new_alpha;
          }
        }

        // Irons-Tuck acceleration
        if (++acc_count == grand_acc) {
          acc_count = 0;
          vec delta1 = x0 - x1;
          vec delta2 = x1 - x2;
          vec delta_diff = delta1 - delta2;
          double denom = dot(delta_diff, delta_diff);
          if (denom > 1e-16) {
            double coef = dot(delta1, delta_diff) / denom;
            x = x0 - coef * delta1;
          }
        }

        x = x_orig - alpha_sum;
        if (convergence_check(x, x0, w, tol))
          break;
      }
    }
    V.col(p) = x;
  }
}

#endif // CAPYBARA_CENTER
