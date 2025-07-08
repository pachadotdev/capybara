#ifndef CAPYBARA_ALPHA
#define CAPYBARA_ALPHA

// FENmlm-style fixed effects recovery: direct group mean solution using
// Armadillo
struct GetAlphaResult {
  field<vec> Alpha;

  cpp11::list to_list() const {
    writable::list Alpha_r(Alpha.n_elem);
    for (size_t k = 0; k < Alpha.n_elem; ++k) {
      Alpha_r[k] = as_doubles_matrix(Alpha(k).eval());
    }
    return Alpha_r;
  }
};

// Core function: pure Armadillo types
inline GetAlphaResult get_alpha(const vec &p, const list &klist, double tol,
                                size_t iter_max) {
  const size_t K = klist.size();
  field<int> list_sizes(K);
  field<field<uvec>> group_indices(K);
  for (size_t k = 0; k < K; ++k) {
    const list &jlist = as_cpp<list>(klist[k]);
    size_t J = jlist.size();
    list_sizes(k) = J;
    group_indices(k).set_size(J);
    for (size_t j = 0; j < J; ++j) {
      group_indices(k)(j) = as_uvec(as_cpp<integers>(jlist[j]));
    }
  }
  field<vec> Alpha(K);
  for (size_t k = 0; k < K; ++k) {
    if (list_sizes(k) > 0) {
      Alpha(k).zeros(list_sizes(k));
    }
  }
  field<vec> Alpha0(K), Alpha1(K), Alpha2(K);
  for (size_t k = 0; k < K; ++k) {
    if (list_sizes(k) > 0) {
      Alpha0(k).zeros(list_sizes(k));
      Alpha1(k).zeros(list_sizes(k));
      Alpha2(k).zeros(list_sizes(k));
    }
  }
  double ratio = 0.0;
  if (K == 2) {
    // Two-way FE specialization: alternate between FEs
    vec resid = p;
    for (size_t iter = 0; iter < iter_max; ++iter) {
      Alpha0 = Alpha;
      for (size_t k = 0; k < 2; ++k) {
        resid = p;
        size_t other = 1 - k;
        // Subtract other FE
        for (size_t j = 0; j < list_sizes(other); ++j) {
          resid.elem(group_indices(other)(j)) -= Alpha(other)(j);
        }
        // Update this FE
        for (size_t j = 0; j < list_sizes(k); ++j) {
          const uvec &idx = group_indices(k)(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }
      double num = 0.0, denom = 0.0;
      for (size_t k = 0; k < 2; ++k) {
        const vec &diff = Alpha(k) - Alpha0(k);
        num += dot(diff, diff);
        denom += dot(Alpha0(k), Alpha0(k));
      }
      ratio = sqrt(num / (denom + 1e-16));
      if (ratio < tol)
        break;
    }
  } else {
    // General k-way FE with Irons-Tuck acceleration
    const int warmup = 10;    // fixest default warmup
    const int grand_acc = 15; // fixest default grand acceleration
    size_t iter = 0;
    // Warmup: simple projections
    for (; iter < std::min<size_t>(warmup, iter_max); ++iter) {
      Alpha0 = Alpha;
      for (size_t k = 0; k < K; ++k) {
        vec resid = p;
        for (size_t l = 0; l < K; ++l) {
          if (l == k || list_sizes(l) == 0)
            continue;
          for (size_t j = 0; j < list_sizes(l); ++j) {
            resid.elem(group_indices(l)(j)) -= Alpha(l)(j);
          }
        }
        for (size_t j = 0; j < list_sizes(k); ++j) {
          const uvec &idx = group_indices(k)(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }
      double num = 0.0, denom = 0.0;
      for (size_t k = 0; k < K; ++k) {
        const vec &diff = Alpha(k) - Alpha0(k);
        num += dot(diff, diff);
        denom += dot(Alpha0(k), Alpha0(k));
      }
      ratio = sqrt(num / (denom + 1e-16));
      if (ratio < tol)
        break;
    }
    // Main loop: alternate projections and Irons-Tuck acceleration
    int acc_count = 0;
    while (iter < iter_max && ratio >= tol) {
      // Save previous states
      Alpha2 = Alpha1;
      Alpha1 = Alpha0;
      Alpha0 = Alpha;
      // Simple projection
      for (size_t k = 0; k < K; ++k) {
        vec resid = p;
        for (size_t l = 0; l < K; ++l) {
          if (l == k || list_sizes(l) == 0)
            continue;
          for (size_t j = 0; j < list_sizes(l); ++j) {
            resid.elem(group_indices(l)(j)) -= Alpha(l)(j);
          }
        }
        for (size_t j = 0; j < list_sizes(k); ++j) {
          const uvec &idx = group_indices(k)(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }
      ++iter;
      // Irons-Tuck acceleration every grand_acc iterations
      if (++acc_count == grand_acc) {
        acc_count = 0;
        // Irons-Tuck step: Alpha = Alpha0 - 2*Alpha1 + Alpha2
        for (size_t k = 0; k < K; ++k) {
          if (list_sizes(k) == 0)
            continue;
          Alpha(k) = Alpha0(k) - 2 * Alpha1(k) + Alpha2(k);
        }
      }
      // Convergence check
      double num = 0.0, denom = 0.0;
      for (size_t k = 0; k < K; ++k) {
        const vec &diff = Alpha(k) - Alpha0(k);
        num += dot(diff, diff);
        denom += dot(Alpha0(k), Alpha0(k));
      }
      ratio = sqrt(num / (denom + 1e-16));
    }
  }
  GetAlphaResult res;
  res.Alpha = Alpha;
  return res;
}

#endif // CAPYBARA_ALPHA
