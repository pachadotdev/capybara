#ifndef CAPYBARA_ALPHA
#define CAPYBARA_ALPHA

// Fixed effects recovery - Direct group mean solution
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

struct AlphaGroupInfo {
  field<uvec> indices;
  size_t n_groups;
  
  AlphaGroupInfo() = default;
  AlphaGroupInfo(const list &jlist) {
    n_groups = jlist.size();
    indices.set_size(n_groups);
    
    for (size_t j = 0; j < n_groups; ++j) {
      indices(j) = as_uvec(as_cpp<integers>(jlist[j]));
    }
  }
};

inline GetAlphaResult get_alpha(const vec &p, const list &klist, double tol,
                                size_t iter_max) {
  const size_t K = klist.size();
  field<AlphaGroupInfo> group_info(K);
  for (size_t k = 0; k < K; ++k) {
    group_info(k) = AlphaGroupInfo(as_cpp<list>(klist[k]));
  }
  field<vec> Alpha(K);
  for (size_t k = 0; k < K; ++k) {
    if (group_info(k).n_groups > 0) {
      Alpha(k).zeros(group_info(k).n_groups);
    }
  }
  field<vec> Alpha0(K), Alpha1(K), Alpha2(K);
  for (size_t k = 0; k < K; ++k) {
    if (group_info(k).n_groups > 0) {
      Alpha0(k).zeros(group_info(k).n_groups);
      Alpha1(k).zeros(group_info(k).n_groups);
      Alpha2(k).zeros(group_info(k).n_groups);
    }
  }
  double ratio = 0.0;
  if (K == 2) {
    // K=2 specialization
    vec resid = p;
    for (size_t iter = 0; iter < iter_max; ++iter) {
      Alpha0 = Alpha;
      for (size_t k = 0; k < 2; ++k) {
        resid = p;
        size_t other = 1 - k;
        const AlphaGroupInfo &gi_other = group_info(other);
        const AlphaGroupInfo &gi_k = group_info(k);
        
        // Subtract other FE
        for (size_t j = 0; j < gi_other.n_groups; ++j) {
          resid.elem(gi_other.indices(j)) -= Alpha(other)(j);
        }
        // Update current FE
        for (size_t j = 0; j < gi_k.n_groups; ++j) {
          const uvec &idx = gi_k.indices(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }
      // Convergence check
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
    // K>2
    const int warmup = 15;
    const int grand_acc = 40;
    size_t iter = 0;
    
    for (; iter < std::min<size_t>(warmup, iter_max); ++iter) {
      Alpha0 = Alpha;
      for (size_t k = 0; k < K; ++k) {
        vec resid = p;
        const AlphaGroupInfo &gi_k = group_info(k);
        
        // Subtract other FEs
        for (size_t l = 0; l < K; ++l) {
          if (l == k || group_info(l).n_groups == 0)
            continue;
          const AlphaGroupInfo &gi_l = group_info(l);
          for (size_t j = 0; j < gi_l.n_groups; ++j) {
            resid.elem(gi_l.indices(j)) -= Alpha(l)(j);
          }
        }
        
        // Update current FE
        for (size_t j = 0; j < gi_k.n_groups; ++j) {
          const uvec &idx = gi_k.indices(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
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
      if (ratio < tol)
        break;
    }
    
    // Main loop - Alternate projections with Irons-Tuck acceleration
    int acc_count = 0;
    while (iter < iter_max && ratio >= tol) {
      // Save previous states
      Alpha2 = Alpha1;
      Alpha1 = Alpha0;
      Alpha0 = Alpha;
      
      // Simple projection
      for (size_t k = 0; k < K; ++k) {
        vec resid = p;
        const AlphaGroupInfo &gi_k = group_info(k);
        
        for (size_t l = 0; l < K; ++l) {
          if (l == k || group_info(l).n_groups == 0)
            continue;
          const AlphaGroupInfo &gi_l = group_info(l);
          for (size_t j = 0; j < gi_l.n_groups; ++j) {
            resid.elem(gi_l.indices(j)) -= Alpha(l)(j);
          }
        }
        
        for (size_t j = 0; j < gi_k.n_groups; ++j) {
          const uvec &idx = gi_k.indices(j);
          if (idx.n_elem == 0)
            continue;
          Alpha(k)(j) = mean(resid.elem(idx));
        }
      }
      ++iter;
      
      // Irons-Tuck acceleration every grand_acc iterations
      if (++acc_count == grand_acc) {
        acc_count = 0;
        for (size_t k = 0; k < K; ++k) {
          if (group_info(k).n_groups == 0)
            continue;
          vec delta1 = Alpha0(k) - Alpha1(k);
          vec delta2 = Alpha1(k) - Alpha2(k);
          vec delta_diff = delta1 - delta2;
          double denom_acc = dot(delta_diff, delta_diff);
          if (denom_acc > 1e-16) {
            double coef = dot(delta1, delta_diff) / denom_acc;
            Alpha(k) = Alpha0(k) - coef * delta1;
          }
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
