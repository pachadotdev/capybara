// Centering using flat observation-to-group mapping
//
// Two centering algorithms are available:
//
// 1. Stammann (default): Alternating projections with RRE-2 acceleration.
//    Each iteration performs a full Gauss-Seidel sweep updating each FE
//    dimension in sequence, then applies RRE-2 (Reduced Rank Extrapolation
//    with 2 columns) acceleration using 4 iterates (X, GX, G2X, G3X).
//    This is based on Ramiere-Helfer's "alternate 2-delta method" which uses
//    a third iteration to improve convergence vs. the simpler Irons-Tuck
//    method.
//
// 2. Berge: Fixed-point reformulation. For 2-FE: defines
//    F(alpha) = f_1(f_2(alpha)), composing both updates into a single map,
//    then solves alpha* = F(alpha*) using RRE-2 acceleration. For K>=3
//    FE: uses a full backward Gauss-Seidel sweep as the composed map F, with
//    RRE-2 acceleration on the first K-1 FE's coefficients.
//
// Both methods precompute in_out[g] = sum_{i: fe[i]==g} w[i] * V[i] once,
// then iterate on coefficient vectors only. Warm-starting is used across
// IRLS calls.

#ifndef CAPYBARA_CENTER_HELPERS_H
#define CAPYBARA_CENTER_HELPERS_H

namespace capybara {

// Centering algorithm selection
enum CenteringMethod { STAMMANN = 0, BERGE = 1 };

// Convert string to CenteringMethod enum
inline CenteringMethod centering_from_string(const std::string &s) {
  if (s == "berge")
    return BERGE;
  return STAMMANN; // default
}

inline bool continue_crit(double a, double b, double diffMax) {
  double diff = std::fabs(a - b);
  return (diff > diffMax) && (diff / (0.1 + std::fabs(a)) > diffMax);
}

inline bool stopping_crit(double a, double b, double diffMax) {
  double diff = std::fabs(a - b);
  return (diff < diffMax) || (diff / (0.1 + std::fabs(a)) < diffMax);
}

// Flat FE structure using std::vector for guaranteed contiguous memory
struct FlatFEMap {
  std::vector<std::vector<uword>>
      fe_map;                   // K x n_obs: fe_map[k][i] = group of obs i
  std::vector<vec> inv_weights; // K: precomputed 1/sum(w) per group
  std::vector<uword> n_groups;  // K: number of groups per FE
  uword n_obs;
  uword K;
  bool structure_built;

  FlatFEMap() : n_obs(0), K(0), structure_built(false) {}

  void build(const field<field<uvec>> &group_indices) {
    K = group_indices.n_elem;
    if (K == 0)
      return;

    n_groups.resize(K);
    n_obs = 0;

    for (uword k = 0; k < K; ++k) {
      n_groups[k] = group_indices(k).n_elem;
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        if (idx.n_elem > 0) {
          n_obs = std::max(n_obs, idx.max() + 1);
        }
      }
    }

    fe_map.resize(K);
    for (uword k = 0; k < K; ++k) {
      fe_map[k].assign(n_obs, 0);
    }

    for (uword k = 0; k < K; ++k) {
      uword *map_k = fe_map[k].data();
      for (uword g = 0; g < n_groups[k]; ++g) {
        const uvec &idx = group_indices(k)(g);
        const uword *idx_ptr = idx.memptr();
        const uword cnt = idx.n_elem;
        for (uword j = 0; j < cnt; ++j) {
          map_k[idx_ptr[j]] = g;
        }
      }
    }

    structure_built = true;
  }

  void update_weights(const vec &w) {
    if (K == 0)
      return;

    inv_weights.resize(K);
    const bool use_w = (w.n_elem == n_obs);
    const double *w_ptr = w.memptr();

    for (uword k = 0; k < K; ++k) {
      inv_weights[k].zeros(n_groups[k]);
      double *inv_w_ptr = inv_weights[k].memptr();
      const uword *map_k = fe_map[k].data();

      if (use_w) {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += w_ptr[i];
        }
      } else {
        for (uword i = 0; i < n_obs; ++i) {
          inv_w_ptr[map_k[i]] += 1.0;
        }
      }

      for (uword g = 0; g < n_groups[k]; ++g) {
        inv_w_ptr[g] = (inv_w_ptr[g] > 1e-12) ? (1.0 / inv_w_ptr[g]) : 0.0;
      }
    }
  }

  // Release all memory when transitioning between models with different
  // dimensions
  void reset() {
    for (auto &m : fe_map) {
      m.clear();
      m.shrink_to_fit();
    }
    fe_map.clear();
    fe_map.shrink_to_fit();

    for (auto &w : inv_weights) {
      w.reset();
    }
    inv_weights.clear();
    inv_weights.shrink_to_fit();

    n_groups.clear();
    n_groups.shrink_to_fit();

    n_obs = 0;
    K = 0;
    structure_built = false;
  }
};

// Warm-start storage for centering across IRLS iterations
struct CenterWarmStart {
  std::vector<mat> alpha; // K coefficient matrices (n_groups[k] x P)
  uword K;
  uword P;
  bool valid;

  // Persistent scratch buffers (sized on first use, reused if dimensions match)
  std::vector<mat> scratch_mats;
  mat scratch_beta; // beta_tmp for 2-FE case
  bool scratch_valid;
  uword scratch_n1, scratch_n2;

  CenterWarmStart()
      : K(0), P(0), valid(false), scratch_valid(false), scratch_n1(0),
        scratch_n2(0), stammann_2fe_valid(false), stammann_2fe_n1(0),
        stammann_2fe_p(0), stammann_kfe_valid(false), stammann_kfe_n0(0),
        stammann_kfe_p(0), berge_kfe_valid(false), berge_kfe_P(0),
        berge_kfe_K(0) {}

  void save(const std::vector<mat> &coeffs, uword n_fe, uword n_cols) {
    K = n_fe;
    P = n_cols;
    alpha.resize(K);
    for (uword k = 0; k < K; ++k) {
      alpha[k] = coeffs[k];
    }
    valid = true;
  }

  bool can_use(uword n_fe, uword n_cols) const {
    if (!valid || n_fe != K || n_cols != P)
      return false;
    return true;
  }

  // For Berge 2-FE: reuse scratch matrices across calls
  void ensure_scratch_2fe(uword n1, uword n2, uword p) {
    if (scratch_valid && scratch_n1 == n1 && scratch_n2 == n2 &&
        scratch_mats.size() >= 7 && scratch_mats[0].n_cols == p) {
      scratch_mats[3].zeros();
      scratch_mats[4].zeros();
      scratch_mats[5].zeros();
      scratch_mats[6].zeros();
      return;
    }
    scratch_mats.resize(7);
    scratch_mats[0].set_size(n1, p); // GX
    scratch_mats[1].set_size(n1, p); // G2X
    scratch_mats[2].set_size(n1, p); // X_it
    scratch_mats[3].zeros(n1, p);    // grand_Y
    scratch_mats[4].zeros(n1, p);    // grand_GY
    scratch_mats[5].zeros(n1, p);    // grand_GGY
    scratch_mats[6].set_size(n1, p); // G3X
    scratch_beta.set_size(n2, p);
    scratch_n1 = n1;
    scratch_n2 = n2;
    scratch_valid = true;
  }

  // Persistent storage for Stammann 2-FE scratch matrices
  std::vector<mat> stammann_2fe_scratch; // X_it, GX_it, G2X_it, G3X_it
  mat stammann_2fe_grand_Y, stammann_2fe_grand_GY;
  bool stammann_2fe_valid;
  uword stammann_2fe_n1, stammann_2fe_p;

  // For Stammann 2-FE: reuse scratch matrices across calls
  void ensure_scratch_stammann_2fe(uword n1, uword p) {
    if (stammann_2fe_valid && stammann_2fe_n1 == n1 && stammann_2fe_p == p) {
      // Reset grand acceleration matrices (they accumulate state)
      stammann_2fe_grand_Y.zeros();
      stammann_2fe_grand_GY.zeros();
      return;
    }
    // Dimensions changed: explicitly release old buffers before reallocating
    // to ensure memory is freed immediately
    for (auto &m : stammann_2fe_scratch) {
      m.reset();
    }
    stammann_2fe_grand_Y.reset();
    stammann_2fe_grand_GY.reset();

    stammann_2fe_scratch.resize(4);
    stammann_2fe_scratch[0].set_size(n1, p); // X_it
    stammann_2fe_scratch[1].set_size(n1, p); // GX_it
    stammann_2fe_scratch[2].set_size(n1, p); // G2X_it
    stammann_2fe_scratch[3].set_size(n1, p); // G3X_it
    stammann_2fe_grand_Y.zeros(n1, p);
    stammann_2fe_grand_GY.zeros(n1, p);
    stammann_2fe_n1 = n1;
    stammann_2fe_p = p;
    stammann_2fe_valid = true;
  }

  // Persistent storage for Stammann K-FE scratch matrices
  std::vector<mat> stammann_kfe_scratch; // X_it, GX_it, G2X_it, G3X_it
  mat stammann_kfe_grand_Y, stammann_kfe_grand_GY;
  bool stammann_kfe_valid;
  uword stammann_kfe_n0, stammann_kfe_p;

  // For Stammann K-FE: reuse scratch matrices across calls
  void ensure_scratch_stammann_kfe(uword n0, uword p) {
    if (stammann_kfe_valid && stammann_kfe_n0 == n0 && stammann_kfe_p == p) {
      // Reset grand acceleration matrices
      stammann_kfe_grand_Y.zeros();
      stammann_kfe_grand_GY.zeros();
      return;
    }
    // Dimensions changed: explicitly release old buffers before reallocating
    // to ensure memory is freed immediately
    for (auto &m : stammann_kfe_scratch) {
      m.reset();
    }
    stammann_kfe_grand_Y.reset();
    stammann_kfe_grand_GY.reset();

    stammann_kfe_scratch.resize(4);
    stammann_kfe_scratch[0].zeros(n0, p); // X_it
    stammann_kfe_scratch[1].zeros(n0, p); // GX_it
    stammann_kfe_scratch[2].zeros(n0, p); // G2X_it
    stammann_kfe_scratch[3].zeros(n0, p); // G3X_it
    stammann_kfe_grand_Y.zeros(n0, p);
    stammann_kfe_grand_GY.zeros(n0, p);
    stammann_kfe_n0 = n0;
    stammann_kfe_p = p;
    stammann_kfe_valid = true;
  }

  // Persistent storage for Berge K-FE col_ptrs_all buffer
  std::vector<std::vector<const double *>> berge_kfe_col_ptrs;
  bool berge_kfe_valid;
  uword berge_kfe_P, berge_kfe_K;

  // For Berge K-FE: reuse col_ptrs_all storage across gs_sweep_backward_kfe
  // calls
  std::vector<std::vector<const double *>> &
  ensure_berge_kfe_col_ptrs(uword p, uword k) {
    if (berge_kfe_valid && berge_kfe_P == p && berge_kfe_K == k) {
      return berge_kfe_col_ptrs;
    }
    // Dimensions changed: reallocate
    berge_kfe_col_ptrs.assign(p, std::vector<const double *>(k, nullptr));
    berge_kfe_P = p;
    berge_kfe_K = k;
    berge_kfe_valid = true;
    return berge_kfe_col_ptrs;
  }

  // Release all scratch buffers to free memory when transitioning between
  // models with different dimensions or when memory pressure is high
  void reset() {
    // Clear coefficient storage
    alpha.clear();
    alpha.shrink_to_fit();
    K = 0;
    P = 0;
    valid = false;

    // Clear Berge 2-FE scratch buffers
    scratch_mats.clear();
    scratch_mats.shrink_to_fit();
    scratch_beta.reset();
    scratch_valid = false;
    scratch_n1 = 0;
    scratch_n2 = 0;

    // Clear Stammann 2-FE scratch buffers
    stammann_2fe_scratch.clear();
    stammann_2fe_scratch.shrink_to_fit();
    stammann_2fe_grand_Y.reset();
    stammann_2fe_grand_GY.reset();
    stammann_2fe_valid = false;
    stammann_2fe_n1 = 0;
    stammann_2fe_p = 0;

    // Clear Stammann K-FE scratch buffers
    stammann_kfe_scratch.clear();
    stammann_kfe_scratch.shrink_to_fit();
    stammann_kfe_grand_Y.reset();
    stammann_kfe_grand_GY.reset();
    stammann_kfe_valid = false;
    stammann_kfe_n0 = 0;
    stammann_kfe_p = 0;

    // Clear Berge K-FE col_ptrs buffer
    berge_kfe_col_ptrs.clear();
    berge_kfe_col_ptrs.shrink_to_fit();
    berge_kfe_valid = false;
    berge_kfe_P = 0;
    berge_kfe_K = 0;
  }
};

// Precompute weighted group sums: in_out_k(g, p) = sum_{i: fe_k[i]==g} w[i] *
// V(i, p)
inline void in_out_(std::vector<mat> &in_out, const mat &V, const double *w_ptr,
                    const FlatFEMap &map) {
  const uword n_obs = V.n_rows;
  const uword P = V.n_cols;
  const uword K = map.K;

  in_out.resize(K);
  for (uword k = 0; k < K; ++k) {
    in_out[k].zeros(map.n_groups[k], P);
    const uword *gk = map.fe_map[k].data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (P > 1)
#endif
    for (uword p = 0; p < P; ++p) {
      const double *v_col = V.colptr(p);
      double *io_col = in_out[k].colptr(p);
      for (uword i = 0; i < n_obs; ++i) {
        io_col[gk[i]] += w_ptr[i] * v_col[i];
      }
    }
  }
}
} // namespace capybara

#endif // CAPYBARA_CENTER_HELPERS_H
