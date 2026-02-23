#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <armadillo4r.hpp>

#ifdef CAPYBARA_DEBUG
#include <chrono>
#endif

using cpp4r::doubles;
using cpp4r::doubles_matrix;
using cpp4r::integers;
using cpp4r::list;
using cpp4r::strings;

// Configure OpenMP threads from configure-time macro
namespace capybara {
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
#ifndef CAPYBARA_DEFAULT_OMP_THREADS
#define CAPYBARA_DEFAULT_OMP_THREADS -1
#endif
inline void set_omp_threads_from_config() {
  static bool done = false;
  if (!done) {
#if defined(_OPENMP) && (CAPYBARA_DEFAULT_OMP_THREADS > 0)
    omp_set_num_threads(CAPYBARA_DEFAULT_OMP_THREADS);
#endif
    done = true;
  }
}
#endif
} // namespace capybara

// Passing parameters from R to C++ functions
// see R/fit_control.R
struct CapybaraParameters {
  double dev_tol;
  double center_tol;
  double center_tol_loose;
  double collin_tol;
  double step_halving_factor;
  double alpha_tol;

  // Separation detection parameters
  double sep_tol;              // Convergence tolerance
  double sep_zero_tol;         // Tolerance for treating values as zero
  size_t sep_max_iter;         // Max iterations for ReLU algorithm
  size_t sep_simplex_max_iter; // Max iterations for simplex algorithm
  bool check_separation;       // Whether to perform separation detection
  bool sep_use_relu;           // Use ReLU algorithm
  bool sep_use_simplex;        // Use simplex algorithm

  size_t iter_max;
  size_t iter_center_max;
  size_t iter_inner_max;
  size_t iter_alpha_max;
  bool return_fe;
  bool keep_tx;

  // Step-halving parameters
  double step_halving_memory;
  size_t max_step_halving;
  double start_inner_tol;

  // Centering acceleration parameters
  size_t grand_acc_period;

  // Centering algorithm: "stammann" (alternating projections) or "berge"
  // (fixed-point)
  std::string centering;

  // Variance-covariance estimator type
  std::string vcov_type;

  CapybaraParameters()
      : dev_tol(1.0e-08), center_tol(1.0e-08), center_tol_loose(1.0e-04),
        collin_tol(1.0e-10), step_halving_factor(0.5), alpha_tol(1.0e-08),
        sep_tol(1.0e-08), sep_zero_tol(1.0e-12), sep_max_iter(200),
        sep_simplex_max_iter(2000), check_separation(true), sep_use_relu(true),
        sep_use_simplex(true), iter_max(25), iter_center_max(10000),
        iter_inner_max(50), iter_alpha_max(10000), return_fe(true),
        keep_tx(false), step_halving_memory(0.9), max_step_halving(2),
        start_inner_tol(1e-06), grand_acc_period(10), centering("stammann"),
        vcov_type("") {}

  explicit CapybaraParameters(const cpp4r::list &control) {
    dev_tol = as_cpp<double>(control["dev_tol"]);
    center_tol = as_cpp<double>(control["center_tol"]);
    center_tol_loose = as_cpp<double>(control["center_tol_loose"]);
    collin_tol = as_cpp<double>(control["collin_tol"]);
    step_halving_factor = as_cpp<double>(control["step_halving_factor"]);
    alpha_tol = as_cpp<double>(control["alpha_tol"]);

    // Separation detection parameters
    sep_tol = as_cpp<double>(control["sep_tol"]);
    sep_zero_tol = as_cpp<double>(control["sep_zero_tol"]);
    sep_max_iter = as_cpp<size_t>(control["sep_max_iter"]);
    sep_simplex_max_iter = as_cpp<size_t>(control["sep_simplex_max_iter"]);
    check_separation = as_cpp<bool>(control["check_separation"]);
    sep_use_relu = as_cpp<bool>(control["sep_use_relu"]);
    sep_use_simplex = as_cpp<bool>(control["sep_use_simplex"]);

    iter_max = as_cpp<size_t>(control["iter_max"]);
    iter_center_max = as_cpp<size_t>(control["iter_center_max"]);
    iter_inner_max = as_cpp<size_t>(control["iter_inner_max"]);
    iter_alpha_max = as_cpp<size_t>(control["iter_alpha_max"]);
    return_fe = as_cpp<bool>(control["return_fe"]);
    keep_tx = as_cpp<bool>(control["keep_tx"]);
    step_halving_memory = as_cpp<double>(control["step_halving_memory"]);
    max_step_halving = as_cpp<size_t>(control["max_step_halving"]);
    start_inner_tol = as_cpp<double>(control["start_inner_tol"]);
    grand_acc_period = as_cpp<size_t>(control["grand_acc_period"]);

    // Extract centering method
    SEXP centering_sexp = control["centering"];
    if (centering_sexp != R_NilValue) {
      centering = as_cpp<std::string>(centering_sexp);
    } else {
      centering = "stammann";
    }

    // Extract vcov_type (optional string parameter)
    SEXP vcov_type_sexp = control["vcov_type"];
    if (vcov_type_sexp != R_NilValue) {
      vcov_type = as_cpp<std::string>(vcov_type_sexp);
    } else {
      vcov_type = "";
    }
  }
};

#include "01_center.h"
#include "02_chol.h"
#include "03_beta.h"
#include "04_alpha.h"
#include "05_separation.h"
#include "06_fit_helpers.h"
#include "07_lm.h"
#include "08_glm.h"
#include "09_negbin.h"

using LMResult = capybara::InferenceLM;
using GLMResult = capybara::InferenceGLM;
using NegBinResult = capybara::InferenceNegBin;

// Convert R indexing to C++ indexing
inline uvec R_1based_to_Cpp_0based_indices(const integers &r_indices) {
  uvec cpp_indices(r_indices.size());

  std::transform(
      r_indices.begin(), r_indices.end(), cpp_indices.begin(),
      [](size_t r_val) -> uword { return static_cast<uword>(r_val - 1); });

  return cpp_indices;
}

// Build FlatFEMap directly from R integer code vectors (the lean path).
// fe_codes is a list of K integer vectors, each of length N, with 0-based
// group codes. This is O(N*K) with zero intermediate allocation — no more
// field<field<uvec>> of hundreds of small heap-allocated uvecs.
inline capybara::FlatFEMap R_codes_to_FlatFEMap(const list &fe_codes) {
  capybara::FlatFEMap map;
  const size_t K = fe_codes.size();
  if (K == 0)
    return map;

  map.K = K;
  map.n_groups.resize(K);
  map.fe_map.resize(K);
  map.n_obs = 0;

  for (size_t k = 0; k < K; ++k) {
    const integers codes_k = as_cpp<integers>(fe_codes[k]);
    const size_t N = codes_k.size();
    if (k == 0)
      map.n_obs = N;

    // Find max code to determine n_groups, and copy into fe_map in one pass
    map.fe_map[k].resize(N);
    uword *map_k = map.fe_map[k].data();
    uword max_code = 0;
    for (size_t i = 0; i < N; ++i) {
      const uword c = static_cast<uword>(codes_k[i]);
      map_k[i] = c;
      if (c > max_code)
        max_code = c;
    }
    map.n_groups[k] = max_code + 1;
  }

  map.structure_built = true;
  return map;
}

// Extract FE level names from an R list of character vectors.
// Returns field<field<string>> for output labeling.
inline void extract_fe_names_and_levels(const list &fe_codes,
                                        const list &fe_levels_r,
                                        field<std::string> &fe_names,
                                        field<field<std::string>> &fe_levels) {
  const size_t K = fe_codes.size();
  fe_names.set_size(K);
  fe_levels.set_size(K);

  // FE variable names
  if (!fe_codes.names().empty()) {
    cpp4r::strings names_r = fe_codes.names();
    for (R_xlen_t i = 0; i < names_r.size(); i++) {
      fe_names(i) = std::string(names_r[i]);
    }
  }

  // Level names per FE
  for (size_t k = 0; k < K; ++k) {
    const cpp4r::strings lvl_k = as_cpp<cpp4r::strings>(fe_levels_r[k]);
    fe_levels(k).set_size(lvl_k.size());
    for (R_xlen_t j = 0; j < lvl_k.size(); j++) {
      fe_levels(k)(j) = std::string(lvl_k[j]);
    }
  }
}

// ============================================================
// Raw data preparation: NA removal + FE coding + cluster coding
// ============================================================

// Result struct for prepare_raw_data
struct PreparedData {
  mat X;
  vec y;
  vec w;
  capybara::FlatFEMap fe_map;
  field<std::string> fe_names;
  field<field<std::string>> fe_levels;
  field<uvec> cluster_groups;
  bool has_clusters;
  uvec obs_indices; // 1-based R indices of kept rows
  size_t nobs_used;
};

// Check if an R SEXP element is NA depending on type
// For INTSXP (including factors): NA_INTEGER
// For REALSXP: R_IsNA or !R_finite
// For STRSXP: STRING_ELT == NA_STRING
// Returns true if the value at index i is NA/NaN/Inf
inline bool sexp_is_na(SEXP col, R_xlen_t i) {
  switch (TYPEOF(col)) {
  case INTSXP:
    return INTEGER(col)[i] == NA_INTEGER;
  case REALSXP:
    return !R_finite(REAL(col)[i]);
  case STRSXP:
    return STRING_ELT(col, i) == NA_STRING;
  case LGLSXP:
    return LOGICAL(col)[i] == NA_LOGICAL;
  default:
    return false;
  }
}

// Format a double to string matching R's as.character() behaviour
// (strips trailing zeros, e.g. 6.0 -> "6", 3.14 -> "3.14")
inline std::string double_to_string(double val) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.15g", val);
  return std::string(buf);
}

// Code a single raw R column into 0-based integer codes + level names.
// Handles: factor (INTSXP with levels attr), integer, real, character, logical.
struct CodedColumn {
  std::vector<uword> codes;
  std::vector<std::string> levels;
};

inline CodedColumn code_column(SEXP col, const uvec &keep_indices) {
  CodedColumn result;
  const size_t n = keep_indices.n_elem;
  result.codes.resize(n);

  // Check if it's a factor (INTSXP with "levels" attribute)
  SEXP levels_attr = Rf_getAttrib(col, R_LevelsSymbol);
  if (levels_attr != R_NilValue && TYPEOF(col) == INTSXP) {
    // Factor: integer codes are 1-based, convert to 0-based
    const int *idata = INTEGER(col);
    R_xlen_t n_levels = Rf_xlength(levels_attr);
    result.levels.resize(n_levels);
    for (R_xlen_t j = 0; j < n_levels; ++j) {
      result.levels[j] = std::string(CHAR(STRING_ELT(levels_attr, j)));
    }
    // We may need to remap if not all levels are present after NA removal.
    // But for now, keep all original levels (matches old droplevels behavior
    // approximately). Actually, let's do a proper remap to drop unused levels.
    std::vector<bool> level_used(n_levels, false);
    for (size_t i = 0; i < n; ++i) {
      int code = idata[keep_indices(i)] - 1; // 0-based
      level_used[code] = true;
    }
    // Build remap
    std::vector<uword> remap(n_levels, 0);
    std::vector<std::string> new_levels;
    new_levels.reserve(n_levels);
    for (R_xlen_t j = 0; j < n_levels; ++j) {
      if (level_used[j]) {
        remap[j] = new_levels.size();
        new_levels.push_back(result.levels[j]);
      }
    }
    result.levels = std::move(new_levels);
    for (size_t i = 0; i < n; ++i) {
      result.codes[i] = remap[idata[keep_indices(i)] - 1];
    }
    return result;
  }

  // Non-factor: use hash map for coding
  if (TYPEOF(col) == INTSXP) {
    const int *idata = INTEGER(col);
    std::unordered_map<int, uword> map;
    for (size_t i = 0; i < n; ++i) {
      int val = idata[keep_indices(i)];
      auto it = map.find(val);
      if (it == map.end()) {
        uword code = map.size();
        map[val] = code;
        result.codes[i] = code;
      } else {
        result.codes[i] = it->second;
      }
    }
    result.levels.resize(map.size());
    for (auto &kv : map) {
      result.levels[kv.second] = std::to_string(kv.first);
    }
  } else if (TYPEOF(col) == REALSXP) {
    const double *ddata = REAL(col);
    std::unordered_map<double, uword> map;
    for (size_t i = 0; i < n; ++i) {
      double val = ddata[keep_indices(i)];
      auto it = map.find(val);
      if (it == map.end()) {
        uword code = map.size();
        map[val] = code;
        result.codes[i] = code;
      } else {
        result.codes[i] = it->second;
      }
    }
    result.levels.resize(map.size());
    for (auto &kv : map) {
      result.levels[kv.second] = double_to_string(kv.first);
    }
  } else if (TYPEOF(col) == STRSXP) {
    std::unordered_map<std::string, uword> map;
    for (size_t i = 0; i < n; ++i) {
      std::string val(CHAR(STRING_ELT(col, keep_indices(i))));
      auto it = map.find(val);
      if (it == map.end()) {
        uword code = map.size();
        map[val] = code;
        result.codes[i] = code;
      } else {
        result.codes[i] = it->second;
      }
    }
    result.levels.resize(map.size());
    for (auto &kv : map) {
      result.levels[kv.second] = kv.first;
    }
  } else if (TYPEOF(col) == LGLSXP) {
    const int *ldata = LOGICAL(col);
    std::unordered_map<int, uword> map;
    for (size_t i = 0; i < n; ++i) {
      int val = ldata[keep_indices(i)];
      auto it = map.find(val);
      if (it == map.end()) {
        uword code = map.size();
        map[val] = code;
        result.codes[i] = code;
      } else {
        result.codes[i] = it->second;
      }
    }
    result.levels.resize(map.size());
    for (auto &kv : map) {
      result.levels[kv.second] = kv.first ? "TRUE" : "FALSE";
    }
  }

  return result;
}

// Build cluster inverted index: field<uvec> where each element contains
// the 0-based row indices for one cluster group
inline field<uvec> build_cluster_groups(SEXP col, const uvec &keep_indices) {
  const size_t n = keep_indices.n_elem;

  // First, code the column
  CodedColumn coded = code_column(col, keep_indices);

  // Count observations per group
  size_t n_groups = coded.levels.size();
  std::vector<size_t> group_counts(n_groups, 0);
  for (size_t i = 0; i < n; ++i) {
    group_counts[coded.codes[i]]++;
  }

  // Allocate and fill
  field<uvec> groups(n_groups);
  std::vector<size_t> offsets(n_groups, 0);
  for (size_t g = 0; g < n_groups; ++g) {
    groups(g).set_size(group_counts[g]);
  }
  for (size_t i = 0; i < n; ++i) {
    uword g = coded.codes[i];
    groups(g)(offsets[g]++) = i; // 0-based index into the clean data
  }

  return groups;
}

// Main preparation function: takes raw R data, returns clean C++ data
inline PreparedData prepare_raw_data(const doubles_matrix<> &X_r,
                                     const doubles &y_r, const doubles &w_r,
                                     const list &fe_cols_r, SEXP cl_col_r) {
  PreparedData out;
  const size_t N = y_r.size();
  const size_t p = X_r.ncol();
  const size_t K = fe_cols_r.size();

  // --- Step 1: Scan for NA/NaN/Inf across y, X, w, FE cols, cluster col ---

  // Start with all valid
  std::vector<bool> valid(N, true);

  // y — access via cpp4r doubles API
  for (size_t i = 0; i < N; ++i) {
    if (!R_finite(static_cast<double>(y_r[i])))
      valid[i] = false;
  }

  // X (column-major) — access via cpp4r doubles_matrix API
  for (size_t j = 0; j < p; ++j) {
    for (size_t i = 0; i < N; ++i) {
      if (valid[i] && !R_finite(static_cast<double>(X_r(i, j))))
        valid[i] = false;
    }
  }

  // w
  for (size_t i = 0; i < N; ++i) {
    if (valid[i] && !R_finite(static_cast<double>(w_r[i])))
      valid[i] = false;
  }

  // FE columns
  for (size_t k = 0; k < K; ++k) {
    SEXP col = VECTOR_ELT(static_cast<SEXP>(fe_cols_r), k);
    for (size_t i = 0; i < N; ++i) {
      if (valid[i] && sexp_is_na(col, i))
        valid[i] = false;
    }
  }

  // Cluster column
  out.has_clusters = (cl_col_r != R_NilValue);
  if (out.has_clusters) {
    for (size_t i = 0; i < N; ++i) {
      if (valid[i] && sexp_is_na(cl_col_r, i))
        valid[i] = false;
    }
  }

  // --- Step 2: Build keep indices (1-based for R, but store 0-based too) ---

  size_t n_valid = 0;
  for (size_t i = 0; i < N; ++i) {
    if (valid[i])
      n_valid++;
  }

  uvec keep_0based(n_valid);
  out.obs_indices.set_size(n_valid);
  size_t idx = 0;
  for (size_t i = 0; i < N; ++i) {
    if (valid[i]) {
      keep_0based(idx) = i;
      out.obs_indices(idx) = i + 1; // 1-based for R
      idx++;
    }
  }
  out.nobs_used = n_valid;

  // If no rows dropped, we can skip subsetting for y, X, w (just copy)
  bool all_valid = (n_valid == N);

  // --- Step 3: Subset y, X, w ---

  if (all_valid) {
    out.y = as_col(y_r);
    out.X = as_mat(X_r);
    out.w = as_col(w_r);
  } else {
    out.y.set_size(n_valid);
    out.X.set_size(n_valid, p);
    out.w.set_size(n_valid);

    for (size_t i = 0; i < n_valid; ++i) {
      size_t src = keep_0based(i);
      out.y(i) = static_cast<double>(y_r[src]);
      out.w(i) = static_cast<double>(w_r[src]);
    }
    for (size_t j = 0; j < p; ++j) {
      for (size_t i = 0; i < n_valid; ++i) {
        out.X(i, j) = static_cast<double>(X_r(keep_0based(i), j));
      }
    }
  }

  // --- Step 4: Code FE columns ---

  out.fe_map.K = K;
  out.fe_map.n_obs = n_valid;
  out.fe_map.n_groups.resize(K);
  out.fe_map.fe_map.resize(K);
  out.fe_names.set_size(K);
  out.fe_levels.set_size(K);

  // Get FE variable names
  if (K > 0 && !fe_cols_r.names().empty()) {
    cpp4r::strings names_r = fe_cols_r.names();
    for (R_xlen_t i = 0; i < static_cast<R_xlen_t>(K); i++) {
      out.fe_names(i) = std::string(names_r[i]);
    }
  }

  for (size_t k = 0; k < K; ++k) {
    SEXP col = VECTOR_ELT(static_cast<SEXP>(fe_cols_r), k);
    CodedColumn coded = code_column(col, keep_0based);

    out.fe_map.fe_map[k] = std::move(coded.codes);
    out.fe_map.n_groups[k] = coded.levels.size();

    out.fe_levels(k).set_size(coded.levels.size());
    for (size_t j = 0; j < coded.levels.size(); ++j) {
      out.fe_levels(k)(j) = std::move(coded.levels[j]);
    }
  }
  out.fe_map.structure_built = (K > 0);

  // --- Step 5: Build cluster inverted index ---

  if (out.has_clusters) {
    out.cluster_groups = build_cluster_groups(cl_col_r, keep_0based);
  }

  return out;
}

// Helper: package FE names and levels into R list for return
inline void add_fe_metadata_to_result(
    writable::list &ret, const field<std::string> &fe_names,
    const field<field<std::string>> &fe_levels, size_t nobs_used,
    const uvec &obs_indices, bool all_valid) {
  // nobs_used
  ret.push_back(
      {"nobs_used"_nm = writable::integers({static_cast<int>(nobs_used)})});

  // obs_indices (1-based) — only if some rows were dropped
  if (!all_valid) {
    writable::integers r_indices(obs_indices.n_elem);
    for (size_t i = 0; i < obs_indices.n_elem; ++i) {
      r_indices[i] = static_cast<int>(obs_indices(i));
    }
    ret.push_back({"obs_indices"_nm = r_indices});
  }

  // nms_fe: list of character vectors (one per FE)
  size_t K = fe_names.n_elem;
  writable::list nms_fe_list(K);
  writable::strings nms_fe_names(K);
  for (size_t k = 0; k < K; ++k) {
    writable::strings lvls(fe_levels(k).n_elem);
    for (size_t j = 0; j < fe_levels(k).n_elem; ++j) {
      lvls[j] = fe_levels(k)(j);
    }
    nms_fe_list[k] = lvls;
    nms_fe_names[k] = fe_names(k);
  }
  nms_fe_list.names() = nms_fe_names;
  ret.push_back({"nms_fe"_nm = nms_fe_list});

  // fe_levels: named integer vector with number of levels per FE
  writable::integers fe_lvl_counts(K);
  for (size_t k = 0; k < K; ++k) {
    fe_lvl_counts[k] = static_cast<int>(fe_levels(k).n_elem);
  }
  fe_lvl_counts.attr("names") = nms_fe_names;
  ret.push_back({"fe_levels"_nm = fe_lvl_counts});
}

[[cpp4r::register]] doubles_matrix<>
center_variables_(const doubles_matrix<> &V_r, const doubles &w_r,
                  const list &fe_codes, const double &tol,
                  const size_t &max_iter, const size_t &grand_acc_period) {
  mat V = as_mat(V_r);
  vec w = as_col(w_r);

  capybara::FlatFEMap map = R_codes_to_FlatFEMap(fe_codes);
  map.update_weights(w);
  capybara::center_variables(V, w, map, tol, max_iter, grand_acc_period);

  return as_doubles_matrix(V);
}

[[cpp4r::register]] list felm_fit_(const doubles_matrix<> &X_r,
                                   const doubles &y_r, const doubles &w_r,
                                   const list &fe_cols_r, SEXP cl_col_r,
                                   SEXP entity1_col_r, SEXP entity2_col_r,
                                   const list &control) {
  CapybaraParameters params(control);

  // Prepare data: NA removal + FE coding + cluster coding
  PreparedData data = prepare_raw_data(X_r, y_r, w_r, fe_cols_r, cl_col_r);

  const field<uvec> *cluster_ptr =
      data.has_clusters ? &data.cluster_groups : nullptr;

  // Prepare entity groups for dyadic clustering if needed
  field<uvec> entity1_groups, entity2_groups;
  const field<uvec> *entity1_ptr = nullptr;
  const field<uvec> *entity2_ptr = nullptr;

  if (params.vcov_type == "m-estimator-dyadic" && entity1_col_r != R_NilValue &&
      entity2_col_r != R_NilValue) {
    // Build entity groups from the raw entity columns, using the same keep
    // indices
    uvec keep_indices = regspace<uvec>(0, data.y.n_elem - 1);
    entity1_groups = build_cluster_groups(entity1_col_r, keep_indices);
    entity2_groups = build_cluster_groups(entity2_col_r, keep_indices);
    entity1_ptr = &entity1_groups;
    entity2_ptr = &entity2_groups;
  }

  capybara::InferenceLM result =
      capybara::felm_fit(data.X, data.y, data.w, data.fe_map, params, nullptr,
                         cluster_ptr, false, 0.0, entity1_ptr, entity2_ptr);

  // Replace collinear coefficients (NaN) with R's NA_REAL in all columns of
  // coef_table
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    uvec collinear_idx = find(collinear_mask);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      uword idx = collinear_idx(i);
      result.coef_table(idx, 0) = NA_REAL; // Estimate
      result.coef_table(idx, 1) = NA_REAL; // Std. Error
      result.coef_table(idx, 2) = NA_REAL; // z value
      result.coef_table(idx, 3) = NA_REAL; // Pr(>|z|)
    }
  }

  bool all_valid = (data.nobs_used == static_cast<size_t>(y_r.size()));

  auto ret = writable::list(
      {"fitted_values"_nm = as_doubles(result.fitted_values),
       "residuals"_nm = as_doubles(result.residuals),
       "weights"_nm = as_doubles(result.weights),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "r_squared"_nm = result.r_squared,
       "adj_r_squared"_nm = result.adj_r_squared,
       "coef_status"_nm = as_integers(result.coef_status),
       "success"_nm = result.success, "has_fe"_nm = result.has_fe});

  // Add fixed effects information if available
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < data.fe_levels.n_elem && data.fe_levels(k).n_elem > 0) {
        writable::strings level_names(data.fe_levels(k).n_elem);
        for (size_t j = 0; j < data.fe_levels(k).n_elem; j++) {
          if (!data.fe_levels(k)(j).empty()) {
            level_names[j] = data.fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < data.fe_names.n_elem && !data.fe_names(k).empty()) {
        fe_list_names[k] = data.fe_names(k);
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;
    ret.push_back({"fixed_effects"_nm = fe_list});
    ret.push_back({"has_fe"_nm = result.has_fe});
  }

  if (!result.iterations.is_empty()) {
    ret.push_back({"iterations"_nm = as_integers(result.iterations)});
  }

  if (params.keep_tx && result.has_tx) {
    ret.push_back({"TX"_nm = as_doubles_matrix(result.TX)});
  }

  // Add metadata for R-side post-processing
  add_fe_metadata_to_result(ret, data.fe_names, data.fe_levels, data.nobs_used,
                            data.obs_indices, all_valid);

  return ret;
}

[[cpp4r::register]] list
feglm_fit_(const doubles &beta_r, const doubles &eta_r, const doubles &y_r,
           const doubles_matrix<> &x_r, const doubles &wt_r,
           const doubles &offset_r, const double &theta,
           const std::string &family, const list &control,
           const list &fe_cols_r, SEXP cl_col_r, SEXP entity1_col_r,
           SEXP entity2_col_r) {
  CapybaraParameters params(control);

  // Prepare data: NA removal + FE coding + cluster coding
  PreparedData data = prepare_raw_data(x_r, y_r, wt_r, fe_cols_r, cl_col_r);
  bool all_valid = (data.nobs_used == static_cast<size_t>(y_r.size()));

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  // Handle beta/eta: if provided, subset to match clean data; else use empty
  vec beta = as_col(beta_r);
  vec eta;
  if (eta_r.size() > 0) {
    if (all_valid) {
      eta = as_col(eta_r);
    } else {
      // Subset eta to valid rows
      eta.set_size(data.nobs_used);
      for (size_t i = 0; i < data.nobs_used; ++i) {
        eta(i) = static_cast<double>(
            eta_r[static_cast<R_xlen_t>(data.obs_indices(i) - 1)]);
      }
    }
  } else {
    eta.set_size(0);
  }

  // Handle offset: subset to match clean data
  vec offset;
  if (all_valid) {
    offset = as_col(offset_r);
  } else {
    offset.set_size(data.nobs_used);
    for (size_t i = 0; i < data.nobs_used; ++i) {
      offset(i) = static_cast<double>(
          offset_r[static_cast<R_xlen_t>(data.obs_indices(i) - 1)]);
    }
  }

  // Replace non-finite offset values with 0 to prevent NaN propagation
  // into eta and subsequently into the Cholesky solver (DLASCL error -4)
  {
    uvec bad_offset = find_nonfinite(offset);
    if (bad_offset.n_elem > 0) {
      offset.elem(bad_offset).zeros();
    }
  }

  // Add offset to eta (the linear predictor is eta = X*beta + alpha + offset)
  if (eta.n_elem > 0) {
    eta += offset;
  }

  // Safety net: if eta contains non-finite values (e.g. because
  // start_guesses_ computed log(NA) when y had NA values that
  // prepare_raw_data subsequently removed), replace with a reasonable
  // starting value derived from the clean y
  if (eta.n_elem > 0) {
    uvec bad_eta = find_nonfinite(eta);
    if (bad_eta.n_elem > 0) {
      double y_mean = mean(data.y);
      // For Poisson/count models the link is log, so log(mean(y)+0.1) is safe
      // For other families this is still a reasonable fallback
      double safe_eta = std::log(y_mean + 0.1);
      if (!std::isfinite(safe_eta))
        safe_eta = 0.0;
      eta.elem(bad_eta).fill(safe_eta);
    }
  }

  // Pass offset pointer so fixed effects can be computed correctly
  const vec *offset_ptr = (any(offset != 0.0)) ? &offset : nullptr;

  const field<uvec> *cluster_ptr =
      data.has_clusters ? &data.cluster_groups : nullptr;

  // Prepare entity groups for dyadic clustering if needed
  field<uvec> entity1_groups, entity2_groups;
  const field<uvec> *entity1_ptr = nullptr;
  const field<uvec> *entity2_ptr = nullptr;

  if (params.vcov_type == "m-estimator-dyadic" && entity1_col_r != R_NilValue &&
      entity2_col_r != R_NilValue) {
    uvec keep_indices = regspace<uvec>(0, data.y.n_elem - 1);
    entity1_groups = build_cluster_groups(entity1_col_r, keep_indices);
    entity2_groups = build_cluster_groups(entity2_col_r, keep_indices);
    entity1_ptr = &entity1_groups;
    entity2_ptr = &entity2_groups;
  }

  capybara::InferenceGLM result = capybara::feglm_fit(
      beta, eta, data.y, data.X, data.w, theta, family_type, data.fe_map,
      params, nullptr, cluster_ptr, offset_ptr, entity1_ptr, entity2_ptr);

  // Replace collinear coefficients (NaN) with R's NA_REAL in all columns of
  // coef_table
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    uvec collinear_idx = find(collinear_mask);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      uword idx = collinear_idx(i);
      result.coef_table(idx, 0) = NA_REAL; // Estimate
      result.coef_table(idx, 1) = NA_REAL; // Std. Error
      result.coef_table(idx, 2) = NA_REAL; // z value
      result.coef_table(idx, 3) = NA_REAL; // Pr(>|z|)
    }
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)})});

  // Add pseudo R-squared for Poisson models
  if (family_type == capybara::POISSON && result.pseudo_rsq > 0.0) {
    out.push_back({"pseudo.rsq"_nm = result.pseudo_rsq});
  }

  // Add separation detection results for Poisson models
  if (family_type == capybara::POISSON && result.has_separation) {
    out.push_back({"has_separation"_nm = writable::logicals({true})});

    // Convert 0-based indices to 1-based for R
    vec separated_obs_r(result.separated_obs.n_elem);
    for (size_t i = 0; i < result.separated_obs.n_elem; ++i) {
      separated_obs_r(i) = static_cast<double>(result.separated_obs(i) + 1);
    }
    out.push_back({"separated_obs"_nm = as_doubles(separated_obs_r)});

    if (result.separation_support.n_elem > 0) {
      out.push_back(
          {"separation_support"_nm = as_doubles(result.separation_support)});
    }
  }

  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < data.fe_levels.n_elem && data.fe_levels(k).n_elem > 0) {
        writable::strings level_names(data.fe_levels(k).n_elem);
        for (size_t j = 0; j < data.fe_levels(k).n_elem; j++) {
          if (!data.fe_levels(k)(j).empty()) {
            level_names[j] = data.fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < data.fe_names.n_elem && !data.fe_names(k).empty()) {
        fe_list_names[k] = data.fe_names(k);
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;
    out.push_back({"fixed_effects"_nm = fe_list});
  }

  if (params.keep_tx && result.has_tx) {
    out.push_back({"TX"_nm = as_doubles_matrix(result.TX)});
  }

  // Add metadata for R-side post-processing
  add_fe_metadata_to_result(out, data.fe_names, data.fe_levels, data.nobs_used,
                            data.obs_indices, all_valid);

  return out;
}

[[cpp4r::register]] doubles
feglm_offset_fit_(const doubles &eta_r, const doubles &y_r,
                  const doubles &offset_r, const doubles &wt_r,
                  const std::string &family, const list &control,
                  const list &fe_codes) {
  vec eta = as_col(eta_r);
  vec y = as_col(y_r);
  vec offset = as_col(offset_r);
  vec w = as_col(wt_r);

  CapybaraParameters params(control);

  capybara::FlatFEMap fe_map = R_codes_to_FlatFEMap(fe_codes);

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  vec result = capybara::feglm_offset_fit(eta, y, offset, w, family_type,
                                          fe_map, params);

  return as_doubles(result);
}

[[cpp4r::register]] list
fenegbin_fit_(const doubles_matrix<> &X_r, const doubles &y_r,
              const doubles &w_r, const list &fe_cols_r,
              const std::string &link, const doubles &beta_r,
              const doubles &eta_r, const double &init_theta,
              const doubles &offset_r, const list &control) {
  CapybaraParameters params(control);

  // Prepare data: NA removal + FE coding (no cluster for negbin)
  PreparedData data = prepare_raw_data(X_r, y_r, w_r, fe_cols_r, R_NilValue);
  bool all_valid = (data.nobs_used == static_cast<size_t>(y_r.size()));

  // Handle offset: subset to match clean data
  vec offset_vec;
  if (all_valid) {
    offset_vec = as_col(offset_r);
  } else {
    offset_vec.set_size(data.nobs_used);
    for (size_t i = 0; i < data.nobs_used; ++i) {
      offset_vec(i) = static_cast<double>(
          offset_r[static_cast<R_xlen_t>(data.obs_indices(i) - 1)]);
    }
  }

  capybara::InferenceNegBin result = capybara::fenegbin_fit(
      data.X, data.y, data.w, data.fe_map, params, offset_vec, init_theta);

  // Replace collinear coefficients (NaN) with R's NA_REAL in all columns of
  // coef_table
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    uvec collinear_idx = find(collinear_mask);
    for (uword i = 0; i < collinear_idx.n_elem; ++i) {
      uword idx = collinear_idx(i);
      result.coef_table(idx, 0) = NA_REAL; // Estimate
      result.coef_table(idx, 1) = NA_REAL; // Std. Error
      result.coef_table(idx, 2) = NA_REAL; // z value
      result.coef_table(idx, 3) = NA_REAL; // Pr(>|z|)
    }
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "hessian"_nm = as_doubles_matrix(result.hessian),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)}),
       "theta"_nm = writable::doubles({result.theta}),
       "iter.outer"_nm =
           writable::integers({static_cast<int>(result.iter_outer)}),
       "conv_outer"_nm = writable::logicals({result.conv_outer})});

  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(result.fixed_effects.n_elem);
    writable::strings fe_list_names(result.fixed_effects.n_elem);

    for (size_t k = 0; k < result.fixed_effects.n_elem; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < data.fe_levels.n_elem && data.fe_levels(k).n_elem > 0) {
        writable::strings level_names(data.fe_levels(k).n_elem);
        for (size_t j = 0; j < data.fe_levels(k).n_elem; j++) {
          if (!data.fe_levels(k)(j).empty()) {
            level_names[j] = data.fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < data.fe_names.n_elem && !data.fe_names(k).empty()) {
        fe_list_names[k] = data.fe_names(k);
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;
    out.push_back({"fixed_effects"_nm = fe_list});
    out.push_back({"has_fe"_nm = result.has_fe});
  }

  if (result.has_tx) {
    out.push_back({"TX"_nm = as_doubles_matrix(result.TX)});
  }

  // Add metadata for R-side post-processing
  add_fe_metadata_to_result(out, data.fe_names, data.fe_levels, data.nobs_used,
                            data.obs_indices, all_valid);

  return out;
}

[[cpp4r::register]] doubles_matrix<> group_sums_(const doubles_matrix<> &M_r,
                                                 const doubles_matrix<> &w_r,
                                                 const list &jlist) {
  const mat M = as_mat(M_r);
  const vec w = vectorise(as_mat(w_r));

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  vec result = capybara::group_sums(M, w, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_spectral_(const doubles_matrix<> &M_r, const doubles_matrix<> &v_r,
                     const doubles_matrix<> &w_r, const size_t K,
                     const list &jlist) {
  const mat M = as_mat(M_r);
  const vec v = vectorise(as_mat(v_r));
  const vec w = vectorise(as_mat(w_r));

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  vec result = capybara::group_sums_spectral(M, v, w, K, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_var_(const doubles_matrix<> &M_r, const list &jlist) {
  const mat M = as_mat(M_r);

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  mat result = capybara::group_sums_var(M, group_indices);

  return as_doubles_matrix(result);
}

[[cpp4r::register]] doubles_matrix<>
group_sums_cov_(const doubles_matrix<> &M_r, const doubles_matrix<> &N_r,
                const list &jlist) {
  const mat M = as_mat(M_r);
  const mat N = as_mat(N_r);

  const size_t J = jlist.size();
  field<uvec> group_indices(J);

  for (size_t j = 0; j < J; ++j) {
    group_indices(j) =
        R_1based_to_Cpp_0based_indices(as_cpp<integers>(jlist[j]));
  }

  mat result = capybara::group_sums_cov(M, N, group_indices);

  return as_doubles_matrix(result);
}
