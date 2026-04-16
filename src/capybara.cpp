#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) ||                 \
    defined(__ARM_FEATURE_FMA)
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif
#endif

#include <armadillo4r.hpp>

#include <optional>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef CAPYBARA_DEBUG
#include <chrono>
#include <fstream>
#include <sstream>
#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#endif
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

#ifdef CAPYBARA_DEBUG
// Get current memory usage in MB
inline double get_memory_usage_mb() {
#if defined(__linux__)
  // Linux: read from /proc/self/status for RSS (Resident Set Size)
  std::ifstream status_file("/proc/self/status");
  std::string line;
  while (std::getline(status_file, line)) {
    if (line.substr(0, 6) == "VmRSS:") {
      std::istringstream iss(line);
      std::string label;
      long mem_kb;
      std::string unit;
      iss >> label >> mem_kb >> unit;
      return mem_kb / 1024.0; // Convert KB to MB
    }
  }
  return -1.0; // Could not read
#elif defined(__APPLE__)
  // macOS: use getrusage
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // ru_maxrss is in bytes on macOS
    return usage.ru_maxrss / (1024.0 * 1024.0); // Convert bytes to MB
  }
  return -1.0;
#elif defined(_WIN32)
  // Windows: use GetProcessMemoryInfo
  PROCESS_MEMORY_COUNTERS_EX pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc,
                           sizeof(pmc))) {
    return pmc.WorkingSetSize / (1024.0 * 1024.0); // Convert bytes to MB
  }
  return -1.0;
#else
  // Unsupported platform
  return -1.0;
#endif
}
#endif

} // namespace capybara

// Passing parameters from R to C++ functions
// see R/fit_control.R
struct CapybaraParameters {
  double dev_tol;
  double center_tol;
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
  bool return_hessian;

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

  // Average Partial Effects computation
  bool compute_apes;
  size_t ape_n_pop;                // 0 = no finite population correction
  std::string ape_panel_structure; // "classic" or "network"
  std::string ape_sampling_fe;     // "independence" or "unrestricted"
  bool ape_weak_exo;

  // Bias correction (Fernández-Val & Weidner 2016)
  bool compute_bias_corr;
  size_t bias_corr_bandwidth;            // L parameter (0 = strict exogeneity)
  std::string bias_corr_panel_structure; // "classic" or "network"

  explicit CapybaraParameters(const cpp4r::list &control) {
    dev_tol = as_cpp<double>(control["dev_tol"]);
    center_tol = as_cpp<double>(control["center_tol"]);
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

    // Optional: skip returning hessian to save memory (vcov is usually enough)
    SEXP return_hessian_sexp = control["return_hessian"];
    if (return_hessian_sexp != R_NilValue) {
      return_hessian = as_cpp<bool>(return_hessian_sexp);
    } else {
      return_hessian = true; // default: return hessian for backward compat
    }
    step_halving_memory = as_cpp<double>(control["step_halving_memory"]);
    max_step_halving = as_cpp<size_t>(control["max_step_halving"]);
    start_inner_tol = as_cpp<double>(control["start_inner_tol"]);
    grand_acc_period = as_cpp<size_t>(control["grand_acc_period"]);

    // Extract centering method
    SEXP centering_sexp = control["centering"];
    if (centering_sexp != R_NilValue) {
      centering = as_cpp<std::string>(centering_sexp);
    } else {
      centering = "berge";
    }

    // Extract vcov_type (optional string parameter)
    SEXP vcov_type_sexp = control["vcov_type"];
    if (vcov_type_sexp != R_NilValue) {
      vcov_type = as_cpp<std::string>(vcov_type_sexp);
    } else {
      vcov_type = "";
    }

    // Extract compute_apes (optional, default false)
    SEXP compute_apes_sexp = control["compute_apes"];
    if (compute_apes_sexp != R_NilValue) {
      compute_apes = as_cpp<bool>(compute_apes_sexp);
    } else {
      compute_apes = false;
    }

    // Extract APE variance parameters
    SEXP ape_n_pop_sexp = control["ape_n_pop"];
    if (ape_n_pop_sexp != R_NilValue && !Rf_isNull(ape_n_pop_sexp)) {
      ape_n_pop = as_cpp<size_t>(ape_n_pop_sexp);
    } else {
      ape_n_pop = 0; // no finite population correction
    }

    SEXP ape_panel_sexp = control["ape_panel_structure"];
    if (ape_panel_sexp != R_NilValue) {
      ape_panel_structure = as_cpp<std::string>(ape_panel_sexp);
    } else {
      ape_panel_structure = "classic";
    }

    SEXP ape_sampling_sexp = control["ape_sampling_fe"];
    if (ape_sampling_sexp != R_NilValue) {
      ape_sampling_fe = as_cpp<std::string>(ape_sampling_sexp);
    } else {
      ape_sampling_fe = "independence";
    }

    SEXP ape_weak_sexp = control["ape_weak_exo"];
    if (ape_weak_sexp != R_NilValue) {
      ape_weak_exo = as_cpp<bool>(ape_weak_sexp);
    } else {
      ape_weak_exo = false;
    }

    // Extract bias correction parameters
    SEXP compute_bias_corr_sexp = control["compute_bias_corr"];
    if (compute_bias_corr_sexp != R_NilValue) {
      compute_bias_corr = as_cpp<bool>(compute_bias_corr_sexp);
    } else {
      compute_bias_corr = false;
    }

    SEXP bias_corr_bw_sexp = control["bias_corr_bandwidth"];
    if (bias_corr_bw_sexp != R_NilValue) {
      bias_corr_bandwidth = as_cpp<size_t>(bias_corr_bw_sexp);
    } else {
      bias_corr_bandwidth = 0;
    }

    SEXP bias_corr_panel_sexp = control["bias_corr_panel_structure"];
    if (bias_corr_panel_sexp != R_NilValue) {
      bias_corr_panel_structure = as_cpp<std::string>(bias_corr_panel_sexp);
    } else {
      bias_corr_panel_structure = "classic";
    }
  }
};

#include "01_01_center_helpers.h"
#include "01_02_center_acceleration.h"
#include "01_03_center_stammann.h"
#include "01_04_center_berge.h"
#include "01_05_center.h"

#include "02_chol.h"
#include "03_beta.h"
#include "04_alpha.h"

#include "05_01_separation_helpers.h"
#include "05_02_separation_relu.h"
#include "05_03_separation_simplex.h"
#include "05_04_separation.h"

#include "06_01_fit_helpers.h"
#include "06_02_fit_deviance.h"
#include "06_03_fit_links.h"
#include "06_04_fit_separation.h"
#include "06_05_fit_vcov.h"
#include "06_06_fit_sums.h"

#include "07_lm.h"
#include "08_glm.h"
#include "09_negbin.h"

#include "10_formula_parser.h"

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
// Uses direct R INTEGER() pointer access to avoid cpp4r wrapper overhead.
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
    // Direct R INTEGER() access - avoids cpp4r integers wrapper allocation
    SEXP codes_sexp = fe_codes[k];
    const int *codes_ptr = INTEGER(codes_sexp);
    const size_t N = static_cast<size_t>(Rf_xlength(codes_sexp));
    if (k == 0)
      map.n_obs = N;

    // Find max code to determine n_groups, and copy into fe_map in one pass
    map.fe_map[k].resize(N);
    uword *map_k = map.fe_map[k].data();
    uword max_code = 0;
    for (size_t i = 0; i < N; ++i) {
      const uword c = static_cast<uword>(codes_ptr[i]);
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

// Build entity groups for two columns sharing the same IDs
// (e.g. ctry1 = exporter, ctry2 = importer).
// Both field<uvec> are coded with the same shared codebook so that
// entity1_groups(e) and entity2_groups(e) refer to the same entities.
// This is required by the dyadic sandwich cross-terms
// B_12 = sum_e S^1_e (S^2_e)'   [entity1_i = entity2_j]
// which only make sense when index e is consistent across dimensions.
inline void build_aligned_entity_groups(SEXP col1, SEXP col2,
                                        const uvec &keep_indices,
                                        field<uvec> &groups1,
                                        field<uvec> &groups2) {
  const size_t n = keep_indices.n_elem;

  // Build shared codebook from the union of both columns ---
  // This works with a shared SEXP types as code_column: integer (incl. factor),
  // real, character, logical.
  // Treat every value as a string (i.e., 1 -> "1") and
  // use a string-keyed map
  // => low overhead vs sandwich of two separate code_column + map + remap steps
  auto to_string = [](SEXP col, R_xlen_t i) -> std::string {
    switch (TYPEOF(col)) {
    case INTSXP: {
      // Check for factor first
      SEXP lvls = Rf_getAttrib(col, R_LevelsSymbol);
      int v = INTEGER(col)[i];
      if (lvls != R_NilValue)
        return std::string(CHAR(STRING_ELT(lvls, v - 1)));
      return std::to_string(v);
    }
    case REALSXP: {
      char buf[64];
      snprintf(buf, sizeof(buf), "%.15g", REAL(col)[i]);
      return std::string(buf);
    }
    case STRSXP:
      return std::string(CHAR(STRING_ELT(col, i)));
    case LGLSXP:
      return LOGICAL(col)[i] ? "TRUE" : "FALSE";
    default:
      return "";
    }
  };

  // First pass: collect all entity names in encounter order from col1 then
  // col2, building a shared map.
  std::unordered_map<std::string, uword> entity_map;
  entity_map.reserve(512);
  std::vector<std::string> ordered_names; // for consistent ordering

  for (size_t i = 0; i < n; ++i) {
    std::string nm = to_string(col1, keep_indices(i));
    if (entity_map.find(nm) == entity_map.end()) {
      entity_map[nm] = ordered_names.size();
      ordered_names.push_back(nm);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    std::string nm = to_string(col2, keep_indices(i));
    if (entity_map.find(nm) == entity_map.end()) {
      entity_map[nm] = ordered_names.size();
      ordered_names.push_back(nm);
    }
  }

  const size_t n_entities = ordered_names.size();

  // Second pass: assign codes and count
  std::vector<uword> codes1(n), codes2(n);
  std::vector<size_t> cnt1(n_entities, 0), cnt2(n_entities, 0);
  for (size_t i = 0; i < n; ++i) {
    codes1[i] = entity_map[to_string(col1, keep_indices(i))];
    cnt1[codes1[i]]++;
    codes2[i] = entity_map[to_string(col2, keep_indices(i))];
    cnt2[codes2[i]]++;
  }

  // Allocate groups
  groups1.set_size(n_entities);
  groups2.set_size(n_entities);
  for (size_t g = 0; g < n_entities; ++g) {
    groups1(g).set_size(cnt1[g]);
    groups2(g).set_size(cnt2[g]);
  }

  std::vector<size_t> off1(n_entities, 0), off2(n_entities, 0);
  for (size_t i = 0; i < n; ++i) {
    uword g1 = codes1[i];
    groups1(g1)(off1[g1]++) = i;
    uword g2 = codes2[i];
    groups2(g2)(off2[g2]++) = i;
  }
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

  // w (skip if empty — means unit weights)
  const bool has_weights = (w_r.size() > 0);
  if (has_weights) {
    for (size_t i = 0; i < N; ++i) {
      if (valid[i] && !R_finite(static_cast<double>(w_r[i])))
        valid[i] = false;
    }
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
    if (has_weights) {
      out.w = as_col(w_r);
    } else {
      out.w.ones(n_valid);
    }
  } else {
    out.y.set_size(n_valid);
    out.X.set_size(n_valid, p);

    for (size_t i = 0; i < n_valid; ++i) {
      size_t src = keep_0based(i);
      out.y(i) = static_cast<double>(y_r[src]);
    }
    if (has_weights) {
      out.w.set_size(n_valid);
      for (size_t i = 0; i < n_valid; ++i) {
        out.w(i) =
            static_cast<double>(w_r[static_cast<R_xlen_t>(keep_0based(i))]);
      }
    } else {
      out.w.ones(n_valid);
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

[[cpp4r::register]] list felm_fit_(const std::string &formula_str, SEXP df,
                                   const doubles &w_r, const list &control) {
  CapybaraParameters params(control);

  // Build matrices directly from formula + data.frame
  const double *weights_ptr = (w_r.size() > 0) ? REAL(w_r) : nullptr;
  size_t weights_len = w_r.size();

  capybara::FormulaMatrixResult fm = capybara::build_matrix_from_formula(
      formula_str, df, weights_ptr, weights_len);

  if (!fm.valid) {
    Rf_error("Formula error: %s", fm.error.c_str());
  }

  // Build weights vector
  vec w;
  if (weights_ptr != nullptr) {
    w.set_size(fm.keep_idx.n_elem);
    for (size_t i = 0; i < fm.keep_idx.n_elem; ++i) {
      w[i] = weights_ptr[fm.keep_idx[i]];
    }
  } else {
    w.ones(fm.keep_idx.n_elem);
  }

  // Update FE map weights
  if (fm.fe_map.structure_built) {
    fm.fe_map.update_weights(w);
  }

  // Build cluster groups from formula's cluster_vars
  const field<uvec> *cluster_ptr = nullptr;
  field<uvec> cluster_groups;
  field<uvec> entity1_groups, entity2_groups;
  const field<uvec> *entity1_ptr = nullptr;
  const field<uvec> *entity2_ptr = nullptr;

  size_t n_cluster_vars = fm.cluster_vars.size();
  if (n_cluster_vars >= 2 && (params.vcov_type == "m-estimator-dyadic" ||
                              params.vcov_type == "two-way")) {
    // Dyadic or two-way clustering: use first two cluster vars as entities
    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < Rf_length(names); ++i) {
      col_idx[CHAR(STRING_ELT(names, i))] = i;
    }

    SEXP col1 = VECTOR_ELT(df, col_idx[fm.cluster_vars[0]]);
    SEXP col2 = VECTOR_ELT(df, col_idx[fm.cluster_vars[1]]);
    build_aligned_entity_groups(col1, col2, fm.keep_idx, entity1_groups,
                                entity2_groups);
    entity1_ptr = &entity1_groups;
    entity2_ptr = &entity2_groups;
  } else if (n_cluster_vars >= 1) {
    // Single cluster variable
    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    int cl_idx = -1;
    for (int i = 0; i < Rf_length(names); ++i) {
      if (CHAR(STRING_ELT(names, i)) == fm.cluster_vars[0]) {
        cl_idx = i;
        break;
      }
    }
    if (cl_idx >= 0) {
      cluster_groups =
          build_cluster_groups(VECTOR_ELT(df, cl_idx), fm.keep_idx);
      cluster_ptr = &cluster_groups;
    }
  }

  // Fit the model
  capybara::InferenceLM result = capybara::felm_fit(
      fm.X, fm.y, w, fm.fe_map, params, nullptr, cluster_ptr, false, 0.0,
      entity1_ptr, entity2_ptr, fm.has_intercept_column, fm.suppress_intercept);

  // Replace collinear coefficients with NA
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    for (uword i = 0; i < result.coef_table.n_rows; ++i) {
      if (collinear_mask(i)) {
        for (uword j = 0; j < result.coef_table.n_cols; ++j) {
          result.coef_table(i, j) = NA_REAL;
        }
      }
    }
  }

  // Build return list
  auto ret = writable::list(
      {"fitted_values"_nm = as_doubles(result.fitted_values),
       "residuals"_nm = as_doubles(result.residuals),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "r_squared"_nm = result.r_squared,
       "adj_r_squared"_nm = result.adj_r_squared,
       "coef_status"_nm = as_integers(result.coef_status),
       "success"_nm = result.success, "has_fe"_nm = result.has_fe});

  // Conditionally include hessian
  if (params.return_hessian) {
    ret.push_back({"hessian"_nm = as_doubles_matrix(result.hessian)});
  }

  // Add fixed effects if computed
  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    size_t K = fm.fe_map.K;
    writable::list fe_list(K);
    writable::strings fe_list_names(K);

    for (size_t k = 0; k < K; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < fm.fe_levels.n_elem && fm.fe_levels(k).n_elem > 0) {
        writable::strings level_names(fm.fe_levels(k).n_elem);
        for (size_t j = 0; j < fm.fe_levels(k).n_elem; ++j) {
          if (!fm.fe_levels(k)(j).empty()) {
            level_names[j] = fm.fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < fm.fe_names.n_elem && !fm.fe_names(k).empty()) {
        fe_list_names[k] = fm.fe_names(k);
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;
    ret.push_back({"fixed_effects"_nm = fe_list});
  }

  if (!result.iterations.is_empty()) {
    ret.push_back({"iterations"_nm = as_integers(result.iterations)});
  }

  if (params.keep_tx && result.has_tx) {
    ret.push_back({"tx"_nm = as_doubles_matrix(result.TX)});
  }

  // Add term names
  size_t p = fm.term_names.size();
  writable::strings term_names_r(p);
  for (size_t i = 0; i < p; ++i) {
    term_names_r[i] = fm.term_names[i];
  }
  ret.push_back({"term_names"_nm = term_names_r});

  // Add observation indices (1-based for R) and metadata
  writable::integers obs_idx_r(fm.keep_idx.n_elem);
  for (size_t i = 0; i < fm.keep_idx.n_elem; ++i) {
    obs_idx_r[i] = static_cast<int>(fm.keep_idx[i] + 1);
  }
  ret.push_back({"obs_indices"_nm = obs_idx_r});
  ret.push_back({"nobs_used"_nm = writable::integers(
                     {static_cast<int>(fm.keep_idx.n_elem)})});

  // Add FE metadata
  size_t K = fm.fe_names.n_elem;
  writable::list nms_fe_list(K);
  writable::strings nms_fe_names(K);
  for (size_t k = 0; k < K; ++k) {
    writable::strings lvls(fm.fe_levels(k).n_elem);
    for (size_t j = 0; j < fm.fe_levels(k).n_elem; ++j) {
      lvls[j] = fm.fe_levels(k)(j);
    }
    nms_fe_list[k] = lvls;
    nms_fe_names[k] = fm.fe_names(k);
  }
  nms_fe_list.names() = nms_fe_names;
  ret.push_back({"nms_fe"_nm = nms_fe_list});

  writable::integers fe_lvl_counts(K);
  for (size_t k = 0; k < K; ++k) {
    fe_lvl_counts[k] = static_cast<int>(fm.fe_levels(k).n_elem);
  }
  fe_lvl_counts.attr("names") = nms_fe_names;
  ret.push_back({"fe_levels"_nm = fe_lvl_counts});

  return ret;
}

[[cpp4r::register]] list
feglm_fit_(const std::string &formula_str, SEXP df, const doubles &beta_r,
           const doubles &eta_r, const doubles &wt_r, const doubles &offset_r,
           const double &theta, const std::string &family,
           const list &control) {
  CapybaraParameters params(control);

  // Build matrices directly from formula + data.frame
  const double *weights_ptr = (wt_r.size() > 0) ? REAL(wt_r) : nullptr;
  size_t weights_len = wt_r.size();

  capybara::FormulaMatrixResult fm = capybara::build_matrix_from_formula(
      formula_str, df, weights_ptr, weights_len);

  if (!fm.valid) {
    Rf_error("Formula error: %s", fm.error.c_str());
  }

  size_t n_valid = fm.keep_idx.n_elem;
  size_t N_orig =
      static_cast<size_t>(Rf_xlength(VECTOR_ELT(df, 0))); // Original row count
  bool all_valid = (n_valid == N_orig);

  // Build weights vector
  vec w;
  if (weights_ptr != nullptr) {
    w.set_size(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      w[i] = weights_ptr[fm.keep_idx[i]];
    }
  } else {
    w.ones(n_valid);
  }

  // Update FE map weights
  if (fm.fe_map.structure_built) {
    fm.fe_map.update_weights(w);
  }

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  // Handle beta/eta: if provided, subset to match clean data
  vec beta = as_col(beta_r);
  vec eta;
  if (eta_r.size() > 0) {
    if (all_valid) {
      eta = as_col(eta_r);
    } else {
      eta.set_size(n_valid);
      for (size_t i = 0; i < n_valid; ++i) {
        eta(i) =
            static_cast<double>(eta_r[static_cast<R_xlen_t>(fm.keep_idx[i])]);
      }
    }
  } else {
    eta.set_size(0);
  }

  // Handle offset
  vec offset;
  if (offset_r.size() == 0) {
    offset.zeros(n_valid);
  } else if (all_valid) {
    offset = as_col(offset_r);
    uvec bad_offset = find_nonfinite(offset);
    if (bad_offset.n_elem > 0) {
      offset.elem(bad_offset).zeros();
    }
  } else {
    offset.set_size(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      offset(i) =
          static_cast<double>(offset_r[static_cast<R_xlen_t>(fm.keep_idx[i])]);
    }
    uvec bad_offset = find_nonfinite(offset);
    if (bad_offset.n_elem > 0) {
      offset.elem(bad_offset).zeros();
    }
  }

  // Add offset to eta
  if (eta.n_elem > 0) {
    eta += offset;
  }

  // Safety net for non-finite eta values
  if (eta.n_elem > 0) {
    uvec bad_eta = find_nonfinite(eta);
    if (bad_eta.n_elem > 0) {
      double y_mean = mean(fm.y);
      double safe_eta = std::log(y_mean + 0.1);
      if (!std::isfinite(safe_eta))
        safe_eta = 0.0;
      eta.elem(bad_eta).fill(safe_eta);
    }
  }

  const vec *offset_ptr = (any(offset != 0.0)) ? &offset : nullptr;

  // Build cluster groups from formula's cluster_vars
  const field<uvec> *cluster_ptr = nullptr;
  field<uvec> cluster_groups;
  field<uvec> entity1_groups, entity2_groups;
  const field<uvec> *entity1_ptr = nullptr;
  const field<uvec> *entity2_ptr = nullptr;

  size_t n_cluster_vars = fm.cluster_vars.size();
  if (n_cluster_vars >= 2 && (params.vcov_type == "m-estimator-dyadic" ||
                              params.vcov_type == "two-way")) {
    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < Rf_length(names); ++i) {
      col_idx[CHAR(STRING_ELT(names, i))] = i;
    }

    SEXP col1 = VECTOR_ELT(df, col_idx[fm.cluster_vars[0]]);
    SEXP col2 = VECTOR_ELT(df, col_idx[fm.cluster_vars[1]]);
    build_aligned_entity_groups(col1, col2, fm.keep_idx, entity1_groups,
                                entity2_groups);
    entity1_ptr = &entity1_groups;
    entity2_ptr = &entity2_groups;
  } else if (n_cluster_vars >= 1) {
    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    int cl_idx = -1;
    for (int i = 0; i < Rf_length(names); ++i) {
      if (CHAR(STRING_ELT(names, i)) == fm.cluster_vars[0]) {
        cl_idx = i;
        break;
      }
    }
    if (cl_idx >= 0) {
      cluster_groups =
          build_cluster_groups(VECTOR_ELT(df, cl_idx), fm.keep_idx);
      cluster_ptr = &cluster_groups;
    }
  }

  capybara::InferenceGLM result = capybara::feglm_fit(
      beta, eta, fm.y, fm.X, w, theta, family_type, fm.fe_map, params, nullptr,
      cluster_ptr, offset_ptr, false, entity1_ptr, entity2_ptr, false,
      fm.suppress_intercept, fm.has_intercept_column);

  // Replace collinear coefficients with NA
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    for (uword i = 0; i < result.coef_table.n_rows; ++i) {
      if (collinear_mask(i)) {
        for (uword j = 0; j < result.coef_table.n_cols; ++j) {
          result.coef_table(i, j) = NA_REAL;
        }
      }
    }
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)})});

  if (params.return_hessian) {
    out.push_back({"hessian"_nm = as_doubles_matrix(result.hessian)});
  }

  if (family_type == capybara::POISSON && result.pseudo_rsq > 0.0) {
    out.push_back({"pseudo.rsq"_nm = result.pseudo_rsq});
  }

  if (result.has_separation) {
    out.push_back({"has_separation"_nm = writable::logicals({true})});
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
    size_t K = fm.fe_map.K;
    writable::list fe_list(K);
    writable::strings fe_list_names(K);

    for (size_t k = 0; k < K; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < fm.fe_levels.n_elem && fm.fe_levels(k).n_elem > 0) {
        writable::strings level_names(fm.fe_levels(k).n_elem);
        for (size_t j = 0; j < fm.fe_levels(k).n_elem; ++j) {
          if (!fm.fe_levels(k)(j).empty()) {
            level_names[j] = fm.fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < fm.fe_names.n_elem && !fm.fe_names(k).empty()) {
        fe_list_names[k] = fm.fe_names(k);
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;
    out.push_back({"fixed_effects"_nm = fe_list});
  }

  if (params.keep_tx && result.has_tx) {
    out.push_back({"tx"_nm = as_doubles_matrix(result.TX)});
  }

  if (result.has_apes && result.ape_delta.n_elem > 0) {
    out.push_back({"ape_delta"_nm = as_doubles(result.ape_delta)});
    out.push_back({"ape_vcov"_nm = as_doubles_matrix(result.ape_vcov)});
    out.push_back({"ape_binary"_nm = as_integers(result.ape_binary)});
    out.push_back({"has_apes"_nm = writable::logicals({true})});
  }

  if (result.has_bias_corr && result.beta_corrected.n_elem > 0) {
    out.push_back({"beta_corrected"_nm = as_doubles(result.beta_corrected)});
    out.push_back({"bias_term"_nm = as_doubles(result.bias_term)});
    out.push_back({"has_bias_corr"_nm = writable::logicals({true})});
  }

  // Add term names
  size_t p = fm.term_names.size();
  writable::strings term_names_r(p);
  for (size_t i = 0; i < p; ++i) {
    term_names_r[i] = fm.term_names[i];
  }
  out.push_back({"term_names"_nm = term_names_r});

  // Add observation indices and metadata
  writable::integers obs_idx_r(n_valid);
  for (size_t i = 0; i < n_valid; ++i) {
    obs_idx_r[i] = static_cast<int>(fm.keep_idx[i] + 1);
  }
  out.push_back({"obs_indices"_nm = obs_idx_r});
  out.push_back(
      {"nobs_used"_nm = writable::integers({static_cast<int>(n_valid)})});

  // Add FE metadata
  size_t K = fm.fe_names.n_elem;
  writable::list nms_fe_list(K);
  writable::strings nms_fe_names(K);
  for (size_t k = 0; k < K; ++k) {
    writable::strings lvls(fm.fe_levels(k).n_elem);
    for (size_t j = 0; j < fm.fe_levels(k).n_elem; ++j) {
      lvls[j] = fm.fe_levels(k)(j);
    }
    nms_fe_list[k] = lvls;
    nms_fe_names[k] = fm.fe_names(k);
  }
  nms_fe_list.names() = nms_fe_names;
  out.push_back({"nms_fe"_nm = nms_fe_list});

  writable::integers fe_lvl_counts(K);
  for (size_t k = 0; k < K; ++k) {
    fe_lvl_counts[k] = static_cast<int>(fm.fe_levels(k).n_elem);
  }
  fe_lvl_counts.attr("names") = nms_fe_names;
  out.push_back({"fe_levels"_nm = fe_lvl_counts});

  return out;
}

// New function that accepts pre-built design matrix from R
// This handles all complex formula operations (poly, cut, as.factor, etc.)
// in R's model.matrix() and passes the result to C++
[[cpp4r::register]] list feglm_fit_matrix_(
    const doubles_matrix<> &X_r, const doubles &y_r, const doubles &beta_r,
    const doubles &eta_r, const doubles &wt_r, const doubles &offset_r,
    const double &theta, const std::string &family, const strings &term_names_r,
    const strings &fe_vars_r, const strings &cluster_vars_r, SEXP df,
    const bool &has_intercept, const list &control) {
  CapybaraParameters params(control);

  // Convert inputs to Armadillo types
  mat X = as_mat(X_r);
  vec y = as_col(y_r);
  size_t n = y.n_elem;
  size_t p = X.n_cols;

  // Find complete cases (no NA in y or X)
  std::vector<bool> valid(n, true);
  for (size_t i = 0; i < n; ++i) {
    if (!R_finite(y(i)))
      valid[i] = false;
    for (size_t j = 0; j < p && valid[i]; ++j) {
      if (!R_finite(X(i, j)))
        valid[i] = false;
    }
  }

  // Check weights for NA
  const double *weights_ptr = (wt_r.size() > 0) ? REAL(wt_r) : nullptr;
  if (weights_ptr != nullptr) {
    for (size_t i = 0; i < n && i < static_cast<size_t>(wt_r.size()); ++i) {
      if (!R_finite(weights_ptr[i]))
        valid[i] = false;
    }
  }

  // Build keep index
  size_t n_valid = 0;
  for (size_t i = 0; i < n; ++i) {
    if (valid[i])
      n_valid++;
  }

  if (n_valid == 0) {
    Rf_error("No complete cases");
  }

  uvec keep_idx(n_valid);
  size_t j = 0;
  for (size_t i = 0; i < n; ++i) {
    if (valid[i])
      keep_idx[j++] = i;
  }

  bool all_valid = (n_valid == n);

  // Subset y and X if needed
  vec y_clean;
  mat X_clean;
  if (all_valid) {
    y_clean = y;
    X_clean = X;
  } else {
    y_clean.set_size(n_valid);
    X_clean.set_size(n_valid, p);
    for (size_t i = 0; i < n_valid; ++i) {
      y_clean(i) = y(keep_idx(i));
      X_clean.row(i) = X.row(keep_idx(i));
    }
  }

  // Build weights vector
  vec w;
  if (weights_ptr != nullptr) {
    w.set_size(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      w[i] = weights_ptr[keep_idx[i]];
    }
  } else {
    w.ones(n_valid);
  }

  // Copy term names
  std::vector<std::string> term_names;
  for (R_xlen_t i = 0; i < term_names_r.size(); ++i) {
    term_names.push_back(std::string(term_names_r[i]));
  }

  // Build FE map from fe_vars
  capybara::FlatFEMap fe_map;
  field<std::string> fe_names;
  field<field<std::string>> fe_levels;

  size_t K = fe_vars_r.size();
  if (K > 0) {
    fe_names.set_size(K);
    fe_levels.set_size(K);
    fe_map.K = K;
    fe_map.n_obs = n_valid;
    fe_map.n_groups.resize(K);
    fe_map.fe_map.resize(K);

    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < Rf_length(names); ++i) {
      col_idx[CHAR(STRING_ELT(names, i))] = i;
    }

    for (size_t k = 0; k < K; ++k) {
      std::string fe_var(fe_vars_r[k]);
      fe_names(k) = fe_var;

      auto it = col_idx.find(fe_var);
      if (it == col_idx.end()) {
        Rf_error("FE variable not found: %s", fe_var.c_str());
      }
      SEXP col = VECTOR_ELT(df, it->second);

      // Code the column
      std::unordered_map<std::string, uword> level_map;
      std::vector<std::string> levels;

      auto val_to_string = [](SEXP col, R_xlen_t i) -> std::string {
        switch (TYPEOF(col)) {
        case INTSXP: {
          int v = INTEGER(col)[i];
          SEXP lvls = Rf_getAttrib(col, R_LevelsSymbol);
          if (lvls != R_NilValue) {
            return CHAR(STRING_ELT(lvls, v - 1));
          }
          return std::to_string(v);
        }
        case REALSXP: {
          char buf[64];
          snprintf(buf, sizeof(buf), "%.15g", REAL(col)[i]);
          return std::string(buf);
        }
        case STRSXP:
          return CHAR(STRING_ELT(col, i));
        default:
          return "";
        }
      };

      fe_map.fe_map[k].resize(n_valid);
      for (size_t i = 0; i < n_valid; ++i) {
        R_xlen_t orig_i = keep_idx[i];
        std::string val = val_to_string(col, orig_i);
        auto it2 = level_map.find(val);
        if (it2 == level_map.end()) {
          uword code = static_cast<uword>(levels.size());
          level_map[val] = code;
          levels.push_back(val);
          fe_map.fe_map[k][i] = code;
        } else {
          fe_map.fe_map[k][i] = it2->second;
        }
      }

      fe_map.n_groups[k] = levels.size();
      fe_levels(k).set_size(levels.size());
      for (size_t l = 0; l < levels.size(); ++l) {
        fe_levels(k)(l) = levels[l];
      }
    }
    fe_map.structure_built = true;
    fe_map.update_weights(w);
  }

  std::string fam = capybara::tidy_family(family);
  capybara::Family family_type = capybara::get_family_type(fam);

  // Handle beta/eta
  vec beta = as_col(beta_r);
  vec eta;
  if (eta_r.size() > 0) {
    if (all_valid) {
      eta = as_col(eta_r);
    } else {
      eta.set_size(n_valid);
      for (size_t i = 0; i < n_valid; ++i) {
        eta(i) = static_cast<double>(eta_r[static_cast<R_xlen_t>(keep_idx[i])]);
      }
    }
  } else {
    eta.set_size(0);
  }

  // Handle offset
  vec offset;
  if (offset_r.size() == 0) {
    offset.zeros(n_valid);
  } else if (all_valid) {
    offset = as_col(offset_r);
    uvec bad_offset = find_nonfinite(offset);
    if (bad_offset.n_elem > 0) {
      offset.elem(bad_offset).zeros();
    }
  } else {
    offset.set_size(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      offset(i) =
          static_cast<double>(offset_r[static_cast<R_xlen_t>(keep_idx[i])]);
    }
    uvec bad_offset = find_nonfinite(offset);
    if (bad_offset.n_elem > 0) {
      offset.elem(bad_offset).zeros();
    }
  }

  if (eta.n_elem > 0) {
    eta += offset;
  }

  if (eta.n_elem > 0) {
    uvec bad_eta = find_nonfinite(eta);
    if (bad_eta.n_elem > 0) {
      double y_mean = mean(y_clean);
      double safe_eta = std::log(y_mean + 0.1);
      if (!std::isfinite(safe_eta))
        safe_eta = 0.0;
      eta.elem(bad_eta).fill(safe_eta);
    }
  }

  const vec *offset_ptr = (any(offset != 0.0)) ? &offset : nullptr;

  // Build cluster groups
  const field<uvec> *cluster_ptr = nullptr;
  field<uvec> cluster_groups;
  field<uvec> entity1_groups, entity2_groups;
  const field<uvec> *entity1_ptr = nullptr;
  const field<uvec> *entity2_ptr = nullptr;

  size_t n_cluster_vars = cluster_vars_r.size();
  if (n_cluster_vars >= 2 && (params.vcov_type == "m-estimator-dyadic" ||
                              params.vcov_type == "two-way")) {
    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < Rf_length(names); ++i) {
      col_idx[CHAR(STRING_ELT(names, i))] = i;
    }

    SEXP col1 = VECTOR_ELT(df, col_idx[std::string(cluster_vars_r[0])]);
    SEXP col2 = VECTOR_ELT(df, col_idx[std::string(cluster_vars_r[1])]);
    build_aligned_entity_groups(col1, col2, keep_idx, entity1_groups,
                                entity2_groups);
    entity1_ptr = &entity1_groups;
    entity2_ptr = &entity2_groups;
  } else if (n_cluster_vars >= 1) {
    SEXP names = Rf_getAttrib(df, R_NamesSymbol);
    int cl_idx = -1;
    for (int i = 0; i < Rf_length(names); ++i) {
      if (CHAR(STRING_ELT(names, i)) == std::string(cluster_vars_r[0])) {
        cl_idx = i;
        break;
      }
    }
    if (cl_idx >= 0) {
      cluster_groups = build_cluster_groups(VECTOR_ELT(df, cl_idx), keep_idx);
      cluster_ptr = &cluster_groups;
    }
  }

  // Determine if intercept should be suppressed
  bool suppress_intercept = !has_intercept;

  capybara::InferenceGLM result = capybara::feglm_fit(
      beta, eta, y_clean, X_clean, w, theta, family_type, fe_map, params,
      nullptr, cluster_ptr, offset_ptr, false, entity1_ptr, entity2_ptr, false,
      suppress_intercept, has_intercept);

  // Replace collinear coefficients with NA
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    for (uword i = 0; i < result.coef_table.n_rows; ++i) {
      if (collinear_mask(i)) {
        for (uword kk = 0; kk < result.coef_table.n_cols; ++kk) {
          result.coef_table(i, kk) = NA_REAL;
        }
      }
    }
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)})});

  if (params.return_hessian) {
    out.push_back({"hessian"_nm = as_doubles_matrix(result.hessian)});
  }

  if (family_type == capybara::POISSON && result.pseudo_rsq > 0.0) {
    out.push_back({"pseudo.rsq"_nm = result.pseudo_rsq});
  }

  if (result.has_separation) {
    out.push_back({"has_separation"_nm = writable::logicals({true})});
    vec separated_obs_r2(result.separated_obs.n_elem);
    for (size_t i = 0; i < result.separated_obs.n_elem; ++i) {
      separated_obs_r2(i) = static_cast<double>(result.separated_obs(i) + 1);
    }
    out.push_back({"separated_obs"_nm = as_doubles(separated_obs_r2)});
    if (result.separation_support.n_elem > 0) {
      out.push_back(
          {"separation_support"_nm = as_doubles(result.separation_support)});
    }
  }

  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    writable::list fe_list(K);
    writable::strings fe_list_names(K);

    for (size_t k = 0; k < K; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < fe_levels.n_elem && fe_levels(k).n_elem > 0) {
        writable::strings level_names(fe_levels(k).n_elem);
        for (size_t l = 0; l < fe_levels(k).n_elem; ++l) {
          if (!fe_levels(k)(l).empty()) {
            level_names[l] = fe_levels(k)(l);
          } else {
            level_names[l] = std::to_string(l + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < fe_names.n_elem && !fe_names(k).empty()) {
        fe_list_names[k] = fe_names(k);
      } else {
        fe_list_names[k] = std::to_string(k + 1);
      }
    }

    fe_list.names() = fe_list_names;
    out.push_back({"fixed_effects"_nm = fe_list});
  }

  if (params.keep_tx && result.has_tx) {
    out.push_back({"tx"_nm = as_doubles_matrix(result.TX)});
  }

  if (result.has_apes && result.ape_delta.n_elem > 0) {
    out.push_back({"ape_delta"_nm = as_doubles(result.ape_delta)});
    out.push_back({"ape_vcov"_nm = as_doubles_matrix(result.ape_vcov)});
    out.push_back({"ape_binary"_nm = as_integers(result.ape_binary)});
    out.push_back({"has_apes"_nm = writable::logicals({true})});
  }

  if (result.has_bias_corr && result.beta_corrected.n_elem > 0) {
    out.push_back({"beta_corrected"_nm = as_doubles(result.beta_corrected)});
    out.push_back({"bias_term"_nm = as_doubles(result.bias_term)});
    out.push_back({"has_bias_corr"_nm = writable::logicals({true})});
  }

  // Add term names
  writable::strings term_names_out(p);
  for (size_t i = 0; i < p; ++i) {
    if (i < term_names.size()) {
      term_names_out[i] = term_names[i];
    } else {
      term_names_out[i] = "V" + std::to_string(i + 1);
    }
  }
  out.push_back({"term_names"_nm = term_names_out});

  // Add observation indices and metadata
  writable::integers obs_idx_r2(n_valid);
  for (size_t i = 0; i < n_valid; ++i) {
    obs_idx_r2[i] = static_cast<int>(keep_idx[i] + 1);
  }
  out.push_back({"obs_indices"_nm = obs_idx_r2});
  out.push_back(
      {"nobs_used"_nm = writable::integers({static_cast<int>(n_valid)})});

  // Add FE metadata
  writable::list nms_fe_list(K);
  writable::strings nms_fe_names(K);
  for (size_t k = 0; k < K; ++k) {
    writable::strings lvls(fe_levels(k).n_elem);
    for (size_t l = 0; l < fe_levels(k).n_elem; ++l) {
      lvls[l] = fe_levels(k)(l);
    }
    nms_fe_list[k] = lvls;
    nms_fe_names[k] = fe_names(k);
  }
  nms_fe_list.names() = nms_fe_names;
  out.push_back({"nms_fe"_nm = nms_fe_list});

  writable::integers fe_lvl_counts(K);
  for (size_t k = 0; k < K; ++k) {
    fe_lvl_counts[k] = static_cast<int>(fe_levels(k).n_elem);
  }
  fe_lvl_counts.attr("names") = nms_fe_names;
  out.push_back({"fe_levels"_nm = fe_lvl_counts});

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

  return as_doubles(capybara::feglm_offset_fit(eta, y, offset, w, family_type,
                                               fe_map, params));
}

[[cpp4r::register]] list
fenegbin_fit_(const std::string &formula_str, SEXP df, const doubles &w_r,
              const std::string &link, const doubles &beta_r,
              const doubles &eta_r, const double &init_theta,
              const doubles &offset_r, const list &control) {
  CapybaraParameters params(control);

  // Build matrices directly from formula + data.frame
  const double *weights_ptr = (w_r.size() > 0) ? REAL(w_r) : nullptr;
  size_t weights_len = w_r.size();

  capybara::FormulaMatrixResult fm = capybara::build_matrix_from_formula(
      formula_str, df, weights_ptr, weights_len);

  if (!fm.valid) {
    Rf_error("Formula error: %s", fm.error.c_str());
  }

  size_t n_valid = fm.keep_idx.n_elem;
  size_t N_orig = static_cast<size_t>(Rf_xlength(VECTOR_ELT(df, 0)));
  bool all_valid = (n_valid == N_orig);

  // Build weights vector
  vec w;
  if (weights_ptr != nullptr) {
    w.set_size(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      w[i] = weights_ptr[fm.keep_idx[i]];
    }
  } else {
    w.ones(n_valid);
  }

  // Update FE map weights
  if (fm.fe_map.structure_built) {
    fm.fe_map.update_weights(w);
  }

  // Handle offset
  vec offset_vec;
  if (offset_r.size() == 0) {
    offset_vec.zeros(n_valid);
  } else if (all_valid) {
    offset_vec = as_col(offset_r);
  } else {
    offset_vec.set_size(n_valid);
    for (size_t i = 0; i < n_valid; ++i) {
      offset_vec(i) =
          static_cast<double>(offset_r[static_cast<R_xlen_t>(fm.keep_idx[i])]);
    }
  }

  capybara::InferenceNegBin result = capybara::fenegbin_fit(
      fm.X, fm.y, w, fm.fe_map, params, offset_vec, init_theta, nullptr,
      fm.suppress_intercept, fm.has_intercept_column);

  // Replace collinear coefficients with NA
  uvec collinear_mask = (result.coef_status == 0);
  if (any(collinear_mask)) {
    for (uword i = 0; i < result.coef_table.n_rows; ++i) {
      if (collinear_mask(i)) {
        for (uword j = 0; j < result.coef_table.n_cols; ++j) {
          result.coef_table(i, j) = NA_REAL;
        }
      }
    }
  }

  auto out = writable::list(
      {"eta"_nm = as_doubles(result.eta),
       "fitted_values"_nm = as_doubles(result.fitted_values),
       "weights"_nm = as_doubles(result.weights),
       "vcov"_nm = as_doubles_matrix(result.vcov),
       "coef_table"_nm = as_doubles_matrix(result.coef_table),
       "deviance"_nm = writable::doubles({result.deviance}),
       "null_deviance"_nm = writable::doubles({result.null_deviance}),
       "conv"_nm = writable::logicals({result.conv}),
       "iter"_nm = writable::integers({static_cast<int>(result.iter + 1)}),
       "theta"_nm = writable::doubles({result.theta}),
       "iter.outer"_nm =
           writable::integers({static_cast<int>(result.iter_outer)}),
       "conv_outer"_nm = writable::logicals({result.conv_outer})});

  if (params.return_hessian) {
    out.push_back({"hessian"_nm = as_doubles_matrix(result.hessian)});
  }

  if (result.has_fe && result.fixed_effects.n_elem > 0) {
    size_t K = fm.fe_map.K;
    writable::list fe_list(K);
    writable::strings fe_list_names(K);

    for (size_t k = 0; k < K; ++k) {
      writable::doubles fe_values = as_doubles(result.fixed_effects(k));

      if (k < fm.fe_levels.n_elem && fm.fe_levels(k).n_elem > 0) {
        writable::strings level_names(fm.fe_levels(k).n_elem);
        for (size_t j = 0; j < fm.fe_levels(k).n_elem; ++j) {
          if (!fm.fe_levels(k)(j).empty()) {
            level_names[j] = fm.fe_levels(k)(j);
          } else {
            level_names[j] = std::to_string(j + 1);
          }
        }
        fe_values.attr("names") = level_names;
      }

      fe_list[k] = fe_values;

      if (k < fm.fe_names.n_elem && !fm.fe_names(k).empty()) {
        fe_list_names[k] = fm.fe_names(k);
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

  // Add term names
  size_t p = fm.term_names.size();
  writable::strings term_names_r(p);
  for (size_t i = 0; i < p; ++i) {
    term_names_r[i] = fm.term_names[i];
  }
  out.push_back({"term_names"_nm = term_names_r});

  // Add observation indices and metadata
  writable::integers obs_idx_r(n_valid);
  for (size_t i = 0; i < n_valid; ++i) {
    obs_idx_r[i] = static_cast<int>(fm.keep_idx[i] + 1);
  }
  out.push_back({"obs_indices"_nm = obs_idx_r});
  out.push_back(
      {"nobs_used"_nm = writable::integers({static_cast<int>(n_valid)})});

  // Add FE metadata
  size_t K = fm.fe_names.n_elem;
  writable::list nms_fe_list(K);
  writable::strings nms_fe_names(K);
  for (size_t k = 0; k < K; ++k) {
    writable::strings lvls(fm.fe_levels(k).n_elem);
    for (size_t j = 0; j < fm.fe_levels(k).n_elem; ++j) {
      lvls[j] = fm.fe_levels(k)(j);
    }
    nms_fe_list[k] = lvls;
    nms_fe_names[k] = fm.fe_names(k);
  }
  nms_fe_list.names() = nms_fe_names;
  out.push_back({"nms_fe"_nm = nms_fe_list});

  writable::integers fe_lvl_counts(K);
  for (size_t k = 0; k < K; ++k) {
    fe_lvl_counts[k] = static_cast<int>(fm.fe_levels(k).n_elem);
  }
  fe_lvl_counts.attr("names") = nms_fe_names;
  out.push_back({"fe_levels"_nm = fe_lvl_counts});

  return out;
}
