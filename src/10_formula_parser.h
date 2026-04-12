// Formula parsing and direct Armadillo matrix building from R data.frame
// No R fallback - everything handled in C++
//
// Supported formula syntax:
//   y ~ a + b + c                    - plain columns
//   log(y) ~ log(a) + sqrt(b)        - transforms
//   y ~ I(a*b) + I(a/b + c)          - arithmetic expressions
//   y ~ poly(x, 3)                   - polynomial expansion
//   y ~ a:b                          - interaction
//   y ~ a*b                          - expands to a + b + a:b
//   y ~ factor(x)                    - dummy coding
//   y ~ x | fe1 + fe2 | cluster      - fixed effects and clusters

#ifndef CAPYBARA_FORMULA_PARSER_H
#define CAPYBARA_FORMULA_PARSER_H

#include <cmath>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace arma;

namespace capybara {

// ============================================================
// Transform types
// ============================================================

enum class Transform {
  NONE,       // plain column
  LOG,        // log(x)
  LOG1P,      // log1p(x) = log(1+x)
  SQRT,       // sqrt(x)
  EXP,        // exp(x)
  ABS,        // abs(x)
  SQUARE,     // x^2
  CUBE,       // x^3
  POLY,       // poly(x, degree) - orthogonal
  POLY_RAW,   // poly(x, degree, raw = TRUE) - raw powers
  IDENTITY,   // I(expr)
  INTERACT,   // a:b
  FACTOR,     // factor(x)
  AS_FACTOR,  // as.factor(x)
  CUT         // cut(x, breaks = n)
};

// ============================================================
// Parsed term structure
// ============================================================

struct ParsedTerm {
  std::string raw;                   // Original: "log(x)"
  std::string column;                // Primary column: "x"
  std::vector<std::string> columns;  // For interactions/I(): ["a", "b"]
  Transform transform = Transform::NONE;
  int poly_degree = 1;
  int cut_breaks = 3;                // For cut(): number of breaks
  bool poly_raw = false;             // For poly(): raw = TRUE
  std::string identity_expr;         // For I(): "a*b + c"

  size_t base_cols() const {
    if (transform == Transform::POLY || transform == Transform::POLY_RAW)
      return static_cast<size_t>(poly_degree);
    if (transform == Transform::FACTOR || transform == Transform::AS_FACTOR ||
        transform == Transform::CUT)
      return 0;  // determined at runtime
    return 1;
  }
};

// ============================================================
// Full parsed formula
// ============================================================

struct ParsedFormula {
  ParsedTerm response;
  std::vector<ParsedTerm> terms;
  std::vector<std::string> fe_vars;
  std::vector<std::string> cluster_vars;
  std::vector<std::string> all_columns;  // All referenced column names
  bool valid = true;
  std::string error;
  bool suppress_intercept = false;  // True when __NO_INTERCEPT__ marker present
};

// ============================================================
// String utilities
// ============================================================

inline std::string str_trim(const std::string &s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos)
    return "";
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

inline std::vector<std::string> str_split_top_level(const std::string &s,
                                                    char delim) {
  std::vector<std::string> parts;
  int paren_depth = 0;
  size_t start = 0;

  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '(')
      paren_depth++;
    else if (c == ')')
      paren_depth--;
    else if (c == delim && paren_depth == 0) {
      parts.push_back(str_trim(s.substr(start, i - start)));
      start = i + 1;
    }
  }
  parts.push_back(str_trim(s.substr(start)));
  return parts;
}

inline bool str_starts_with(const std::string &s, const std::string &prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

// ============================================================
// Safe numeric column access (handles REALSXP and INTSXP)
// ============================================================

// Get numeric values from column, converting integers if needed
// Copies into dst, applying keep_idx filter
inline void copy_numeric_column(SEXP col, const uvec &keep_idx, double *dst) {
  size_t n = keep_idx.n_elem;
  if (TYPEOF(col) == REALSXP) {
    const double *src = REAL(col);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = src[keep_idx[i]];
    }
  } else if (TYPEOF(col) == INTSXP) {
    const int *src = INTEGER(col);
    for (size_t i = 0; i < n; ++i) {
      int val = src[keep_idx[i]];
      dst[i] = (val == NA_INTEGER) ? NA_REAL : static_cast<double>(val);
    }
  } else {
    // Fill with NA for unsupported types
    for (size_t i = 0; i < n; ++i) {
      dst[i] = NA_REAL;
    }
  }
}

// Get pointer to raw REALSXP data, or nullptr if not REALSXP
// For use with expression evaluator (which converts ints separately)
inline const double *get_real_ptr_or_null(SEXP col) {
  return (TYPEOF(col) == REALSXP) ? REAL(col) : nullptr;
}

// Get integer pointer or nullptr
inline const int *get_int_ptr_or_null(SEXP col) {
  return (TYPEOF(col) == INTSXP) ? INTEGER(col) : nullptr;
}

// Copy numeric column without index (for full column access)
inline void copy_numeric_column_full(SEXP col, size_t n, double *dst) {
  if (TYPEOF(col) == REALSXP) {
    const double *src = REAL(col);
    std::memcpy(dst, src, n * sizeof(double));
  } else if (TYPEOF(col) == INTSXP) {
    const int *src = INTEGER(col);
    for (size_t i = 0; i < n; ++i) {
      int val = src[i];
      dst[i] = (val == NA_INTEGER) ? NA_REAL : static_cast<double>(val);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = NA_REAL;
    }
  }
}

// Get a single numeric value from a column at index i
inline double get_numeric_value(SEXP col, size_t i) {
  if (TYPEOF(col) == REALSXP) {
    return REAL(col)[i];
  } else if (TYPEOF(col) == INTSXP) {
    int val = INTEGER(col)[i];
    return (val == NA_INTEGER) ? NA_REAL : static_cast<double>(val);
  }
  return NA_REAL;
}

// Extract variable names from expression (for I() and interactions)
inline std::vector<std::string> extract_var_names(const std::string &expr) {
  std::vector<std::string> vars;
  std::unordered_set<std::string> seen;

  // Match valid C/R identifiers
  std::regex var_re("[a-zA-Z_][a-zA-Z0-9_\\.]*");
  std::sregex_iterator it(expr.begin(), expr.end(), var_re);
  std::sregex_iterator end;

  // Reserved words to skip
  static const std::unordered_set<std::string> reserved = {
      "log", "log1p", "sqrt", "exp", "abs", "poly", "factor", "I", "TRUE",
      "FALSE", "NA", "NaN", "Inf"};

  while (it != end) {
    std::string var = it->str();
    if (reserved.find(var) == reserved.end() && seen.find(var) == seen.end()) {
      vars.push_back(var);
      seen.insert(var);
    }
    ++it;
  }
  return vars;
}

// ============================================================
// Parse a single term
// ============================================================

inline ParsedTerm parse_term(const std::string &raw) {
  ParsedTerm t;
  t.raw = raw;
  std::string s = str_trim(raw);

  // Check for interaction: a:b (but not inside I())
  if (s.find(':') != std::string::npos && !str_starts_with(s, "I(")) {
    t.transform = Transform::INTERACT;
    t.columns = str_split_top_level(s, ':');
    return t;
  }

  // Check for I(...) - identity/arithmetic
  if (str_starts_with(s, "I(") && s.back() == ')') {
    t.transform = Transform::IDENTITY;
    t.identity_expr = s.substr(2, s.size() - 3);
    t.columns = extract_var_names(t.identity_expr);
    return t;
  }

  // Check for poly() with possible raw = TRUE argument
  if (str_starts_with(s, "poly(") && s.back() == ')') {
    std::string inner = s.substr(5, s.size() - 6);
    auto args = str_split_top_level(inner, ',');
    t.column = str_trim(args[0]);
    t.poly_degree = 1;
    t.poly_raw = false;
    
    for (size_t i = 1; i < args.size(); ++i) {
      std::string arg = str_trim(args[i]);
      // Check for raw = TRUE or raw=TRUE
      if (arg.find("raw") != std::string::npos && 
          (arg.find("TRUE") != std::string::npos || arg.find("T") != std::string::npos)) {
        t.poly_raw = true;
      } else {
        // Try to parse as degree
        try {
          // Remove any "degree =" prefix
          size_t eq = arg.find('=');
          std::string val = (eq != std::string::npos) ? str_trim(arg.substr(eq + 1)) : arg;
          t.poly_degree = std::stoi(val);
        } catch (...) {}
      }
    }
    t.transform = t.poly_raw ? Transform::POLY_RAW : Transform::POLY;
    return t;
  }

  // Check for cut() function
  if (str_starts_with(s, "cut(") && s.back() == ')') {
    std::string inner = s.substr(4, s.size() - 5);
    auto args = str_split_top_level(inner, ',');
    t.column = str_trim(args[0]);
    t.cut_breaks = 3;  // default
    
    for (size_t i = 1; i < args.size(); ++i) {
      std::string arg = str_trim(args[i]);
      // Check for breaks = N
      if (arg.find("breaks") != std::string::npos) {
        size_t eq = arg.find('=');
        if (eq != std::string::npos) {
          try {
            t.cut_breaks = std::stoi(str_trim(arg.substr(eq + 1)));
          } catch (...) {}
        }
      } else {
        // Just a number - assume it's breaks
        try {
          t.cut_breaks = std::stoi(arg);
        } catch (...) {}
      }
    }
    t.transform = Transform::CUT;
    return t;
  }

  // Check for function transforms: func(col) or func(col, arg)
  auto try_parse_func = [&](const std::string &prefix, Transform tf) -> bool {
    if (str_starts_with(s, prefix) && s.back() == ')') {
      std::string inner = s.substr(prefix.size(), s.size() - prefix.size() - 1);
      auto args = str_split_top_level(inner, ',');
      t.column = str_trim(args[0]);
      t.transform = tf;
      return true;
    }
    return false;
  };

  if (try_parse_func("log1p(", Transform::LOG1P))
    return t;
  if (try_parse_func("log(", Transform::LOG))
    return t;
  if (try_parse_func("sqrt(", Transform::SQRT))
    return t;
  if (try_parse_func("exp(", Transform::EXP))
    return t;
  if (try_parse_func("abs(", Transform::ABS))
    return t;
  if (try_parse_func("factor(", Transform::FACTOR))
    return t;
  if (try_parse_func("as.factor(", Transform::AS_FACTOR))
    return t;

  // Check for power: x^2, x^3
  size_t caret = s.find('^');
  if (caret != std::string::npos) {
    t.column = str_trim(s.substr(0, caret));
    int power = std::stoi(str_trim(s.substr(caret + 1)));
    if (power == 2) {
      t.transform = Transform::SQUARE;
    } else if (power == 3) {
      t.transform = Transform::CUBE;
    } else {
      t.transform = Transform::POLY;
      t.poly_degree = power;
    }
    return t;
  }

  // Plain column name
  t.column = s;
  t.transform = Transform::NONE;
  return t;
}

// ============================================================
// Parse full formula string
// ============================================================

inline ParsedFormula parse_formula(const std::string &formula) {
  ParsedFormula pf;
  std::unordered_set<std::string> all_cols_set;

  auto add_col = [&](const std::string &col) {
    if (!col.empty() && all_cols_set.find(col) == all_cols_set.end()) {
      pf.all_columns.push_back(col);
      all_cols_set.insert(col);
    }
  };

  // Split on ~
  size_t tilde = formula.find('~');
  if (tilde == std::string::npos) {
    pf.valid = false;
    pf.error = "Formula must contain ~";
    return pf;
  }

  std::string lhs = str_trim(formula.substr(0, tilde));
  std::string rhs = str_trim(formula.substr(tilde + 1));

  // Parse response
  pf.response = parse_term(lhs);
  if (!pf.response.column.empty())
    add_col(pf.response.column);
  for (const auto &c : pf.response.columns)
    add_col(c);

  // Split RHS on | for multi-part formula
  auto parts = str_split_top_level(rhs, '|');

  // Part 0: model terms
  auto terms = str_split_top_level(parts[0], '+');
  for (const auto &term_str : terms) {
    if (term_str.empty())
      continue;

    // Check for intercept suppression marker
    if (term_str == "__NO_INTERCEPT__") {
      pf.suppress_intercept = true;
      continue;
    }

    // Helper to extract actual column names from a term
    // (handles transforms like log(x) -> x)
    auto add_cols_from_term = [&](const ParsedTerm &pt) {
      if (pt.transform == Transform::INTERACT) {
        // For interactions, recursively extract columns from each component
        for (const auto &c : pt.columns) {
          ParsedTerm sub = parse_term(c);
          if (!sub.column.empty()) {
            add_col(sub.column);
          }
          for (const auto &sc : sub.columns) {
            add_col(sc);
          }
        }
      } else {
        if (!pt.column.empty()) {
          add_col(pt.column);
        }
        for (const auto &c : pt.columns) {
          add_col(c);
        }
      }
    };

    // Handle * expansion: a*b -> a + b + a:b
    if (term_str.find('*') != std::string::npos &&
        !str_starts_with(term_str, "I(")) {
      auto vars = str_split_top_level(term_str, '*');
      // Add main effects
      for (const auto &v : vars) {
        ParsedTerm pt = parse_term(v);
        pf.terms.push_back(pt);
        add_cols_from_term(pt);
      }
      // Add interaction
      std::string interact = vars[0];
      for (size_t i = 1; i < vars.size(); ++i) {
        interact += ":" + vars[i];
      }
      ParsedTerm pt = parse_term(interact);
      pf.terms.push_back(pt);
      add_cols_from_term(pt);
    } else {
      ParsedTerm pt = parse_term(term_str);
      pf.terms.push_back(pt);
      add_cols_from_term(pt);
    }
  }

  // Part 1: fixed effects
  if (parts.size() > 1) {
    auto fe = str_split_top_level(parts[1], '+');
    for (const auto &f : fe) {
      if (!f.empty()) {
        pf.fe_vars.push_back(f);
        add_col(f);
      }
    }
  }

  // Part 2: cluster
  if (parts.size() > 2) {
    auto cl = str_split_top_level(parts[2], '+');
    for (const auto &c : cl) {
      if (!c.empty()) {
        pf.cluster_vars.push_back(c);
        add_col(c);
      }
    }
  }

  return pf;
}

// ============================================================
// Apply element-wise transforms
// ============================================================

inline void apply_transform_inplace(double *data, size_t n, Transform tf) {
  switch (tf) {
  case Transform::LOG:
    for (size_t i = 0; i < n; ++i)
      data[i] = std::log(data[i]);
    break;
  case Transform::LOG1P:
    for (size_t i = 0; i < n; ++i)
      data[i] = std::log1p(data[i]);
    break;
  case Transform::SQRT:
    for (size_t i = 0; i < n; ++i)
      data[i] = std::sqrt(data[i]);
    break;
  case Transform::EXP:
    for (size_t i = 0; i < n; ++i)
      data[i] = std::exp(data[i]);
    break;
  case Transform::ABS:
    for (size_t i = 0; i < n; ++i)
      data[i] = std::abs(data[i]);
    break;
  case Transform::SQUARE:
    for (size_t i = 0; i < n; ++i)
      data[i] = data[i] * data[i];
    break;
  case Transform::CUBE:
    for (size_t i = 0; i < n; ++i)
      data[i] = data[i] * data[i] * data[i];
    break;
  default:
    break;
  }
}

// ============================================================
// Orthogonal polynomial expansion (matches R's poly() exactly)
// Uses QR decomposition like R's make.poly()
// ============================================================

inline mat orthogonal_poly(const vec &x, int degree) {
  size_t n = x.n_elem;
  mat result(n, degree);
  
  if (degree < 1 || n == 0) {
    return result;
  }

  // R's poly() algorithm:
  // 1. Center x by subtracting mean
  // 2. Build raw polynomial matrix [1, xc, xc^2, ..., xc^degree]
  // 3. QR decomposition
  // 4. Ensure diagonal of R is positive (flip Q column signs if needed)
  // 5. Drop first column (constant)
  // 6. Normalize each column so sum(col^2) = 1
  
  double mean_x = arma::mean(x);
  vec xc = x - mean_x;
  
  // Build raw polynomial matrix (Vandermonde-like)
  mat X(n, degree + 1);
  X.col(0).ones();  // x^0 = 1
  for (int d = 1; d <= degree; ++d) {
    X.col(d) = arma::pow(xc, d);
  }
  
  // QR decomposition
  mat Q, R;
  arma::qr_econ(Q, R, X);
  
  // Ensure diagonal of R is positive (R's sign convention)
  // This makes the result deterministic and matches R's output
  for (int j = 0; j <= degree; ++j) {
    if (R(j, j) < 0) {
      Q.col(j) = -Q.col(j);
    }
  }
  
  // Drop first column (constant) and normalize
  for (int d = 0; d < degree; ++d) {
    vec col = Q.col(d + 1);  // Skip first column
    double norm = std::sqrt(arma::dot(col, col));
    if (norm > 1e-10) {
      result.col(d) = col / norm;
    } else {
      result.col(d) = col;
    }
  }
  
  return result;
}

// ============================================================
// Raw polynomial expansion (like R's poly(x, raw = TRUE))
// ============================================================

inline mat raw_poly(const vec &x, int degree) {
  size_t n = x.n_elem;
  mat result(n, degree);
  
  for (int d = 0; d < degree; ++d) {
    result.col(d) = arma::pow(x, d + 1);
  }
  
  return result;
}

// ============================================================
// Cut function (like R's cut())
// Bins continuous variable into equal-width intervals
// ============================================================

struct CutInfo {
  mat dummies;
  std::vector<std::string> level_names;
};

inline CutInfo cut_dummies(SEXP col, const uvec &keep_idx, int n_breaks) {
  CutInfo info;
  size_t n = keep_idx.n_elem;
  
  // Extract numeric values
  vec values(n);
  if (TYPEOF(col) == REALSXP) {
    const double *src = REAL(col);
    for (size_t i = 0; i < n; ++i) {
      values[i] = src[keep_idx[i]];
    }
  } else if (TYPEOF(col) == INTSXP) {
    const int *src = INTEGER(col);
    for (size_t i = 0; i < n; ++i) {
      int val = src[keep_idx[i]];
      values[i] = (val == NA_INTEGER) ? NA_REAL : static_cast<double>(val);
    }
  }
  
  // Get range - R extends by 0.1% on each end
  double min_val = values.min();
  double max_val = values.max();
  double range = max_val - min_val;
  double ext = 0.001 * range;  // 0.1% extension
  
  // Create break points matching R's behavior
  std::vector<double> breaks(n_breaks + 1);
  double ext_min = min_val - ext;
  double ext_max = max_val + ext;
  double width = (ext_max - ext_min) / n_breaks;
  for (int i = 0; i <= n_breaks; ++i) {
    breaks[i] = ext_min + i * width;
  }
  
  // Create level names like R does: "(a,b]" with 2 decimal places
  info.level_names.resize(n_breaks);
  for (int i = 0; i < n_breaks; ++i) {
    char buf[128];
    snprintf(buf, sizeof(buf), "(%.2f,%.2f]", breaks[i], breaks[i + 1]);
    info.level_names[i] = buf;
  }
  
  // Assign each value to a bin
  std::vector<int> bin_idx(n);
  for (size_t i = 0; i < n; ++i) {
    double v = values[i];
    int bin = 0;
    for (int b = 0; b < n_breaks; ++b) {
      if (v > breaks[b] && v <= breaks[b + 1]) {
        bin = b;
        break;
      }
    }
    // First bin includes left boundary
    if (v == min_val) bin = 0;
    bin_idx[i] = bin;
  }
  
  // Create K-1 dummies (first level is reference)
  if (n_breaks <= 1) {
    return info;
  }
  
  info.dummies.set_size(n, n_breaks - 1);
  info.dummies.zeros();
  
  for (size_t i = 0; i < n; ++i) {
    int bin = bin_idx[i];
    if (bin > 0) {
      info.dummies(i, bin - 1) = 1.0;
    }
  }
  
  return info;
}

// ============================================================
// Helper to check if a column is a factor
// ============================================================

inline bool is_factor_column(SEXP col) {
  // Check for R factor (has levels attribute)
  SEXP lvls = Rf_getAttrib(col, R_LevelsSymbol);
  if (lvls != R_NilValue) return true;
  
  // Check for string column (treat as factor)
  if (TYPEOF(col) == STRSXP) return true;
  
  return false;
}

// ============================================================
// Get number of factor levels for a column
// ============================================================

inline size_t get_factor_levels_count(SEXP col, const uvec &keep_idx) {
  std::unordered_set<std::string> levels;
  size_t n = keep_idx.n_elem;
  
  auto val_to_string = [](SEXP col, R_xlen_t i) -> std::string {
    switch (TYPEOF(col)) {
    case INTSXP: {
      int v = INTEGER(col)[i];
      if (v == NA_INTEGER) return "__NA__";
      SEXP lvls = Rf_getAttrib(col, R_LevelsSymbol);
      if (lvls != R_NilValue) {
        return CHAR(STRING_ELT(lvls, v - 1));
      }
      return std::to_string(v);
    }
    case REALSXP: {
      double v = REAL(col)[i];
      if (!R_finite(v)) return "__NA__";
      char buf[64];
      snprintf(buf, sizeof(buf), "%.15g", v);
      return std::string(buf);
    }
    case STRSXP: {
      SEXP s = STRING_ELT(col, i);
      if (s == NA_STRING) return "__NA__";
      return CHAR(s);
    }
    default:
      return "";
    }
  };
  
  for (size_t i = 0; i < n; ++i) {
    R_xlen_t orig_i = keep_idx[i];
    levels.insert(val_to_string(col, orig_i));
  }
  
  return levels.size();
}

// ============================================================
// Factor dummy coding (K-1 dummies, first level = reference)
// ============================================================

struct FactorInfo {
  mat dummies;
  std::vector<std::string> level_names;
};

inline FactorInfo factor_dummies(SEXP col, const uvec &keep_idx) {
  FactorInfo info;
  size_t n = keep_idx.n_elem;

  // Build level map from actual values
  std::unordered_map<std::string, size_t> level_map;
  std::vector<std::string> levels;

  auto val_to_string = [](SEXP col, R_xlen_t i) -> std::string {
    switch (TYPEOF(col)) {
    case INTSXP: {
      int v = INTEGER(col)[i];
      if (v == NA_INTEGER)
        return "__NA__";
      // Check if factor
      SEXP lvls = Rf_getAttrib(col, R_LevelsSymbol);
      if (lvls != R_NilValue) {
        return CHAR(STRING_ELT(lvls, v - 1));
      }
      return std::to_string(v);
    }
    case REALSXP: {
      double v = REAL(col)[i];
      if (!R_finite(v))
        return "__NA__";
      char buf[64];
      snprintf(buf, sizeof(buf), "%.15g", v);
      return std::string(buf);
    }
    case STRSXP: {
      SEXP s = STRING_ELT(col, i);
      if (s == NA_STRING)
        return "__NA__";
      return CHAR(s);
    }
    default:
      return "";
    }
  };

  // First pass: collect unique levels
  std::set<std::string> unique_levels;
  for (size_t i = 0; i < n; ++i) {
    R_xlen_t orig_i = keep_idx[i];
    std::string val = val_to_string(col, orig_i);
    unique_levels.insert(val);
  }
  
  // Sort levels the way R does: alphabetically/lexicographically
  // (set is already sorted)
  for (const auto &lvl : unique_levels) {
    level_map[lvl] = levels.size();
    levels.push_back(lvl);
  }

  info.level_names = levels;
  size_t n_levels = levels.size();

  // K-1 dummies
  if (n_levels <= 1) {
    return info;
  }

  info.dummies.set_size(n, n_levels - 1);
  info.dummies.zeros();

  for (size_t i = 0; i < n; ++i) {
    R_xlen_t orig_i = keep_idx[i];
    std::string val = val_to_string(col, orig_i);
    size_t lvl = level_map[val];
    if (lvl > 0) {
      info.dummies(i, lvl - 1) = 1.0;
    }
  }

  return info;
}

// ============================================================
// Expression evaluator for I()
// Supports: +, -, *, /, ^, numeric literals
// ============================================================

inline vec evaluate_identity_expr(
    const std::string &expr,
    const std::unordered_map<std::string, const double *> &col_data,
    size_t n,
    const uvec &keep_idx) {
  vec result(n);

  // Tokenize: find the lowest-precedence operator outside parentheses
  auto find_binary_op = [](const std::string &s) -> std::pair<char, size_t> {
    int paren = 0;
    // Scan for + or - (lowest precedence) right to left
    for (size_t i = s.size(); i > 0; --i) {
      char c = s[i - 1];
      if (c == ')')
        paren++;
      else if (c == '(')
        paren--;
      else if (paren == 0 && (c == '+' || c == '-')) {
        return {c, i - 1};
      }
    }
    // Then * or /
    paren = 0;
    for (size_t i = s.size(); i > 0; --i) {
      char c = s[i - 1];
      if (c == ')')
        paren++;
      else if (c == '(')
        paren--;
      else if (paren == 0 && (c == '*' || c == '/')) {
        return {c, i - 1};
      }
    }
    // Then ^
    paren = 0;
    for (size_t i = s.size(); i > 0; --i) {
      char c = s[i - 1];
      if (c == ')')
        paren++;
      else if (c == '(')
        paren--;
      else if (paren == 0 && c == '^') {
        return {c, i - 1};
      }
    }
    return {0, std::string::npos};
  };

  std::function<vec(const std::string &)> eval_rec =
      [&](const std::string &s) -> vec {
    std::string trimmed = str_trim(s);

    // Remove outer parentheses
    while (trimmed.size() >= 2 && trimmed.front() == '(' &&
           trimmed.back() == ')') {
      // Check if they match
      int paren = 0;
      bool matched = true;
      for (size_t i = 0; i < trimmed.size() - 1; ++i) {
        if (trimmed[i] == '(')
          paren++;
        else if (trimmed[i] == ')')
          paren--;
        if (paren == 0) {
          matched = false;
          break;
        }
      }
      if (matched)
        trimmed = str_trim(trimmed.substr(1, trimmed.size() - 2));
      else
        break;
    }

    auto [op, pos] = find_binary_op(trimmed);

    if (op != 0 && pos != std::string::npos) {
      std::string left = str_trim(trimmed.substr(0, pos));
      std::string right = str_trim(trimmed.substr(pos + 1));
      vec l = eval_rec(left);
      vec r = eval_rec(right);

      switch (op) {
      case '+':
        return l + r;
      case '-':
        return l - r;
      case '*':
        return l % r;
      case '/':
        return l / r;
      case '^':
        return arma::pow(l, r);
      }
    }

    // Try as column name
    auto it = col_data.find(trimmed);
    if (it != col_data.end()) {
      vec v(n);
      const double *src = it->second;
      for (size_t i = 0; i < n; ++i) {
        v[i] = src[keep_idx[i]];
      }
      return v;
    }

    // Try as number
    try {
      double val = std::stod(trimmed);
      vec v(n);
      v.fill(val);
      return v;
    } catch (...) {
    }

    // Unknown - return zeros with warning
    vec v(n);
    v.zeros();
    return v;
  };

  return eval_rec(expr);
}

// ============================================================
// Main: Build y and X directly from R data.frame
// ============================================================

struct FormulaMatrixResult {
  vec y;
  mat X;
  uvec keep_idx;                        // 0-based indices of kept rows
  std::vector<std::string> term_names;  // Column names for X
  FlatFEMap fe_map;
  field<std::string> fe_names;
  field<field<std::string>> fe_levels;
  std::vector<std::string> cluster_vars;
  bool valid = true;
  std::string error;
  bool suppress_intercept = false;  // True when formula has -1 or 0+
};

inline FormulaMatrixResult build_matrix_from_formula(
    const std::string &formula_str,
    SEXP df,
    const double *weights_ptr,  // NULL for unit weights
    size_t weights_len) {
  FormulaMatrixResult result;

  // Parse formula
  ParsedFormula pf = parse_formula(formula_str);
  if (!pf.valid) {
    result.valid = false;
    result.error = pf.error;
    return result;
  }

  // Get data.frame column names and build index map
  SEXP names = Rf_getAttrib(df, R_NamesSymbol);
  std::unordered_map<std::string, int> col_idx;
  for (int i = 0; i < Rf_length(names); ++i) {
    col_idx[CHAR(STRING_ELT(names, i))] = i;
  }

  // Validate all columns exist
  for (const auto &col : pf.all_columns) {
    if (col_idx.find(col) == col_idx.end()) {
      result.valid = false;
      result.error = "undefined columns: " + col;
      return result;
    }
  }

  size_t N = static_cast<size_t>(Rf_xlength(VECTOR_ELT(df, 0)));

  // ============================================================
  // Pass 1: Find complete cases
  // ============================================================

  std::vector<bool> valid(N, true);

  auto check_na = [&](int idx) {
    SEXP col = VECTOR_ELT(df, idx);
    switch (TYPEOF(col)) {
    case REALSXP: {
      const double *ptr = REAL(col);
      for (size_t i = 0; i < N; ++i) {
        if (!R_finite(ptr[i]))
          valid[i] = false;
      }
      break;
    }
    case INTSXP: {
      const int *ptr = INTEGER(col);
      for (size_t i = 0; i < N; ++i) {
        if (ptr[i] == NA_INTEGER)
          valid[i] = false;
      }
      break;
    }
    case STRSXP: {
      for (size_t i = 0; i < N; ++i) {
        if (STRING_ELT(col, i) == NA_STRING)
          valid[i] = false;
      }
      break;
    }
    default:
      break;
    }
  };

  // Check all referenced columns
  for (const auto &col_name : pf.all_columns) {
    check_na(col_idx[col_name]);
  }

  // Check weights if provided
  if (weights_ptr != nullptr) {
    for (size_t i = 0; i < N && i < weights_len; ++i) {
      if (!R_finite(weights_ptr[i]))
        valid[i] = false;
    }
  }

  // Build keep index
  size_t n_valid = 0;
  for (size_t i = 0; i < N; ++i) {
    if (valid[i])
      n_valid++;
  }

  if (n_valid == 0) {
    result.valid = false;
    result.error = "No complete cases";
    return result;
  }

  result.keep_idx.set_size(n_valid);
  size_t j = 0;
  for (size_t i = 0; i < N; ++i) {
    if (valid[i])
      result.keep_idx[j++] = i;
  }

  // ============================================================
  // Pass 2: Build response vector y
  // ============================================================

  result.y.set_size(n_valid);

  if (pf.response.transform == Transform::IDENTITY) {
    // Build column data map for expression evaluation
    // Need to convert integer columns to temporary storage
    std::unordered_map<std::string, const double *> col_data;
    std::unordered_map<std::string, std::vector<double>> int_storage;
    for (const auto &c : pf.response.columns) {
      int idx = col_idx[c];
      SEXP col = VECTOR_ELT(df, idx);
      if (TYPEOF(col) == REALSXP) {
        col_data[c] = REAL(col);
      } else if (TYPEOF(col) == INTSXP) {
        // Convert to double storage
        const int *src = INTEGER(col);
        size_t N = static_cast<size_t>(Rf_xlength(col));
        int_storage[c].resize(N);
        for (size_t i = 0; i < N; ++i) {
          int_storage[c][i] = (src[i] == NA_INTEGER) ? NA_REAL : static_cast<double>(src[i]);
        }
        col_data[c] = int_storage[c].data();
      }
    }
    result.y = evaluate_identity_expr(pf.response.identity_expr, col_data,
                                      n_valid, result.keep_idx);
  } else {
    // Simple column with optional transform
    int y_idx = col_idx[pf.response.column];
    SEXP y_col = VECTOR_ELT(df, y_idx);
    copy_numeric_column(y_col, result.keep_idx, result.y.memptr());

    apply_transform_inplace(result.y.memptr(), n_valid, pf.response.transform);
  }

  // ============================================================
  // Pass 3: Count total columns needed for X
  // ============================================================

  size_t total_cols = 0;
  std::vector<size_t> term_col_counts;

  // Helper to count columns for an interaction term
  auto count_interact_cols = [&](const ParsedTerm &term) -> size_t {
    // Count how many factor components and their levels
    size_t n_factors = 0;
    size_t total_factor_levels = 1;
    
    for (const auto &c : term.columns) {
      ParsedTerm sub = parse_term(c);
      std::string col_name = sub.column.empty() ? c : sub.column;
      auto it = col_idx.find(col_name);
      if (it != col_idx.end()) {
        SEXP col = VECTOR_ELT(df, it->second);
        if (is_factor_column(col)) {
          n_factors++;
          size_t n_levels = get_factor_levels_count(col, result.keep_idx);
          total_factor_levels *= n_levels;
        }
      }
    }
    
    if (n_factors == 0) {
      // Pure numeric interaction - 1 column
      return 1;
    } else if (n_factors == 1) {
      // numeric:factor - (levels - 1) columns
      return total_factor_levels - 1;
    } else {
      // factor:factor - all level combinations 
      return total_factor_levels;
    }
  };

  for (const auto &term : pf.terms) {
    if (term.transform == Transform::POLY || term.transform == Transform::POLY_RAW) {
      term_col_counts.push_back(term.poly_degree);
      total_cols += term.poly_degree;
    } else if (term.transform == Transform::FACTOR || 
               term.transform == Transform::AS_FACTOR) {
      // Need to scan column to count levels
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      FactorInfo fi = factor_dummies(col, result.keep_idx);
      size_t ncols = fi.dummies.n_cols;
      term_col_counts.push_back(ncols);
      total_cols += ncols;
    } else if (term.transform == Transform::CUT) {
      // cut() produces n_breaks - 1 dummies
      size_t ncols = (term.cut_breaks > 1) ? term.cut_breaks - 1 : 0;
      term_col_counts.push_back(ncols);
      total_cols += ncols;
    } else if (term.transform == Transform::INTERACT) {
      // Count columns for interactions - need to handle:
      // - factors: K-1 levels for factor:numeric, K levels for factor:factor
      // - poly() terms (degree columns)
      // - plain numeric (1 column)
      
      // First count how many factors
      size_t num_factors = 0;
      for (const auto &c : term.columns) {
        ParsedTerm sub = parse_term(c);
        std::string col_name = sub.column.empty() ? c : sub.column;
        if (sub.transform != Transform::POLY && sub.transform != Transform::POLY_RAW) {
          auto it = col_idx.find(col_name);
          if (it != col_idx.end()) {
            SEXP col = VECTOR_ELT(df, it->second);
            if (is_factor_column(col)) {
              num_factors++;
            }
          }
        }
      }
      bool use_full_factors = (num_factors >= 2);
      
      size_t ncols = 1;
      for (const auto &c : term.columns) {
        ParsedTerm sub = parse_term(c);
        std::string col_name = sub.column.empty() ? c : sub.column;
        
        // Check for poly terms first
        if (sub.transform == Transform::POLY || sub.transform == Transform::POLY_RAW) {
          ncols *= sub.poly_degree;
        } else {
          auto it = col_idx.find(col_name);
          if (it != col_idx.end()) {
            SEXP col = VECTOR_ELT(df, it->second);
            if (is_factor_column(col)) {
              size_t n_levels = get_factor_levels_count(col, result.keep_idx);
              if (use_full_factors) {
                // factor:factor uses all K levels
                ncols *= n_levels;
              } else {
                // factor:numeric uses K-1 levels
                ncols *= (n_levels - 1);
              }
            }
          }
        }
      }
      term_col_counts.push_back(ncols);
      total_cols += ncols;
    } else {
      // Transform::NONE - check if it's actually a factor column
      auto it = col_idx.find(term.column);
      if (it != col_idx.end()) {
        SEXP col = VECTOR_ELT(df, it->second);
        if (is_factor_column(col)) {
          FactorInfo fi = factor_dummies(col, result.keep_idx);
          size_t ncols = fi.dummies.n_cols;
          term_col_counts.push_back(ncols);
          total_cols += ncols;
        } else {
          term_col_counts.push_back(1);
          total_cols += 1;
        }
      } else {
        term_col_counts.push_back(1);
        total_cols += 1;
      }
    }
  }

  // ============================================================
  // Pass 4: Build design matrix X
  // ============================================================

  result.X.set_size(n_valid, total_cols);
  result.term_names.reserve(total_cols);

  size_t col_offset = 0;
  for (size_t t = 0; t < pf.terms.size(); ++t) {
    const auto &term = pf.terms[t];
    (void)term_col_counts[t]; // Columns tracked via col_offset

    if (term.transform == Transform::NONE) {
      // Plain column - check if it's actually a factor
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      
      if (is_factor_column(col)) {
        // Convert factor to dummies (k-1 columns, first level = reference)
        FactorInfo fi = factor_dummies(col, result.keep_idx);
        for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
          result.X.col(col_offset + d) = fi.dummies.col(d);
          // Use column name + level name (e.g., "cyl_factor6")
          result.term_names.push_back(term.column + fi.level_names[d + 1]);
        }
        col_offset += fi.dummies.n_cols;
      } else {
        // Numeric column
        double *dst = result.X.colptr(col_offset);
        copy_numeric_column(col, result.keep_idx, dst);
        result.term_names.push_back(term.column);
        col_offset++;
      }

    } else if (term.transform == Transform::LOG ||
               term.transform == Transform::LOG1P ||
               term.transform == Transform::SQRT ||
               term.transform == Transform::EXP ||
               term.transform == Transform::ABS ||
               term.transform == Transform::SQUARE ||
               term.transform == Transform::CUBE) {
      // Simple transform
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      double *dst = result.X.colptr(col_offset);
      copy_numeric_column(col, result.keep_idx, dst);
      apply_transform_inplace(dst, n_valid, term.transform);
      result.term_names.push_back(term.raw);
      col_offset++;

    } else if (term.transform == Transform::POLY) {
      // Orthogonal polynomial expansion
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);

      vec x(n_valid);
      copy_numeric_column(col, result.keep_idx, x.memptr());
      mat poly_mat = orthogonal_poly(x, term.poly_degree);

      for (int d = 0; d < term.poly_degree; ++d) {
        result.X.col(col_offset + d) = poly_mat.col(d);
        result.term_names.push_back("poly(" + term.column + ", " +
                                    std::to_string(term.poly_degree) + ")" +
                                    std::to_string(d + 1));
      }
      col_offset += term.poly_degree;

    } else if (term.transform == Transform::POLY_RAW) {
      // Raw polynomial expansion
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);

      vec x(n_valid);
      copy_numeric_column(col, result.keep_idx, x.memptr());
      mat poly_mat = raw_poly(x, term.poly_degree);

      for (int d = 0; d < term.poly_degree; ++d) {
        result.X.col(col_offset + d) = poly_mat.col(d);
        result.term_names.push_back(term.raw + std::to_string(d + 1));
      }
      col_offset += term.poly_degree;

    } else if (term.transform == Transform::FACTOR ||
               term.transform == Transform::AS_FACTOR) {
      // Factor dummies
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      FactorInfo fi = factor_dummies(col, result.keep_idx);

      for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
        result.X.col(col_offset + d) = fi.dummies.col(d);
        result.term_names.push_back(term.raw + fi.level_names[d + 1]);
      }
      col_offset += fi.dummies.n_cols;

    } else if (term.transform == Transform::CUT) {
      // Cut into bins
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      CutInfo ci = cut_dummies(col, result.keep_idx, term.cut_breaks);

      for (size_t d = 0; d < ci.dummies.n_cols; ++d) {
        result.X.col(col_offset + d) = ci.dummies.col(d);
        result.term_names.push_back(term.raw + ci.level_names[d + 1]);
      }
      col_offset += ci.dummies.n_cols;

    } else if (term.transform == Transform::INTERACT) {
      // Handle interactions between any combination of:
      // - factors (K-1 columns for factor:numeric, K columns for factor:factor)
      // - poly() terms (degree columns)
      // - plain numeric (1 column)
      
      // First, determine how many factors we have (for deciding K vs K-1)
      size_t num_factors = 0;
      for (const auto &c : term.columns) {
        ParsedTerm sub_term = parse_term(c);
        std::string col_name = sub_term.column.empty() ? c : sub_term.column;
        if (sub_term.transform != Transform::POLY && sub_term.transform != Transform::POLY_RAW) {
          auto it = col_idx.find(col_name);
          if (it != col_idx.end()) {
            SEXP col = VECTOR_ELT(df, it->second);
            if (is_factor_column(col)) {
              num_factors++;
            }
          }
        }
      }
      bool use_full_factors = (num_factors >= 2);  // factor:factor uses all levels
      
      // For each component, store its expansion info
      struct ComponentExpansion {
        std::string base_name;      // e.g., "hp" or "poly(wt, 2)"
        mat columns;                // The actual data columns
        std::vector<std::string> suffixes;  // e.g., ["1", "2"] for poly, ["6", "8"] for factor
        bool is_factor;
      };
      
      std::vector<ComponentExpansion> expansions;
      
      for (const auto &c : term.columns) {
        ParsedTerm sub_term = parse_term(c);
        std::string col_name = sub_term.column.empty() ? c : sub_term.column;
        
        ComponentExpansion exp;
        exp.base_name = c;  // Use full term string as base name
        exp.is_factor = false;
        
        if (sub_term.transform == Transform::POLY || sub_term.transform == Transform::POLY_RAW) {
          // Poly term - expand to degree columns
          auto it = col_idx.find(col_name);
          if (it == col_idx.end()) {
            result.valid = false;
            result.error = "undefined columns: " + col_name;
            return result;
          }
          
          SEXP col = VECTOR_ELT(df, it->second);
          vec x(n_valid);
          copy_numeric_column(col, result.keep_idx, x.memptr());
          
          if (sub_term.transform == Transform::POLY) {
            exp.columns = orthogonal_poly(x, sub_term.poly_degree);
          } else {
            exp.columns = raw_poly(x, sub_term.poly_degree);
          }
          
          for (int d = 0; d < sub_term.poly_degree; ++d) {
            exp.suffixes.push_back(std::to_string(d + 1));
          }
          
        } else {
          // Check if it's a factor
          auto it = col_idx.find(col_name);
          if (it == col_idx.end()) {
            result.valid = false;
            result.error = "undefined columns: " + col_name;
            return result;
          }
          
          SEXP col = VECTOR_ELT(df, it->second);
          
          if (is_factor_column(col)) {
            exp.is_factor = true;
            FactorInfo fi = factor_dummies(col, result.keep_idx);
            
            if (use_full_factors) {
              // Factor:factor interaction - use ALL K levels
              // Create full indicator matrix including reference level
              size_t k = fi.level_names.size();
              exp.columns.set_size(n_valid, k);
              
              // First column: indicator for reference level (all zeros in dummies)
              vec ref_indicator(n_valid);
              for (size_t i = 0; i < n_valid; ++i) {
                bool is_ref = true;
                for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
                  if (fi.dummies(i, d) == 1.0) {
                    is_ref = false;
                    break;
                  }
                }
                ref_indicator[i] = is_ref ? 1.0 : 0.0;
              }
              exp.columns.col(0) = ref_indicator;
              
              // Remaining columns: the K-1 dummies
              for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
                exp.columns.col(d + 1) = fi.dummies.col(d);
              }
              
              // All level names
              for (size_t d = 0; d < fi.level_names.size(); ++d) {
                exp.suffixes.push_back(fi.level_names[d]);
              }
            } else {
              // Factor:numeric interaction - use K-1 dummies
              exp.columns = fi.dummies;
              for (size_t d = 1; d < fi.level_names.size(); ++d) {
                exp.suffixes.push_back(fi.level_names[d]);
              }
            }
          } else {
            // Plain numeric - single column
            exp.columns.set_size(n_valid, 1);
            copy_numeric_column(col, result.keep_idx, exp.columns.colptr(0));
            if (sub_term.transform != Transform::NONE && 
                sub_term.transform != Transform::POLY && 
                sub_term.transform != Transform::POLY_RAW) {
              apply_transform_inplace(exp.columns.colptr(0), n_valid, sub_term.transform);
            }
            exp.suffixes.push_back("");  // No suffix for plain numeric
          }
        }
        
        expansions.push_back(exp);
      }
      
      // Generate all combinations of columns
      // For poly(wt, 2):hp, this gives: poly(wt,2)1:hp, poly(wt,2)2:hp
      if (expansions.size() == 2) {
        // Two-component interaction
        // For factor:factor, R iterates second factor in outer loop
        // For other cases, use natural order
        bool both_factors = expansions[0].is_factor && expansions[1].is_factor;
        
        if (both_factors) {
          // factor:factor - second factor outer, first factor inner
          for (size_t j = 0; j < expansions[1].columns.n_cols; ++j) {
            for (size_t i = 0; i < expansions[0].columns.n_cols; ++i) {
              vec product = expansions[0].columns.col(i) % expansions[1].columns.col(j);
              result.X.col(col_offset) = product;
              
              std::string name = expansions[0].base_name + expansions[0].suffixes[i] + 
                                ":" + expansions[1].base_name + expansions[1].suffixes[j];
              result.term_names.push_back(name);
              col_offset++;
            }
          }
        } else {
          // Other cases - first component outer
          for (size_t i = 0; i < expansions[0].columns.n_cols; ++i) {
            for (size_t j = 0; j < expansions[1].columns.n_cols; ++j) {
              vec product = expansions[0].columns.col(i) % expansions[1].columns.col(j);
              result.X.col(col_offset) = product;
              
              std::string name;
              if (expansions[0].suffixes[i].empty()) {
                name = expansions[0].base_name;
              } else {
                name = expansions[0].base_name + expansions[0].suffixes[i];
              }
              name += ":";
              if (expansions[1].suffixes[j].empty()) {
                name += expansions[1].base_name;
              } else {
                name += expansions[1].base_name + expansions[1].suffixes[j];
              }
              result.term_names.push_back(name);
              col_offset++;
            }
          }
        }
      } else if (expansions.size() == 1) {
        // Single component (shouldn't happen for interactions, but handle it)
        for (size_t i = 0; i < expansions[0].columns.n_cols; ++i) {
          result.X.col(col_offset) = expansions[0].columns.col(i);
          result.term_names.push_back(expansions[0].base_name + expansions[0].suffixes[i]);
          col_offset++;
        }
      } else {
        // Fallback for 3+ components: multiply all together as single column
        vec product(n_valid);
        product.ones();
        for (const auto &exp : expansions) {
          if (exp.columns.n_cols == 1) {
            product %= exp.columns.col(0);
          }
        }
        result.X.col(col_offset) = product;
        result.term_names.push_back(term.raw);
        col_offset++;
      }

    } else if (term.transform == Transform::IDENTITY) {
      // I() expression - need to convert integer columns to temporary storage
      std::unordered_map<std::string, const double *> col_data;
      std::unordered_map<std::string, std::vector<double>> int_storage;
      for (const auto &c : term.columns) {
        int idx = col_idx[c];
        SEXP col = VECTOR_ELT(df, idx);
        if (TYPEOF(col) == REALSXP) {
          col_data[c] = REAL(col);
        } else if (TYPEOF(col) == INTSXP) {
          // Convert to double storage
          const int *src = INTEGER(col);
          size_t N = static_cast<size_t>(Rf_xlength(col));
          int_storage[c].resize(N);
          for (size_t ii = 0; ii < N; ++ii) {
            int_storage[c][ii] = (src[ii] == NA_INTEGER) ? NA_REAL : static_cast<double>(src[ii]);
          }
          col_data[c] = int_storage[c].data();
        }
      }

      result.X.col(col_offset) = evaluate_identity_expr(
          term.identity_expr, col_data, n_valid, result.keep_idx);
      result.term_names.push_back(term.raw);
      col_offset++;
    }
  }

  // ============================================================
  // Pass 5: Build FE map
  // ============================================================

  size_t K = pf.fe_vars.size();
  result.fe_names.set_size(K);
  result.fe_levels.set_size(K);

  if (K > 0) {
    result.fe_map.K = K;
    result.fe_map.n_obs = n_valid;
    result.fe_map.n_groups.resize(K);
    result.fe_map.fe_map.resize(K);

    for (size_t k = 0; k < K; ++k) {
      const std::string &fe_var = pf.fe_vars[k];
      result.fe_names(k) = fe_var;

      int idx = col_idx[fe_var];
      SEXP col = VECTOR_ELT(df, idx);

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

      result.fe_map.fe_map[k].resize(n_valid);
      for (size_t i = 0; i < n_valid; ++i) {
        R_xlen_t orig_i = result.keep_idx[i];
        std::string val = val_to_string(col, orig_i);
        auto it = level_map.find(val);
        if (it == level_map.end()) {
          uword code = static_cast<uword>(levels.size());
          level_map[val] = code;
          levels.push_back(val);
          result.fe_map.fe_map[k][i] = code;
        } else {
          result.fe_map.fe_map[k][i] = it->second;
        }
      }

      result.fe_map.n_groups[k] = levels.size();
      result.fe_levels(k).set_size(levels.size());
      for (size_t l = 0; l < levels.size(); ++l) {
        result.fe_levels(k)(l) = levels[l];
      }
    }
    result.fe_map.structure_built = true;
  }

  result.cluster_vars = pf.cluster_vars;
  result.suppress_intercept = pf.suppress_intercept;

  return result;
}

}  // namespace capybara

#endif  // CAPYBARA_FORMULA_PARSER_H
