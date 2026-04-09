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
  NONE,      // plain column
  LOG,       // log(x)
  LOG1P,     // log1p(x) = log(1+x)
  SQRT,      // sqrt(x)
  EXP,       // exp(x)
  ABS,       // abs(x)
  SQUARE,    // x^2
  CUBE,      // x^3
  POLY,      // poly(x, degree)
  IDENTITY,  // I(expr)
  INTERACT,  // a:b
  FACTOR     // factor(x)
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
  std::string identity_expr;  // For I(): "a*b + c"

  size_t base_cols() const {
    if (transform == Transform::POLY)
      return static_cast<size_t>(poly_degree);
    if (transform == Transform::FACTOR)
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

  // Check for function transforms: func(col) or func(col, arg)
  auto try_parse_func = [&](const std::string &prefix, Transform tf) -> bool {
    if (str_starts_with(s, prefix) && s.back() == ')') {
      std::string inner = s.substr(prefix.size(), s.size() - prefix.size() - 1);
      auto args = str_split_top_level(inner, ',');
      t.column = str_trim(args[0]);
      t.transform = tf;

      if (tf == Transform::POLY && args.size() > 1) {
        t.poly_degree = std::stoi(str_trim(args[1]));
      }
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
  if (try_parse_func("poly(", Transform::POLY))
    return t;
  if (try_parse_func("factor(", Transform::FACTOR))
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

    // Handle * expansion: a*b -> a + b + a:b
    if (term_str.find('*') != std::string::npos &&
        !str_starts_with(term_str, "I(")) {
      auto vars = str_split_top_level(term_str, '*');
      // Add main effects
      for (const auto &v : vars) {
        ParsedTerm pt = parse_term(v);
        pf.terms.push_back(pt);
        if (!pt.column.empty())
          add_col(pt.column);
      }
      // Add interaction
      std::string interact = vars[0];
      for (size_t i = 1; i < vars.size(); ++i) {
        interact += ":" + vars[i];
      }
      ParsedTerm pt = parse_term(interact);
      pf.terms.push_back(pt);
      for (const auto &c : pt.columns)
        add_col(c);
    } else {
      ParsedTerm pt = parse_term(term_str);
      pf.terms.push_back(pt);
      if (!pt.column.empty())
        add_col(pt.column);
      for (const auto &c : pt.columns)
        add_col(c);
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
// Orthogonal polynomial expansion (like R's poly())
// ============================================================

inline mat orthogonal_poly(const vec &x, int degree) {
  size_t n = x.n_elem;
  mat result(n, degree);

  // Center x
  double mean_x = arma::mean(x);
  vec xc = x - mean_x;

  // Raw polynomials
  result.col(0) = xc;
  for (int d = 1; d < degree; ++d) {
    result.col(d) = arma::pow(xc, d + 1);
  }

  // Gram-Schmidt orthogonalization
  for (int j = 0; j < degree; ++j) {
    vec v = result.col(j);
    for (int k = 0; k < j; ++k) {
      vec u = result.col(k);
      double proj = arma::dot(v, u) / arma::dot(u, u);
      v -= proj * u;
    }
    double norm_v = arma::norm(v);
    if (norm_v > 1e-10) {
      result.col(j) = v / norm_v;
    }
  }

  return result;
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

  // First pass: collect unique levels in order
  for (size_t i = 0; i < n; ++i) {
    R_xlen_t orig_i = keep_idx[i];
    std::string val = val_to_string(col, orig_i);
    if (level_map.find(val) == level_map.end()) {
      level_map[val] = levels.size();
      levels.push_back(val);
    }
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

  for (const auto &term : pf.terms) {
    if (term.transform == Transform::POLY) {
      term_col_counts.push_back(term.poly_degree);
      total_cols += term.poly_degree;
    } else if (term.transform == Transform::FACTOR) {
      // Need to scan column to count levels
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      FactorInfo fi = factor_dummies(col, result.keep_idx);
      size_t ncols = fi.dummies.n_cols;
      term_col_counts.push_back(ncols);
      total_cols += ncols;
    } else {
      term_col_counts.push_back(1);
      total_cols += 1;
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
      // Plain column
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      double *dst = result.X.colptr(col_offset);
      copy_numeric_column(col, result.keep_idx, dst);
      result.term_names.push_back(term.column);
      col_offset++;

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
      // Polynomial expansion
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

    } else if (term.transform == Transform::FACTOR) {
      // Factor dummies
      int idx = col_idx[term.column];
      SEXP col = VECTOR_ELT(df, idx);
      FactorInfo fi = factor_dummies(col, result.keep_idx);

      for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
        result.X.col(col_offset + d) = fi.dummies.col(d);
        result.term_names.push_back(term.column + fi.level_names[d + 1]);
      }
      col_offset += fi.dummies.n_cols;

    } else if (term.transform == Transform::INTERACT) {
      // Interaction: multiply columns element-wise
      vec product(n_valid);
      product.ones();

      for (const auto &c : term.columns) {
        int idx = col_idx[c];
        SEXP col = VECTOR_ELT(df, idx);
        vec col_vals(n_valid);
        copy_numeric_column(col, result.keep_idx, col_vals.memptr());
        product %= col_vals;
      }
      result.X.col(col_offset) = product;
      result.term_names.push_back(term.raw);
      col_offset++;

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

  return result;
}

}  // namespace capybara

#endif  // CAPYBARA_FORMULA_PARSER_H
