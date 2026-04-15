// Formula parsing and direct Armadillo matrix building from R data.frame
//
// capybara only supports simple formulas:
//   y ~ x1 + x2 + x3                  - plain columns
//   y ~ x1 + x2 | fe1 + fe2           - with fixed effects
//   y ~ x1 + x2 | fe1 + fe2 | cl      - with fixed effects and clusters
//   y ~ x1:x2                         - interaction (element-wise product)
//   y ~ x1*x2                         - expands to x1 + x2 + x1:x2
//
// Functions like log(), sqrt(), poly(), I(), factor() are NOT supported.
// Use dplyr/data.table to transform data before fitting.

#ifndef CAPYBARA_FORMULA_PARSER_H
#define CAPYBARA_FORMULA_PARSER_H

namespace capybara {

// ============================================================
// Parsed term structure
// ============================================================

struct ParsedTerm {
  std::string raw;                   // Original term string
  std::string column;                // Column name (for simple terms)
  std::vector<std::string> columns;  // For interactions: ["a", "b"]
  bool is_interaction = false;
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
// Detect if term contains a function call
// Returns the function name if found, empty string otherwise
// ============================================================

inline std::string detect_function_call(const std::string &term) {
  std::string s = str_trim(term);
  
  // Look for pattern: identifier followed by (
  // This catches: log(x), sqrt(x), poly(x, 2), I(x*y), factor(x), etc.
  std::regex func_re("([a-zA-Z_][a-zA-Z0-9_\\.]*)\\s*\\(");
  std::smatch match;
  if (std::regex_search(s, match, func_re)) {
    return match[1].str();
  }
  
  return "";
}

// Helper to build the "unsupported function" error message
inline std::string unsupported_function_error(const std::string &func_name) {
  return "capybara does not support functions in formulas (found '" + 
         func_name + "'). Use dplyr/data.table to transform your data before fitting. " +
         "Supported syntax: y ~ x1 + x2 + x3 | fe1 + fe2 | cl";
}

// ============================================================
// Safe numeric column access (handles REALSXP and INTSXP)
// ============================================================

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
    for (size_t i = 0; i < n; ++i) {
      dst[i] = NA_REAL;
    }
  }
}

inline double get_numeric_value(SEXP col, size_t i) {
  if (TYPEOF(col) == REALSXP) {
    return REAL(col)[i];
  } else if (TYPEOF(col) == INTSXP) {
    int val = INTEGER(col)[i];
    return (val == NA_INTEGER) ? NA_REAL : static_cast<double>(val);
  }
  return NA_REAL;
}

// ============================================================
// Parse a single term
// ============================================================

inline ParsedTerm parse_term(const std::string &raw) {
  ParsedTerm t;
  t.raw = raw;
  std::string s = str_trim(raw);

  // Check for interaction: a:b
  if (s.find(':') != std::string::npos) {
    t.is_interaction = true;
    t.columns = str_split_top_level(s, ':');
    return t;
  }

  // Plain column name
  t.column = s;
  t.is_interaction = false;
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

  // Check for functions in response
  std::string func = detect_function_call(lhs);
  if (!func.empty()) {
    pf.valid = false;
    pf.error = unsupported_function_error(func);
    return pf;
  }

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

    // Check for functions in this term
    func = detect_function_call(term_str);
    if (!func.empty()) {
      pf.valid = false;
      pf.error = unsupported_function_error(func);
      return pf;
    }

    // Handle * expansion: a*b -> a + b + a:b
    if (term_str.find('*') != std::string::npos) {
      auto vars = str_split_top_level(term_str, '*');
      // Add main effects
      for (const auto &v : vars) {
        ParsedTerm pt = parse_term(v);
        pf.terms.push_back(pt);
        if (!pt.column.empty())
          add_col(pt.column);
        for (const auto &c : pt.columns)
          add_col(c);
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
        // Check for functions in FE vars
        func = detect_function_call(f);
        if (!func.empty()) {
          pf.valid = false;
          pf.error = unsupported_function_error(func);
          return pf;
        }
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
        // Check for functions in cluster vars
        func = detect_function_call(c);
        if (!func.empty()) {
          pf.valid = false;
          pf.error = unsupported_function_error(func);
          return pf;
        }
        pf.cluster_vars.push_back(c);
        add_col(c);
      }
    }
  }

  return pf;
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
  bool suppress_intercept = false;
  bool has_intercept_column = false;
};

inline FormulaMatrixResult build_matrix_from_formula(
    const std::string &formula_str,
    SEXP df,
    const double *weights_ptr,
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
  int y_idx = col_idx[pf.response.column];
  SEXP y_col = VECTOR_ELT(df, y_idx);
  copy_numeric_column(y_col, result.keep_idx, result.y.memptr());

  // ============================================================
  // Pass 3: Count total columns needed for X
  // Cache factor expansions to avoid recomputing
  // ============================================================

  size_t total_cols = 0;
  std::vector<size_t> term_col_counts;
  std::unordered_map<std::string, FactorInfo> factor_cache;

  for (size_t t = 0; t < pf.terms.size(); ++t) {
    const auto &term = pf.terms[t];
    
    if (term.is_interaction) {
      // Interaction: count columns for each component
      size_t num_factors = 0;
      for (const auto &c : term.columns) {
        auto it = col_idx.find(c);
        if (it != col_idx.end()) {
          SEXP col = VECTOR_ELT(df, it->second);
          if (is_factor_column(col)) {
            num_factors++;
          }
        }
      }
      bool use_full_factors = (num_factors >= 2);
      
      size_t ncols = 1;
      for (const auto &c : term.columns) {
        auto it = col_idx.find(c);
        if (it != col_idx.end()) {
          SEXP col = VECTOR_ELT(df, it->second);
          if (is_factor_column(col)) {
            size_t n_levels = get_factor_levels_count(col, result.keep_idx);
            if (use_full_factors) {
              ncols *= n_levels;
            } else {
              ncols *= (n_levels - 1);
            }
          }
        }
      }
      term_col_counts.push_back(ncols);
      total_cols += ncols;
    } else {
      // Simple column - check if it's a factor
      auto it = col_idx.find(term.column);
      if (it != col_idx.end()) {
        SEXP col = VECTOR_ELT(df, it->second);
        if (is_factor_column(col)) {
          FactorInfo fi = factor_dummies(col, result.keep_idx);
          size_t ncols = fi.dummies.n_cols;
          factor_cache[term.column] = std::move(fi);
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

  bool needs_intercept = (pf.fe_vars.size() == 0) && !pf.suppress_intercept;
  size_t intercept_cols = needs_intercept ? 1 : 0;
  
  result.X.set_size(n_valid, total_cols + intercept_cols);
  result.term_names.reserve(total_cols + intercept_cols);

  size_t col_offset = 0;
  
  // Fill intercept column first (if needed)
  if (needs_intercept) {
    result.X.col(0).ones();
    result.has_intercept_column = true;
    col_offset = 1;
  }

  for (size_t t = 0; t < pf.terms.size(); ++t) {
    const auto &term = pf.terms[t];

    if (term.is_interaction) {
      // Handle interactions
      size_t num_factors = 0;
      for (const auto &c : term.columns) {
        auto it = col_idx.find(c);
        if (it != col_idx.end()) {
          SEXP col = VECTOR_ELT(df, it->second);
          if (is_factor_column(col)) {
            num_factors++;
          }
        }
      }
      bool use_full_factors = (num_factors >= 2);
      
      // Build component expansions
      struct ComponentExpansion {
        std::string base_name;
        mat columns;
        std::vector<std::string> suffixes;
        bool is_factor;
      };
      
      std::vector<ComponentExpansion> expansions;
      
      for (const auto &c : term.columns) {
        ComponentExpansion exp;
        exp.base_name = c;
        exp.is_factor = false;
        
        auto it = col_idx.find(c);
        if (it == col_idx.end()) {
          result.valid = false;
          result.error = "undefined columns: " + c;
          return result;
        }
        
        SEXP col = VECTOR_ELT(df, it->second);
        
        if (is_factor_column(col)) {
          exp.is_factor = true;
          FactorInfo fi;
          auto cache_it = factor_cache.find(c);
          if (cache_it != factor_cache.end()) {
            fi = cache_it->second;
          } else {
            fi = factor_dummies(col, result.keep_idx);
            factor_cache[c] = fi;
          }
          
          if (use_full_factors) {
            // Factor:factor - use ALL K levels
            size_t k = fi.level_names.size();
            exp.columns.set_size(n_valid, k);
            
            // First column: reference level indicator
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
            
            for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
              exp.columns.col(d + 1) = fi.dummies.col(d);
            }
            
            for (size_t d = 0; d < fi.level_names.size(); ++d) {
              exp.suffixes.push_back(fi.level_names[d]);
            }
          } else {
            // Factor:numeric - use K-1 dummies
            exp.columns = fi.dummies;
            for (size_t d = 1; d < fi.level_names.size(); ++d) {
              exp.suffixes.push_back(fi.level_names[d]);
            }
          }
        } else {
          // Plain numeric
          exp.columns.set_size(n_valid, 1);
          copy_numeric_column(col, result.keep_idx, exp.columns.colptr(0));
          exp.suffixes.push_back("");
        }
        
        expansions.push_back(exp);
      }
      
      // Generate all combinations
      if (expansions.size() == 2) {
        bool both_factors = expansions[0].is_factor && expansions[1].is_factor;
        
        if (both_factors) {
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
        for (size_t i = 0; i < expansions[0].columns.n_cols; ++i) {
          result.X.col(col_offset) = expansions[0].columns.col(i);
          result.term_names.push_back(expansions[0].base_name + expansions[0].suffixes[i]);
          col_offset++;
        }
      } else {
        // 3+ components: multiply all together
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
      
    } else {
      // Simple column
      auto cache_it = factor_cache.find(term.column);
      if (cache_it != factor_cache.end()) {
        // Use cached factor dummies
        const FactorInfo &fi = cache_it->second;
        for (size_t d = 0; d < fi.dummies.n_cols; ++d) {
          result.X.col(col_offset + d) = fi.dummies.col(d);
          result.term_names.push_back(term.column + fi.level_names[d + 1]);
        }
        col_offset += fi.dummies.n_cols;
      } else {
        // Numeric column
        int idx = col_idx[term.column];
        SEXP col = VECTOR_ELT(df, idx);
        double *dst = result.X.colptr(col_offset);
        copy_numeric_column(col, result.keep_idx, dst);
        result.term_names.push_back(term.column);
        col_offset++;
      }
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
