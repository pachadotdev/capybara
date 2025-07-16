# Debug formula parsing
devtools::load_all()

cat("=== Formula parsing debug ===\n")
formula <- mpg ~ wt | cyl

cat("Full formula:", deparse(formula), "\n")

# Check what terms() extracts
cat("Terms with rhs = 1 (regressors):\n")
rhs1_terms <- terms(formula, rhs = 1L)
print(attr(rhs1_terms, "term.labels"))

cat("\nTerms with rhs = 2 (fixed effects):\n")
rhs2_terms <- suppressWarnings(terms(formula, rhs = 2L))
print(attr(rhs2_terms, "term.labels"))

# Check the original formula structure
cat("\nFormula length:", length(formula), "\n")
cat("Formula parts:\n")
for (i in seq_along(formula)) {
  cat("Part", i, ":", deparse(formula[[i]]), "\n")
}

# Check if there's a bar in the formula
if (length(formula) == 3 && grepl("\\|", deparse(formula[[3]]))) {
  cat("\nFormula has fixed effects separator '|'\n")
  
  # Split the RHS on the | symbol
  rhs_full <- deparse(formula[[3]])
  cat("Full RHS:", rhs_full, "\n")
  
  if (grepl("\\|", rhs_full)) {
    parts <- strsplit(rhs_full, "\\|")[[1]]
    cat("Split parts:\n")
    for (i in seq_along(parts)) {
      cat("  Part", i, ":", trimws(parts[i]), "\n")
    }
  }
}
