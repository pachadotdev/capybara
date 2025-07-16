devtools::load_all()

cat("=== Debugging formula parsing and model fitting ===\n")

# Test 1: No fixed effects
cat("\n=== Test 1: No FE (mpg ~ wt) ===\n")
formula1 <- mpg ~ wt
cat("Formula:", deparse(formula1), "\n")

tryCatch({
  fe_vars <- parse_formula_pipe_(formula1)
  cat("Extracted FE vars:", paste(fe_vars, collapse=", "), "\n")
  cat("Length:", length(fe_vars), "\n")
  
  # Test if the problem is in felm
  cat("Attempting felm...\n")
  result1 <- felm(formula1, mtcars)
  cat("SUCCESS - No FE case works!\n")
  cat("Coefficients:", result1$coefficients, "\n")
}, error = function(e) {
  cat("ERROR in no FE case:", conditionMessage(e), "\n")
  cat("Call stack:\n")
  print(sys.calls())
})

# Test 2: Single fixed effect  
cat("\n=== Test 2: Single FE (mpg ~ wt | cyl) ===\n")
formula2 <- mpg ~ wt | cyl
cat("Formula:", deparse(formula2), "\n")

tryCatch({
  fe_vars <- parse_formula_pipe_(formula2)
  cat("Extracted FE vars:", paste(fe_vars, collapse=", "), "\n")
  cat("Length:", length(fe_vars), "\n")
  
  cat("Attempting felm...\n")
  result2 <- felm(formula2, mtcars)
  cat("SUCCESS - Single FE case works!\n")
  cat("Coefficients:", result2$coefficients, "\n")
}, error = function(e) {
  cat("ERROR in single FE case:", conditionMessage(e), "\n")
  cat("Call stack:\n")
  print(sys.calls())
})

# Test 3: Two fixed effects
cat("\n=== Test 3: Two FE (mpg ~ wt | cyl + am) ===\n")
formula3 <- mpg ~ wt | cyl + am
cat("Formula:", deparse(formula3), "\n")

tryCatch({
  fe_vars <- parse_formula_pipe_(formula3)
  cat("Extracted FE vars:", paste(fe_vars, collapse=", "), "\n")
  cat("Length:", length(fe_vars), "\n")
  
  cat("Attempting felm...\n")
  result3 <- felm(formula3, mtcars)
  cat("SUCCESS - Two FE case works!\n")
  cat("Coefficients:", result3$coefficients, "\n")
}, error = function(e) {
  cat("ERROR in two FE case:", conditionMessage(e), "\n")
  cat("Call stack:\n")
  print(sys.calls())
})

# Test 4: GLM binomial
cat("\n=== Test 4: GLM binomial (am ~ wt + disp | cyl) ===\n")
formula4 <- am ~ wt + disp | cyl
cat("Formula:", deparse(formula4), "\n")

tryCatch({
  fe_vars <- parse_formula_pipe_(formula4)
  cat("Extracted FE vars:", paste(fe_vars, collapse=", "), "\n")
  cat("Length:", length(fe_vars), "\n")
  
  cat("Attempting feglm...\n")
  result4 <- feglm(formula4, mtcars, family = binomial())
  cat("SUCCESS - GLM binomial case works!\n")
  cat("Coefficients:", result4$coefficients, "\n")
}, error = function(e) {
  cat("ERROR in GLM binomial case:", conditionMessage(e), "\n")
  cat("Call stack:\n")
  print(sys.calls())
})

cat("\n=== Debug complete ===\n")
