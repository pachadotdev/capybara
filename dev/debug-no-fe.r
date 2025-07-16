# Debug what happens with no fixed effects
devtools::load_all()

cat("=== Debugging no FE case ===\n")
data(mtcars)

# Create a simple test to see what k_list looks like
formula <- mpg ~ wt
mf <- model.frame(formula, mtcars)
cat("Model frame created\n")

# Check what felm_helpers.R creates for k_list when no FE
source("R/felm_helpers.R")

# Try to replicate what happens inside felm
tryCatch({
  # This is what felm() does internally
  data <- mtcars
  control <- list(center_tol = 1e-8, collin_tol = 1e-10, iter_center_max = 10000L, 
                  iter_interrupt = 10000L, iter_ssr = 5000L)
  
  # Extract response and model matrix
  y <- model.response(mf)
  x <- model.matrix(formula, mf)
  
  cat("y length:", length(y), "\n")
  cat("x dimensions:", dim(x), "\n")
  
  # What does the helper create for k_list?
  # Look at how model_response_ works
  cat("Calling model_response_...\n")
  
  # This should tell us what k_list looks like
  k_list <- list()  # Empty list for no FE
  cat("k_list length:", length(k_list), "\n")
  cat("k_list class:", class(k_list), "\n")
  
  # Try the C++ function directly
  wt <- rep(1.0, nrow(mtcars))
  
  cat("About to call felm_fit_...\n")
  result <- felm_fit_(y, x, wt, control, k_list)
  cat("SUCCESS!\n")
  
}, error = function(e) {
  cat("Error:", e$message, "\n")
  cat("Call stack:\n")
  print(traceback())
})
