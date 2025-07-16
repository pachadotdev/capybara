# Debug the core issue step by step
devtools::load_all()

cat("=== Testing model.frame creation ===\n")
data(mtcars)

# Test basic model.frame creation
tryCatch({
  formula_test <- mpg ~ wt
  mf <- model.frame(formula_test, mtcars)
  cat("Basic model.frame SUCCESS\n")
  cat("Dimensions:", dim(mf), "\n")
  
  # Test model.response
  y <- model.response(mf)
  cat("y length:", length(y), "\n")
  cat("y first few values:", head(y), "\n")
  
  # Test model.matrix
  x <- model.matrix(formula_test, mf)[, -1L, drop = FALSE]
  cat("x dimensions:", dim(x), "\n")
  cat("x first few values:", head(x[,1]), "\n")
  
}, error = function(e) {
  cat("Error in model frame creation:", e$message, "\n")
})

# Test the model_response_ helper function
cat("\n=== Testing model_response_ helper ===\n")
tryCatch({
  # Initialize variables
  y <- NULL
  x <- NULL
  nms_sp <- NULL
  p <- NULL
  
  # Call the helper
  capybara:::model_response_(mtcars, mpg ~ wt)
  
  cat("After model_response_ call:\n")
  cat("y defined:", exists("y"), "length:", if(exists("y")) length(y) else "NA", "\n")
  cat("x defined:", exists("x"), "dimensions:", if(exists("x")) paste(dim(x), collapse="x") else "NA", "\n")
  cat("nms_sp defined:", exists("nms_sp"), "value:", if(exists("nms_sp")) nms_sp else "NA", "\n")
  cat("p defined:", exists("p"), "value:", if(exists("p")) p else "NA", "\n")
  
}, error = function(e) {
  cat("Error in model_response_:", e$message, "\n")
})

# Test the C++ function call directly
cat("\n=== Testing C++ function call ===\n")
tryCatch({
  # Use simple data
  y_test <- mtcars$mpg
  x_test <- as.matrix(mtcars$wt)
  wt_test <- rep(1.0, nrow(mtcars))
  control_test <- list(
    center_tol = 1e-8,
    collin_tol = 1e-10,
    iter_center_max = 10000L,
    iter_interrupt = 1000L,
    iter_ssr = 100L
  )
  k_list_test <- list(list(`1` = seq_len(nrow(mtcars)) - 1L))
  
  cat("Input dimensions:\n")
  cat("y:", length(y_test), "\n")
  cat("x:", paste(dim(x_test), collapse="x"), "\n")
  cat("wt:", length(wt_test), "\n")
  cat("k_list length:", length(k_list_test), "\n")
  cat("k_list[[1]] length:", length(k_list_test[[1]]), "\n")
  cat("k_list[[1]][[1]] length:", length(k_list_test[[1]][[1]]), "\n")
  cat("k_list[[1]][[1]] range:", range(k_list_test[[1]][[1]]), "\n")
  
  # Call the C++ function
  result <- capybara:::felm_fit_(y_test, x_test, wt_test, control_test, k_list_test)
  cat("C++ call SUCCESS\n")
  cat("Result names:", names(result), "\n")
  
}, error = function(e) {
  cat("Error in C++ call:", e$message, "\n")
  cat("Traceback:\n")
  traceback()
})
