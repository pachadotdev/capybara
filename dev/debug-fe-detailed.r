# Debug fixed effects processing in detail
devtools::load_all()

cat("=== Detailed FE debugging ===\n")
data(mtcars)

# Simple case with minimal data
small_data <- mtcars[1:6, c("mpg", "wt", "cyl")]
cat("Small data:\n")
print(small_data)

formula <- mpg ~ wt | cyl

# Test the individual steps
cat("\n=== Step 1: check_formula_ ===\n")
tryCatch({
  check_formula_(formula)
  cat("Formula check - SUCCESS\n")
  cat("k_vars:", k_vars, "\n")
  cat("has_fe:", has_fe, "\n")
}, error = function(e) {
  cat("ERROR in check_formula_:", e$message, "\n")
})

cat("\n=== Step 2: model_frame_ ===\n")
tryCatch({
  # Initialize variables
  lhs <- NA
  nobs_na <- NA  
  nobs_full <- NA
  weights_vec <- NA
  weights_col <- NA
  
  model_frame_(small_data, formula, NULL)
  cat("Model frame - SUCCESS\n")
  cat("Data dimensions after model_frame_:", dim(data), "\n")
  print(head(data))
}, error = function(e) {
  cat("ERROR in model_frame_:", e$message, "\n")
})

cat("\n=== Step 3: transform_fe_ ===\n")
tryCatch({
  if (has_fe) {
    tmp_var <- temp_var_(data)
    data <- transform_fe_(data, formula, k_vars)
    cat("Transform FE - SUCCESS\n")
    cat("Data after transform_fe_:\n")
    print(str(data))
  }
}, error = function(e) {
  cat("ERROR in transform_fe_:", e$message, "\n")
})

cat("\n=== Step 4: model_response_ ===\n")
tryCatch({
  nms_sp <- NA
  p <- NA
  model_response_(data, formula)
  cat("Model response - SUCCESS\n")
  cat("y length:", length(y), "\n")
  cat("x dimensions:", dim(x), "\n")
  cat("y values:", y, "\n")
  cat("x values:\n")
  print(x)
}, error = function(e) {
  cat("ERROR in model_response_:", e$message, "\n")
})

cat("\n=== Step 5: FE structure ===\n")
tryCatch({
  if (has_fe) {
    nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
    lvls_k <- vapply(nms_fe, length, integer(1))
    cat("nms_fe:\n")
    print(nms_fe)
    cat("lvls_k:", lvls_k, "\n")
    
    k_list <- get_index_list_(k_vars, data)
    cat("k_list structure:\n")
    cat("Length of k_list:", length(k_list), "\n")
    for (i in seq_along(k_list)) {
      cat("FE", i, "- groups:", length(k_list[[i]]), "\n")
      for (j in seq_along(k_list[[i]])) {
        cat("  Group", j, "- indices:", k_list[[i]][[j]], "\n")
      }
    }
  }
}, error = function(e) {
  cat("ERROR in FE structure:", e$message, "\n")
})

cat("\n=== Step 6: C++ call ===\n")
tryCatch({
  control <- list(center_tol = 1e-8, collin_tol = 1e-10, iter_center_max = 10000L, 
                  iter_interrupt = 10000L, iter_ssr = 5000L, keep_mx = FALSE)
  wt <- rep(1.0, length(y))
  
  cat("About to call felm_fit_ with:\n")
  cat("y length:", length(y), "\n") 
  cat("x dimensions:", dim(x), "\n")
  cat("wt length:", length(wt), "\n")
  cat("k_list structure provided\n")
  
  result <- felm_fit_(y, x, wt, control, k_list)
  cat("C++ call - SUCCESS\n")
}, error = function(e) {
  cat("ERROR in C++ call:", e$message, "\n")
})
