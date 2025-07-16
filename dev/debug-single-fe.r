# Debug single FE case step by step
devtools::load_all()

cat("=== Debugging single FE step by step ===\n")
data(mtcars)

# Follow exactly what happens in felm() with FE
formula <- mpg ~ wt | cyl
data <- mtcars

# Check validity of data 
check_data_(data)

# Generate model.frame (this sets some variables in parent environment)
lhs <- NA
nobs_na <- NA  
nobs_full <- NA
weights_vec <- NA
weights_col <- NA
model_frame_(data, formula, weights = NULL)

cat("After model_frame_:\n")
cat("nobs_full:", nobs_full, "\n")
cat("nobs_na:", nobs_na, "\n")

# Get k_vars
k_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
cat("k_vars:", k_vars, "\n")

has_fe <- length(k_vars) >= 1L
cat("has_fe:", has_fe, "\n")

if (has_fe) {
  # Generate temporary variable
  tmp_var <- temp_var_(data)
  cat("tmp_var:", tmp_var, "\n")
  
  cat("About to call transform_fe_...\n")
  tryCatch({
    data <- transform_fe_(data, formula, k_vars)
    cat("transform_fe_ successful\n")
  }, error = function(e) {
    cat("ERROR in transform_fe_:", e$message, "\n")
    return()
  })
}

# Continue with data processing
nt <- nrow(data)
cat("nt (nrow after processing):", nt, "\n")

# Extract response and regressor matrix
nms_sp <- NA
p <- NA

cat("About to call model_response_...\n")
tryCatch({
  model_response_(data, formula)
  cat("model_response_ successful\n")
  cat("y length:", length(y), "\n")
  cat("x dimensions:", dim(x), "\n")
}, error = function(e) {
  cat("ERROR in model_response_:", e$message, "\n")
  return()
})

# Get FE structure
if (has_fe) {
  cat("Getting FE structure...\n")
  tryCatch({
    nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
    lvls_k <- vapply(nms_fe, length, integer(1))
    cat("lvls_k:", lvls_k, "\n")
    
    cat("About to call get_index_list_...\n")
    k_list <- get_index_list_(k_vars, data)
    cat("k_list created successfully\n")
    cat("k_list structure:\n")
    cat("Length:", length(k_list), "\n")
    if (length(k_list) > 0) {
      cat("First element length:", length(k_list[[1]]), "\n")
      if (length(k_list[[1]]) > 0) {
        cat("First group indices:", k_list[[1]][[1]], "\n")
        cat("Indices range:", range(k_list[[1]][[1]]), "\n")
      }
    }
  }, error = function(e) {
    cat("ERROR in FE processing:", e$message, "\n")
    return()
  })
} else {
  nms_fe <- list()
  lvls_k <- integer(0)
  k_list <- list()
}

# Try the C++ call
control <- list(center_tol = 1e-8, collin_tol = 1e-10, iter_center_max = 10000L, 
                iter_interrupt = 10000L, iter_ssr = 5000L)
wt <- rep(1.0, nt)

cat("\nTrying felm_fit_ call...\n")
cat("y length:", length(y), "\n")
cat("x dimensions:", dim(x), "\n") 
cat("wt length:", length(wt), "\n")
cat("k_list length:", length(k_list), "\n")

tryCatch({
  result <- felm_fit_(y, x, wt, control, k_list)
  cat("SUCCESS!\n")
}, error = function(e) {
  cat("ERROR in felm_fit_:", e$message, "\n")
  cat("Traceback:\n")
  traceback()
})
