# Debug the missing_fe issue
devtools::load_all()

cat("=== Debugging missing_fe case ===\n")
data(mtcars)

# Replicate exactly what felm() does for no FE case
formula <- mpg ~ wt
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

# Get k_vars (should be empty, so will set to "missing_fe")
k_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
cat("k_vars:", k_vars, "\n")

if (length(k_vars) < 1L) {
  k_vars <- "missing_fe"
  data[, `:=`("missing_fe", 1L)]
  cat("Added missing_fe column\n")
}

# Transform fixed effects
data <- transform_fe_(data, formula, k_vars)

# Get nt
nt <- nrow(data)
cat("nt (nrow after processing):", nt, "\n")

# Extract response and regressor matrix
nms_sp <- NA
p <- NA
model_response_(data, formula)

cat("After model_response_:\n")
cat("y length:", length(y), "\n")
cat("x dimensions:", dim(x), "\n")

# Get lvls_k and k_list
nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
if (length(nms_fe) > 0L) {
  lvls_k <- vapply(nms_fe, length, integer(1))
} else {
  lvls_k <- c("missing_fe" = 1L)
}

cat("lvls_k:", lvls_k, "\n")

# Generate k_list
if (!any(lvls_k %in% "missing_fe")) {
  k_list <- get_index_list_(k_vars, data)
} else {
  k_list <- list(list(`1` = seq_len(nt) - 1L))
}

cat("k_list structure:\n")
cat("Length:", length(k_list), "\n")
cat("First element length:", length(k_list[[1]]), "\n")
cat("Indices range:", range(k_list[[1]][[1]]), "\n")
cat("y length for comparison:", length(y), "\n")

# Try the C++ call
control <- list(center_tol = 1e-8, collin_tol = 1e-10, iter_center_max = 10000L, 
                iter_interrupt = 10000L, iter_ssr = 5000L)
wt <- rep(1.0, nt)

cat("\nTrying felm_fit_ call...\n")
tryCatch({
  result <- felm_fit_(y, x, wt, control, k_list)
  cat("SUCCESS!\n")
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
})
