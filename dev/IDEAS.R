#' @title Bulk QUF preprocessing for fixed effects
#' @description Preprocesses all fixed effects variables using QUF for optimal performance.
#'   This function should be called before running multiple models on the same dataset.
#' @param data Data frame
#' @param fe_vars Character vector of fixed effects variable names
#' @param min_size Minimum dataset size to trigger QUF optimization (default: 1000)
#' @param min_levels Minimum number of unique levels to trigger QUF optimization (default: 100)
#' @return Modified data frame with QUF-optimized factors
#' @noRd
preprocess_fe_quf_ <- function(data, fe_vars, min_size = 1000, min_levels = 100) {
  if (nrow(data) < min_size) {
    return(data)  # Skip optimization for small datasets
  }

  optimized_vars <- character(0)
  total_speedup <- 0

  for (var in fe_vars) {
    if (!var %in% names(data)) next

    var_data <- data[[var]]
    n_unique <- length(unique(var_data))

    if (n_unique > min_levels) {
      # Apply QUF optimization
      time_start <- Sys.time()
      quf_result <- quf(var_data)
      time_quf <- as.numeric(difftime(Sys.time(), time_start, units = "secs"))

      # Create optimized factor
      factor_levels <- quf_result$unique[order(quf_result$unique)]
      optimized_factor <- factor(quf_result$unique[quf_result$quf], levels = factor_levels)

      # Store QUF metadata
      attr(optimized_factor, "quf_codes") <- quf_result$quf
      attr(optimized_factor, "quf_unique") <- quf_result$unique
      attr(optimized_factor, "quf_n_unique") <- quf_result$n_unique
      attr(optimized_factor, "quf_time") <- time_quf

      # Update data
      data[[var]] <- optimized_factor
      optimized_vars <- c(optimized_vars, var)
      total_speedup <- total_speedup + time_quf
    }
  }

  if (length(optimized_vars) > 0) {
    message(sprintf("QUF preprocessing applied to %d variables (%s) in %.3f seconds",
                   length(optimized_vars),
                   paste(optimized_vars, collapse = ", "),
                   total_speedup))
  }

  return(data)
}

#' Optimize Dataset for Fixed Effects Models using QUF
#'
#' Preprocesses fixed effects variables using QUF (Quick Unclass Factor)
#' optimization for improved performance in subsequent model fitting. This is
#' particularly beneficial for large datasets with high-cardinality factor
#' variables.
#'
#' @param data A data.frame or data.table containing the dataset
#' @param fe_vars Character vector of fixed effects variable names to optimize
#' @param min_size Minimum dataset size to trigger QUF optimization
#'   (default: 1000)
#' @param min_levels Minimum number of unique levels to trigger optimization
#'   (default: 100)
#' @param verbose Logical indicating whether to print optimization details
#'   (default: TRUE)
#'
#' @return The input dataset with QUF-optimized fixed effects variables.
#'   Optimized variables will have QUF metadata stored as attributes.
#'
#' @details
#' This function applies the QUF (Quick Unclass Factor) algorithm to
#' preprocess fixed effects variables, providing significant performance
#' improvements for:
#'
#' - Large datasets (> 1,000 observations)
#' - High-cardinality factors (> 100 unique levels)
#' - Repeated model fitting on the same dataset
#'
#' The QUF algorithm uses radix-based sorting and intelligent algorithm
#' selection to optimize factor-to-integer conversion, which is a bottleneck
#' in fixed effects model estimation.
#'
#' @examples
#' \dontrun{
#' library(data.table)
#'
#' # Large dataset with high-cardinality factors
#' dt <- data.table(
#'   firm_id = sample(1:1000, 50000, replace = TRUE),
#'   year_id = sample(2000:2020, 50000, replace = TRUE),
#'   y = rnorm(50000),
#'   x = rnorm(50000)
#' )
#'
#' # Optimize for better performance
#' dt_optimized <- optimize_for_feglm(dt, c("firm_id", "year_id"))
#'
#' # Run models - will be faster due to preprocessing
#' mod1 <- feglm(y ~ x | firm_id + year_id, data = dt_optimized)
#' mod2 <- feglm(y ~ I(x^2) | firm_id + year_id, data = dt_optimized)
#' }
#'
#' @seealso \code{\link{quf}} for the underlying QUF implementation
#' @export
optimize_for_feglm <- function(data, fe_vars, min_size = 1000,
                               min_levels = 100, verbose = TRUE) {
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame or data.table", call. = FALSE)
  }

  if (!is.character(fe_vars) || length(fe_vars) == 0) {
    stop("'fe_vars' must be a non-empty character vector", call. = FALSE)
  }

  # Check that all variables exist
  missing_vars <- setdiff(fe_vars, names(data))
  if (length(missing_vars) > 0) {
    stop("Variables not found in data: ", paste(missing_vars, collapse = ", "),
         call. = FALSE)
  }

  # Convert to data.table for efficient operations
  dt <- data.table::as.data.table(data)

  # Skip optimization for small datasets
  if (nrow(dt) < min_size) {
    if (verbose) {
      message(sprintf(
        "Dataset too small (%d < %d observations), skipping QUF optimization",
        nrow(dt), min_size
      ))
    }
    return(data)
  }

  optimized_vars <- character(0)
  total_time <- 0
  total_compression <- 0

  for (var in fe_vars) {
    var_data <- dt[[var]]
    n_unique <- data.table::uniqueN(var_data)

    if (n_unique > min_levels) {
      if (verbose) {
        cat(sprintf("Optimizing %s (%d levels)... ", var, n_unique))
      }

      # Apply QUF optimization
      time_start <- Sys.time()
      quf_result <- quf(var_data)
      time_elapsed <- as.numeric(difftime(Sys.time(), time_start,
                                          units = "secs"))

      # Create optimized factor with proper ordering
      factor_levels <- quf_result$unique[order(quf_result$unique)]
      optimized_factor <- factor(quf_result$unique[quf_result$quf],
                                 levels = factor_levels)

      # Store QUF metadata for reuse
      attr(optimized_factor, "quf_codes") <- quf_result$quf
      attr(optimized_factor, "quf_unique") <- quf_result$unique
      attr(optimized_factor, "quf_n_unique") <- quf_result$n_unique
      attr(optimized_factor, "quf_time") <- time_elapsed
      attr(optimized_factor, "quf_compression") <- length(var_data) /
        quf_result$n_unique

      # Update data
      dt[[var]] <- optimized_factor
      optimized_vars <- c(optimized_vars, var)
      total_time <- total_time + time_elapsed
      total_compression <- total_compression +
        attr(optimized_factor, "quf_compression")

      if (verbose) {
        cat(sprintf("%.3f sec (%.1fx compression)\n", time_elapsed,
                    attr(optimized_factor, "quf_compression")))
      }
    } else if (verbose) {
      cat(sprintf("Skipping %s (%d levels < %d threshold)\n", var,
                  n_unique, min_levels))
    }
  }

  if (length(optimized_vars) > 0 && verbose) {
    avg_compression <- total_compression / length(optimized_vars)
    message(sprintf(
      "\n✅ QUF optimization completed for %d variables in %.3f seconds",
      length(optimized_vars), total_time
    ))
    message(sprintf("   Average compression ratio: %.1fx", avg_compression))
    message("   Optimized variables:", paste(optimized_vars, collapse = ", "))
    message("   Subsequent feglm() calls on this dataset will be faster!")
  }

  # Return in original format
  if (data.table::is.data.table(data)) {
    return(dt)
  } else {
    return(as.data.frame(dt))
  }
}
