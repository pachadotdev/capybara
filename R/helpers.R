#' srr_stats
#' @srrstats {G2.0} Validates the integrity of inputs such as factors, formulas, data, and control parameters.
#' @srrstats {G2.1a} Ensures inputs have expected types and structures, such as formulas being of class `formula` and data being a `data.frame`.
#' @srrstats {G2.3a} Implements strict argument validation for ranges and constraints (e.g., numeric weights must be non-negative).
#' @srrstats {G2.3b} Converts inputs (e.g., character vectors) to appropriate formats when required, ensuring consistency.
#' @srrstats {G2.4a} Validates input arguments to ensure they meet expected formats and values, providing meaningful error messages for invalid inputs to guide users.
#' @srrstats {G2.4b} Implements checks to detect incompatible parameter combinations, preventing runtime errors and ensuring consistent function behavior.
#' @srrstats {G2.4c} Ensures numeric inputs (e.g., convergence thresholds, tolerances) are within acceptable ranges to avoid unexpected results.
#' @srrstats {G2.4d} Verifies the structure and completeness of input data, including the absence of missing values and correct dimensionality for matrices.
#' @srrstats {G2.4e} Issues warnings when deprecated or redundant arguments are used, encouraging users to adopt updated practices while maintaining backward compatibility.
#' @srrstats {G2.7} The input accepts data frames, tibbles and data table objects, from which it creates the design matrix.
#' @srrstats {G2.8} The pre-processing for all main functions (e.g., `feglm`, `felm`, `fepois`, `fenegbin`) is the same. The helper functions discard unusable observations dependening on the link function, and then create the design matrix.
#' @srrstats {G2.10} For data frames, tibbles and data tables the column-extraction operations are consistent.
#' @srrstats {G2.11} `data.frame`-like tabular objects which have can have atypical columns (i.e., `vector`) do not error without reason.
#' @srrstats {G2.13} Checks for and handles missing data in input datasets.
#' @srrstats {G2.14a} Issues informative errors for invalid inputs, such as incorrect link functions or missing data.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G2.15} The functions check for unusable observations (i.e., one column has an NA), and these are discarded before creating the design matrix.
#' @srrstats {G2.16} `NaN`, `Inf` and `-Inf` cannot be used for the design matrix, and all observations with these values are removed.
#' @srrstats {G5.2a} Ensures that all error and warning messages are unique and descriptive.
#' @srrstats {G5.4a} Includes tests for edge cases, such as binary and continuous response variables, and validates all input arguments.
#' @srrstats {RE4.4} The model is specified using a formula object, or a character-type object convertible to a formula, which is then used to create the design matrix.
#' @srrstats {RE4.5} Fitted models have an nobs element that can be called with `nobs()`.
#' @srrstats {RE4.12} The `check_data_()` function drops observations that are not useable with link function or that do not contribute to the log-likelihood.
#' @noRd
NULL

#' @title Transform factor
#' @description Checks if variable is a factor and transforms if necessary
#' @param x Variable to be checked
#' @noRd
check_factor_ <- function(x) {
  if (is.factor(x)) {
    droplevels(x)
  } else {
    factor(x)
  }
}

#' @title Temporary variable
#' @description Generates a temporary variable name
#' @param data Data frame
#' @noRd
temp_var_ <- function(data) {
  tmp_var <- "capybara_temp12345"
  while (tmp_var %in% colnames(data)) {
    tmp_var <- paste0("capybara_temp", sample(letters, 5, replace = TRUE))
  }
  tmp_var
}

#' @title Check formula
#' @description Checks formulas for LM/GLM/NegBin models
#' @param formula Formula object
#' @noRd
check_formula_ <- function(formula) {
  if (is.null(formula)) {
    stop("'formula' has to be specified.", call. = FALSE)
  } else if (!inherits(formula, "formula")) {
    stop("'formula' has to be of class 'formula'.", call. = FALSE)
  }

  formula <- Formula(formula)

  if (!any(grepl("\\|", formula[[3L]]))) {
    message(
      paste(
        "Perhaps you forgot to add the fixed effects like 'mpg ~ wt | cyl'",
        "You are better off using the 'lm()' function from base R."
      )
    )
  }

  assign("formula", formula, envir = parent.frame())
}

#' @title Check data
#' @description Checks data for LM/GLM/NegBin models
#' @param data Data frame
#' @noRd
check_data_ <- function(data) {
  if (is.null(data)) stop("'data' must be specified.", call. = FALSE)
  if (!is.data.frame(data)) stop("'data' must be a data.frame.", call. = FALSE)
  if (nrow(data) == 0L) stop("'data' has zero observations.", call. = FALSE)

  setDT(data) # Convert to data.table
}

#' @title Column types
#' @description Returns the column types of a data frame
#' @param data Data frame
#' @noRd
col_types <- function(data) {
  vapply(data, class, character(1L), USE.NAMES = FALSE)
}

#' @title Model frame
#' @description Creates model frame for LM/GLM/NegBin models
#' @param data Data frame
#' @param formula Formula object
#' @param weights Weights
#' @noRd
model_frame_ <- function(data, formula, weights) {
  # Necessary columns
  formula_vars <- all.vars(formula)

  # Handle different ways weights might be specified
  if (is.null(weights)) {
    # No weights specified
    weight_col <- NULL
    needed_cols <- formula_vars
  } else if (is.character(weights) && length(weights) == 1) {
    # Weights as column name
    weight_col <- weights
    needed_cols <- c(formula_vars, weight_col)
  } else if (inherits(weights, "formula")) {
    # Weights as formula like ~cyl
    weight_col <- all.vars(weights)
    needed_cols <- c(formula_vars, weight_col)
    # Store the extracted column name for later use
    assign("weights_col", weight_col, envir = parent.frame())
  } else if (is.numeric(weights)) {
    # Weights as vector - store for later use
    weight_col <- NULL
    needed_cols <- formula_vars
    assign("weights_vec", weights, envir = parent.frame())
  } else {
    stop("'weights' must be a column name, formula, or numeric vector", call. = FALSE)
  }

  # Extract needed columns
  data <- data[, .SD, .SDcols = needed_cols]

  lhs <- names(data)[1L]
  nobs_full <- nrow(data)
  data <- na.omit(data)

  # Convert columns of type "units" to numeric
  unit_cols <- names(data)[vapply(data, inherits, what = "units", logical(1))]
  if (length(unit_cols) > 0) {
    data[, (unit_cols) := lapply(.SD, as.numeric), .SDcols = unit_cols]
  }

  nobs_na <- nobs_full - nrow(data)

  assign("data", data, envir = parent.frame())
  assign("lhs", lhs, envir = parent.frame())
  assign("nobs_na", nobs_na, envir = parent.frame())
  assign("nobs_full", nobs_full, envir = parent.frame())
}

#' @title Transform fixed effects
#' @description Transforms fixed effects that are factors
#' @param data Data frame
#' @param formula Formula object
#' @param k_vars Fixed effects
#' @noRd
transform_fe_ <- function(data, formula, k_vars) {
  data[, (k_vars) := lapply(.SD, check_factor_), .SDcols = k_vars]

  if (length(formula)[[2L]] > 2L) {
    add_vars <- attr(terms(formula, rhs = 3L), "term.labels")
    data[, (add_vars) := lapply(.SD, check_factor_), .SDcols = add_vars]
  }

  return(data)
}

#' @title Number of observations
#' @description Computes the number of observations
#' @param nobs_full Number of observations in the full data set
#' @param nobs_na Number of observations with missing values (NA values)
#' @param y Dependent variable
#' @param yhat Predicted values
#' @noRd
nobs_ <- function(nobs_full, nobs_na, y, yhat) {
  # Use tolerance for floating-point comparisons
  tol <- sqrt(.Machine$double.eps)

  # Count observations with perfect prediction
  nobs_pc <- sum(abs(y - yhat) < tol, na.rm = TRUE)

  # Number of observations used in the model (length of predictions)
  nobs_used <- length(yhat)

  # Total missing observations (original NA values + dropped during fitting)
  total_missing <- nobs_full - nobs_used

  c(
    nobs_full = nobs_full, # Original dataset size
    nobs_na   = total_missing, # Total missing (NA + dropped)
    nobs_pc   = nobs_pc, # Perfect classification count
    nobs      = nobs_used # Observations used in model
  )
}

#' @title Model response
#' @description Computes the model response
#' @param data Data frame
#' @param formula Formula object
#' @noRd
model_response_ <- function(data, formula) {
  # Get the full model frame with BOTH response and predictors
  mf <- model.frame(formula, data)

  # Extract the transformed response
  y <- model.response(mf)

  # Create the model matrix for predictors
  x <- model.matrix(formula, mf)[, -1L, drop = FALSE]
  nms_sp <- colnames(x)
  attr(x, "dimnames") <- NULL

  # Check for Inf values in both model matrix and response
  if (any(is.infinite(x))) {
    stop("Infinite values detected in model matrix. This often happens when applying log() to zero or negative values in predictors.", call. = FALSE)
  }

  if (any(is.infinite(y))) {
    stop("Infinite values detected in response variable. This often happens when applying log() to zero or negative values.", call. = FALSE)
  }

  # Assign results to parent environment
  assign("y", y, envir = parent.frame())
  assign("x", x, envir = parent.frame())
  assign("nms_sp", nms_sp, envir = parent.frame())
  assign("p", ncol(x), envir = parent.frame())
  assign("data", data, envir = parent.frame()) # Update data frame
}

#' @title Check weights
#' @description Checks if weights are valid
#' @param wt Weights
#' @noRd
check_weights_ <- function(wt) {
  if (!is.numeric(wt) || anyNA(wt)) {
    stop("Weights must be numeric and non-missing.", call. = FALSE)
  }
  if (any(wt < 0)) {
    stop("Negative weights are not allowed.", call. = FALSE)
  }
}

#' @title Get index list
#' @description Generates an auxiliary list of indexes to project out the fixed
#'  effects
#' @param k_vars Fixed effects
#' @param data Data frame
#' @noRd
get_index_list_ <- function(k_vars, data) {
  indexes <- seq.int(0L, nrow(data) - 1L)
  lapply(k_vars, function(x, indexes, data) {
    split(indexes, data[[x]])
  }, indexes = indexes, data = data)
}

# time_function <- function(func, func_name, ...) {
#   start_time <- Sys.time()
#   result <- tryCatch(
#     {
#       func(...)
#     },
#     error = function(e) {
#       # If timing wrapper causes issues, call directly
#       func(...)
#     }
#   )
#   end_time <- Sys.time()
#   elapsed_ms <- as.numeric(difftime(end_time, start_time, units = "secs")) * 1000
#   cat(sprintf("[R timing] %s: %.0f ms\n", func_name, elapsed_ms), file = stderr())
#   return(result)
# }
