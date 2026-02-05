#' srr_stats
#' @srrstats {G1.0} Implements controls for efficient and numerically stable fitting of generalized linear models with
#'  fixed effects.
#' @srrstats {G2.0} Validates the integrity of inputs such as factors, formulas, data, and control parameters.
#' @srrstats {G2.0a} Gives informative errors (e.g. "tolerance must be unidimensional").
#' @srrstats {G2.1a} Ensures inputs have expected types and structures, such as formulas being of class `formula` and
#'  data being a `data.frame`. Ensures the proper data types for arguments (e.g., integer for `iter_max`).
#' @srrstats {G2.3a} Implements strict argument validation for ranges and constraints (e.g., numeric weights must be
#'  non-negative).
#' @srrstats {G2.3b} Converts inputs (e.g., character vectors) to appropriate formats when required, ensuring
#'  consistency.
#' @srrstats {G2.4a} Validates input arguments to ensure they meet expected formats and values, providing meaningful
#'  error messages for invalid inputs to guide users.
#' @srrstats {G2.4b} Implements checks to detect incompatible parameter combinations, preventing runtime errors and
#'  ensuring consistent function behavior.
#' @srrstats {G2.4c} Ensures numeric inputs (e.g., convergence thresholds, tolerances) are within acceptable ranges to
#'  avoid unexpected results.
#' @srrstats {G2.4d} Verifies the structure and completeness of input data, including the absence of missing values and
#'  correct dimensionality for matrices.
#' @srrstats {G2.4e} Issues warnings when deprecated or redundant arguments are used, encouraging users to adopt updated
#'  practices while maintaining backward compatibility.
#' @srrstats {G2.7} The input accepts data frames, tibbles and data table objects, from which it creates the design
#'  matrix.
#' @srrstats {G2.8} The pre-processing for all main functions (e.g., `feglm`, `felm`, `fepois`, `fenegbin`) is the same.
#'  The helper functions discard unusable observations dependening on the link function, and then create the design
#'  matrix.
#' @srrstats {G2.10} For data frames, tibbles and data tables the column-extraction operations are consistent.
#' @srrstats {G2.11} `data.frame`-like tabular objects which have can have atypical columns (i.e., `vector`) do not
#'  error without reason.
#' @srrstats {G2.13} Checks for and handles missing data in input datasets.
#' @srrstats {G2.14a} Issues informative errors for invalid inputs, such as incorrect link functions or missing data.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G2.15} The functions check for unusable observations (i.e., one column has an NA), and these are discarded
#'  before creating the design matrix.
#' @srrstats {G2.16} `NaN`, `Inf` and `-Inf` cannot be used for the design matrix, and all observations with these
#'  values are removed.
#' @srrstats {G5.2a} Ensures that all error and warning messages are unique and descriptive. All parameter validations
#'  provide clear error messages
#' @srrstats {G5.4a} Includes tests for edge cases, such as binary and continuous response variables, and validates all
#'  input arguments.
#' @srrstats {RE3.0} If the deviance difference between 2 iterations is not less than tolerance after the max number of
#'  iterations, it prints a convergence warning.
#' @srrstats {RE4.4} The model is specified using a formula object, or a character-type object convertible to a formula,
#'  which is then used to create the design matrix.
#' @srrstats {RE4.5} Fitted models have an nobs element that can be called with `nobs()`.
#' @srrstats {RE4.12} The `check_data_()` function drops observations that are not useable with link function or that do
#'  not contribute to the log-likelihood.
#' @srrstats {RE5.0} Supports control over algorithmic complexity, such as dropping perfectly separated observations
#'  (`drop_pc`) and optional matrix storage (`keep_dmx`).
#' @noRd
NULL



#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing
#'  imputation methods would bias the estimation (i.e., replacing all missing
#'  values with the median).
#' @noRd
NULL

#' @title Get index list
#' @description Generates an auxiliary list of indexes to project out the fixed effects (on C++ side the outputs are
#'  0-indexed)
#' @param k_vars Fixed effects
#' @param data Data frame
#' @noRd
get_index_list_ <- function(k_vars, data) {
  n <- nrow(data)
  lapply(k_vars, function(v) split(seq_len(n), data[[v]]))
}

#' @title Model frame
#' @description Creates model frame for GLM/NegBin models
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
    stop(
      "'weights' must be a column name, formula, or numeric vector",
      call. = FALSE
    )
  }

  # Extract needed columns (base R)
  data <- data[, needed_cols, drop = FALSE]

  lhs <- names(data)[1L]
  nobs_full <- nrow(data)
  data <- na.omit(data)

  # Convert columns of type "units" to numeric (base R)
  unit_cols <- names(data)[vapply(data, inherits, what = "units", logical(1))]
  if (length(unit_cols) > 0) {
    data[unit_cols] <- lapply(data[unit_cols], as.numeric)
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
  if (length(k_vars) > 0) {
    data[k_vars] <- lapply(data[k_vars], check_factor_)
  }

  if (length(formula)[[2L]] > 2L) {
    add_vars <- attr(terms(formula, rhs = 3L), "term.labels")
    # Only transform cluster variables if they exist in data
    # (they're not needed for prediction, only for fitting)
    if (length(add_vars) > 0) {
      existing_vars <- add_vars[add_vars %in% colnames(data)]
      if (length(existing_vars) > 0) {
        data[existing_vars] <- lapply(data[existing_vars], check_factor_)
      }
    }
  }

  return(data)
}


#' @title Number of observations
#' @description Computes the number of observations
#' @param nobs_full Number of observations in the full data set
#' @param nobs_na Number of observations with missing values (NA values)
#' @param y Dependent variable
#' @param yhat Predicted values
#' @param num_separated Number of separated observations (default 0)
#' @noRd
nobs_ <- function(nobs_full, nobs_na, y, yhat, num_separated = 0) {
  # Use tolerance for floating-point comparisons
  tol <- sqrt(.Machine$double.eps)

  # Count non-NA fitted values (excludes separated/dropped observations)
  nobs_used <- sum(!is.na(yhat))

  # Count observations with perfect prediction (among non-NA)
  if (nobs_used > 0) {
    nobs_pc <- sum(
      abs(y[!is.na(yhat)] - yhat[!is.na(yhat)]) < tol,
      na.rm = TRUE
    )
  } else {
    nobs_pc <- 0
  }

  # Separated observations are tracked separately from missing
  # Total dropped = original NA + singletons (separated tracked separately)
  total_dropped <- nobs_full - nobs_used - num_separated

  c(
    nobs_full = nobs_full, # Original dataset size
    nobs_na = total_dropped, # Missing/dropped (NA + singletons, excluding separated)
    nobs_separated = num_separated, # Separated observations
    nobs_pc = nobs_pc, # Perfect classification count
    nobs = nobs_used # Observations used in model
  )
}

#' @title Ensure fixed effects variables
#' @description Extracts fixed effect variable names from formula
#' @param formula Formula object
#' @param data Data frame
#' @return Character vector of fixed effect variable names (empty if none)
#' @noRd
check_fe_ <- function(formula, data) {
  fe_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
  if (length(fe_vars) < 1L) {
    fe_vars <- character(0)
  }
  fe_vars
}
