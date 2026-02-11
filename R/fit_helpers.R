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

#' @title Get FE codes
#' @description Returns flat 0-based integer factor codes for each FE variable
#'  plus level names. Uses match() + unique() to avoid expensive factor()
#'  conversion. Works on character, numeric, or factor columns equally.
#' @param k_vars Fixed effects variable names
#' @param data Data frame
#' @return A named list with two elements:
#'   \code{codes}: a list of K integer vectors (0-based codes, length N)
#'   \code{levels}: a list of K character vectors (unique level names)
#' @noRd
get_index_list_ <- function(k_vars, data) {
  codes <- vector("list", length(k_vars))
  lvls <- vector("list", length(k_vars))
  names(codes) <- k_vars
  names(lvls) <- k_vars
  for (i in seq_along(k_vars)) {
    v <- k_vars[i]
    x <- data[[v]]
    if (is.factor(x)) {
      x <- droplevels(x)
      codes[[i]] <- as.integer(x) - 1L
      lvls[[i]] <- levels(x)
    } else {
      u <- unique(x)
      codes[[i]] <- match(x, u) - 1L
      lvls[[i]] <- as.character(u)
    }
  }
  list(codes = codes, levels = lvls)
}

#' @title Get cluster index list
#' @description Generates an auxiliary list of indexes for cluster-robust
#'  standard errors. Keeps the old inverted-index format (list of integer
#'  vectors, one per group) since clusters use a different C++ path.
#' @param cl_var Single cluster variable name (character of length 1)
#' @param data Data frame
#' @return A list of integer vectors (1-based observation indices per group)
#' @noRd
get_cluster_list_ <- function(cl_var, data) {
  n <- nrow(data)
  split(seq_len(n), data[[cl_var]])
}

#' @title Model frame
#' @description Extracts needed columns from the data frame and handles weight
#'  extraction. NA removal is deferred to C++ for performance.
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

  # Extract needed columns only
  data <- data[, needed_cols, drop = FALSE]

  lhs <- names(data)[1L]
  nobs_full <- nrow(data)

  # Convert columns of type "units" to numeric (base R)
  unit_cols <- names(data)[vapply(data, inherits, what = "units", logical(1))]
  if (length(unit_cols) > 0) {
    data[unit_cols] <- lapply(data[unit_cols], as.numeric)
  }

  assign("data", data, envir = parent.frame())
  assign("lhs", lhs, envir = parent.frame())
  assign("nobs_full", nobs_full, envir = parent.frame())
}

#' @title Transform fixed effects
#' @description Drops unused factor levels for FE/cluster columns that are
#'  already factors. Non-factor columns (character, numeric) are left as-is
#'  since get_index_list_() and get_cluster_list_() handle them directly.
#' @param data Data frame
#' @param formula Formula object
#' @param k_vars Fixed effects
#' @noRd
transform_fe_ <- function(data, formula, k_vars) {
  # Only droplevels for columns that are already factors
  if (length(k_vars) > 0) {
    for (v in k_vars) {
      if (is.factor(data[[v]])) {
        data[[v]] <- droplevels(data[[v]])
      }
    }
  }

  if (length(formula)[[2L]] > 2L) {
    add_vars <- attr(terms(formula, rhs = 3L), "term.labels")
    if (length(add_vars) > 0) {
      existing_vars <- add_vars[add_vars %in% colnames(data)]
      for (v in existing_vars) {
        if (is.factor(data[[v]])) {
          data[[v]] <- droplevels(data[[v]])
        }
      }
    }
  }

  return(data)
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
