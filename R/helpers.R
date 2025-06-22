#' @title Transform factor
#' @description Checks if variable is a factor and transforms if necessary
#' @param x Variable to be checked
#' @noRd
check_factor_ <- function(x) {
  if (is.factor(x)) {
    # Only call droplevels if there are actually unused levels
    # This is much faster than always calling droplevels
    lev <- levels(x)
    if (length(lev) > length(unique(x))) {
      droplevels(x)
    } else {
      x
    }
  } else {
    factor(x)
  }
}

#' @title Temporary variable
#' @description Generates a temporary variable name
#' @param data Data frame
#' @noRd
temp_var_ <- function(data) {
  tmp_var <- paste0("capybara_", gsub("\\.", "", gsub("\\s+|-|\\:", "", Sys.time())))
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
  }

  if (!inherits(formula, "formula")) {
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
#' @description Checks data for GLM/NegBin models
#' @param data Data frame
#' @noRd
check_data_ <- function(data) {
  if (is.null(data)) stop("'data' must be specified.", call. = FALSE)
  if (!is.data.frame(data)) stop("'data' must be a data.frame.", call. = FALSE)
  if (nrow(data) == 0L) stop("'data' has zero observations.", call. = FALSE)

  setDT(data) # Convert to data.table
}

#' @title Check control
#' @description Checks control for GLM/NegBin models and merges with defaults
#' @param control Control list
#' @noRd
check_control_ <- function(control) {
  default_control <- do.call(fit_control, list())

  if (is.null(control)) {
    assign("control", default_control, envir = parent.frame())
  } else if (!inherits(control, "list")) {
    stop("'control' has to be a list.", call. = FALSE)
  } else {
    # merge user-provided values with defaults
    merged_control <- default_control

    for (param_name in names(control)) {
      if (param_name %in% names(default_control)) {
        merged_control[[param_name]] <- control[[param_name]]
      } else {
        warning(sprintf("Unknown control parameter: '%s'", param_name), call. = FALSE)
      }
    }

    # checks
    # 1. non-negative params
    non_neg_params <- c(
      "dev_tol", "center_tol", "iter_max", "iter_center_max",
      "iter_inner_max", "iter_interrupt", "limit"
    )
    for (param_name in non_neg_params) {
      if (merged_control[[param_name]] <= 0) {
        stop(sprintf("'%s' must be greater than zero.", param_name), call. = FALSE)
      }
    }
    # 2. logical params
    logical_params <- c("trace", "drop_pc", "keep_mx")
    for (param_name in logical_params) {
      if (!is.logical(merged_control[[param_name]])) {
        stop(sprintf("'%s' must be logical.", param_name), call. = FALSE)
      }
    }

    assign("control", merged_control, envir = parent.frame())
  }
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
    stop("'weights' must be a column name, formula, or numeric vector", call. = FALSE)
  }

  # Extract needed columns more efficiently
  # Use intersect to avoid errors if columns don't exist
  available_cols <- intersect(needed_cols, names(data))
  if (length(available_cols) < length(needed_cols)) {
    missing <- setdiff(needed_cols, available_cols)
    stop(paste("Missing columns:", paste(missing, collapse = ", ")), call. = FALSE)
  }

  # More efficient subsetting
  data <- data[, .SD, .SDcols = available_cols]

  # Get lhs before NA removal
  lhs <- names(data)[1L]
  nobs_full <- nrow(data)

  # complete.cases is faster than na.omit and doesn't add attributes
  complete_rows <- complete.cases(data)
  nobs_na <- sum(!complete_rows)

  if (nobs_na > 0L) {
    data <- data[complete_rows]
  }

  # Convert columns of type "units" to numeric more efficiently
  # Check first to avoid unnecessary vapply if no units columns
  has_units <- FALSE
  for (j in seq_along(data)) {
    if (inherits(data[[j]], "units")) {
      has_units <- TRUE
      break
    }
  }

  if (has_units) {
    unit_cols <- names(data)[vapply(data, inherits, what = "units", logical(1L))]
    # Use set() for true in-place modification
    for (col in unit_cols) {
      set(data, j = col, value = as.numeric(data[[col]]))
    }
  }

  # Assign all values to parent frame
  assign("data", data, envir = parent.frame())
  assign("lhs", lhs, envir = parent.frame())
  assign("nobs_na", nobs_na, envir = parent.frame())
  assign("nobs_full", nobs_full, envir = parent.frame())
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

#' @title Transform fixed effects
#' @description Transforms fixed effects that are factors
#' @param data Data frame
#' @param formula Formula object
#' @param k_vars Fixed effects
#' @noRd
transform_fe_ <- function(data, formula, k_vars) {
  # Check which k_vars exist in data
  k_vars_exist <- intersect(k_vars, names(data))

  # Apply check_factor_ to all k_vars at once using lapply
  if (length(k_vars_exist) > 0L) {
    data[, (k_vars_exist) := lapply(.SD, check_factor_), .SDcols = k_vars_exist]
  }

  # Handle additional variables if present
  if (length(formula)[[2L]] > 2L) {
    add_vars <- attr(terms(formula, rhs = 3L), "term.labels")
    add_vars_exist <- intersect(add_vars, names(data))

    if (length(add_vars_exist) > 0L) {
      data[, (add_vars_exist) := lapply(.SD, check_factor_), .SDcols = add_vars_exist]
    }
  }

  data
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
#' @title Get index list
#' @description Generates an auxiliary list of indexes to project out the fixed effects
#' @param k_vars Fixed effects
#' @param data Data frame
#' @noRd
get_index_list_ <- function(k_vars, data) {
  indexes <- seq.int(0L, nrow(data) - 1L)
  lapply(k_vars, function(x, indexes, data) {
    split(indexes, data[[x]])
  }, indexes = indexes, data = data)
}
