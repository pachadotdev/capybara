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

  # Check that formula has a left-hand side (response variable)
  if (length(formula)[1] == 0) {
    stop(
      "'formula' must have a response variable (left-hand side).",
      call. = FALSE
    )
  }

  assign("formula", formula, envir = parent.frame())
}

#' @title Check data
#' @description Checks data for GLM/NegBin models
#' @param data Data frame
#' @noRd
check_data_ <- function(data) {
  if (is.null(data)) {
    stop("'data' must be specified.", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame.", call. = FALSE)
  }
  if (nrow(data) == 0L) stop("'data' has zero observations.", call. = FALSE)
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

    invisible(lapply(names(control), function(param_name) {
      if (param_name %in% names(default_control)) {
        merged_control[[param_name]] <<- control[[param_name]]
      } else {
        warning(
          sprintf("Unknown control parameter: '%s'", param_name),
          call. = FALSE
        )
      }
    }))

    # checks
    # 1. non-negative params
    non_neg_params <- c(
      "dev_tol",
      "center_tol",
      "collin_tol",
      "step_halving_factor",
      "alpha_tol",
      "sep_tol",
      "iter_max",
      "iter_center_max",
      "iter_inner_max",
      "iter_interrupt",
      "sep_max_iter",
      "iter_alpha_max",
      "step_halving_memory",
      "start_inner_tol"
    )
    invisible(lapply(non_neg_params, function(param_name) {
      if (
        param_name %in%
          names(merged_control) &&
          merged_control[[param_name]] <= 0
      ) {
        stop(
          sprintf("'%s' must be greater than zero.", param_name),
          call. = FALSE
        )
      }
    }))
    # 2. logical params
    logical_params <- c("return_fe", "keep_tx", "check_separation")
    invisible(lapply(logical_params, function(param_name) {
      if (
        param_name %in%
          names(merged_control) &&
          !is.logical(merged_control[[param_name]])
      ) {
        stop(sprintf("'%s' must be logical.", param_name), call. = FALSE)
      }
    }))

    assign("control", merged_control, envir = parent.frame())
  }
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
      if (v %in% colnames(data) && is.factor(data[[v]])) {
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

#' @title Process vcov argument
#' @description Validates and processes the vcov argument, updating control
#' @param vcov Character string specifying vcov type
#' @param control Control list to update
#' @return Named list with vcov_label and updated control
#' @noRd
process_vcov_ <- function(vcov, control) {
  vcov_label <- NULL
  if (!is.null(vcov)) {
    vcov <- match.arg(vcov, c("iid", "hetero", "cluster", "m-estimator", "dyadic"))
    vcov_label <- vcov
    if (vcov == "iid") {
      control$vcov_type <- NULL
    } else if (vcov == "hetero") {
      control$vcov_type <- "hetero"
    } else if (vcov == "cluster") {
      control$vcov_type <- NULL
    } else if (vcov == "m-estimator") {
      control$vcov_type <- "m-estimator"
    } else if (vcov == "dyadic") {
      control$vcov_type <- "m-estimator-dyadic"
    }
  }
  list(vcov_label = vcov_label, control = control)
}

#' @title Extract weight column name
#' @description Extracts the weight column name from weights argument
#' @param weights Weights specification (NULL, character, formula, or numeric)
#' @return Character string or NULL
#' @noRd
extract_weight_col_ <- function(weights) {
  if (is.null(weights)) {
    NULL
  } else if (is.character(weights) && length(weights) == 1L) {
    weights
  } else if (inherits(weights, "formula")) {
    all.vars(weights)
  } else {
    NULL
  }
}

#' @title Get needed columns from formula
#' @description Determines which columns are needed for model fitting and validates they exist.
#'   Expands the . operator using terms() and includes weights/offset columns.
#' @param formula Formula object
#' @param data Data frame
#' @param weights Weights specification (NULL, character, formula, or numeric)
#' @param offset Offset specification (NULL, formula, or numeric)
#' @return Named list with formula_vars (character vector) and needed_cols (character vector)
#' @noRd
get_needed_cols_ <- function(formula, data, weights = NULL, offset = NULL) {
  # Use all.vars first to check for . operator
  formula_vars_raw <- all.vars(formula)
  
  # If formula contains ".", we need to expand it using terms()
  # This handles formulas like y ~ . or y ~ . - x
  if ("." %in% formula_vars_raw) {
    # Extract base formula (before |) to expand . operator
    fml_chr <- deparse1(formula)
    parts <- trimws(strsplit(fml_chr, "\\|")[[1L]])
    base_part <- parts[[1L]]
    base_fml <- as.formula(base_part, env = environment(formula))
    tt <- terms(base_fml, data = data)
    formula_vars <- all.vars(tt)
  } else {
    formula_vars <- formula_vars_raw
  }
  
  weight_col <- extract_weight_col_(weights)
  offset_cols <- if (!is.null(offset) && inherits(offset, "formula")) {
    all.vars(offset)
  } else {
    NULL
  }
  needed_cols <- unique(c(formula_vars, weight_col, offset_cols))
  
  # Validate columns exist before subsetting
  missing_cols <- setdiff(needed_cols, names(data))
  if (length(missing_cols) > 0L) {
    stop("undefined columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  
  list(formula_vars = formula_vars, needed_cols = needed_cols)
}

#' @title Extract offset vector
#' @description Extracts offset from formula or numeric specification
#' @param offset Offset specification (NULL, formula, or numeric)
#' @param data Data frame
#' @param nobs Number of observations in data
#' @return Numeric vector or NULL
#' @noRd
extract_offset_ <- function(offset, data, nobs) {
  if (is.null(offset)) {
    return(NULL)
  }
  if (inherits(offset, "formula")) {
    offset_vars <- attr(terms(offset, data = data), "term.labels")
    if (length(offset_vars) != 1L) {
      stop("Offset formula must specify exactly one term.", call. = FALSE)
    }
    # Direct evaluation instead of eval(parse()) for speed
    offset_expr <- str2lang(offset_vars)
    eval(offset_expr, envir = data, enclos = parent.frame())
  } else if (is.numeric(offset)) {
    if (length(offset) != nobs) {
      stop("Length of offset must equal number of observations.", call. = FALSE)
    }
    offset
  } else {
    stop("Offset must be NULL, a formula, or a numeric vector.", call. = FALSE)
  }
}

#' @title Prepare data for fitting
#' @description Subsets columns, handles units, removes NAs
#' @param data Data frame
#' @param needed_cols Character vector of column names to keep
#' @param offset_vec Optional offset vector to subset in parallel
#' @param weights_vec Optional weights vector to subset in parallel
#' @return Named list with data, lhs, nobs_full, complete_idx, offset_vec, weights_vec
#' @noRd
prepare_data_ <- function(data, needed_cols, offset_vec = NULL, weights_vec = NULL) {
  # Validate all needed columns exist
  missing_cols <- setdiff(needed_cols, names(data))
  if (length(missing_cols) > 0L) {
    stop("undefined columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  # Preserve rownames before conversion
  orig_rn <- rownames(data)

  # Subset to needed columns
  data <- data[, needed_cols, drop = FALSE]

  lhs <- names(data)[[1L]]
  nobs_full <- nrow(data)

  # Convert "units" columns to numeric
  unit_cols <- names(data)[vapply(data, inherits, what = "units", logical(1))]
  for (uc in unit_cols) {
    data[[uc]] <- as.numeric(data[[uc]])
  }

  # Remove NA rows early (before creating y, X)
  complete_idx <- which(complete.cases(data))
  if (length(complete_idx) < nobs_full) {
    data <- data[complete_idx]
    if (!is.null(orig_rn)) orig_rn <- orig_rn[complete_idx]
    if (!is.null(offset_vec)) offset_vec <- offset_vec[complete_idx]
    if (!is.null(weights_vec)) weights_vec <- weights_vec[complete_idx]
  }

  # Store surviving rownames
  if (!is.null(orig_rn)) {
    attr(data, ".rownames") <- orig_rn
  }

  list(
    data = data,
    lhs = lhs,
    nobs_full = nobs_full,
    complete_idx = complete_idx,
    offset_vec = offset_vec,
    weights_vec = weights_vec
  )
}

#' @title Extract weights vector
#' @description Extracts weights from data or uses provided vector
#' @param weights Weights specification
#' @param weight_col Column name (or NULL)
#' @param data Data frame
#' @param nt Number of observations (after NA removal)
#' @param nobs_full Original number of observations
#' @param complete_idx Indices of complete cases
#' @return Numeric vector of weights
#' @noRd
extract_weights_ <- function(weights, weight_col, data, nt, nobs_full, complete_idx) {
  if (is.null(weights)) {
    rep(1.0, nt)
  } else if (is.numeric(weights)) {
    if (length(weights) != nobs_full) {
      stop("Length of weights vector must equal number of observations.", call. = FALSE)
    }
    if (length(complete_idx) < nobs_full) weights[complete_idx] else weights
  } else {
    data[[weight_col]]
  }
}

#' @title Build design matrix
#' @description Builds design matrix using fast or slow path
#' @param data Data frame
#' @param formula Formula object
#' @return Named list with X (matrix), nms_sp (column names), p (ncol)
#' @noRd
build_design_matrix_ <- function(data, formula) {
  f1 <- formula(formula, lhs = 1L, rhs = 1L)
  tt <- terms(f1)
  rhs_labels <- attr(tt, "term.labels")

  # Determine fast vs slow path
  use_fast <- FALSE
  if (length(rhs_labels) > 0L) {
    # Fast path only when all rhs terms are plain column names (no transformations)
    all_are_columns <- all(rhs_labels %in% colnames(data))
    if (all_are_columns) {
      all_numeric <- all(vapply(data[rhs_labels], is.numeric, logical(1)))
      has_interaction <- any(grepl(":", rhs_labels, fixed = TRUE))
      use_fast <- all_numeric && !has_interaction
    }
  }

  if (use_fast) {
    # Fast path: extract columns directly as matrix
    if (length(rhs_labels) == 1L) {
      X <- matrix(data[[rhs_labels]], ncol = 1L)
    } else {
      X <- as.matrix(data[rhs_labels])
    }
    nms_sp <- rhs_labels
  } else {
    # Slow path: model.frame + model.matrix
    # Use data directly instead of subsetting (avoids copy)
    mf <- model.frame(f1, data, na.action = na.pass)
    X <- model.matrix(tt, mf)[, -1L, drop = FALSE]
    nms_sp <- colnames(X)
    attr(X, "dimnames") <- NULL
  }

  list(X = X, nms_sp = nms_sp, p = ncol(X), terms = tt)
}

#' @title Extract response variable
#' @description Extracts and converts response from data
#' @param data Data frame
#' @param formula Formula object
#' @return Numeric response vector
#' @noRd
extract_response_ <- function(data, formula) {
  f1 <- formula(formula, lhs = 1L, rhs = 1L)
  tt <- terms(f1)
  resp_call <- attr(tt, "variables")[[2L]]
  y <- eval(resp_call, data)
  if (is.integer(y)) y <- as.numeric(y)
  y
}

#' @title Get FE and cluster variable names
#' @description Extracts FE and cluster variable names from formula
#' @param formula Formula object
#' @return Named list with fe_vars and cl_vars
#' @noRd
get_fe_cl_vars_ <- function(formula) {
  fe_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
  if (length(fe_vars) < 1L) fe_vars <- character(0)

  cl_vars <- suppressWarnings(attr(terms(formula, rhs = 3L), "term.labels"))
  if (length(cl_vars) < 1L) cl_vars <- character(0)

  list(fe_vars = fe_vars, cl_vars = cl_vars)
}
