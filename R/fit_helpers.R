#' srr_stats
#' #' @srrstats {G1.0} Implements controls for efficient and numerically stable
#'  fitting of generalized linear models with fixed effects.
#' @srrstats {G2.0} Validates the integrity of inputs such as
#'  factors, formulas, data, and control parameters.
#' @srrstats {G2.0a} Gives informative errors (e.g. "tolerance must be
#'  unidimensional").
#' @srrstats {G2.1a} Ensures inputs have expected types and structures, such as
#'  formulas being of class `formula` and data being a `data.frame`.
#'  Ensures the proper data types for arguments (e.g., integer for `iter_max`).
#' @srrstats {G2.3a} Implements strict argument validation for ranges and
#'  constraints (e.g., numeric weights must be non-negative).
#' @srrstats {G2.3b} Converts inputs (e.g., character vectors) to appropriate
#'  formats when required, ensuring consistency.
#' @srrstats {G2.4a} Validates input arguments to ensure they meet expected
#'  formats and values, providing meaningful error messages for invalid inputs
#'  to guide users.
#' @srrstats {G2.4b} Implements checks to detect incompatible parameter
#'  combinations, preventing runtime errors and ensuring consistent function
#'  behavior.
#' @srrstats {G2.4c} Ensures numeric inputs (e.g., convergence thresholds,
#'  tolerances) are within acceptable ranges to avoid unexpected results.
#' @srrstats {G2.4d} Verifies the structure and completeness of input data,
#'  including the absence of missing values and correct dimensionality for
#'  matrices.
#' @srrstats {G2.4e} Issues warnings when deprecated or redundant arguments are
#'  used, encouraging users to adopt updated practices while maintaining
#'  backward compatibility.
#' @srrstats {G2.7} The input accepts data frames, tibbles and data table
#'  objects, from which it creates the design matrix.
#' @srrstats {G2.8} The pre-processing for all main functions (e.g., `feglm`,
#'  `felm`, `fepois`, `fenegbin`) is the same. The helper functions discard
#'  unusable observations dependening on the link function, and then create the
#'  design matrix.
#' @srrstats {G2.10} For data frames, tibbles and data tables the
#'  column-extraction operations are consistent.
#' @srrstats {G2.11} `data.frame`-like tabular objects which have can have
#'  atypical columns (i.e., `vector`) do not error without reason.
#' @srrstats {G2.13} Checks for and handles missing data in input datasets.
#' @srrstats {G2.14a} Issues informative errors for invalid inputs, such as
#'  incorrect link functions or missing data.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is
#'  incompatible with the model requirements.
#' @srrstats {G2.15} The functions check for unusable observations (i.e.,
#'  one column has an NA), and these are discarded before creating the design
#'  matrix.
#' @srrstats {G2.16} `NaN`, `Inf` and `-Inf` cannot be used for the design
#'  matrix, and all observations with these values are removed.
#' @srrstats {G5.2a} Ensures that all error and warning messages are unique and
#'  descriptive. All parameter validations provide clear error messages
#' @srrstats {G5.4a} Includes tests for edge cases, such as binary and
#'  continuous response variables, and validates all input arguments.
#' @srrstats {RE3.0} If the deviance difference between 2 iterations is not less
#'  than tolerance after the max number of iterations, it
#'  prints a convergence warning.
#' @srrstats {RE4.4} The model is specified using a formula object, or a
#'  character-type object convertible to a formula, which is then used to create
#'  the design matrix.
#' @srrstats {RE4.5} Fitted models have an nobs element that can be called with
#'  `nobs()`.
#' @srrstats {RE4.12} The `check_data_()` function drops observations that are
#'  not useable with link function or that do not contribute to the
#'  log-likelihood.
#' @srrstats {RE5.0} Supports control over algorithmic complexity, such as
#'  dropping perfectly separated observations (`drop_pc`) and optional matrix
#'  storage (`keep_dmx`).
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing
#'  imputation methods would bias the estimation (i.e., replacing all missing
#'  values with the median).
#' @noRd
NULL

#' @title Set \code{feglm} and \code{felm} Control Parameters
#'
#' @description Set and change parameters used for fitting \code{\link{feglm}}
#'  and \code{felm}. Termination conditions are similar to
#'  \code{\link[stats]{glm}}.
#'
#' @param dev_tol tolerance level for the first stopping condition of the
#'  maximization routine. The stopping condition is based on the relative change
#'  of the deviance in iteration \eqn{r} and can be expressed as follows:
#'  \eqn{|dev_{r} - dev_{r - 1}| / (0.1 + |dev_{r}|) < tol}{|dev - devold| /
#'  (0.1 + |dev|) < tol}. The default is \code{1.0e-08}.
#' @param demean_tol tolerance level for the stopping condition of the demeaning
#'  algorithm. The default is \code{1.0e-08}.
#' @param collin_tol tolerance level for detecting collinearity. The default is
#'  \code{1.0e-07}.
#' @param iter_max maximum number of GLM iterations. The default is \code{25L}.
#' @param iter_max_cluster maximum number of iterations for cluster
#'  coefficient convergence in (negative) binomial models. The default is \code{100L}.
#' @param iter_full_dicho maximum number of full Newton-Raphson iterations
#'  before switching to dichotomy for cluster coefficient convergence in
#'  (negative) binomial models. The default is \code{10L}.
#' @param iter_demean_max unsigned integer indicating the maximum number of
#'  iterations in the demeaning algorithm. The default is \code{10000L}.
#' @param iter_inner_max maximum number of step-halving iterations. The default
#'  is \code{50L}.
#' @param iter_interrupt interruption frequency for user interrupt checks.
#'  The default is \code{1000L}.
#' @param iter_ssr frequency for SSR-based convergence checks. The default
#'  is \code{10L}.
#' @param rel_tol_denom denominator for relative tolerance in convergence
#'  criterion. The default is \code{0.1}.
#' @param convergence_iter_max maximum iterations for cluster coefficient
#'  convergence (Newton-Raphson + dichotomy). The default is \code{100L}.
#' @param convergence_iter_full_dicho number of iterations using full
#'  Newton-Raphson before switching to dichotomy. The default is \code{10L}.
#' @param step_halving_factor step size reduction factor in step-halving
#'  algorithm. The default is \code{0.5}.
#' @param binomial_mu_min minimum value for mu in binomial family. The default
#'  is \code{0.001}.
#' @param binomial_mu_max maximum value for mu in binomial family. The default
#'  is \code{0.999}.
#' @param safe_clamp_min minimum value for safe clamping operations. The default
#'  is \code{1.0e-15}.
#' @param safe_clamp_max maximum value for safe clamping operations. The default
#'  is \code{1.0e12}.
#' @param direct_qr_threshold threshold for using direct QR vs Cholesky
#'  decomposition. The default is \code{0.9}.
#' @param qr_collin_tol_multiplier multiplier for QR collinearity tolerance.
#'  The default is \code{1.0}.
#' @param chol_stability_threshold threshold for Cholesky stability check.
#'  The default is \code{1.0e-12}.
#' @param safe_division_min minimum value for safe division operations.
#'  The default is \code{1.0e-12}.
#' @param safe_log_min minimum value for safe logarithm operations.
#'  The default is \code{1.0e-12}.
#' @param newton_raphson_tol tolerance for Newton-Raphson convergence.
#'  The default is \code{1.0e-08}.
#' @param irons_tuck_eps tolerance for Irons-Tuck acceleration numerical
#'  convergence. The default is \code{1.0e-14}.
#' @param alpha_convergence_tol tolerance for fixed effects (alpha) convergence.
#'  The default is \code{1.0e-08}.
#' @param alpha_iter_max maximum iterations for fixed effects computation.
#'  The default is \code{10000L}.
#' @param demean_extra_projections number of extra projections in demeaning
#'  algorithm. The default is \code{0L}.
#' @param demean_warmup_iterations number of warmup iterations in demeaning.
#'  The default is \code{15L}.
#' @param demean_projections_after_acc projections after acceleration in
#'  demeaning. The default is \code{5L}.
#' @param demean_grand_acc_frequency frequency of grand acceleration in
#'  demeaning. The default is \code{20L}.
#' @param demean_ssr_check_frequency frequency of SSR checks in demeaning.
#'  The default is \code{40L}.
#' @param keep_dmx logical indicating if the demeaned design matrix should be
#'  stored. The demeaned design matrix is required for some covariance
#'  estimators, bias corrections, and average partial effects. This option saves
#'  some computation time at the cost of memory. The default is \code{FALSE}.
#' @param use_weights logical indicating whether to use weights. If \code{FALSE},
#'  weights are ignored for performance. The default is \code{TRUE}.
#'
#' @return A named list of control parameters.
#'
#' @examples
#' fit_control(dev_tol = 1e-6, demean_tol = 1e-6)
#'
#' @seealso \code{\link{feglm}}
#'
#' @export
fit_control <- function(
    # Core tolerance parameters
    dev_tol = 1.0e-8,
    demean_tol = 1.0e-8,
    collin_tol = 1.0e-7,
    # Iteration parameters
    iter_max = 25L,
    iter_max_cluster = 100L,
    iter_full_dicho = 10L,
    iter_demean_max = 10000L,
    iter_inner_max = 50L,
    iter_interrupt = 1000L,
    iter_ssr = 10L,
    # Previously hardcoded parameters now configurable
    rel_tol_denom = 0.1,
    irons_tuck_eps = 1.0e-14,
    safe_division_min = 1.0e-12,
    safe_log_min = 1.0e-12,
    newton_raphson_tol = 1.0e-8,
    # Convergence parameters
    convergence_iter_max = 100L,
    convergence_iter_full_dicho = 10L,
    # GLM parameters
    step_halving_factor = 0.5,
    binomial_mu_min = 0.001,
    binomial_mu_max = 0.999,
    safe_clamp_min = 1.0e-15,
    safe_clamp_max = 1.0e12,
    # Algorithm configuration
    direct_qr_threshold = 0.9,
    qr_collin_tol_multiplier = 1.0,
    chol_stability_threshold = 1.0e-12,
    # Alpha computation
    alpha_convergence_tol = 1.0e-8,
    alpha_iter_max = 10000L,
    # Demean algorithm
    demean_extra_projections = 0L,
    demean_warmup_iterations = 15L,
    demean_projections_after_acc = 5L,
    demean_grand_acc_frequency = 20L,
    demean_ssr_check_frequency = 40L,
    # Configuration flags
    keep_dmx = FALSE,
    use_weights = TRUE) {
  iter_max <- as.integer(iter_max)
  iter_max_cluster <- as.integer(iter_max_cluster)
  iter_full_dicho <- as.integer(iter_full_dicho)

  iter_demean_max <- as.integer(iter_demean_max)

  convergence_iter_max <- as.integer(convergence_iter_max)

  convergence_iter_full_dicho <- as.integer(convergence_iter_full_dicho)

  alpha_iter_max <- as.integer(alpha_iter_max)

  # Return list with control parameters
  list(
    # Core tolerance parameters
    dev_tol = dev_tol,
    demean_tol = demean_tol,
    collin_tol = collin_tol,

    # Iteration parameters
    iter_max = iter_max,
    iter_max_cluster = iter_max_cluster,
    iter_full_dicho = iter_full_dicho,
    iter_demean_max = iter_demean_max,
    iter_inner_max = as.integer(iter_inner_max),
    iter_interrupt = as.integer(iter_interrupt),
    iter_ssr = as.integer(iter_ssr),

    # Algorithm parameters
    direct_qr_threshold = direct_qr_threshold,
    qr_collin_tol_multiplier = qr_collin_tol_multiplier,
    chol_stability_threshold = chol_stability_threshold,
    safe_division_min = safe_division_min,
    safe_log_min = safe_log_min,
    newton_raphson_tol = newton_raphson_tol,

    # Demean algorithm parameters
    demean_extra_projections = as.integer(demean_extra_projections),
    demean_warmup_iterations = as.integer(demean_warmup_iterations),
    demean_projections_after_acc = as.integer(demean_projections_after_acc),
    demean_grand_acc_frequency = as.integer(demean_grand_acc_frequency),
    demean_ssr_check_frequency = as.integer(demean_ssr_check_frequency),

    # Convergence algorithm parameters
    irons_tuck_eps = irons_tuck_eps,
    alpha_convergence_tol = alpha_convergence_tol,
    alpha_iter_max = alpha_iter_max,

    # Previously hardcoded parameters
    rel_tol_denom = rel_tol_denom,
    convergence_iter_max = convergence_iter_max,
    convergence_iter_full_dicho = convergence_iter_full_dicho,
    step_halving_factor = step_halving_factor,
    binomial_mu_min = binomial_mu_min,
    binomial_mu_max = binomial_mu_max,
    safe_clamp_min = safe_clamp_min,
    safe_clamp_max = safe_clamp_max,

    # Configuration parameters
    keep_dmx = as.logical(keep_dmx),
    use_weights = as.logical(use_weights)
  )
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
      "dev_tol", "demean_tol", "iter_max", "iter_demean_max",
      "iter_inner_max", "iter_interrupt", "iter_ssr", "limit"
    )
    for (param_name in non_neg_params) {
      if (merged_control[[param_name]] <= 0) {
        stop(sprintf("'%s' must be greater than zero.", param_name), call. = FALSE)
      }
    }
    # 2. logical params
    logical_params <- c("trace", "drop_pc", "keep_dmx")
    for (param_name in logical_params) {
      if (!is.logical(merged_control[[param_name]])) {
        stop(sprintf("'%s' must be logical.", param_name), call. = FALSE)
      }
    }

    assign("control", merged_control, envir = parent.frame())
  }
}

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
#' @param fe_names Fixed effects
#' @noRd
transform_fe_ <- function(data, formula, fe_names) {
  data[, (fe_names) := lapply(.SD, check_factor_), .SDcols = fe_names]

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
  X <- model.matrix(formula, mf)[, -1L, drop = FALSE]
  nms_sp <- colnames(X)
  attr(X, "dimnames") <- NULL

  # Check for Inf values in both model matrix and response
  if (any(is.infinite(X))) {
    stop("Infinite values detected in model matrix. This often happens when applying log() to zero or negative values in predictors.", call. = FALSE)
  }

  if (any(is.infinite(y))) {
    stop("Infinite values detected in response variable. This often happens when applying log() to zero or negative values.", call. = FALSE)
  }

  # Assign results to parent environment
  assign("y", y, envir = parent.frame())
  assign("X", X, envir = parent.frame())
  assign("nms_sp", nms_sp, envir = parent.frame())
  assign("p", ncol(X), envir = parent.frame())
  assign("data", data, envir = parent.frame()) # Update data frame
}

#' @title Check weights
#' @description Checks if weights are valid
#' @param w Weights
#' @noRd
check_weights_ <- function(w) {
  if (!is.numeric(w) || anyNA(w)) {
    stop("Weights must be numeric and non-missing.", call. = FALSE)
  }
  if (any(w < 0)) {
    stop("Negative weights are not allowed.", call. = FALSE)
  }
}

#' @title Get index list
#' @description Generates an auxiliary list of indexes to project out the fixed
#'  effects
#' @param fe_names Fixed effects
#' @param data Data frame
#' @noRd
get_index_list_ <- function(fe_names, data) {
  indexes <- seq.int(1L, nrow(data)) # Generate 1-based indices for R consistency
  lapply(fe_names, function(x, indexes, data) {
    split(indexes, data[[x]])
  }, indexes = indexes, data = data)
}
