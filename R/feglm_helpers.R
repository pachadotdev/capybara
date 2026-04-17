#' srr_stats
#' @srrstats {G1.0} Provides modular helper functions for internal checks and computations in generalized linear models
#'  with fixed effects.
#' @srrstats {G2.0} Validates the integrity of inputs such as factors, formulas, data, and control parameters.
#' @srrstats {G2.1a} Ensures inputs have expected types and structures, such as formulas being of class `formula` and
#'  data being a `data.frame`.
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
#' @srrstats {G5.2a} Ensures that all error and warning messages are unique and descriptive.
#' @srrstats {G5.4a} Includes tests for edge cases, such as binary and continuous response variables, and validates all
#'  input arguments.
#' @srrstats {RE4.4} The model is specified using a formula object, or a character-type object convertible to a formula
#'   which is then used to create the design matrix.
#' @srrstats {RE4.5} Fitted models have an nobs element that can be called with `nobs()`.
#' @srrstats {RE4.8} The response variable is checked and some observations are dropped if the response is not
#'  compatible with the link (i.e., negative values and log-link).
#' @srrstats {RE4.12} The `check_data_()` function drops observations that are not useable with link function or that do
#'  not contribute to the log-likelihood.
#' @srrstats {RE4.13} Observations with a dependent variable that is incompatible with the link function are removed.
#' @srrstats {RE5.0} Supports internal optimizations, including centering variables and reducing computational
#'  redundancy.
#' @srrstats {RE5.1} Implements computational safeguards for iterative processes, such as weight validation and
#'  convergence checks.
#' @srrstats {RE5.2} Provides utilities for scalable and efficient computation of GLM derivatives and score matrices.
#' @noRd
NULL

#' @title Probit regression family
#'
#' @description Creates a family object for probit regression (binomial with probit link).
#'  This is a binary response model using the standard normal CDF as the link function.
#'
#' @return A list with class \code{"family"} containing:
#'  \item{family}{The family name ("probit")}
#'  \item{link}{The link function name ("probit")}
#'  \item{linkinv}{Inverse link function (standard normal CDF)}
#'  \item{mu.eta}{Derivative of inverse link (standard normal PDF)}
#'  \item{variance}{Variance function (mu * (1 - mu))}
#'  \item{validmu}{Function to validate mu values}
#'  \item{valideta}{Function to validate eta values}
#'
#' @examples
#' # Probit regression with fixed effects
#' mod <- feglm(am ~ wt | cyl, mtcars, family = probit())
#' summary(mod)
#'
#' @export
probit <- function() {
  structure(
    list(
      family = "probit",
      link = "probit",
      linkinv = function(eta) pnorm(eta),
      mu.eta = function(eta) dnorm(eta),
      variance = function(mu) mu * (1 - mu),
      validmu = function(mu) all(mu > 0 & mu < 1),
      valideta = function(eta) TRUE
    ),
    class = "family"
  )
}

#' @title Normalize multi-part formula for C++
#' @description Expands formula operators (*, ^, -, /, %in%, .) using R's
#'   terms() machinery, then rebuilds a simplified formula string that
#'   uses only + and : operators that the C++ parser can handle.
#' @param formula Original formula (possibly with |)
#' @param data Data frame (needed for . expansion)
#' @return A character string of the normalized formula
#' @noRd
normalize_formula_ <- function(formula, data) {
  # Split formula on | for fixed effects and clusters
  fml_chr <- deparse1(formula)
  parts <- trimws(strsplit(fml_chr, "\\|")[[1L]])

  base_part <- parts[[1L]]
  fe_part <- if (length(parts) >= 2L) parts[[2L]] else NULL
  cl_part <- if (length(parts) >= 3L) parts[[3L]] else NULL

  # Handle "0" meaning no fixed effects - convert to empty string
  # but preserve structure when cluster is present
  if (!is.null(fe_part) && fe_part == "0") {
    fe_part <- ""
  }
  if (!is.null(cl_part) && cl_part == "0") {
    cl_part <- ""
  }

  # Compute terms for base formula
  base_fml <- as.formula(base_part, env = environment(formula))
  tt <- terms(base_fml, data = data)

  # Get LHS (response)
  response_idx <- attr(tt, "response")
  all_vars <- attr(tt, "variables")
  lhs_expr <- if (response_idx > 0L) {
    deparse1(all_vars[[response_idx + 1L]])
  } else {
    "y"
  }

  # Get expanded term labels (RHS)
  term_labels <- attr(tt, "term.labels")

  # Check if intercept is included
  has_intercept <- attr(tt, "intercept") == 1L

  # Build RHS string
  # Note: C++ adds intercept by default when no FEs, so we use
  # __NO_INTERCEPT__ as a special marker for intercept suppression
  if (length(term_labels) == 0L) {
    if (has_intercept) {
      rhs_str <- "1"
    } else {
      # No terms and no intercept - edge case
      rhs_str <- "__NO_INTERCEPT__"
    }
  } else {
    rhs_str <- paste(term_labels, collapse = " + ")
    if (!has_intercept) {
      # Prepend marker that C++ will recognize
      rhs_str <- paste0("__NO_INTERCEPT__ + ", rhs_str)
    }
  }

  # Rebuild the formula string
  new_fml <- paste(lhs_expr, "~", rhs_str)

  # Add FE and cluster parts back
  # Use empty string for "0" to preserve | positions (e.g., | | cluster)
  if (!is.null(fe_part)) {
    new_fml <- paste(new_fml, "|", fe_part)
  }
  if (!is.null(cl_part)) {
    new_fml <- paste(new_fml, "|", cl_part)
  }

  new_fml
}

#' @title Transform factor
#' @description Checks if variable is a factor and transforms if necessary
#' @param X Variable to be checked
#' @noRd
check_factor_ <- function(X) {
  if (is.factor(X)) {
    droplevels(X)
  } else {
    factor(X)
  }
}

#' @title Second order derivative
#' @description Helper for the partial_mu_eta function
#' @param eta Eta value
#' @param mu_eta Mu.eta value
#' @param family Family object
#' @noRd
second_order_derivative_ <- function(eta, mu_eta, family) {
  if (family[["link"]] == "logit") {
    return(mu_eta * (1.0 - 2.0 * family[["linkinv"]](eta)))
  } else if (family[["link"]] == "probit") {
    return(-eta * mu_eta)
  } else if (family[["link"]] == "cloglog") {
    return(mu_eta * (1.0 - exp(eta)))
  } else {
    return(-2.0 * eta / (1.0 + eta^2) * mu_eta)
  }
}

#' @title Third order derivative
#' @description Helper for the partial_mu_eta function
#' @param eta Eta value
#' @param mu_eta Mu.eta value
#' @param family Family object
#' @noRd
third_order_derivative_ <- function(eta, mu_eta, family) {
  if (family[["link"]] == "logit") {
    linkinv_eta <- family[["linkinv"]](eta)
    return(mu_eta * ((1.0 - 2.0 * linkinv_eta)^2 - 2.0 * mu_eta))
  } else if (family[["link"]] == "probit") {
    return((eta^2 - 1.0) * mu_eta)
  } else if (family[["link"]] == "cloglog") {
    return(mu_eta * (1.0 - exp(eta)) * (2.0 - exp(eta)) - mu_eta)
  } else {
    return((6.0 * eta^2 - 2.0) / (1.0 + eta^2)^2 * mu_eta)
  }
}

#' @title Second or third order derivative
#' @description Computes the second or third order derivative of the link
#'  function
#' @param eta Linear predictor
#' @param family Family object
#' @param order Order of the derivative (2 or 3)
#' @noRd
partial_mu_eta_ <- function(eta, family, order) {
  # Safeguard eta if necessary
  if (family[["link"]] != "logit") {
    eta <- family[["linkfun"]](family[["linkinv"]](eta))
  }

  mu_eta <- family[["mu.eta"]](eta)

  if (order == 2L) {
    return(second_order_derivative_(eta, mu_eta, family))
  } else {
    return(third_order_derivative_(eta, mu_eta, family))
  }
}

#' @title Check family
#' @description Checks family for GLM/NegBin models
#' @param family Family object
#' @noRd
check_family_ <- function(family) {
  if (startsWith(family[["family"]], "Negative Binomial")) {
    stop("use 'fenegbin' instead.", call. = FALSE)
  }

  allowed_families <- c(
    "gaussian",
    "binomial",
    "probit",
    "poisson",
    "Gamma",
    "inverse.gaussian"
  )

  # Handle binomial(link = "probit") - convert to probit family
  if (family[["family"]] == "binomial" && !is.null(family[["link"]]) &&
    family[["link"]] == "probit") {
    family[["family"]] <- "probit"
  }

  family[["family"]] <- match.arg(family[["family"]], allowed_families)

  if (family[["family"]] == "binomial" && family[["link"]] != "logit") {
    stop(
      "The binomial family only supports 'logit' link. For probit link, use ",
      "probit() or binomial(link = 'probit').",
      call. = FALSE
    )
  }
}

#' @title Temporary variable
#' @description Generates a collision-free temporary column name
#' @param data Data frame
#' @noRd
temp_var_ <- function(data) {
  tmp_var <- "capybara_temp12345"
  while (tmp_var %in% colnames(data)) {
    tmp_var <- paste0("capybara_temp", paste(sample(letters, 5L, replace = TRUE), collapse = ""))
  }
  tmp_var
}

#' @title Check response
#' @description Checks response for GLM/NegBin models (validation only, no mutation)
#' @param data Data frame
#' @param lhs Left-hand side of the formula
#' @param family Family object
#' @noRd
check_response_ <- function(data, lhs, family) {
  y <- data[[lhs]]
  
  if (family[["family"]] %in% c("binomial", "probit")) {
    if (is.numeric(y)) {
      if (any(y < 0.0 | y > 1.0, na.rm = TRUE)) {
        stop("Model response must be within [0,1].")
      }
    } else {
      # Factor/character: validate two levels
      y_fac <- check_factor_(y)
      if (length(levels(y_fac)) != 2L) {
        stop("Model response has to be binary.")
      }
    }
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    if (any(y <= 0.0, na.rm = TRUE)) {
      stop("Model response has to be positive.", call. = FALSE)
    }
  } else if (family[["family"]] != "gaussian") {
    if (any(y < 0.0, na.rm = TRUE)) {
      stop("Model response has to be strictly positive.", call. = FALSE)
    }
  }
}

#' @title Get score matrix
#' @description Computes the score matrix
#' @param object Result list
#' @noRd
get_score_matrix_feglm_ <- function(object) {
  # Update weights and dependent variable
  y <- object[["data"]][[1L]]
  mu <- object[["family"]][["linkinv"]](object[["eta"]])
  mu_eta <- object[["family"]][["mu.eta"]](object[["eta"]])
  w <- (object[["weights"]] * mu_eta^2) / object[["family"]][["variance"]](mu)
  nu <- (y - mu) / mu_eta

  # Center regressor matrix (if required)
  if (object[["control"]][["keep_tx"]]) {
    X <- object[["tx"]]
  } else {
    # Generate flat FE codes to project out the fixed effects
    k_list <- get_index_list_(object[["nms_fe"]], object[["data"]])

    # Extract regressor matrix
    X <- model.matrix(object[["formula"]], object[["data"]], rhs = 1L)[,
      -1L,
      drop = FALSE
    ]
    nms_sp <- attr(X, "dimnames")[[2L]]
    attr(X, "dimnames") <- NULL

    # Center variables
    X <- center_variables_(
      X,
      w,
      k_list[["codes"]],
      object[["control"]][["center_tol"]],
      object[["control"]][["iter_center_max"]]
    )
    colnames(X) <- nms_sp
  }

  # Return score matrix
  X * (nu * w)
}

#' @title Gamma computation
#' @description Computes the gamma matrix for the APES function
#' @param tx Regressor matrix
#' @param h Hessian matrix
#' @param j Jacobian matrix
#' @param ppsi Psi matrix
#' @param v Vector of weights
#' @param nt Number of observations
#' @noRd
gamma_ <- function(tx, h, j, ppsi, v, nt) {
  (tx %*% solve(h / nt, j) - ppsi) * v / nt
}
