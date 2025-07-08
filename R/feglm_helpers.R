#' srr_stats
#' @srrstats {G1.0} Provides modular helper functions for internal checks and computations in generalized linear models with fixed effects.
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
#' @srrstats {RE4.8} The response variable is checked and some observations are dropped if the response is not compatible with the link (i.e., negative values and log-link).
#' @srrstats {RE4.12} The `check_data_()` function drops observations that are not useable with link function or that do not contribute to the log-likelihood.
#' @srrstats {RE4.13} Observations with a dependent variable that is incompatible with the link function are removed.
#' @srrstats {RE5.0} Supports internal optimizations, including centering variables and reducing computational redundancy.
#' @srrstats {RE5.1} Implements computational safeguards for iterative processes, such as weight validation and convergence checks.
#' @srrstats {RE5.2} Provides utilities for scalable and efficient computation of GLM derivatives and score matrices.
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

#' @title Second order derivative
#' @description Helper for the partial_mu_eta function
#' @param eta Eta value
#' @param mu_eta Mu.eta value
#' @param family Family object
#' @noRd
second_order_derivative_ <- function(eta, mu_eta, family) {
  link <- family[["link"]]
  linkinv_eta <- family[["linkinv"]](eta)

  if (link == "logit") {
    return(mu_eta * (1.0 - 2.0 * linkinv_eta))
  } else if (link == "probit") {
    return(-eta * mu_eta)
  } else if (link == "cloglog") {
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
  link <- family[["link"]]
  linkinv_eta <- family[["linkinv"]](eta)

  if (link == "logit") {
    return(mu_eta * ((1.0 - 2.0 * linkinv_eta)^2 - 2.0 * mu_eta))
  } else if (link == "probit") {
    return((eta^2 - 1.0) * mu_eta)
  } else if (link == "cloglog") {
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
  default_control <- do.call(feglm_control, list())

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
      "iter_inner_max", "iter_interrupt", "iter_ssr", "limit"
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

#' @title Check family
#' @description Checks family for GLM/NegBin models
#' @param family Family object
#' @noRd
check_family_ <- function(family) {
  if (startsWith(family[["family"]], "Negative Binomial")) {
    stop("use 'fenegbin' instead.", call. = FALSE)
  }

  allowed_families <- c("gaussian", "binomial", "poisson", "Gamma", "inverse.gaussian")
  family[["family"]] <- match.arg(family[["family"]], allowed_families)

  if (family[["family"]] == "binomial" && family[["link"]] != "logit") {
    stop(
      "The current version only supports logit in the binomial family.
       This is because I had to rewrite the links in C++ to use those with
       Armadillo. Send me a Pull Request or open an issue if you need Probit.",
      call. = FALSE
    )
  }
}

#' @title Column types
#' @description Returns the column types of a data frame
#' @param data Data frame
#' @noRd
col_types <- function(data) {
  vapply(data, class, character(1L), USE.NAMES = FALSE)
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

#' @title Check response
#' @description Checks response for GLM/NegBin models
#' @param data Data frame
#' @param lhs Left-hand side of the formula
#' @param family Family object
#' @noRd
check_response_ <- function(data, lhs, family) {
  if (family[["family"]] == "binomial") {
    # Check if 'y' is numeric
    y <- data[[lhs]]
    if (is.numeric(y)) {
      if (any(y < 0 | y > 1)) stop("Model response must be within [0,1].")
    } else {
      # Check if 'y' is factor and transform otherwise
      y <- check_factor_(y)

      # Check if the number of levels equals two
      if (nlevels(y) != 2) stop("Model response has to be binary.")

      # Ensure 'y' is 0-1 encoded
      y <- as.numeric(y) - 1

      data[[lhs]] <- y
    }
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    # Check if 'y' is strictly positive
    if (nrow(data[get(lhs) <= 0.0]) > 0L) {
      stop("Model response has to be positive.", call. = FALSE)
    }
  } else if (family[["family"]] != "gaussian") {
    # Check if 'y' is positive
    if (nrow(data[get(lhs) < 0.0]) > 0L) {
      stop("Model response has to be strictly positive.", call. = FALSE)
    }
  }
}

#' @title Drop by link type
#' @description Drops observations that do not contribute to the log-likelihood
#'  for binomial and poisson models
#' @param data Data frame
#' @param lhs Left-hand side of the formula
#' @param family Family object
#' @param tmp_var Temporary variable
#' @param k_vars Fixed effects
#' @param control Control list
#' @noRd
drop_by_link_type_ <- function(data, lhs, family, tmp_var, k_vars, control) {
  if (family[["family"]] %in% c("binomial", "poisson") && isTRUE(control[["drop_pc"]])) {
    # Convert response to numeric if it's an integer
    if (is.integer(data[[lhs]])) {
      data[, (lhs) := as.numeric(get(lhs))]
    }

    ncheck <- 0
    nrow_data <- nrow(data)

    while (ncheck != nrow_data) {
      ncheck <- nrow_data

      for (j in k_vars) {
        data[, (tmp_var) := mean(as.numeric(get(lhs))), by = j]

        # Filter rows based on family type
        if (family[["family"]] == "binomial") {
          data <- data[get(tmp_var) > 0 & get(tmp_var) < 1]
        } else {
          data <- data[get(tmp_var) > 0]
        }

        data[, (tmp_var) := NULL]
      }

      nrow_data <- nrow(data)
    }
  }

  data
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
#' @param nobs_na Number of observations with missing values
#' @param nt Number of observations after dropping
#' @noRd
nobs_ <- function(nobs_full, nobs_na, nt) {
  c(
    nobs_full = nobs_full,
    nobs_na   = nobs_na,
    nobs_pc   = nobs_full - nt,
    nobs      = nobs_full + nobs_na
  )
}

#' @title Model response
#' @description Computes the model response
#' @param data Data frame
#' @param formula Formula object
#' @noRd
model_response_ <- function(data, formula) {
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms_sp <- colnames(x)
  attr(x, "dimnames") <- NULL

  assign("y", data[[1L]], envir = parent.frame())
  assign("x", x, envir = parent.frame())
  assign("nms_sp", nms_sp, envir = parent.frame())
  assign("p", ncol(x), envir = parent.frame())
}

#' @title Check weights
#' @description Checks if weights are valid
#' @param y Dependent variable
#' @param x Regressors matrix
#' @param p Number of parameters
#' @noRd
check_linear_dependence_ <- function(y, x, p) {
  # if (qr(x)$rank < p) {
  #   stop("Linear dependent terms detected.", call. = FALSE)
  # }

  if (check_linear_dependence_qr_(y, x, p) == 1L) {
    stop("Linear dependent terms detected.", call. = FALSE)
  }

  return(TRUE)
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

#' @title Check starting theta
#' @description Checks if starting theta is valid for NegBin models
#' @param init_theta Initial theta value
#' @param link Link function
#' @noRd
init_theta_ <- function(init_theta, link) {
  if (is.null(init_theta)) {
    family <- poisson(link)
  } else {
    # Validity of input argument (beta_start)
    if (length(init_theta) != 1L) {
      stop("'init_theta' has to be a scalar.", call. = FALSE)
    } else if (init_theta <= 0.0) {
      stop("'init_theta' has to be strictly positive.", call. = FALSE)
    }
    family <- negative.binomial(init_theta, link)
  }

  family
}

#' @title Check starting guesses
#' @description Checks if starting guesses are valid
#' @param beta_start Starting values for beta
#' @param eta_start Starting values for eta
#' @param y Dependent variable
#' @param x Regressor matrix
#' @param beta Beta values
#' @param nt Number of observations
#' @param wt Weights
#' @param p Number parameters
#' @param family Family object
#' @noRd
start_guesses_ <- function(
    beta_start, eta_start, y, x, beta, nt, wt, p, family) {
  if (!is.null(beta_start) || !is.null(eta_start)) {
    # If both are specified, ignore eta_start
    if (!is.null(beta_start) && !is.null(eta_start)) {
      warning(
        "'beta_start' and 'eta_start' are specified. Ignoring 'eta_start'.",
        call. = FALSE
      )
    }

    # Compute and check starting guesses
    if (!is.null(beta_start)) {
      # Validity of input argument (beta_start)
      if (length(beta_start) != p) {
        stop(
          paste(
            "Length of 'beta_start' has to be equal to the number of",
            "structural parameters."
          ),
          call. = FALSE
        )
      }

      # Set starting guesses
      beta <- beta_start
      eta <- x %*% beta
    } else {
      # Validity of input argument (eta_start)
      if (length(eta_start) != nt) {
        stop(
          paste(
            "Length of 'eta_start' has to be equal to the number of",
            "observations."
          ),
          call. = FALSE
        )
      }

      # Set starting guesses
      beta <- numeric(p)
      eta <- eta_start
    }
  } else {
    # Compute starting guesses if not user specified
    beta <- numeric(p)
    if (family[["family"]] == "binomial") {
      eta <- rep(family[["linkfun"]](sum(wt * (y + 0.5) / 2.0) / sum(wt)), nt)
    } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
      eta <- rep(family[["linkfun"]](sum(wt * y) / sum(wt)), nt)
    } else {
      eta <- rep(family[["linkfun"]](sum(wt * (y + 0.1)) / sum(wt)), nt)
    }
  }

  assign("beta", beta, envir = parent.frame())
  assign("eta", eta, envir = parent.frame())
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

#' @title Get score matrix
#' @description Computes the score matrix
#' @param object Result list
#' @noRd
get_score_matrix_feglm_ <- function(object) {
  # Extract required quantities from result list
  control <- object[["control"]]
  data <- object[["data"]]
  eta <- object[["eta"]]
  wt <- object[["weights"]]
  family <- object[["family"]]

  # Update weights and dependent variable
  y <- data[[1L]]
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  w <- (wt * mu_eta^2) / family[["variance"]](mu)
  nu <- (y - mu) / mu_eta

  # Center regressor matrix (if required)
  if (control[["keep_mx"]]) {
    mx <- object[["mx"]]
  } else {
    # Extract additional required quantities from result list
    formula <- object[["formula"]]
    k_vars <- names(object[["lvls_k"]])

    # Generate auxiliary list of indexes to project out the fixed effects
    k_list <- get_index_list_(k_vars, data)

    # Extract regressor matrix
    x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(x, "dimnames")[[2L]]
    attr(x, "dimnames") <- NULL

    # Center variables
    x <- center_variables_r_(x, w, k_list, control[["center_tol"]], control[["iter_max"]], control[["interrupt_iter"]], control[["iter_ssr"]])
    colnames(x) <- nms_sp
  }

  # Return score matrix
  x * (nu * w)
}

#' @title Gamma computation
#' @description Computes the gamma matrix for the APES function
#' @param mx Regressor matrix
#' @param h Hessian matrix
#' @param j Jacobian matrix
#' @param ppsi Psi matrix
#' @param v Vector of weights
#' @param nt Number of observations
#' @noRd
gamma_ <- function(mx, h, j, ppsi, v, nt) {
  inv_nt <- 1.0 / nt
  (mx %*% solve(h * inv_nt, j) - ppsi) * v * inv_nt
}
