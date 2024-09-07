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
#' @description Computes the second or third order derivative of the link function
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
  repeat {
    tmp_var <- paste0("capybara_internal_variable_", sample(letters, 5L, replace = TRUE), collapse = "")
    if (!(tmp_var %in% colnames(data))) {
      break
    }
  }
  tmp_var
}

#' @title Check formula
#' @description Checks if formula for GLM/NegBin models
#' @param formula Formula object
#' @noRd
check_formula_ <- function(formula) {
  if (is.null(formula)) {
    stop("'formula' has to be specified.", call. = FALSE)
  } else if (!inherits(formula, "formula")) {
    stop("'formula' has to be of class 'formula'.", call. = FALSE)
  }
}

#' @title Check data
#' @description Checks data for GLM/NegBin models
#' @param data Data frame
#' @noRd
check_data_ <- function(data) {
  if (is.null(data)) {
    stop("'data' has to be specified.", call. = FALSE)
  } else if (nrow(data) == 0L) {
    stop("'data' has zero observations.", call. = FALSE)
  } else if (!inherits(data, "data.frame")) {
    stop("'data' has to be of class data.frame.", call. = FALSE)
  }
}

#' @title Check control
#' @description Checks control for GLM/NegBin models
#' @param control Control list
#' @noRd
check_control_ <- function(control) {
  if (is.null(control)) {
    control <- list()
  } else if (!inherits(control, "list")) {
    stop("'control' has to be a list.", call. = FALSE)
  }

  do.call(feglm_control, control)
}

#' @title Check family
#' @description Checks family for GLM/NegBin models
#' @param family Family object
#' @noRd
check_family_ <- function(family) {
  if (!inherits(family, "family")) {
    stop("'family' has to be of class family", call. = FALSE)
  } else if (family[["family"]] %in%
    c("quasi", "quasipoisson", "quasibinomial")) {
    stop("Quasi-variants of 'family' are not supported.", call. = FALSE)
  } else if (startsWith(family[["family"]], "Negative Binomial")) {
    stop("Please use 'fenegbin' instead.", call. = FALSE)
  }

  if (family[["family"]] == "binomial" && family[["link"]] != "logit") {
    stop("The current version only supports logit in the binomial family.
    This is because I had to rewrite the links in C++ to use those with Armadillo.
    Send me a Pull Request or open an issue if you need Probit.", call. = FALSE)
  }
}

#' @title Update formula
#' @description Updates formula for GLM/NegBin models
#' @param formula Formula object
#' @noRd
update_formula_ <- function(formula) {
  formula <- Formula(formula)

  if (length(formula)[[2L]] < 2L || length(formula)[[1L]] > 1L) {
    stop(paste(
      "'formula' incorrectly specified. Perhaps you forgot to add the",
      "fixed effects as 'mpg ~ wt | cyl' or similar."
    ), call. = FALSE)
  }

  formula
}

#' @title Model frame
#' @description Creates model frame for GLM/NegBin models
#' @param data Data frame
#' @param formula Formula object
#' @param weights Weights
#' @noRd
model_frame_ <- function(data, formula, weights) {
  data <- select(ungroup(data), all_of(c(all.vars(formula), weights)))

  lhs <- names(data)[[1L]]

  nobs_full <- nrow(data)

  data <- na.omit(data)

  nobs_na <- nobs_full - nrow(data)
  nobs_full <- nrow(data)

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
    if (is.numeric(pull(select(data, !!sym(lhs))))) {
      # Check if 'y' is in [0, 1]
      if (nrow(filter(data, !!sym(lhs) < 0.0 | !!sym(lhs) > 1.0)) > 0L) {
        stop("Model response has to be within the unit interval.",
          call. = FALSE
        )
      }
    } else {
      # Check if 'y' is factor and transform otherwise
      data <- mutate(data, !!sym(lhs) := check_factor_(!!sym(lhs)))

      # Check if the number of levels equals two
      if (nrow(summarise(data, n_levels = nlevels(!!sym(lhs)))) != 2L) {
        stop("Model response has to be binary.", call. = FALSE)
      }

      # Ensure 'y' is 0-1 encoded
      ## if lhs is not numeric, convert it
      if (!is.numeric(pull(select(data, !!sym(lhs))))) {
        data <- mutate(data, !!sym(lhs) := as.numeric(!!sym(lhs)) - 1.0)
      } else {
        data <- mutate(data, !!sym(lhs) := !!sym(lhs) - 1.0)
      }
    }
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    # Check if 'y' is strictly positive
    if (nrow(filter(data, !!sym(lhs) <= 0.0)) > 0L) {
      stop("Model response has to be strictly positive.", call. = FALSE)
    }
  } else if (family[["family"]] != "gaussian") {
    # Check if 'y' is positive
    if (nrow(filter(data, !!sym(lhs) < 0.0)) > 0L) {
      stop("Model response has to be positive.", call. = FALSE)
    }
  }
}

#' @title Drop by link type
#' @description Drops observations that do not contribute to the log-likelihood for binomial and poisson models
#' @param data Data frame
#' @param lhs Left-hand side of the formula
#' @param family Family object
#' @param tmp_var Temporary variable
#' @param k_vars Fixed effects
#' @param control Control list
#' @noRd
drop_by_link_type_ <- function(data, lhs, family, tmp_var, k_vars, control) {
  if (family[["family"]] %in% c("binomial", "poisson")) {
    if (control[["drop_pc"]]) {
      repeat {
        # Drop observations that do not contribute to the log-likelihood
        ncheck <- nrow(data)
        for (j in k_vars) {
          data <- data %>%
            group_by(!!sym(j)) %>%
            mutate(!!sym(tmp_var) := mean(!!sym(lhs))) %>%
            ungroup()
          if (family[["family"]] == "binomial") {
            data <- filter(data, !!sym(tmp_var) > 0.0 & !!sym(tmp_var) < 1.0)
          } else {
            data <- filter(data, !!sym(tmp_var) > 0.0)
          }
          data <- select(data, -!!sym(tmp_var))
        }

        # Check termination
        if (ncheck == nrow(data)) {
          break
        }
      }
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
  data <- mutate(data, across(all_of(k_vars), check_factor_))

  if (length(formula)[[2L]] > 2L) {
    add.vars <- attr(terms(formula, rhs = 3L), "term.labels")
    data <- mutate(data, across(all_of(add.vars), check_factor_))
  }

  data
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
  y <- data[[1L]]
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms_sp <- attr(x, "dimnames")[[2L]]
  attr(x, "dimnames") <- NULL
  p <- ncol(x)

  assign("y", y, envir = parent.frame())
  assign("x", x, envir = parent.frame())
  assign("nms_sp", nms_sp, envir = parent.frame())
  assign("p", p, envir = parent.frame())
}

check_linear_dependence_ <- function(x, p) {
  if (qr(x)$rank < p) {
    stop("Linear dependent terms detected.", call. = FALSE)
  }
}

#' @title Check weights
#' @description Checks if weights are valid
#' @param wt Weights
#' @noRd
check_weights_ <- function(wt) {
  if (!is.numeric(wt)) {
    stop("weights must be numeric.", call. = FALSE)
  }
  if (any(wt < 0.0)) {
    stop("negative weights are not allowed.", call. = FALSE)
  }
}

#' @title Check starting theta
#' @description Checks if starting theta is valid for NegBin models
#' @param init.theta Initial theta value
#' @param link Link function
#' @noRd
init_theta_ <- function(init.theta, link) {
  if (is.null(init.theta)) {
    family <- poisson(link)
  } else {
    # Validity of input argument (beta_start)
    if (length(init.theta) != 1L) {
      stop("'init.theta' has to be a scalar.", call. = FALSE)
    } else if (init.theta <= 0.0) {
      stop("'init.theta' has to be strictly positive.", call. = FALSE)
    }
    family <- negative.binomial(init.theta, link)
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

#' @title Get score matrix
#' @description Computes the score matrix
#' @param object Result list
#' @noRd
get_score_matrix_ <- function(object) {
  # Extract required quantities from result list
  control <- object[["control"]]
  data <- object[["data"]]
  eta <- object[["eta"]]
  wt <- object[["weights"]]
  family <- object[["family"]]

  # Update weights and dependent variable
  y <- data[[1L]]
  mu <- family[["linkinv"]](eta)
  mu.eta <- family[["mu.eta"]](eta)
  w <- (wt * mu.eta^2) / family[["variance"]](mu)
  nu <- (y - mu) / mu.eta

  # Center regressor matrix (if required)
  if (control[["keep_mx"]]) {
    mx <- object[["mx"]]
  } else {
    # Extract additional required quantities from result list
    formula <- object[["formula"]]
    k_vars <- names(object[["lvls_k"]])

    # Generate auxiliary list of indexes to project out the fixed effects
    k.list <- get_index_list_(k_vars, data)

    # Extract regressor matrix
    x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(x, "dimnames")[[2L]]
    attr(x, "dimnames") <- NULL

    # Center variables
    mx <- center_variables_r_(x, w, k.list, control[["center_tol"]], 10000L)
    colnames(mx) <- nms_sp
  }

  # Return score matrix
  mx * (nu * w)
}

#' @title Gamma computation
#' @description Computes the gamma matrix for the APES function
#' @param mx Regressor matrix
#' @param H Hessian matrix
#' @param J Jacobian matrix
#' @param PPsi Psi matrix
#' @param v Vector of weights
#' @param nt Number of observations
#' @noRd
gamma_ <- function(mx, H, J, PPsi, v, nt) {
  inv_nt <- 1.0 / nt
  (mx %*% solve(H * inv_nt, J) - PPsi) * v * inv_nt
}
