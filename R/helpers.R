# Checks if variable is a factor and transforms if necessary ---

check_factor_ <- function(x) {
  if (is.factor(x)) {
    droplevels(x)
  } else {
    factor(x)
  }
}

# Higher-order partial derivatives ----

second_order_derivative_ <- function(eta, f, family) {
  link <- family[["link"]]
  linkinv_eta <- family[["linkinv"]](eta)

  if (link == "logit") {
    return(f * (1.0 - 2.0 * linkinv_eta))
  } else if (link == "probit") {
    return(-eta * f)
  } else if (link == "cloglog") {
    return(f * (1.0 - exp(eta)))
  } else {
    return(-2.0 * eta / (1.0 + eta^2) * f)
  }
}

third_order_derivative_ <- function(eta, f, family) {
  link <- family[["link"]]
  linkinv_eta <- family[["linkinv"]](eta)

  if (link == "logit") {
    return(f * ((1.0 - 2.0 * linkinv_eta)^2 - 2.0 * f))
  } else if (link == "probit") {
    return((eta^2 - 1.0) * f)
  } else if (link == "cloglog") {
    return(f * (1.0 - exp(eta)) * (2.0 - exp(eta)) - f)
  } else {
    return((6.0 * eta^2 - 2.0) / (1.0 + eta^2)^2 * f)
  }
}

partial_mu_eta_ <- function(eta, family, order) {
  # Safeguard eta if necessary
  if (family[["link"]] != "logit") {
    eta <- family[["linkfun"]](family[["linkinv"]](eta))
  }

  f <- family[["mu.eta"]](eta)

  if (order == 2L) {
    return(second_order_derivative_(eta, f, family))
  } else {
    return(third_order_derivative_(eta, f, family))
  }
}

# Returns suitable name for a temporary variable ----

temp_var_ <- function(data) {
  repeat {
    tmp.var <- paste0("capybara_internal_variable_", sample(letters, 5L, replace = TRUE), collapse = "")
    if (!(tmp.var %in% colnames(data))) {
      break
    }
  }
  tmp.var
}

# GLM/NegBin ----

check_formula_ <- function(formula) {
  if (is.null(formula)) {
    stop("'formula' has to be specified.", call. = FALSE)
  } else if (!inherits(formula, "formula")) {
    stop("'formula' has to be of class 'formula'.", call. = FALSE)
  }
}

check_data_ <- function(data) {
  if (is.null(data)) {
    stop("'data' has to be specified.", call. = FALSE)
  } else if (!inherits(data, "data.frame")) {
    stop("'data' has to be of class data.frame.", call. = FALSE)
  }
}

check_control_ <- function(control) {
  if (is.null(control)) {
    control <- list()
  } else if (!inherits(control, "list")) {
    stop("'control' has to be a list.", call. = FALSE)
  }

  do.call(feglm_control, control)
}

check_family_ <- function(family) {
  if (!inherits(family, "family")) {
    stop("'family' has to be of class family", call. = FALSE)
  } else if (family[["family"]] %in%
    c("quasi", "quasipoisson", "quasibinomial")) {
    stop("Quasi-variants of 'family' are not supported.", call. = FALSE)
  } else if (startsWith(family[["family"]], "Negative Binomial")) {
    stop("Please use 'fenegbin' instead.", call. = FALSE)
  }
}

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

model_frame_ <- function(data, formula, weights) {
  data <- select(ungroup(data), all_of(c(all.vars(formula), weights)))

  lhs <- names(data)[[1L]]

  nobs.full <- nrow(data)

  data <- na.omit(data)

  nobs.na <- nobs.full - nrow(data)
  nobs.full <- nrow(data)

  assign("data", data, envir = parent.frame())
  assign("lhs", lhs, envir = parent.frame())
  assign("nobs.na", nobs.na, envir = parent.frame())
  assign("nobs.full", nobs.full, envir = parent.frame())
}

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

drop_by_link_type_ <- function(data, lhs, family, tmp.var, k.vars, control) {
  if (family[["family"]] %in% c("binomial", "poisson")) {
    if (control[["drop_pc"]]) {
      repeat {
        # Drop observations that do not contribute to the log-likelihood
        ncheck <- nrow(data)
        for (j in k.vars) {
          data <- data %>%
            group_by(!!sym(j)) %>%
            mutate(!!sym(tmp.var) := mean(!!sym(lhs))) %>%
            ungroup()
          if (family[["family"]] == "binomial") {
            data <- filter(data, !!sym(tmp.var) > 0.0 & !!sym(tmp.var) < 1.0)
          } else {
            data <- filter(data, !!sym(tmp.var) > 0.0)
          }
          data <- select(data, -!!sym(tmp.var))
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

transform_fe_ <- function(data, formula, k.vars) {
  data <- mutate(data, across(all_of(k.vars), check_factor_))

  if (length(formula)[[2L]] > 2L) {
    add.vars <- attr(terms(formula, rhs = 3L), "term.labels")
    data <- mutate(data, across(all_of(add.vars), check_factor_))
  }

  data
}

nobs_ <- function(nobs.full, nobs.na, nt) {
  c(
    nobs.full = nobs.full,
    nobs.na   = nobs.na,
    nobs.pc   = nobs.full - nt,
    nobs      = nt
  )
}

model_response_ <- function(data, formula) {
  y <- data[[1L]]
  X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms_sp <- attr(X, "dimnames")[[2L]]
  attr(X, "dimnames") <- NULL
  p <- ncol(X)

  assign("y", y, envir = parent.frame())
  assign("X", X, envir = parent.frame())
  assign("nms_sp", nms_sp, envir = parent.frame())
  assign("p", p, envir = parent.frame())
}

check_linear_dependence_ <- function(X, p) {
  if (rank_(X) < p) {
    stop("Linear dependent terms detected.", call. = FALSE)
  }
}

check_weights_ <- function(wt) {
  if (!is.numeric(wt)) {
    stop("weights must be numeric.", call. = FALSE)
  }
  if (any(wt < 0.0)) {
    stop("negative weights are not allowed.", call. = FALSE)
  }
}

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

start_guesses_ <- function(
    beta_start, eta_start, y, X, beta, nt, wt, p, family) {
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
      eta <- solve_y_(X, beta)
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

# Generate auxiliary list of indexes for different sub panels ----

get_index_list_ <- function(k.vars, data) {
  indexes <- seq.int(0L, nrow(data) - 1L)
  lapply(k.vars, function(x, indexes, data) {
    split(indexes, data[[x]])
  }, indexes = indexes, data = data)
}

# Compute score matrix ----

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
  # nu <- (y - mu) / mu.eta
  nu <- update_nu_(y, mu, mu.eta)

  # Center regressor matrix (if required)
  if (control[["keep_mx"]]) {
    MX <- object[["MX"]]
  } else {
    # Extract additional required quantities from result list
    formula <- object[["formula"]]
    k.vars <- names(object[["lvls.k"]])

    # Generate auxiliary list of indexes to project out the fixed effects
    k.list <- get_index_list_(k.vars, data)

    # Extract regressor matrix
    X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(X, "dimnames")[[2L]]
    attr(X, "dimnames") <- NULL

    # Center variables
    MX <- center_variables_(X, NA_real_, w, k.list, control[["center.tol"]], 10000L, FALSE)
    colnames(MX) <- nms_sp
  }

  # Return score matrix
  MX * (nu * w)
}

# Suitable name for a temporary variable ----

temp_var_ <- function(data) {
  repeat {
    tmp.var <- paste0(sample(letters, 5L, replace = TRUE), collapse = "")
    if (!(tmp.var %in% colnames(data))) {
      break
    }
  }
  tmp.var
}
