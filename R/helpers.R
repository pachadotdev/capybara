# Checks if variable is a factor and transforms if necessary ---

check_factor_ <- function(x) {
  if (is.factor(x)) {
    droplevels(x)
  } else {
    factor(x)
  }
}

# Higher-order partial derivatives ----

partial_mu_eta_ <- function(eta, family, order) {
  f <- family[["mu.eta"]](eta)
  if (order == 2L) {
    if (family[["link"]] == "logit") {
      f * (1.0 - 2.0 * family[["linkinv"]](eta))
    } else if (family[["link"]] == "probit") {
      -eta * f
    } else {
      f * (1.0 - exp(eta))
    }
  } else {
    if (family[["link"]] == "logit") {
      f * ((1.0 - 2.0 * family[["linkinv"]](eta))^2 - 2.0 * f)
    } else if (family[["link"]] == "probit") {
      (eta^2 - 1.0) * f
    } else {
      f * (1.0 - exp(eta)) * (2.0 - exp(eta)) - f
    }
  }
}

# Returns suitable name for a temporary variable ----

temp_var_ <- function(data) {
  repeat {
    tmp.var <- paste0(sample(letters, 5L, replace = TRUE), collapse = "")
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

  return(TRUE)
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

  control <- do.call(feglm_control, control)

  return(control)
}

check_family_ <- function(family) {
  if (!inherits(family, "family")) {
    stop("'family' has to be of class family", call. = FALSE)
  } else if (family[["family"]] %in% c("quasi", "quasipoisson", "quasibinomial")) {
    stop("Quasi-variants of 'family' are not supported.", call. = FALSE)
  } else if (startsWith(family[["family"]], "Negative Binomial")) {
    stop("Please use 'feglm.nb' instead.", call. = FALSE)
  }

  return(TRUE)
}

update_formula_ <- function(formula) {
  formula <- Formula(formula)

  if (length(formula)[[2L]] < 2L || length(formula)[[1L]] > 1L) {
    stop("'formula' uncorrectly specified. Perhaps you forgot to add the fixed effects as 'mpg ~ wt | cyl' or similar.", call. = FALSE)
  }

  return(formula)
}

model_frame_ <- function(data, formula, weights) {
  data <- select(data, all.vars(formula), weights)

  lhs <- names(data)[[1L]]

  nobs.full <- nrow(data)

  data <- na.omit(data)

  nobs.na <- nobs.full - nrow(data)
  nobs.full <- nrow(data)

  assign("data", data, envir = parent.frame())
  assign("lhs", lhs, envir = parent.frame())
  assign("nobs.na", nobs.na, envir = parent.frame())
  assign("nobs.full", nobs.full, envir = parent.frame())

  return(TRUE)
}

check_response_ <- function(data, lhs, family) {
  if (family[["family"]] == "binomial") {
    # Check if 'y' is numeric
    if (is.numeric(select(data, !!sym(lhs)))) {
      # Check if 'y' is in [0, 1]
      if (nrow(filter(data, !!sym(lhs) < 0.0 | !!sym(lhs) > 1.0)) > 0L) {
        stop("Model response has to be within the unit interval.", call. = FALSE)
      }
    } else {
      # Check if 'y' is factor and transform otherwise
      data <- mutate(data, !!lhs := check_factor_(!!sym(lhs)))

      # Check if the number of levels equals two
      if (summarise(data, n_levels = nlevels(!!sym(lhs))) != 2L) {
        stop("Model response has to be binary.", call. = FALSE)
      }

      # Ensure 'y' is 0-1 encoded
      data <- mutate_at(data, vars(lhs), ~ as.numeric(.) - 1.0)
    }
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    # Check if 'y' is strictly positive
    if (nrow(filter(data, !!sym(lhs) <= 0.0)) > 0L) {
      stop("Model response has to be strictly positive.", call. = FALSE)
    }
  } else {
    # Check if 'y' is positive
    if (nrow(filter(data, !!sym(lhs) < 0.0)) > 0L) {
      stop("Model response has to be positive.", call. = FALSE)
    }
  }

  return(TRUE)
}

drop_by_link_type_ <- function(data, lhs, family, tmp.var, k.vars, control) {
  if (family[["family"]] %in% c("binomial", "poisson")) {
    if (control[["drop.pc"]]) {
      repeat {
        # Drop observations that do not contribute to the log-likelihood
        ncheck <- nrow(data)
        for (j in k.vars) {
          data <- data %>%
            group_by(!!sym(j)) %>%
            mutate(!!tmp.var := mean(!!sym(lhs)))
          if (family[["family"]] == "binomial") {
            data <- filter(data, !!sym(tmp.var) > 0.0, !!sym(tmp.var) < 1.0)
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

  return(data)
}

transform_fe_ <- function(data, formula, k.vars) {
  data <- data %>%
    mutate_at(vars(k.vars), check_factor_)

  if (length(formula)[[2L]] > 2L) {
    add.vars <- attr(terms(formula, rhs = 3L), "term.labels")
    data <- data %>%
      mutate_at(vars(add.vars), check_factor_)
  }

  return(data)
}

nobs_ <- function(nobs.full, nobs.na, nt) {
  return(
    c(
      nobs.full = nobs.full,
      nobs.na   = nobs.na,
      nobs.pc   = nobs.full - nt,
      nobs      = nt
    )
  )
}

model_response_ <- function(data, formula) {
  y <- data[[1L]]
  X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms.sp <- attr(X, "dimnames")[[2L]]
  attr(X, "dimnames") <- NULL
  p <- ncol(X)

  assign("y", y, envir = parent.frame())
  assign("X", X, envir = parent.frame())
  assign("nms.sp", nms.sp, envir = parent.frame())
  assign("p", p, envir = parent.frame())

  return(TRUE)
}

check_linear_dependence_ <- function(X, p) {
  if (qr(X)[["rank"]] < p) {
    stop("Linear dependent terms detected.", call. = FALSE)
  }

  return(TRUE)
}

check_weights_ <- function(wt) {
  if (!is.numeric(wt)) {
    stop("weights must be numeric.", call. = FALSE)
  }
  if (any(wt < 0.0)) {
    stop("negative weights are not allowed.", call. = FALSE)
  }

  return(TRUE)
}

init_theta_ <- function(init.theta, link) {
  if (is.null(init.theta)) {
    family <- poisson(link)
  } else {
    # Validity of input argument (beta.start)
    if (length(init.theta) != 1L) {
      stop("'init.theta' has to be a scalar.", call. = FALSE)
    } else if (init.theta <= 0.0) {
      stop("'init.theta' has to be strictly positive.", call. = FALSE)
    }
    family <- negative.binomial(init.theta, link)
  }

  return(family)
}

start_guesses_ <- function(beta.start, eta.start, y, X, beta, nt, wt, p, family) {
  if (!is.null(beta.start) || !is.null(eta.start)) {
    # If both are specified, ignore eta.start
    if (!is.null(beta.start) && !is.null(eta.start)) {
      warning("'beta.start' and 'eta.start' are specified. Ignoring 'eta.start'.", call. = FALSE)
    }

    # Compute and check starting guesses
    if (!is.null(beta.start)) {
      # Validity of input argument (beta.start)
      if (length(beta.start) != p) {
        stop("Length of 'beta.start' has to be equal to the number of structural parameters.", call. = FALSE)
      }

      # Set starting guesses
      beta <- beta.start
      eta <- as.vector(X %*% beta)
    } else {
      # Validity of input argument (eta.start)
      if (length(eta.start) != nt) {
        stop("Length of 'eta.start' has to be equal to the number of observations.", call. = FALSE)
      }

      # Set starting guesses
      beta <- numeric(p)
      eta <- eta.start
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

  return(TRUE)
}
