check_formula <- function(formula) {
  if (is.null(formula)) {
    stop("'formula' has to be specified.", call. = FALSE)
  } else if (!inherits(formula, "formula")) {
    stop("'formula' has to be of class 'formula'.", call. = FALSE)
  }
}

check_data <- function(data) {
  if (is.null(data)) {
    stop("'data' has to be specified.", call. = FALSE)
  } else if (!inherits(data, "data.frame")) {
    stop("'data' has to be of class data.frame.", call. = FALSE)
  }
}

check_control <- function(control) {
  if (is.null(control)) {
    control <- list()
  } else if (!inherits(control, "list")) {
    stop("'control' has to be a list.", call. = FALSE)
  }

  control <- do.call(feglmControl, control)

  control
}

check_family <- function(family) {
  if (!inherits(family, "family")) {
    stop("'family' has to be of class family", call. = FALSE)
  } else if (family[["family"]] %in% c("quasi", "quasipoisson", "quasibinomial")) {
    stop("Quasi-variants of 'family' are not supported.", call. = FALSE)
  } else if (startsWith(family[["family"]], "Negative Binomial")) {
    stop("Please use 'feglm.nb' instead.", call. = FALSE)
  }
}

update_formula <- function(formula) {
  formula <- Formula(formula)

  if (length(formula)[[2L]] < 2L || length(formula)[[1L]] > 1L) {
    stop("'formula' uncorrectly specified.", call. = FALSE)
  }

  formula
}

generate_model_frame <- function(data, formula, weights) {
  setDT(data)
  data <- data[, c(all.vars(formula), weights), with = FALSE]

  lhs <- names(data)[[1L]]

  nobs.full <- nrow(data)

  data <- na.omit(data)

  nobs.na <- nobs.full - nrow(data)
  nobs.full <- nrow(data)

  list(data = data, lhs = lhs, nobs.na = nobs.na, nobs.full = nobs.full)
}

check_response <- function(data, lhs, family) {
  if (family[["family"]] == "binomial") {
    # Check if 'y' is numeric
    if (data[, is.numeric(get(lhs))]) {
      # Check if 'y' is in [0, 1]
      if (data[, any(get(lhs) < 0.0 | get(lhs) > 1.0)]) {
        stop("Model response has to be within the unit interval.", call. = FALSE)
      }
    } else {
      # Check if 'y' is factor and transform otherwise
      data[, (1L) := checkFactor(get(lhs))]

      # Check if the number of levels equals two
      if (data[, length(levels(get(lhs)))] != 2L) {
        stop("Model response has to be binary.", call. = FALSE)
      }

      # Ensure 'y' is 0-1 encoded
      data[, (1L) := as.numeric(get(lhs)) - 1.0]
    }
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    # Check if 'y' is strictly positive
    if (data[, any(get(lhs) <= 0.0)]) {
      stop("Model response has to be strictly positive.", call. = FALSE)
    }
  } else {
    # Check if 'y' is positive
    if (data[, any(get(lhs) < 0.0)]) {
      stop("Model response has to be positive.", call. = FALSE)
    }
  }
}

drop_by_link_type <- function(data, lhs, family, tmp.var, k.vars, control) {
  if (family[["family"]] %in% c("binomial", "poisson")) {
    if (control[["drop.pc"]]) {
      repeat {
        # Drop observations that do not contribute to the log-likelihood
        ncheck <- nrow(data)
        for (j in k.vars) {
          data[, (tmp.var) := mean(get(lhs)), by = eval(j)]
          if (family[["family"]] == "binomial") {
            data <- data[get(tmp.var) > 0.0 & get(tmp.var) < 1.0]
          } else {
            data <- data[get(tmp.var) > 0.0]
          }
          data[, (tmp.var) := NULL]
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

transform_fe <- function(data, formula, k.vars) {
  data[, (k.vars) := lapply(.SD, checkFactor), .SDcols = k.vars]

  if (length(formula)[[2L]] > 2L) {
    add.vars <- attr(terms(formula, rhs = 3L), "term.labels")
    data[, (add.vars) := lapply(.SD, checkFactor), .SDcols = add.vars]
  }

  data
}

generate_nobs <- function(nobs.full, nobs.na, nt) {
  c(
    nobs.full = nobs.full,
    nobs.na   = nobs.na,
    nobs.pc   = nobs.full - nt,
    nobs      = nt
  )
}

generate_model_response <- function(data, formula) {
  y <- data[[1L]]
  X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms.sp <- attr(X, "dimnames")[[2L]]
  attr(X, "dimnames") <- NULL
  p <- ncol(X)

  list(y = y, X = X, nms.sp = nms.sp, p = p)
}

check_linear_dependence <- function(X, p) {
  if (qr(X)[["rank"]] < p) {
    stop("Linear dependent terms detected.", call. = FALSE)
  }
}

check_weights <- function(wt) {
  if (!is.numeric(wt)) {
    stop("weights must be numeric.", call. = FALSE)
  }
  if (any(wt < 0.0)) {
    stop("negative weights are not allowed.", call. = FALSE)
  }
}

generate_family_init_theta <- function(init.theta, link) {
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

  family
}

generate_start_guesses <- function(beta.start, eta.start, y, X, beta, nt, wt, p, family) {
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

  list(beta = beta, eta = eta)
}
