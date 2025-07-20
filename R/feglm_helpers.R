#' srr_stats
#' @srrstats {G1.0} Provides modular helper functions for internal checks and computations in generalized linear models with fixed effects.
#' @srrstats {RE4.8} The response variable is checked and some observations are dropped if the response is not compatible with the link (i.e., negative values and log-link).
#' @srrstats {RE4.13} Observations with a dependent variable that is incompatible with the link function are removed.
#' @srrstats {RE5.0} Supports internal optimizations, including centering variables and reducing computational redundancy.
#' @srrstats {RE5.1} Implements computational safeguards for iterative processes, such as weight validation and convergence checks.
#' @srrstats {RE5.2} Provides utilities for scalable and efficient computation of GLM derivatives and score matrices.
#' @noRd
NULL

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
#' @param X Regressor matrix
#' @param y Dependent variable
#' @param w Weights
#' @param beta Beta values
#' @param family Family object
#' @param nt Number of observations
#' @param p Number parameters
#' @param beta_start Starting values for beta
#' @param eta_start Starting values for eta
#' @noRd
start_guesses_ <- function(X, y, w, beta, family, nt, p, beta_start, eta_start) {
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
      eta <- X %*% beta
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
      eta <- rep(family[["linkfun"]](sum(w * (y + 0.5) / 2.0) / sum(w)), nt)
    } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
      eta <- rep(family[["linkfun"]](sum(w * y) / sum(w)), nt)
    } else if (family[["family"]] == "poisson") {
      # Better starting values for Poisson regression
      # Use a small offset to avoid log(0)
      y_mean <- sum(w * y) / sum(w)
      y_offset <- pmax(y, 0.1) # Avoid log(0)
      eta <- family[["linkfun"]](y_offset)

      # If we have a regressor matrix, try to get better starting values
      if (p > 0) {
        # Simple linear regression on log scale as starting point
        tryCatch(
          {
            lm_fit <- lm(log(y_offset) ~ X - 1, weights = w)
            beta <- as.numeric(coef(lm_fit))
            beta[is.na(beta)] <- 0 # Replace NA with 0
            eta <- X %*% beta
          },
          error = function(e) {
            # Fall back to simple mean if lm fails
            beta <- numeric(p)
            eta <- rep(family[["linkfun"]](y_mean + 0.1), nt)
          }
        )
      } else {
        eta <- rep(family[["linkfun"]](y_mean + 0.1), nt)
      }
    } else {
      eta <- rep(family[["linkfun"]](sum(w * (y + 0.1)) / sum(w)), nt)
    }
  }

  assign("beta", beta, envir = parent.frame())
  assign("eta", eta, envir = parent.frame())
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
  w <- object[["weights"]]
  family <- object[["family"]]

  # Update weights and dependent variable
  y <- data[[1L]]
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  w <- (w * mu_eta^2) / family[["variance"]](mu)
  nu <- (y - mu) / mu_eta

  # Center regressor matrix (if required)
  if (control[["keep_dmx"]]) {
    X_dm <- object[["X_dm"]]
  } else {
    # Extract additional required quantities from result list
    formula <- object[["formula"]]
    k_vars <- names(object[["lvls_k"]])

    # Generate auxiliary list of indexes to project out the fixed effects
    k_list <- get_index_list_(k_vars, data)

    # Extract regressor matrix
    X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
    nms_sp <- attr(X, "dimnames")[[2L]]
    attr(X, "dimnames") <- NULL

    # Center variables
    X <- demean_variables_(
      X, w, k_list, control[["center_tol"]],
      control[["iter_max"]], control[["iter_interrupt"]],
      control[["iter_ssr"]], family[["family"]]
    )
    colnames(X) <- nms_sp
  }

  # Return score matrix
  X * (nu * w)
}

#' @title Gamma computation
#' @description Computes the gamma matrix for the APES function
#' @param X_dm Regressor matrix
#' @param h Hessian matrix
#' @param j Jacobian matrix
#' @param ppsi Psi matrix
#' @param v Vector of weights
#' @param nt Number of observations
#' @noRd
gamma_ <- function(X_dm, h, j, ppsi, v, nt) {
  inv_nt <- 1.0 / nt
  (X_dm %*% solve(h * inv_nt, j) - ppsi) * v * inv_nt
}
