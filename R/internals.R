# Transform factor ----

check_factor_ <- function(x) {
  if (is.factor(x)) {
    droplevels(x)
  } else {
    factor(x)
  }
}

# Fitting algorithm (similar to lm.fit) ----

felm_fit_ <- function(y, X, wt, k.list, control) {
  # Extract control arguments
  center.tol <- control[["center.tol"]]
  keep.mx <- control[["keep.mx"]]

  # Generate temporary variables
  nt <- length(y)
  MX <- X

  # Centering variables
  MX <- center_variables_(MX, NA_real_, wt, k.list, center.tol, 10000L, FALSE)

  # Compute the OLS estimate
  # beta <- as.vector(qr.solve(MX, y, epsilon))
  beta <- solve_beta_(MX, y, NA_real_, FALSE)

  # Generate result list
  reslist <- list(
    coefficients = beta
  )

  # Update result list
  if (keep.mx) reslist[["MX"]] <- MX

  # Return result list
  reslist
}

# Fitting algorithm (similar to glm.fit) ----

# feglm_fit_ <- function(beta, eta, y, X, wt, k.list, family, control) {
#   # Extract control arguments
#   center.tol <- control[["center.tol"]]
#   dev.tol <- control[["dev.tol"]]
#   iter.max <- control[["iter.max"]]
#   keep.mx <- control[["keep.mx"]]

#   # Compute initial quantities for the maximization routine
#   nt <- length(y)
#   mu <- family[["linkinv"]](eta)
#   dev <- sum(family[["dev.resids"]](y, mu, wt))
#   null.dev <- sum(family[["dev.resids"]](y, mean(y), wt))

#   # Generate temporary variables
#   Mnu <- as.matrix(numeric(nt))
#   MX <- X

#   # Start maximization of the log-likelihood
#   conv <- FALSE
#   for (iter in seq.int(iter.max)) {
#     # Store \eta, \beta, and deviance of the previous iteration
#     eta.old <- eta
#     beta.old <- beta
#     dev.old <- dev

#     # Compute weights and dependent variable
#     mu.eta <- family[["mu.eta"]](eta)
#     w <- (wt * mu.eta^2) / family[["variance"]](mu)
#     nu <- (y - mu) / mu.eta

#     # Centering variables
#     Mnu <- center_variables_(Mnu, nu, w, k.list, center.tol, 10000L, TRUE)
#     MX <- center_variables_(MX, NA_real_, w, k.list, center.tol, 10000L, FALSE)

#     # Compute update step and update eta
    
#     # Step-halving with three checks
#     # 1. finite deviance
#     # 2. valid \eta and \mu
#     # 3. improvement as in glm2
    
#     # if (is.integer(y)) y <- as.double(y)
#     # theta <- ifelse(is.null(family[["theta"]]), 0.0, family[["theta"]])
#     # sh <- step_halving_(
#     #   MX, Mnu, nu, w, eta.old, beta.old, y, wt,
#     #   theta, family[["family"]], dev.old, dev.tol
#     # )

#     # dev <- sh[["dev"]]
#     # eta <- sh[["eta"]]
#     # beta <- sh[["beta"]]
#     # mu <- sh[["mu"]]
#     # imp.crit <- sh[["imp.crit"]]
#     # sh <- NULL

#     beta.upd <- solve_beta_(MX, Mnu, w, TRUE)
#     # print(beta.upd)
#     eta.upd <- solve_eta_(MX, Mnu, nu, beta.upd)

#     rho <- 1.0

#     for (inner.iter in seq.int(50L)) {
#       eta <- update_beta_eta_(eta.old, eta.upd, rho)
#       beta <- update_beta_eta_(beta.old, beta.upd, rho)
#       mu <- family[["linkinv"]](eta)
#       dev <- sum(family[["dev.resids"]](y, mu, wt))
#       dev.crit <- is.finite(dev)
#       val.crit <- family[["valideta"]](eta) && family[["validmu"]](mu)
#       imp.crit <- (dev - dev.old) / (0.1 + abs(dev)) <= -dev.tol
#       # print(c(dev.crit, val.crit, imp.crit))
#       if (dev.crit && val.crit && imp.crit) break
#       rho <- rho * 0.5
#     }

#     # Check if step-halving failed (deviance and invalid \eta or \mu)
#     if (!dev.crit || !val.crit) {
#       stop("Inner loop failed; cannot correct step size.", call. = FALSE)
#     }

#     # Stop if we do not improve
#     if (!imp.crit) {
#       eta <- eta.old
#       beta <- beta.old
#       dev <- dev.old
#       mu <- family[["linkinv"]](eta)
#     }

#     # Check convergence
#     dev.crit <- abs(dev - dev.old) / (0.1 + abs(dev))
#     if (dev.crit < dev.tol) {
#       conv <- TRUE
#       break
#     }

#     # Update starting guesses for acceleration
#     Mnu <- Mnu - nu
#   }

#   # Information if convergence failed
#   if (!conv) cat("Algorithm did not converge.\n")

#   # Update weights and dependent variable
#   mu.eta <- family[["mu.eta"]](eta)
#   w <- (wt * mu.eta^2) / family[["variance"]](mu)

#   # Center variables
#   MX <- center_variables_(X, NA_real_, w, k.list, center.tol, 10000L, FALSE)
#   # Recompute Hessian
#   H <- crossprod_(MX, w, TRUE, TRUE)

#   # Generate result list
#   reslist <- list(
#     coefficients  = beta,
#     eta           = eta,
#     weights       = wt,
#     Hessian       = H,
#     deviance      = dev,
#     null.deviance = null.dev,
#     conv          = conv,
#     iter          = iter
#   )

#   # Update result list
#   if (keep.mx) reslist[["MX"]] <- MX

#   # Return result list
#   reslist
# }

# Efficient offset algorithm to update the linear predictor ----

feglm_offset_ <- function(object, offset) {
  # Check validity of 'object'
  if (!inherits(object, "feglm")) {
    stop("'feglm_offset_' called on a non-'feglm' object.")
  }

  # Extract required quantities from result list
  control <- object[["control"]]
  data <- object[["data"]]
  wt <- object[["weights"]]
  family <- object[["family"]]
  formula <- object[["formula"]]
  lvls.k <- object[["lvls.k"]]
  nt <- object[["nobs"]][["nobs"]]
  k.vars <- names(lvls.k)

  # Extract dependent variable
  y <- data[[1L]]

  # Extract control arguments
  center.tol <- control[["center.tol"]]
  dev.tol <- control[["dev.tol"]]
  iter.max <- control[["iter.max"]]

  # Generate auxiliary list of indexes to project out the fixed effects
  k.list <- get_index_list_(k.vars, data)

  # Compute starting guess for \eta
  if (family[["family"]] == "binomial") {
    eta <- rep(family[["linkfun"]](sum(wt * (y + 0.5) / 2.0) / sum(wt)), nt)
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    eta <- rep(family[["linkfun"]](sum(wt * y) / sum(wt)), nt)
  } else {
    eta <- rep(family[["linkfun"]](sum(wt * (y + 0.1)) / sum(wt)), nt)
  }

  # Compute initial quantities for the maximization routine
  mu <- family[["linkinv"]](eta)
  dev <- sum(family[["dev.resids"]](y, mu, wt))
  Myadj <- as.matrix(numeric(nt))

  # Start maximization of the log-likelihood
  for (iter in seq.int(iter.max)) {
    # Store \eta, \beta, and deviance of the previous iteration
    eta.old <- eta
    dev.old <- dev

    # Compute weights and dependent variable
    mu.eta <- family[["mu.eta"]](eta)
    w <- (wt * mu.eta^2) / family[["variance"]](mu)
    yadj <- (y - mu) / mu.eta + eta - offset

    # Centering dependent variable and compute \eta update
    Myadj <- center_variables_(Myadj, yadj, w, k.list, center.tol, 10000L, TRUE)
    # eta.upd <- yadj - drop(Myadj) + offset - eta
    eta.upd <- solve_eta2_(yadj, Myadj, offset, eta)

    # Step-halving with three checks
    # 1. finite deviance
    # 2. valid \eta and \mu
    # 3. improvement as in glm2
    rho <- 1.0
    for (inner.iter in seq.int(50L)) {
      # eta <- eta.old + rho * eta.upd
      eta <- update_beta_eta_(eta.old, eta.upd, rho)
      mu <- family[["linkinv"]](eta)
      dev <- sum(family[["dev.resids"]](y, mu, wt))
      dev.crit <- is.finite(dev)
      val.crit <- family[["valideta"]](eta) && family[["validmu"]](mu)
      imp.crit <- (dev - dev.old) / (0.1 + abs(dev)) <= -dev.tol
      if (dev.crit && val.crit && imp.crit) break
      rho <- rho / 2.0
    }

    # Check if step-halving failed
    if (!dev.crit || !val.crit) {
      stop("Inner loop failed; cannot correct step size.", call. = FALSE)
    }

    # Check termination condition
    if (abs(dev - dev.old) / (0.1 + abs(dev)) < dev.tol) break

    # Update starting guesses for acceleration
    Myadj <- Myadj - yadj
  }

  # Return \eta
  eta
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
  if (control[["keep.mx"]]) {
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

# Returns suitable name for a temporary variable
temp_var_ <- function(data) {
  repeat {
    tmp.var <- paste0(sample(letters, 5L, replace = TRUE), collapse = "")
    if (!(tmp.var %in% colnames(data))) {
      break
    }
  }
  tmp.var
}
