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
  lvls_k <- object[["lvls_k"]]
  nt <- object[["nobs"]][["nobs"]]
  k.vars <- names(lvls_k)

  # Extract dependent variable
  y <- data[[1L]]

  # Extract control arguments
  center_tol <- control[["center_tol"]]
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
    Myadj <- center_variables_(Myadj, yadj, w, k.list, center_tol, 10000L, TRUE)
    eta.upd <- yadj - drop(Myadj) + offset - eta

    # Step-halving with three checks
    # 1. finite deviance
    # 2. valid \eta and \mu
    # 3. improvement as in glm2
    rho <- 1.0
    for (inner.iter in seq.int(50L)) {
      eta <- eta.old + rho * eta.upd
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

  # Return eta
  eta
}
