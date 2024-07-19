# Fitting algorithm (similar to glm.fit) ----

feglm_fit_ <- function(beta, eta, y, X, wt, k.list, family, control) {
  # Extract control arguments
  center.tol <- control[["center.tol"]]
  dev.tol <- control[["dev.tol"]]
  epsilon <- max(min(1.0e-07, dev.tol / 1000.0), .Machine[["double.eps"]])
  iter.max <- control[["iter.max"]]
  trace <- control[["trace"]]
  keep.mx <- control[["keep.mx"]]

  # Compute initial quantities for the maximization routine
  nt <- length(y)
  mu <- family[["linkinv"]](eta)
  dev <- sum(family[["dev.resids"]](y, mu, wt))
  null.dev <- sum(family[["dev.resids"]](y, mean(y), wt))

  # Generate temporary variables
  Mnu <- as.matrix(numeric(nt))
  MX <- X

  # Start maximization of the log-likelihood
  conv <- FALSE
  for (iter in seq.int(iter.max)) {
    # Store \eta, \beta, and deviance of the previous iteration
    eta.old <- eta
    beta.old <- beta
    dev.old <- dev

    # Compute weights and dependent variable
    mu.eta <- family[["mu.eta"]](eta)
    w <- (wt * mu.eta^2) / family[["variance"]](mu)
    nu <- (y - mu) / mu.eta

    # Centering variables
    Mnu <- center_variables_(Mnu, nu, w, k.list, center.tol, 10000L, TRUE)
    MX <- center_variables_(MX, NA_real_, w, k.list, center.tol, 10000L, FALSE)

    # Compute update step and update eta
    # beta.upd <- as.vector(qr.solve(MX * w.tilde, Mnu * w.tilde, epsilon))
    # eta.upd <- nu - as.vector(Mnu - MX %*% beta.upd)
    beta.upd <- solve_beta_(MX, Mnu, w, TRUE)
    eta.upd <- solve_eta_(MX, Mnu, nu, beta.upd)

    # Step-halving with three checks
    # 1. finite deviance
    # 2. valid \eta and \mu
    # 3. improvement as in glm2
    rho <- 1.0

    for (inner.iter in seq.int(50L)) {
      # eta <- eta.old + rho * eta.upd
      # beta <- beta.old + rho * beta.upd
      eta <- update_beta_eta_(eta.old, eta.upd, rho)
      beta <- update_beta_eta_(beta.old, beta.upd, rho)
      mu <- family[["linkinv"]](eta)
      dev <- sum(family[["dev.resids"]](y, mu, wt))
      dev.crit <- is.finite(dev)
      val.crit <- family[["valideta"]](eta) && family[["validmu"]](mu)
      imp.crit <- (dev - dev.old) / (0.1 + abs(dev)) <= -dev.tol
      if (dev.crit && val.crit && imp.crit) break
      rho <- rho * 0.5
    }

    # Check if step-halving failed (deviance and invalid \eta or \mu)
    if (!dev.crit || !val.crit) {
      stop("Inner loop failed; cannot correct step size.", call. = FALSE)
    }

    # Stop if we do not improve
    if (!imp.crit) {
      eta <- eta.old
      beta <- beta.old
      dev <- dev.old
      mu <- family[["linkinv"]](eta)
    }

    # Progress information
    if (trace) {
      cat(
        "Deviance=", format(dev, digits = 5L, nsmall = 2L), "Iterations -",
        iter, "\n"
      )
      cat("Estimates=", format(beta, digits = 3L, nsmall = 2L), "\n")
    }

    # Check convergence
    dev.crit <- abs(dev - dev.old) / (0.1 + abs(dev))
    if (trace) cat("Stopping criterion=", dev.crit, "\n")
    if (dev.crit < dev.tol) {
      if (trace) cat("Convergence\n")
      conv <- TRUE
      break
    }

    # Update starting guesses for acceleration
    Mnu <- Mnu - nu
  }

  # Information if convergence failed
  if (!conv && trace) cat("Algorithm did not converge.\n")

  # Update weights and dependent variable
  mu.eta <- family[["mu.eta"]](eta)
  w <- (wt * mu.eta^2) / family[["variance"]](mu)

  # Center variables
  MX <- center_variables_(X, NA_real_, w, k.list, center.tol, 10000L, FALSE)
  # Recompute Hessian
  H <- crossprod_(MX, w, TRUE, TRUE)

  # Generate result list
  reslist <- list(
    coefficients  = beta,
    eta           = eta,
    weights       = wt,
    Hessian       = H,
    deviance      = dev,
    null.deviance = null.dev,
    conv          = conv,
    iter          = iter
  )

  # Update result list
  if (keep.mx) reslist[["MX"]] <- MX

  # Return result list
  reslist
}
