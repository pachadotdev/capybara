#' @title Negative Binomial model fitting with high-dimensional k-way fixed
#'  effects
#' @description A routine that uses the same internals as \code{\link{feglm}}.
#' @inheritParams feglm
#' @param init.theta an optional initial value for the theta parameter (see
#'  \code{\link[MASS]{glm.nb}}).
#' @param link the link function. Must be one of \code{"log"}, \code{"sqrt"}, or
#'  \code{"identity"}.
#' @examples
#' # same as the example in fepoisson but with overdispersion/underdispersion
#' mod <- fenegbin(
#'   trade ~ dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_panel
#' )
#'
#' summary(mod)
#' @export
fenegbin <- function(
    formula = NULL,
    data = NULL,
    weights = NULL,
    beta.start = NULL,
    eta.start = NULL,
    init.theta = NULL,
    link = c("log", "identity", "sqrt"),
    control = NULL) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of link ----
  link <- match.arg(link)

  # Check validity of control + Extract control list ----
  control <- check_control_(control)

  # Update formula and do further validity check ----
  formula <- update_formula_(formula)

  # Generate model.frame
  model_frame_(data, formula, weights)

  # Check starting guess of theta ----
  family <- init_theta_(init.theta, link)
  rm(init.theta)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables and sort ----
  k.vars <- attr(terms(formula, rhs = 2L), "term.labels")
  k <- length(k.vars)
  setkeyv(data, k.vars)

  # Generate temporary variable ----
  tmp.var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp.var, k.vars, control)

  # Transform fixed effects variables and potential cluster variables to factors ----
  data <- transform_fe_(data, formula, k.vars)

  # Determine the number of dropped observations ----
  nt <- nrow(data)
  nobs <- nobs_(nobs.full, nobs.na, nt)

  # Extract model response and regressor matrix ----
  model_response_(data, formula)

  # Check for linear dependence in 'X' ----
  check_linear_dependence_(X, p)

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else {
    wt <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(wt)

  # Compute and check starting guesses ----
  start_guesses_(beta.start, eta.start, y, X, beta, nt, wt, p, family)

  # Get names and number of levels in each fixed effects category ----
  nms.fe <- lapply(data[, k.vars, with = FALSE], levels)
  lvls.k <- vapply(nms.fe, length, integer(1))

  # Generate auxiliary list of indexes for different sub panels ----
  k.list <- get_index_list_(k.vars, data)

  # Extract control arguments ----
  tol <- control[["dev.tol"]]
  limit <- control[["limit"]]
  iter.max <- control[["iter.max"]]
  trace <- control[["trace"]]

  # Initial negative binomial fit ----
  fit <- feglm_fit_(
    beta, eta, y, X, wt, k.list, family, control
  )

  beta <- fit[["coefficients"]]
  eta <- fit[["eta"]]
  dev <- fit[["deviance"]]
  theta <- suppressWarnings(
    theta.ml(
      y     = y,
      mu    = family[["linkinv"]](eta),
      n     = nt,
      limit = limit,
      trace = trace
    )
  )

  # Alternate between fitting glm and \theta ----
  conv <- FALSE
  for (iter in seq.int(iter.max)) {
    # Fit negative binomial model
    dev.old <- dev
    theta.old <- theta
    family <- negative.binomial(theta, link)
    fit <- feglm_fit_(beta, eta, y, X, wt, k.list, family, control)
    beta <- fit[["coefficients"]]
    eta <- fit[["eta"]]
    dev <- fit[["deviance"]]
    theta <- suppressWarnings(
      theta.ml(
        y     = y,
        mu    = family[["linkinv"]](eta),
        n     = nt,
        limit = limit,
        trace = trace
      )
    )

    # Progress information
    if (trace) {
      cat("Outer Iteration=", iter, "\n")
      cat("Deviance=", format(dev, digits = 5L, nsmall = 2L), "\n")
      cat("theta=", format(theta, digits = 5L, nsmall = 2L), "\n")
      cat("Estimates=", format(beta, digits = 3L, nsmall = 2L), "\n")
    }

    # Check termination condition
    dev.crit <- abs(dev - dev.old) / (0.1 + abs(dev))
    theta.crit <- abs(theta - theta.old) / (0.1 + abs(theta.old))
    if (dev.crit <= tol && theta.crit <= tol) {
      if (trace) {
        cat("Convergence\n")
      }
      conv <- TRUE
      break
    }
  }
  y <- NULL
  X <- NULL
  eta <- NULL

  # Information if convergence failed ----
  if (!conv && trace) cat("Algorithm did not converge.\n")

  # Add names to beta, Hessian, and MX (if provided) ----
  names(fit[["coefficients"]]) <- nms.sp
  if (control[["keep.mx"]]) {
    colnames(fit[["MX"]]) <- nms.sp
  }
  dimnames(fit[["Hessian"]]) <- list(nms.sp, nms.sp)

  # Generate result list ----
  reslist <- c(
    fit, list(
      theta      = theta,
      iter.outer = iter,
      conv.outer = conv,
      nobs       = nobs,
      lvls.k     = lvls.k,
      nms.fe     = nms.fe,
      formula    = formula,
      data       = data,
      family     = family,
      control    = control
    )
  )

  # Return result list ----
  structure(reslist, class = c("feglm", "feglm.nb"))
}
