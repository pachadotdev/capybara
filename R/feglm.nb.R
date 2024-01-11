#' @title
#' Efficiently fit negative binomial glm's with high-dimensional \eqn{k}-way fixed effects
#' @description
#' \code{feglm.nb} can be used to fit negative binomial generalized linear models with many
#' high-dimensional fixed effects (see \code{\link{feglm}}).
#' @param
#' formula,data,weights,beta.start,eta.start,control see \code{\link{feglm}}.
#' @param
#' init.theta an optional initial value for the theta parameter (see \code{\link[MASS]{glm.nb}}).
#' @param
#' link the link function. Must be one of \code{"log"}, \code{"sqrt"}, or \code{"identity"}.
#' @details
#' If \code{feglm.nb} does not converge this is usually a sign of linear dependence between one or
#' more regressors and a fixed effects category. In this case, you should carefully inspect your
#' model specification.
#' @return
#' The function \code{feglm.nb} returns a named list of class \code{"feglm"}.
#' @references
#' Gaure, S. (2013). "OLS with Multiple High Dimensional Category Variables". Computational
#' Statistics and Data Analysis. 66.
#' @references
#' Marschner, I. (2011). "glm2: Fitting generalized linear models with convergence problems".
#' The R Journal, 3(2).
#' @references
#' Stammann, A., F. Heiss, and D. McFadden (2016). "Estimating Fixed Effects Logit Models with Large
#' Panel Data". Working paper.
#' @references
#' Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with
#' High-Dimensional k-Way Fixed Effects". ArXiv e-prints.
#' @seealso
#' \code{\link[MASS]{glm.nb}}, \code{\link{feglm}}
#' @export
feglm.nb <- function(
    formula = NULL,
    data = NULL,
    weights = NULL,
    beta.start = NULL,
    eta.start = NULL,
    init.theta = NULL,
    link = c("log", "identity", "sqrt"),
    control = NULL) {
  # Check validity of formula ----
  check_formula(formula)

  # Check validity of data ----
  check_data(data)

  # Check validity of link ----
  link <- match.arg(link)

  # Check validity of control + Extract control list ----
  control <- check_control(control)

  # Update formula and do further validity check ----
  formula <- update_formula(formula)

  # Generate model.frame
  model_frame <- generate_model_frame(data, formula, weights)
  rm(data)

  # Check starting guess of theta ----
  family <- generate_family_init_theta(init.theta, link)
  rm(init.theta)

  # Ensure that model response is in line with the chosen model ----
  check_response(model_frame$data, model_frame$lhs, family)

  # Get names of the fixed effects variables and sort ----
  k.vars <- attr(terms(formula, rhs = 2L), "term.labels")
  k <- length(k.vars)
  setkeyv(model_frame$data, k.vars)

  # Generate temporary variable ----
  tmp.var <- tempVar(model_frame$data)

  # Drop observations that do not contribute to the log likelihood ----
  model_frame$data <- drop_by_link_type(model_frame$data, model_frame$lhs, family, tmp.var, k.vars, control)

  # Transform fixed effects variables and potential cluster variables to factors ----
  model_frame$data <- transform_fe(model_frame$data, formula, k.vars)

  # Determine the number of dropped observations ----
  nt <- nrow(model_frame$data)
  nobs <- generate_nobs(model_frame$nobs.full, model_frame$nobs.na, nt)

  # Extract model response and regressor matrix ----
  model_response <- generate_model_response(model_frame$data, formula)

  # Check for linear dependence in 'X' ----
  check_linear_dependence(model_response$X, model_response$p)

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else {
    wt <- model_frame$data[[weights]]
  }

  # Check validity of weights ----
  check_weights(wt)

  # Compute and check starting guesses ----
  start_guesses <- generate_start_guesses(
    beta.start, eta.start, model_response$y, model_response$X, beta, nt, wt, model_response$p, family
  )
  rm(beta.start, eta.start)

  # Get names and number of levels in each fixed effects category ----
  nms.fe <- lapply(model_frame$data[, k.vars, with = FALSE], levels)
  lvls.k <- vapply(nms.fe, length, integer(1))

  # Generate auxiliary list of indexes for different sub panels ----
  k.list <- getIndexList(k.vars, model_frame$data)

  # Extract control arguments ----
  tol <- control[["dev.tol"]]
  limit <- control[["limit"]]
  iter.max <- control[["iter.max"]]
  trace <- control[["trace"]]

  # Initial negative binomial fit ----
  fit <- feglmFit(
    start_guesses$beta, start_guesses$eta, model_response$y, model_response$X, wt, k.list, family, control
  )

  beta <- fit[["coefficients"]]
  eta <- fit[["eta"]]
  dev <- fit[["deviance"]]
  theta <- suppressWarnings(
    theta.ml(
      y     = model_response$y,
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
    fit <- feglmFit(beta, eta, model_response$y, model_response$X, wt, k.list, family, control)
    beta <- fit[["coefficients"]]
    eta <- fit[["eta"]]
    dev <- fit[["deviance"]]
    theta <- suppressWarnings(
      theta.ml(
        y     = model_response$y,
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
  model_response$y <- NULL
  model_response$X <- NULL
  start_guesses$eta <- NULL

  # Information if convergence failed ----
  if (!conv && trace) cat("Algorithm did not converge.\n")

  # Add names to beta, Hessian, and MX (if provided) ----
  names(fit[["coefficients"]]) <- model_response$nms.sp
  if (control[["keep.mx"]]) {
    colnames(fit[["MX"]]) <- model_response$nms.sp
  }
  dimnames(fit[["Hessian"]]) <- list(model_response$nms.sp, model_response$nms.sp)

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
