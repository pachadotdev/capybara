#' @title Negative Binomial model fitting with high-dimensional k-way fixed
#'  effects
#' @description A routine that uses the same internals as \code{\link{feglm}}.
#' @inheritParams feglm
#' @param init_theta an optional initial value for the theta parameter (see
#'  \code{\link[MASS]{glm.nb}}).
#' @param link the link function. Must be one of \code{"log"}, \code{"sqrt"}, or
#'  \code{"identity"}.
#' @examples
#' # same as the example in fepoisson but with overdispersion/underdispersion
#' mod <- fenegbin(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_panel
#' )
#'
#' summary(mod)
#' 
#' @return A named list of class \code{"feglm"}.
#' 
#' @export
fenegbin <- function(
    formula = NULL,
    data = NULL,
    weights = NULL,
    beta_start = NULL,
    eta_start = NULL,
    init_theta = NULL,
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
  lhs <- NA # just to avoid global variable warning
  nobs_na <- NA
  nobs_full <- NA
  model_frame_(data, formula, weights)

  # Check starting guess of theta ----
  family <- init_theta_(init_theta, link)
  rm(init_theta)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables and sort ----
  k_vars <- attr(terms(formula, rhs = 2L), "term.labels")
  k <- length(k_vars)

  # Generate temporary variable ----
  tmp.var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp.var, k_vars, control)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, k_vars)

  # Determine the number of dropped observations ----
  nt <- nrow(data)
  nobs <- nobs_(nobs_full, nobs_na, nt)

  # Extract model response and regressor matrix ----
  nms_sp <- NA
  p <- NA
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
  start_guesses_(beta_start, eta_start, y, X, beta, nt, wt, p, family)

  # Get names and number of levels in each fixed effects category ----
  nms_fe <- lapply(select(data, all_of(k_vars)), levels)
  lvls_k <- vapply(nms_fe, length, integer(1))

  # Generate auxiliary list of indexes for different sub panels ----
  k_list <- get_index_list_(k_vars, data)

  # Extract control arguments ----
  tol <- control[["dev_tol"]]
  limit <- control[["limit"]]
  iter_max <- control[["iter_max"]]
  trace <- control[["trace"]]

  # Initial negative binomial fit ----

  theta <- suppressWarnings(
    theta.ml(
      y     = y,
      mu    = family[["linkinv"]](eta),
      n     = nt,
      limit = limit,
      trace = trace
    )
  )

  fit <- feglm_fit_(
    beta, eta, y, X, wt, theta, family[["family"]], control, k_list
  )

  beta <- fit[["coefficients"]]
  eta <- fit[["eta"]]
  dev <- fit[["deviance"]]

  # Alternate between fitting glm and \theta ----
  conv <- FALSE
  for (iter in seq.int(iter_max)) {
    # Fit negative binomial model
    dev.old <- dev
    theta.old <- theta
    family <- negative.binomial(theta, link)
    theta <- suppressWarnings(
      theta.ml(
        y     = y,
        mu    = family[["linkinv"]](eta),
        n     = nt,
        limit = limit,
        trace = trace
      )
    )
    fit <- feglm_fit_(beta, eta, y, X, wt, theta, family[["family"]], control, k_list)
    beta <- fit[["coefficients"]]
    eta <- fit[["eta"]]
    dev <- fit[["deviance"]]

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

  # Add names to beta, hessian, and MX (if provided) ----
  names(fit[["coefficients"]]) <- nms_sp
  if (control[["keep_mx"]]) {
    colnames(fit[["MX"]]) <- nms_sp
  }
  dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)

  # Generate result list ----
  reslist <- c(
    fit, list(
      theta      = theta,
      iter.outer = iter,
      conv.outer = conv,
      nobs       = nobs,
      lvls_k     = lvls_k,
      nms_fe     = nms_fe,
      formula    = formula,
      data       = data,
      family     = family,
      control    = control
    )
  )

  # Return result list ----
  structure(reslist, class = c("feglm", "fenegbin"))
}
