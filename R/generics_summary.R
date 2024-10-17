#' @title Summary method for fixed effects APEs
#' @inherit vcov.apes
#' @export
#' @srrstats {RE4.6} *The variance-covariance matrix of the model parameters (via `vcov()`)*
#' @srrstats {RE4.5} *Numbers of observations submitted to model (via `nobs()`)*
#' @srrstats {RE4.7} *Where appropriate, convergence statistics*
#' @srrstats {RE4.11} *Goodness-of-fit and other statistics associated such as effect sizes with model coefficients.*
#' @srrstats {RE4.18} *Regression Software may also implement `summary` methods for model objects, and in particular should implement distinct `summary` methods for any cases in which calculation of summary statistics is computationally non-trivial (for example, for bootstrapped estimates of confidence intervals).*
#' @noRd
summary.apes <- function(object, ...) {
  # Compute coefficent matrix
  est <- object[["delta"]]
  se <- sqrt(diag(object[["vcov"]]))
  z <- est / se
  p <- 2.0 * pnorm(-abs(z))
  cm <- cbind(est, se, z, p)
  rownames(cm) <- names(est)
  colnames(cm) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")

  # Return coefficient matrix
  structure(list(cm = cm), class = "summary.apes")
}

#' @title Summary method for fixed effects GLMs
#' @inherit vcov.feglm
#' @export
#' @srrstats {RE4.6} *The variance-covariance matrix of the model parameters (via `vcov()`)*
#' @srrstats {RE4.5} *Numbers of observations submitted to model (via `nobs()`)*
#' @srrstats {RE4.7} *Where appropriate, convergence statistics*
#' @srrstats {RE4.11} *Goodness-of-fit and other statistics associated such as effect sizes with model coefficients.*
#' @noRd
summary.feglm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    ...) {
  # Compute coefficients matrix
  est <- object[["coefficients"]]
  se <- sqrt(diag(vcov(object, type)))
  z <- est / se
  p <- 2.0 * pnorm(-abs(z))
  cm <- cbind(est, se, z, p)
  rownames(cm) <- names(est)
  colnames(cm) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")

  # Generate result list
  res <- list(
    cm            = cm,
    deviance      = object[["deviance"]],
    null_deviance = object[["null_deviance"]],
    iter          = object[["iter"]],
    nobs          = object[["nobs"]],
    lvls_k        = object[["lvls_k"]],
    formula       = object[["formula"]],
    family        = object[["family"]]
  )

  if (object[["family"]][["family"]] == "poisson") {
    # Compute pseudo R-squared
    # http://personal.lse.ac.uk/tenreyro/r2.do
    # pass matrix with y and yhat as columns
    res[["pseudo.rsq"]] <- (kendall_cor(
      unlist(object$data[, 1], use.names = FALSE),
      predict(object, type = "response")
    ))^2
  }

  if (inherits(object, "fenegbin")) {
    res[["theta"]] <- object[["theta"]]
    res[["iter.outer"]] <- object[["iter.outer"]]
  }

  # Return list
  structure(res, class = "summary.feglm")
}

#' @title Summary method for fixed effects LMs
#' @inherit vcov.felm
#' @export
#' @srrstats {RE4.6} *The variance-covariance matrix of the model parameters (via `vcov()`)*
#' @srrstats {RE4.5} *Numbers of observations submitted to model (via `nobs()`)*
#' @srrstats {RE4.7} *Where appropriate, convergence statistics*
#' @srrstats {RE4.11} *Goodness-of-fit and other statistics associated such as effect sizes with model coefficients.*
#' @srrstats {RE4.18} *Regression Software may also implement `summary` methods for model objects, and in particular should implement distinct `summary` methods for any cases in which calculation of summary statistics is computationally non-trivial (for example, for bootstrapped estimates of confidence intervals).*
#' @noRd
summary.felm <- function(
    object,
    type = "hessian",
    ...) {
  # Compute coefficients matrix
  est <- object[["coefficients"]]
  se <- sqrt(diag(vcov(object, type)))
  z <- est / se
  p <- 2.0 * pnorm(-abs(z))
  cm <- cbind(est, se, z, p)
  rownames(cm) <- names(est)
  colnames(cm) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")

  y <- unlist(object[["data"]][, 1], use.names = FALSE)
  w <- object[["weights"]]
  ydemeaned_sq <- (y - mean(y))^2
  e_sq <- (y - object[["fitted.values"]])^2
  tss <- sum(w * ydemeaned_sq)
  rss <- sum(w * e_sq)
  n <- unname(object[["nobs"]]["nobs_full"])
  k <- length(object[["coefficients"]]) +
    sum(vapply(object[["nms_fe"]], length, integer(1)))
  rsq <- 1 - (rss / tss)

  # Generate result list
  res <- list(
    cm            = cm,
    nobs          = object[["nobs"]],
    lvls_k        = object[["lvls_k"]],
    formula       = object[["formula"]],
    r.squared     = rsq,
    adj.r.squared = 1 - (1 - rsq) * (n - 1) / (n - k + 1)
  )

  # Return list
  structure(res, class = "summary.felm")
}
