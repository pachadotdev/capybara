#' @title Summary method for fixed effects APEs
#' @inherit vcov.apes
#' @export
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
