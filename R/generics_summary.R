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
  colnames(cm) <- c("Estimate", "Std. error", "z value", "Pr(> |z|)")

  # Return coefficient matrix
  structure(list(cm = cm), class = "summary.apes")
}

#' @export
#' @noRd
summary.feglm <- function(
    object,
    type = c("hessian", "outer.product", "sandwich", "clustered"),
    cluster = NULL,
    ...) {
  # Compute coefficients matrix
  est <- object[["coefficients"]]
  se <- sqrt(diag(vcov(object, type, cluster)))
  z <- est / se
  p <- 2.0 * pnorm(-abs(z))
  cm <- cbind(est, se, z, p)
  rownames(cm) <- names(est)
  colnames(cm) <- c("Estimate", "Std. error", "z value", "Pr(> |z|)")

  # Generate result list
  res <- list(
    cm            = cm,
    deviance      = object[["deviance"]],
    null.deviance = object[["null.deviance"]],
    iter          = object[["iter"]],
    nobs          = object[["nobs"]],
    lvls.k        = object[["lvls.k"]],
    formula       = object[["formula"]],
    family        = object[["family"]]
  )
  if (inherits(object, "feglm.nb")) {
    res[["theta"]] <- object[["theta"]]
    res[["iter.outer"]] <- object[["iter.outer"]]
  }

  # Return list
  structure(res, class = "summary.feglm")
}

#' @export
#' @noRd
summary.felm <- function(
    object,
    type = "hessian",
    cluster = NULL,
    ...) {
  # Compute coefficients matrix
  est <- object[["coefficients"]]
  se <- sqrt(diag(vcov(object, type, cluster)))
  z <- est / se
  p <- 2.0 * pnorm(-abs(z))
  cm <- cbind(est, se, z, p)
  rownames(cm) <- names(est)
  colnames(cm) <- c("Estimate", "Std. error", "z value", "Pr(> |z|)")

  y <- unlist(object$data[, 1], use.names = FALSE)
  # yhat <- object$fitted.values
  # ybar <- mean(y)
  w <- object$weights
  ydemeaned_sq <- (y - mean(y))^2
  e_sq <- (y - object$fitted.values)^2
  tss <- sum(w * ydemeaned_sq)
  rss <- sum(w * e_sq)
  n <- unname(object[["nobs"]]["nobs.full"])
  k <- length(object$coefficients) +
    sum(vapply(object$nms.fe, length, integer(1)))

  # r.squared <- 1 - (rss / tss)
  # no -1 in the denominator because the FE estimation does not include the
  # "grand mean"
  # adj.r.squared <- 1 - (1 - object$r.squared) * ((n - 1) / (n - k))

  # Generate result list
  res <- list(
    cm            = cm,
    nobs          = object[["nobs"]],
    lvls.k        = object[["lvls.k"]],
    formula       = object[["formula"]],
    r.squared     = 1 - (rss / tss),
    adj.r.squared = 1 - (rss / tss) * ((n - 1) / (n - k))
  )

  # Return list
  structure(res, class = "summary.felm")
}
