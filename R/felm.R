felm <- function(formula = NULL, data = NULL, weights = NULL) {
  # Use 'feglm' to estimate the model
  reslist <- feglm(formula = formula, data = data, weights = weights, family = gaussian())

  yhat <- fitted.values(reslist)
  y <- unlist(reslist$data[, 1], use.names = FALSE)
  ybar <- mean(y)

  w <- reslist$weights

  ydemeaned_sq <- (y - ybar)^2
  e_sq <- (y - yhat)^2

  if (is.null(w)) {
    tss <- sum(ydemeaned_sq)
    rss <- sum(e_sq)
  } else {
    tss <- sum(w * ydemeaned_sq)
    rss <- sum(w * e_sq)
  }

  n <- length(yhat)
  k <- length(reslist$coefficients) + sum(vapply(reslist$nms.fe, length, integer(1)))
  
  reslist$r.squared <- 1 - (rss / tss)

  # no -1 in the denominator because the FE estimation does not include the "grand mean"
  reslist$adj.r.squared <- 1 - (1 - reslist$r.squared) * ((n - 1) / (n - k))

  # Return result list
  structure(reslist, class = "felm")
}
