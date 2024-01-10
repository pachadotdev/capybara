felm <- function(formula = NULL, data = NULL, weights = NULL) {
  # Use 'feglm' to estimate the model
  reslist <- feglm(formula = formula, data = data, weights = weights, family = gaussian())
  # reslist[["deviance"]] <- NULL
  # reslist[["null.deviance"]] <- NULL
  # reslist[["eta"]] <- NULL
  # reslist[["weights"]] <- NULL
  # reslist[["conv"]] <- NULL
  # reslist[["iter"]] <- NULL
  # reslist[["control"]] <- NULL

  f <- fitted.values(reslist)
  r <- reslist$data[, 1] - f
  w <- reslist$weights

  quantile(unlist(r))

  if (is.null(w)) {
    mss <- sum(f^2)
    rss <- sum(r^2)
  } else {
    mss <- sum(w * f^2)
    rss <- sum(w * r^2)
    r <- sqrt(w) * r
  }

  reslist$r.squared <- mss / (mss + rss)
  reslist$adj.r.squared <- 1 - (1 - reslist$r.squared) * ((n - df.int) / rdf)

  # Return result list
  structure(reslist, class = "felm")
}
