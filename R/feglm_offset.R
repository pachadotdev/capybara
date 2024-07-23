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
  k_vars <- names(lvls_k)

  # Extract dependent variable
  y <- data[[1L]]

  # Extract control arguments
  center_tol <- control[["center_tol"]]
  dev_tol <- control[["dev_tol"]]
  iter_max <- control[["iter_max"]]

  # Generate auxiliary list of indexes to project out the fixed effects
  k_list <- get_index_list_(k_vars, data)

  # Compute starting guess for eta
  if (family[["family"]] == "binomial") {
    eta <- rep(family[["linkfun"]](sum(wt * (y + 0.5) / 2.0) / sum(wt)), nt)
  } else if (family[["family"]] %in% c("Gamma", "inverse.gaussian")) {
    eta <- rep(family[["linkfun"]](sum(wt * y) / sum(wt)), nt)
  } else {
    eta <- rep(family[["linkfun"]](sum(wt * (y + 0.1)) / sum(wt)), nt)
  }

  # Return eta
  if (is.integer(y)) { y <- as.numeric(y) }
  feglm_offset_fit_(eta, y, offset, wt, family[["family"]], control,
    k_list)
}
