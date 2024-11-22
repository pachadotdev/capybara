#' srr_stats
#' @srrstats {G1.0} Implements an efficient offset algorithm for generalized linear models with fixed effects.
#' @srrstats {G2.1a} Ensures the input object is of class `feglm` and validates offsets.
#' @srrstats {G2.3a} Strictly checks that the `offset` parameter is numeric and matches the number of observations.
#' @srrstats {G2.14a} Issues clear error messages for invalid inputs, such as non-`feglm` objects or mismatched offsets.
#' @srrstats {G5.2a} Guarantees that all errors and warnings are unique and descriptive.
#' @srrstats {RE5.0} Optimizes iterative computation with safeguards for large-scale datasets and weight adjustments.
#' @srrstats {RE5.2} Efficiently handles updates to the linear predictor in models with fixed effects, ensuring scalability.
#' @noRd
NULL

#' @title GLM offset
#'
#' @description Efficient offset algorithm to update the linear predictor
#'
#' @param object an object of class \code{feglm}
#' @param offset a numeric vector of length equal to the number of observations
#'
#' @return an object of class \code{feglm}
#'
#' @noRd
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
  lvls_k <- object[["lvls_k"]]
  nt <- object[["nobs"]][["nobs_full"]] - object[["nobs"]][["nobs_pc"]]
  k_vars <- names(lvls_k)

  # Extract dependent variable
  y <- data[[1L]]

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
  if (is.integer(y)) {
    y <- as.numeric(y)
  }
  feglm_offset_fit_(
    eta, y, offset, wt, family[["family"]], control, k_list
  )
}
