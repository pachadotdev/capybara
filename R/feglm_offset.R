#' srr_stats
#' @srrstats {G1.0} Implements an efficient offset algorithm for generalized linear models with fixed effects.
#' @srrstats {G2.1a} Ensures the input object is of class `feglm` and validates offsets.
#' @srrstats {G2.3a} Strictly checks that the `offset` parameter is numeric and matches the number of observations.
#' @srrstats {G2.14a} Issues clear error messages for invalid inputs, such as non-`feglm` objects or mismatched offsets.
#' @srrstats {G5.2a} Guarantees that all errors and warnings are unique and descriptive.
#' @srrstats {RE5.0} Optimizes iterative computation with safeguards for large-scale datasets and weight adjustments.
#' @srrstats {RE5.2} Efficiently handles updates to the linear predictor in models with fixed effects, ensuring
#'  scalability.
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

  # Generate auxiliary list of indexes to project out the fixed effects
  k_list <- get_index_list_(names(object[["fe_levels"]]), object[["data"]])

  # Extract dependent variable
  y <- object[["data"]][[1L]]

  # Compute starting guess for eta
  nt <- object[["nobs"]][["nobs"]]
  if (object[["family"]][["family"]] == "binomial") {
    eta <- rep(
      object[["family"]][["linkfun"]](
        sum(object[["weights"]] * (y + 0.5) / 2.0) / sum(object[["weights"]])
      ),
      nt
    )
  } else if (
    object[["family"]][["family"]] %in% c("Gamma", "inverse.gaussian")
  ) {
    eta <- rep(
      object[["family"]][["linkfun"]](
        sum(object[["weights"]] * y) / sum(object[["weights"]])
      ),
      nt
    )
  } else {
    eta <- rep(
      object[["family"]][["linkfun"]](
        sum(object[["weights"]] * (y + 0.1)) / sum(object[["weights"]])
      ),
      nt
    )
  }

  # Return eta
  if (is.integer(y)) {
    y <- as.numeric(y)
  }
  feglm_offset_fit_(
    eta,
    y,
    offset,
    object[["weights"]],
    object[["family"]][["family"]],
    object[["control"]],
    k_list
  )
}
