#' Extract fixed effects from a fitted model
#'
#' @description Accessor for the fixed-effect coefficients stored in a
#'   \code{felm} or \code{feglm} object, so that they can be retrieved
#'   without reaching into the model object's internal list structure (e.g.
#'   \code{object$fixed_effects$dimension}).
#' @param object A \code{felm} or \code{feglm} object fitted with
#'   \code{control = fit_control(return_fe = TRUE)} (the default).
#' @param which Optional string with the name of a single fixed-effect
#'   dimension to extract (e.g. \code{"exp_year"}). If \code{NULL} (default),
#'   the full named list of fixed-effect vectors is returned.
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' fit <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
#'
#' fixed_effects(fit, "ctry1")
#' @return A named numeric vector (single dimension) or a named list of
#'   numeric vectors (all dimensions).
#' @export
fixed_effects <- function(object, which = NULL) {
  if (!inherits(object, c("felm", "feglm"))) {
    stop("`object` is not a felm or feglm object")
  }

  fe <- object[["fixed_effects"]]
  if (is.null(fe)) {
    stop(
      "Model has no fixed effects. ",
      "Refit with control = fit_control(return_fe = TRUE)."
    )
  }

  if (is.null(which)) {
    return(fe)
  }

  if (!is.character(which) || length(which) != 1L) {
    stop("`which` must be a single string naming a fixed-effect dimension")
  }

  if (!which %in% names(fe)) {
    stop(
      "'", which, "' is not a fixed-effect dimension in this model. ",
      "Available dimensions: ", paste(names(fe), collapse = ", ")
    )
  }

  fe[[which]]
}
