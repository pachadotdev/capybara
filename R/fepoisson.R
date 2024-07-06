#' @title Poisson model fitting high-dimensional with k-way fixed effects
#' @description A wrapper for \code{\link{feglm}} with
#'  \code{family = poisson()}.
#' @inheritParams feglm
#' @examples
#' # same as the example in feglm but with less typing
#' mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_panel
#' )
#'
#' summary(mod)
#' 
#' @return A named list of class \code{"feglm"}.
#' 
#' @export
fepoisson <- function(
    formula = NULL,
    data = NULL,
    weights = NULL,
    beta.start = NULL,
    eta.start = NULL,
    control = NULL) {
  feglm(
    formula = formula, data = data, weights = weights, family = poisson(),
    beta.start = beta.start, eta.start = eta.start, control = control
  )
}

# fequasipoisson <- function(
#     formula = NULL,
#     data = NULL,
#     weights = NULL,
#     beta.start = NULL,
#     eta.start = NULL,
#     control = NULL) {
#   # Fit the model using standard Poisson assumptions
#   fit <- feglm(
#     formula = formula, data = data, weights = weights, family = poisson(),
#     beta.start = beta.start, eta.start = eta.start, control = control
#   )

#   # Estimate the dispersion parameter (phi)
#   fitted_values <- predict(object, type = "response")
#   residuals <- unlist(object$data[, 1], use.names = FALSE) - fitted_values
#   phi <- sum((residuals^2) / fitted_values) / fit$df.residual?

#   # Adjust model diagnostics for Quasi Poisson
#   fit$std.errors <- sqrt(phi) * fit$std.errors

#   return(fit)
# }
