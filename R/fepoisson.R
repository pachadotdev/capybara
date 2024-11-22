#' srr_stats
#' @srrstats {G1.0} Implements Poisson regression with high-dimensional fixed effects via `feglm`.
#' @srrstats {G2.1a} Validates input `formula` to ensure correct specification of fixed effects.
#' @srrstats {G2.1b} Ensures `data` is appropriately formatted and contains sufficient observations.
#' @srrstats {G2.3a} Uses internally validated arguments (`control` and starting guesses) for consistency.
#' @srrstats {G3.1a} Supports canonical log link function for Poisson family.
#' @srrstats {G3.1b} Provides detailed outputs including coefficients, deviance, and convergence diagnostics.
#' @srrstats {G5.2a} Issues informative error messages when inputs or configurations are invalid.
#' @srrstats {RE5.0} Optimized for high-dimensional data with scalability in fixed effect estimation.
#' @srrstats {RE5.1} Relies on robust `feglm` fitting procedure to handle large-scale Poisson models.
#' @srrstats {RE5.2} Ensures consistent convergence reporting via `feglm`.
#' @noRd
NULL

#' @title Poisson model fitting high-dimensional with k-way fixed effects
#'
#' @description A wrapper for \code{\link{feglm}} with
#'  \code{family = poisson()}.
#'
#' @inheritParams feglm
#'
#' @examples
#' # check the feglm examples for the details about clustered standard errors
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
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
    beta_start = NULL,
    eta_start = NULL,
    control = NULL) {
  feglm(
    formula = formula, data = data, weights = weights, family = poisson(),
    beta_start = beta_start, eta_start = eta_start, control = control
  )
}
