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
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 1000), ]
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
