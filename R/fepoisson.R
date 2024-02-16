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
#' @export
fepoisson <- function(
    formula = NULL,
    data = NULL,
    weights = NULL,
    beta.start = NULL,
    eta.start = NULL,
    control = NULL) {
  return(
    feglm(
      formula = formula, data = data, weights = weights, family = poisson(),
      beta.start = beta.start, eta.start = eta.start, control = control
    )
  )
}
