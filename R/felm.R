#' @title LM fitting with high-dimensional k-way fixed effects
#'
#' @description A wrapper for \code{\link{feglm}} with
#'  \code{family = gaussian()}.
#'
#' @inheritParams feglm
#'
#' @return The function \code{\link{felm}} returns a named list of class
#'  \code{"felm"}.
#'
#' @references Gaure, S. (2013). "OLS with Multiple High Dimensional Category
#'  Variables". Computational Statistics and Data Analysis, 66.
#' @references Marschner, I. (2011). "glm2: Fitting generalized linear models
#'  with convergence problems". The R Journal, 3(2).
#' @references Stammann, A., F. Heiss, and D. McFadden (2016). "Estimating Fixed
#'  Effects Logit Models with Large Panel Data". Working paper.
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized
#'  Linear Models with High-Dimensional k-Way Fixed Effects". ArXiv e-prints.
#'
#' @examples
#' mod <- felm(
#'   log(trade) ~ dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_panel
#' )
#'
#' summary(mod)
#' @export
felm <- function(formula = NULL, data = NULL, weights = NULL) {
  # Use 'feglm' to estimate the model
  # Using felm_fit_ directly leads to the incorrect yhat = Xb
  # we need iteratively reweighted least squares
  reslist <- feglm(
    formula = formula, data = data, weights = weights, family = gaussian()
  )

  names(reslist)[which(names(reslist) == "eta")] <- "fitted.values"

  # reslist[["Hessian"]] <- NULL
  reslist[["family"]] <- NULL
  reslist[["deviance"]] <- NULL

  # Return result list
  structure(reslist, class = "felm")
}
