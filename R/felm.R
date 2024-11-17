#' srr_stats (tests)
#' @srrstats {G1.0} Statistical Software should list at least one primary
#'  reference from published academic literature.
#' @srrstats {G1.3} All statistical terminology should be clarified and
#'  unambiguously defined.
#' @srrstats {RE4.0} Regression Software should return some form of "model"
#'  object, generally through using or modifying existing class structures for
#'  model objects (such as `lm`, `glm`, or model objects from other packages),
#'  or creating a new class of model objects.
#' @noRd
NULL

#' @title LM fitting with high-dimensional k-way fixed effects
#'
#' @description A wrapper for \code{\link{feglm}} with
#'  \code{family = gaussian()}.
#'
#' @inheritParams feglm
#' 
#' @return A named list of class \code{"felm"}. The list contains the following
#'  eleven elements:
#'  \item{coefficients}{a named vector of the estimated coefficients}
#'  \item{fitted.values}{a vector of the estimated dependent variable}
#'  \item{weights}{a vector of the weights used in the estimation}
#'  \item{hessian}{a matrix with the numerical second derivatives}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{nobs}{a named vector with the number of observations used in the
#'    estimation indicating the dropped and perfectly predicted observations}
#'  \item{lvls_k}{a named vector with the number of levels in each fixed
#'    effect}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing
#'   observations}
#'  \item{control}{the control list used in the model}
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
#' # check the feglm examples for the details about clustered standard errors
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- felm(
#'   log(trade) ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
#' summary(mod)
#'
#' @export
felm <- function(formula = NULL, data = NULL, weights = NULL) {
  # Use 'feglm' to estimate the model
  # Using felm_fit_ directly leads to the incorrect yhat = Xb
  # we need iteratively reweighted least squares
  reslist <- feglm(
    formula = formula, data = data, weights = weights, family = gaussian()
  )

  names(reslist)[which(names(reslist) == "eta")] <- "fitted.values"

  reslist[["conv"]] <- NULL
  reslist[["iter"]] <- NULL
  reslist[["family"]] <- NULL
  reslist[["deviance"]] <- NULL

  # Return result list
  structure(reslist, class = "felm")
}
