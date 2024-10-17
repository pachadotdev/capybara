#' @importFrom generics augment
#' @export
generics::augment

#' @title Broom Integration
#'
#' @srrstats {RE4.10} *Model Residuals, including sufficient documentation to enable interpretation of residuals, and to enable users to submit residuals to their own tests.*
#' 
#' @description The provided `broom` methods do the following:
#'  1. `augment`: takes the input data and adds additional columns with the
#'      fitted values and residuals.
#'  2. `glance`: Extracts the deviance, null deviance, and the number of
#'      observations.`
#'  3. `tidy`: Extracts the estimated coefficients and their standard errors.
#'
#' @param x A fitted model object.
#' @param newdata Optional argument to use data different from the data used to
#'  fit the model.
#' @param conf_int Logical indicating whether to include the confidence
#'  interval.
#' @param conf_level The confidence level for the confidence interval.
#' @param ... Additional arguments passed to the method.
#'
#' @return A tibble with the respective information for the `augment`, `glance`,
#'  and `tidy` methods.
#'
#' @rdname broom
#'
#' @examples
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
#' broom::augment(mod)
#' broom::glance(mod)
#' broom::tidy(mod)
#'
#' @export
augment.feglm <- function(x, newdata = NULL, ...) {
  if (is.null(newdata)) {
    res <- x$data
  } else {
    res <- newdata
  }

  res[[".fitted"]] <- predict(x, type = "response")
  res[[".residuals"]] <- res[[names(x$data)[1]]] - res[[".fitted"]]

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @rdname broom
#' @export
augment.felm <- function(x, newdata = NULL, ...) {
  augment.feglm(x, newdata, ...)
}
