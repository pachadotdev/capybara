#' Augment method for fepoisson (Broom)
#'
#' @param x A fitted model object.
#' @param newdata Optional argument to use data different from the data used to fit
#'  the model.
#' @param ... Additional arguments passed to the method.
#'
#' @return A tibble with the input data and additional columns for the fitted
#' values and residuals.
#'
#' @rdname broom
#'
#' @examples
#' if (require("broom")) {
#'  set.seed(123)
#'  trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#'  trade_2006 <- trade_2006[sample(nrow(trade_2006), 1000), ]
#'
#'  mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#'  )
#'
#'  broom::augment(mod)
#' }

#' Glance method for fepoisson (Broom)
#'
#' @param x A fitted model object.
#' @param ... Additional arguments passed to the method.
#'
#' @return A tibble with the deviance, null deviance, and the number of
#' observations (full rows, missing values, and perfectly classified).
#'
#' @rdname broom
#'
#' @examples
#' if (require("broom")) {
#'   set.seed(123)
#'   trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#'   trade_2006 <- trade_2006[sample(nrow(trade_2006), 1000), ]
#'
#'   mod <- fepoisson(
#'     trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'     trade_2006
#'   )
#'
#'   broom::glance(mod)
#' }

#' Tidy method for fepoisson (Broom)
#'
#' @param x A fitted model object.
#' @param conf.int Logical indicating whether to include the confidence interval.
#' @param conf.level The confidence level for the confidence interval.
#' @param ... Additional arguments passed to the method.
#'
#' @return A tibble with the estimated coefficients, standard errors, test
#' statistics, p-values, and optionally the lower and upper bounds of the
#' confidence interval.
#'
#' @rdname broom
#'
#' @examples
#' if (require("broom")) {
#'   set.seed(123)
#'   trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#'   trade_2006 <- trade_2006[sample(nrow(trade_2006), 1000), ]
#'
#'   mod <- fepoisson(
#'     trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'     trade_2006
#'   )
#'
#'   broom::tidy(mod)
#' }
