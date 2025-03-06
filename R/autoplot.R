utils::globalVariables(c("term", "estimate", "conf_low", "conf_high"))

#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' srr_stats
#' @srrstats {G1.0} Adheres to R standards for extending methods like `autoplot` for custom classes (`feglm` and `felm`).
#' @srrstats {G2.1a} Ensures input objects are of the expected classes (`feglm` or `felm`), stopping otherwise.
#' @srrstats {G2.3a} Provides validation for optional arguments like `conf_level`, ensuring their correctness.
#' @srrstats {G2.3b} Handles potential case sensitivity issues for user-specified arguments.
#' @srrstats {G2.14a} Issues errors if required packages (`ggplot2`) are missing, ensuring dependencies are installed.
#' @srrstats {G2.14b} Provides default values for optional arguments when missing.
#' @srrstats {G3.1a} Supports customizable confidence intervals via user-provided `conf_level`.
#' @srrstats {G5.2a} Produces unique and informative error messages when preconditions are not met.
#' @srrstats {G5.4a} Includes validation against common edge cases, like missing required input or invalid argument values.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {RE6.2} Considering that the data tends to be very large, it made more sense to add a method to plot the coefficients instead of millions of predicted data points.
#' @srrstatsNA {RE6.3} We plot the estimated coefficients without the fixed effects. Plotting millions of points would only add visual clutter and not provide any additional information.
#' @noRd
NULL

#' @title Autoplot method for feglm objects
#'
#' @description Extracts the estimated coefficients and their confidence
#'  intervals.
#'
#' @rdname autoplot
#'
#' @param object A fitted model object.
#' @param ... Additional arguments passed to the method. In this case,
#'  the additional argument is `conf_level`, which is the confidence level for
#'  the confidence interval.
#'
#' @return A ggplot object with the estimated coefficients and their confidence
#'  intervals.
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
#' autoplot(mod, conf_level = 0.99)
#'
#' @export
autoplot.feglm <- function(object, ...) {
  # stop if ggplot2 is not installed
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required to use this function")
  }

  # stop if the object is not of class feglm or felm
  if (!inherits(object, "feglm")) {
    stop("The object must be of class 'feglm'")
  }

  # if conf_level is not provided, set it to 0.95
  if (!"conf_level" %in% names(list(...))) {
    conf_level <- 0.95
  } else {
    conf_level <- list(...)$conf_level
  }

  # check that conf_level is between 0 and 1
  if (conf_level <= 0 || conf_level >= 1) {
    stop("The confidence level must be between 0 and 1")
  }

  # Extract the summary of the feglm object
  res <- summary(object)$cm
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  # Calculate the critical value for the specified confidence conf_level
  alpha <- 1 - conf_level
  z <- qnorm(1 - alpha / 2)

  # Compute the confidence intervals
  conf_data <- data.frame(
    term = rownames(res),
    estimate = res[, "estimate"],
    conf_low = res[, "estimate"] - z * res[, "std.error"],
    conf_high = res[, "estimate"] + z * res[, "std.error"]
  )

  p <- ggplot(conf_data, aes(x = term, y = estimate)) +
    geom_errorbar(
      aes(
        ymin = conf_low,
        ymax = conf_high
      ),
      width = 0.1,
      color = "#165976"
    ) +
    geom_point() +
    labs(
      title = sprintf(
        "Coefficient Estimates with Confidence Intervals at %s%%",
        round(conf_level * 100, 0)
      ),
      x = "Term",
      y = "Estimate"
    ) +
    theme_minimal() +
    coord_flip()

  p
}

#' @title Autoplot method for felm objects
#'
#' @description Extracts the estimated coefficients and their confidence
#'
#' @rdname autoplot
#'
#' @return A ggplot object with the estimated coefficients and their confidence
#' intervals.
#'
#' @examples
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[trade_2006$trade > 0, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#' trade_2006$log_trade <- log(trade_2006$trade)
#'
#' mod <- felm(
#'   log_trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
#' autoplot(mod, conf_level = 0.90)
#'
#' @export
autoplot.felm <- function(object, ...) {
  # stop if ggplot2 is not installed
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required to use this function")
  }

  # stop if the object is not of class feglm or felm
  if (!inherits(object, "felm")) {
    stop("The object must be of class 'felm'")
  }

  # if conf_level is not provided, set it to 0.95
  if (!"conf_level" %in% names(list(...))) {
    conf_level <- 0.95
  } else {
    conf_level <- list(...)$conf_level
  }

  # check that conf_level is between 0 and 1
  if (conf_level <= 0 || conf_level >= 1) {
    stop("The confidence level must be between 0 and 1")
  }

  # Extract the summary of the felm object
  res <- summary(object)$cm
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  # Calculate the critical value for the specified confidence conf_level
  alpha <- 1 - conf_level
  z <- qnorm(1 - alpha / 2)

  # Compute the confidence intervals
  conf_data <- data.frame(
    term = rownames(res),
    estimate = res[, "estimate"],
    conf_low = res[, "estimate"] - z * res[, "std.error"],
    conf_high = res[, "estimate"] + z * res[, "std.error"]
  )

  p <- ggplot(conf_data, aes(x = term, y = estimate)) +
    geom_errorbar(
      aes(
        ymin = conf_low,
        ymax = conf_high
      ),
      width = 0.1,
      color = "#165976"
    ) +
    geom_point() +
    labs(
      title = sprintf(
        "Coefficient Estimates with Confidence Intervals at %s%%",
        round(conf_level * 100, 0)
      ),
      x = "Term",
      y = "Estimate"
    ) +
    theme_minimal() +
    coord_flip()

  p
}
