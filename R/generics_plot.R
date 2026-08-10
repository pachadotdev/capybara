# silence R CMD check: conf_low/conf_high are evaluated by tinyplot::tinyplot()
# against the `data = conf_data` argument, not looked up in this file's scope
utils::globalVariables(c("conf_low", "conf_high"))

# srr_stats
# {G1.0} Adheres to R standards for extending base R's `plot` generic for custom classes (`feglm` and `felm`).
# {G2.1a} Ensures input objects are of the expected classes (`feglm` or `felm`), stopping otherwise.
# {G2.3a} Provides validation for optional arguments like `conf_level`, ensuring their correctness.
# {G2.3b} Handles potential case sensitivity issues for user-specified arguments.
# {G2.14a} Issues errors if required packages (`tinyplot`) are missing, ensuring dependencies are installed.
# {G2.14b} Provides default values for optional arguments when missing.
# {G3.1a} Supports customizable confidence intervals via user-provided `conf_level`.
# {G5.2a} Produces unique and informative error messages when preconditions are not met.
# {G5.4a} Includes validation against common edge cases, like missing required input or invalid argument values.

# NA_standards
# @srrstatsNA {RE6.2} Considering that the data tends to be very large, it made more sense to add a method to plot the coefficients instead of millions of predicted data points.
# @srrstatsNA {RE6.3} We plot the estimated coefficients without the fixed effects. Plotting millions of points would only add visual clutter and not provide any additional information.

#' @title Plot method for feglm objects
#'
#' @description Plots the estimated coefficients and their confidence
#'  intervals.
#'
#' @param x A fitted model object of class \code{feglm}.
#' @param ... Additional arguments passed to the method. In this case, the additional argument is `conf_level`, which is
#'  the confidence level for the confidence interval.
#'
#' @return No return value, called for the side effect of producing a plot
#'  of the estimated coefficients and their confidence intervals.
#'
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' ross2004_subset <- ross2004_subset[ross2004_subset$ltrade >
#'   quantile(ross2004_subset$ltrade, 0.75), ]
#'
#' fit <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)
#'
#' plot(fit, conf_level = 0.99)
#'
#' @exportS3Method
plot.feglm <- function(x, ...) {
  # stop if tinyplot is not installed
  if (!requireNamespace("tinyplot", quietly = TRUE)) {
    stop("The 'tinyplot' package is required to use this function")
  }

  # stop if the object is not of class feglm or felm
  if (!inherits(x, "feglm")) {
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

  # Extract the coefficient matrix from the summary
  res <- coef(summary(x))
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  # Calculate the critical value and compute confidence intervals
  z_crit <- qnorm(1 - (1 - conf_level) / 2)

  # Compute the confidence intervals
  conf_data <- data.frame(
    term = rownames(res),
    estimate = res[, "estimate"],
    conf_low = res[, "estimate"] - z_crit * res[, "std.error"],
    conf_high = res[, "estimate"] + z_crit * res[, "std.error"]
  )

  tinyplot::tinyplot(
    estimate ~ term,
    ymin = conf_low, ymax = conf_high,
    data = conf_data,
    type = "pointrange",
    pch = 19,
    col = "#165976",
    flip = TRUE,
    grid = TRUE,
    xlab = "Term",
    ylab = "Estimate",
    main = sprintf(
      "Coefficient Estimates with Confidence Intervals at %s%%",
      round(conf_level * 100, 0)
    )
  )

  invisible(NULL)
}

#' @title Plot method for felm objects
#'
#' @description Plots the estimated coefficients and their confidence intervals.
#'
#' @param x A fitted model object of class \code{felm}.
#' @param ... Additional arguments passed to the method. In this case, the additional argument is `conf_level`, which is
#'  the confidence level for the confidence interval.
#'
#' @return No return value, called for the side effect of producing a plot
#'  of the estimated coefficients and their confidence intervals.
#'
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' ross2004_subset <- ross2004_subset[ross2004_subset$ltrade >
#'   quantile(ross2004_subset$ltrade, 0.75), ]
#'
#' fit <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
#'
#' plot(fit, conf_level = 0.99)
#'
#' @exportS3Method
plot.felm <- function(x, ...) {
  # stop if tinyplot is not installed
  if (!requireNamespace("tinyplot", quietly = TRUE)) {
    stop("The 'tinyplot' package is required to use this function")
  }

  # stop if the object is not of class feglm or felm
  if (!inherits(x, "felm")) {
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

  # Extract the coefficient matrix from the summary
  res <- coef(summary(x))
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  # Calculate the critical value and compute confidence intervals
  z_crit <- qnorm(1 - (1 - conf_level) / 2)

  # Compute the confidence intervals
  conf_data <- data.frame(
    term = rownames(res),
    estimate = res[, "estimate"],
    conf_low = res[, "estimate"] - z_crit * res[, "std.error"],
    conf_high = res[, "estimate"] + z_crit * res[, "std.error"]
  )

  tinyplot::tinyplot(
    estimate ~ term,
    ymin = conf_low, ymax = conf_high,
    data = conf_data,
    type = "pointrange",
    pch = 19,
    col = "#165976",
    flip = TRUE,
    grid = TRUE,
    xlab = "Term",
    ylab = "Estimate",
    main = sprintf(
      "Coefficient Estimates with Confidence Intervals at %s%%",
      round(conf_level * 100, 0)
    )
  )

  invisible(NULL)
}
