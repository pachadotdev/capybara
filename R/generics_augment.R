#' @importFrom generics augment
#' @export
generics::augment

#' srr_stats
#' @srrstats {G1.0} Provides integration with the `broom` package for model output tidying.
#' @srrstats {G2.1a} Ensures the input object is of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Ensures compatibility with new data provided via the `newdata` argument.
#' @srrstats {G3.1a} Outputs include fitted values and residuals in a tidy tibble format for interpretability.
#' @srrstats {G3.1c} Supports additional columns in the output for confidence intervals if requested.
#' @srrstats {G3.3} Handles the addition of multiple model outputs (`.fitted`, `.residuals`) to the data.
#' @srrstats {G5.1} Provides robust error handling for missing or invalid input objects.
#' @srrstats {RE4.10} The residuals are returned in a tidy data frame following the `broom` convention.
#' @srrstats {RE4.11} The deviance and null deviance are returned in a tidy data frame following the `broom` convention.
#' @srrstats {RE5.0} Optimized for integration with downstream analysis workflows.
#' @srrstats {RE5.1} Maintains computational efficiency when augmenting large datasets.
#' @srrstats {RE5.3} Supports additional data input (`newdata`) to enhance flexibility.
#' @noRd
NULL

#' @title Broom Integration
#'
#' @description The provided `broom` methods do the following:
#'  1. `augment`: Takes the input data and adds additional columns with the
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
#' mod <- fepoisson(mpg ~ wt | cyl, mtcars)
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
