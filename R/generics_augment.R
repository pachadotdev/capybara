#' @importFrom generics augment
#' @export
generics::augment

# srr_stats
# {G1.0} Provides integration with the `broom` package for model output tidying.
# {G2.1a} Ensures the input object is of the expected class (`feglm` or `felm`).
# {G2.3a} Ensures compatibility with new data provided via the `newdata` argument.
# {G3.1a} Outputs include fitted values and residuals in a tidy tibble format for interpretability.
# {G3.1c} Supports additional columns in the output for confidence intervals if requested.
# {G3.3} Handles the addition of multiple model outputs (`.fitted`, `.residuals`) to the data.
# {G5.1} Provides robust error handling for missing or invalid input objects.
# {RE4.10} The residuals are returned in a tidy data frame following the `broom` convention.
# {RE4.11} The deviance and null deviance are returned in a tidy data frame following the `broom` convention.
# {RE5.0} Optimized for integration with downstream analysis workflows.
# {RE5.1} Maintains computational efficiency when augmenting large datasets.
# {RE5.3} Supports additional data input (`newdata`) to enhance flexibility.

#' @title Broom Integration
#'
#' @description The provided `broom` methods do the following:
#'  1. `augment`: Takes the input data and adds additional columns with the fitted values and residuals.
#'  2. `glance`: Extracts the deviance, null deviance, and the number of observations.`
#'  3. `tidy`: Extracts the estimated coefficients and their standard errors.
#'
#' @param x A fitted model object.
#' @param newdata Optional argument to use data different from the data used to fit the model.
#' @param conf_int Logical indicating whether to include the confidence interval.
#' @param conf_level The confidence level for the confidence interval.
#' @param ... Additional arguments passed to the method.
#'
#' @return A tibble with the respective information for the `augment`, `glance`, and `tidy` methods.
#'
#' @rdname broom
#'
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' ross2004_subset <- ross2004_subset[ross2004_subset$ltrade >
#'   quantile(ross2004_subset$ltrade, 0.75), ]
#'
#' fit <- fepoisson(ltrade ~ ldist, ross2004_subset,
#'   control = fit_control(keep_data = TRUE)
#' )
#'
#' broom::augment(fit)
#' broom::glance(fit)
#' broom::tidy(fit)
#'
#' @export
augment.feglm <- function(x, newdata = NULL, ...) {
  if (is.null(newdata)) {
    if (is.null(x$data)) {
      stop(
        "augment() requires `keep_data = TRUE` in fit_control() ",
        "or a newdata argument.",
        call. = FALSE
      )
    }
    # Create a copy to avoid mutating x$data
    res <- x$data
    resp_name <- names(x$data)[1L]
  } else {
    res <- newdata
    # Get response name from formula
    resp_name <- all.vars(x$formula)[1L]
  }

  res[[".fitted"]] <- predict(x, newdata = newdata, type = "response")
  if (resp_name %in% names(res)) {
    res[[".residuals"]] <- res[[resp_name]] - res[[".fitted"]]
  }

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @rdname broom
#' @export
augment.felm <- function(x, newdata = NULL, ...) {
  augment.feglm(x, newdata, ...)
}
