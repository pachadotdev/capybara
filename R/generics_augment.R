#' @importFrom generics augment
#' @export
generics::augment

#' @title Augment method for 'feglm' objects
#' @description Integration with the 'broom' package
#' @export
#' @noRd
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

#' @title Augement method for 'felm' objects
#' @description Integration with the 'broom' package
#' @export
#' @noRd
augment.felm <- function(x, newdata = NULL, ...) {
  augment.feglm(x, newdata, ...)
}
