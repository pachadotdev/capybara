#' @importFrom generics augment
#' @export
generics::augment

#' @export
#' @noRd
augment.feglm <- function(x, data = x$data, newdata = NULL, ...) {
  if (is.null(newdata)) {
    res <- data
  } else {
    res <- newdata
  }

  res[[".fitted"]] <- predict(x, type = "response")
  res[[".residuals"]] <- res[[names(x$data)[1]]] - res[[".fitted"]]

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}
