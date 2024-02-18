#' @importFrom generics augment
#' @export
generics::augment

#' @export
#' @noRd
augment.feglm <- function(x, data = x$data, newdata = NULL, ...) {
  if (is.null(newdata)) {
    aug <- as_tibble(data)
  } else {
    aug <- as_tibble(newdata)
  }

  aug[[".fitted"]] <- predict(x, type = "response")
  aug[[".residuals"]] <- aug[[names(x$data)[1]]] - aug[[".fitted"]]
}
