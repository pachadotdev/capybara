# Helper function for MAPE calculation
mape <- function(y, yhat) {
  mean(abs(y - yhat) / abs(y))
}
