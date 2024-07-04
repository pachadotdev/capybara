# not exported, it is not of interest to the user and it is just for the
# pseudo-R2 computation in the summary method
kendall_cor <- function(x, y) {
  arr <- cbind(x, y)
  storage.mode(arr) <- "double"
  arr <- arr[complete.cases(arr), ]
  kendall_cor_(arr)
}
