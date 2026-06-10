skip_on_cran <- function () {
    env <- Sys.getenv("NOT_CRAN")
    if (identical(env, "")) {
        !interactive()
    } else {
        !isTRUE(as.logical(env))
    }
}

# Helper function for MAPE calculation
mape <- function(y, yhat) {
    mean(abs(y - yhat) / abs(y))
}
