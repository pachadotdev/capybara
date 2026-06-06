skip_on_cran <- function () {
    env <- Sys.getenv("NOT_CRAN")
    if (identical(env, "")) {
        !interactive()
    } else {
        !isTRUE(as.logical(env))
    }
}
