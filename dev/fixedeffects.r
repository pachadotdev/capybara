load_all()

object <- fepoisson(cyl ~ mpg | am, data = mtcars)

alpha.tol <- 1.0e-08

if (is.null(object)) {
  stop("'object' has to be specified.", call. = FALSE)
} else if (isFALSE(inherits(object, "felm")) &&
  isFALSE(inherits(object, "feglm"))) {
  stop(
    "'fixed_effects' called on a non-'felm' or non-'feglm' object.",
    call. = FALSE
  )
}

# Extract required quantities from result list
beta <- object[["coefficients"]]
data <- object[["data"]]
formula <- object[["formula"]]
lvls.k <- object[["lvls.k"]]
nms.fe <- object[["nms.fe"]]
k.vars <- names(lvls.k)
k <- length(lvls.k)
eta <- object[["eta"]]

# Extract regressor matrix
X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
nms.sp <- attr(X, "dimnames")[[2L]]
attr(X, "dimnames") <- NULL

# Generate auxiliary list of indexes for different sub panels
k.list <- get_index_list_(k.vars, data)

# Recover fixed effects by alternating the solutions of normal equations
pie <- eta - solve_y_(X, beta)

fe.list <- as.list(get_alpha_(pie, k.list, alpha.tol))

for (i in seq.int(k)) {
  fe.list[[i]] <- as.vector(fe.list[[i]])
  names(fe.list[[i]]) <- nms.fe[[i]]
}
names(fe.list) <- k.vars

fe.list
