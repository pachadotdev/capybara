#' @title Recover the estimates of the fixed effects after fitting GLMs
#' @description The system might not have a unique solution since we do not take
#'  collinearity into account. If the solution is not unique, an estimable
#'  function has to be applied to our solution to get meaningful estimates of
#'  the fixed effects.
#' @param object an object of class \code{"feglm"}.
#' @param alpha.tol tolerance level for the stopping condition. The algorithm is
#'  stopped at iteration \eqn{i} if \eqn{||\boldsymbol{\alpha}_{i} -
#'  \boldsymbol{\alpha}_{i - 1}||_{2} < tol ||\boldsymbol{\alpha}_{i - 1}||
#'  {2}}{||\Delta \alpha|| < tol ||\alpha_old||}. Default is \code{1.0e-08}.
#' @return A named list containing named vectors of estimated fixed effects.
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized
#'  Linear Models with High-Dimensional k-way Fixed Effects". ArXiv e-prints.
#' @references Gaure, S. (n. d.). "Multicollinearity, identification, and
#'  estimable functions". Unpublished.
#' @seealso \code{\link{felm}}, \code{\link{feglm}}
#' @export
fixed_effects <- function(object = NULL, alpha.tol = 1.0e-08) {
  # Check validity of 'object'
  if (is.null(object)) {
    stop("'object' has to be specified.", call. = FALSE)
  } else if (isFALSE(inherits(object, "felm")) && isFALSE(inherits(object, "feglm"))) {
    stop("'fixed_effects' called on a non-'felm' or non-'feglm' object.", call. = FALSE)
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

  # Recover fixed effects by alternating between the solutions of normal equations
  pie <- eta - as.vector(X %*% beta)
  fe.list <- as.list(get_alpha_(pie, k.list, alpha.tol))

  # Assign names to the different fixed effects categories
  for (i in seq.int(k)) {
    fe.list[[i]] <- as.vector(fe.list[[i]])
    names(fe.list[[i]]) <- nms.fe[[i]]
  }
  names(fe.list) <- k.vars

  # Return list of estimated fixed effects
  fe.list
}
