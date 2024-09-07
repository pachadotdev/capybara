#' @title Recover the estimates of the fixed effects after fitting (G)LMs
#'
#' @description The system might not have a unique solution since we do not take
#'  collinearity into account. If the solution is not unique, an estimable
#'  function has to be applied to our solution to get meaningful estimates of
#'  the fixed effects.
#'
#' @param object an object of class \code{"feglm"}.
#' @param alpha_tol tolerance level for the stopping condition. The algorithm is
#'  stopped at iteration \eqn{i} if \eqn{||\boldsymbol{\alpha}_{i} -
#'  \boldsymbol{\alpha}_{i - 1}||_{2} < tol ||\boldsymbol{\alpha}_{i - 1}||
#'  {2}}{||\Delta \alpha|| < tol ||\alpha_old||}. Default is \code{1.0e-08}.
#'
#' @return A named list containing named vectors of estimated fixed effects.
#'
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized
#'  Linear Models with High-Dimensional k-way Fixed Effects". ArXiv e-prints.
#' @references Gaure, S. (n. d.). "Multicollinearity, identification, and
#'  estimable functions". Unpublished.
#'
#' @seealso \code{\link{felm}}, \code{\link{feglm}}
#'
#' @examples
#' # check the feglm examples for the details about clustered standard errors
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 1000), ]
#'
#' mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
#' fixed_effects(mod)
#'
#' @export
fixed_effects <- function(object = NULL, alpha_tol = 1.0e-08) {
  # Check validity of 'object'
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
  lvls_k <- object[["lvls_k"]]
  nms_fe <- object[["nms_fe"]]
  k_vars <- names(lvls_k)
  k <- length(lvls_k)
  eta <- object[["eta"]]

  # Extract regressor matrix
  X <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  nms_sp <- attr(X, "dimnames")[[2L]]
  attr(X, "dimnames") <- NULL

  # Generate auxiliary list of indexes for different sub panels
  k_list <- get_index_list_(k_vars, data)

  # Recover fixed effects by alternating the solutions of normal equations
  if (inherits(object, "feglm")) {
    pie <- eta - X %*% beta
  } else {
    pie <- fitted.values(object) - X %*% beta
  }
  fe_list <- as.list(get_alpha_(pie, k_list, alpha_tol))

  # Assign names to the different fixed effects categories
  for (i in seq.int(k)) {
    colnames(fe_list[[i]]) <- k_vars[i]
    rownames(fe_list[[i]]) <- nms_fe[[i]]
  }
  names(fe_list) <- k_vars

  # Return list of estimated fixed effects
  fe_list
}
