#' @title GLM fitting with high-dimensional k-way fixed effects
#'
#' @description \code{\link{feglm}} can be used to fit generalized linear models
#'  with many high-dimensional fixed effects. The estimation procedure is based
#'  on unconditional maximum likelihood and can be interpreted as a
#'  \dQuote{weighted demeaning} approach.
#'
#' \strong{Remark:} The term fixed effect is used in econometrician's sense of
#'  having intercepts for each level in each category.
#'
#' @param formula an object of class \code{"formula"}: a symbolic description of
#'  the model to be fitted. \code{formula} must be of type \code{y ~ x | k},
#'  where the second part of the formula refers to factors to be concentrated
#'  out. It is also possible to pass clustering variables to \code{\link{feglm}}
#'  as \code{y ~ x | k | c}.
#' @param data an object of class \code{"data.frame"} containing the variables
#'  in the model. The expected input is a dataset with the variables specified
#'  in \code{formula} and a number of rows at least equal to the number of
#'  variables in the model.
#' @param family the link function to be used in the model. Similar to
#'  \code{\link[stats]{glm.fit}} this has to be the result of a call to a family
#'  function. Default is \code{gaussian()}. See \code{\link[stats]{family}} for
#'  details of family functions.
#' @param weights an optional string with the name of the 'prior weights'
#'  variable in \code{data}.
#' @param beta_start an optional vector of starting values for the structural
#'  parameters in the linear predictor. Default is
#'  \eqn{\boldsymbol{\beta} = \mathbf{0}}{\beta = 0}.
#' @param eta_start an optional vector of starting values for the linear
#'  predictor.
#' @param control a named list of parameters for controlling the fitting
#'  process. See \code{\link{feglm_control}} for details.
#'
#' @details If \code{\link{feglm}} does not converge this is often a sign of
#'  linear dependence between one or more regressors and a fixed effects
#'  category. In this case, you should carefully inspect your model
#'  specification.
#'
#' @return A named list of class \code{"feglm"}. The list contains the following
#'  fifteen elements:
#'  \item{coefficients}{a named vector of the estimated coefficients}
#'  \item{eta}{a vector of the linear predictor}
#'  \item{weights}{a vector of the weights used in the estimation}
#'  \item{hessian}{a matrix with the numerical second derivatives}
#'  \item{deviance}{the deviance of the model}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{conv}{a logical indicating whether the model converged}
#'  \item{iter}{the number of iterations needed to converge}
#'  \item{nobs}{a named vector with the number of observations used in the
#'   estimation indicating the dropped and perfectly predicted observations}
#'  \item{lvls_k}{a named vector with the number of levels in each fixed
#'   effects}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing
#'   observations}
#'  \item{family}{the family used in the model}
#'  \item{control}{the control list used in the model}
#'
#' @references Gaure, S. (2013). "OLS with Multiple High Dimensional Category
#'  Variables". Computational Statistics and Data Analysis, 66.
#' @references Marschner, I. (2011). "glm2: Fitting generalized linear models
#'  with convergence problems". The R Journal, 3(2).
#' @references Stammann, A., F. Heiss, and D. McFadden (2016). "Estimating Fixed
#'  Effects Logit Models with Large Panel Data". Working paper.
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized
#'  Linear Models with High-Dimensional k-Way Fixed Effects". ArXiv e-prints.
#'
#' @examples
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- feglm(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006,
#'   family = poisson(link = "log")
#' )
#'
#' summary(mod)
#'
#' mod <- feglm(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year | pair,
#'   trade_panel,
#'   family = poisson(link = "log")
#' )
#'
#' summary(mod, type = "clustered")
#'
#' @export
feglm <- function(
    formula = NULL,
    data = NULL,
    family = gaussian(),
    weights = NULL,
    beta_start = NULL,
    eta_start = NULL,
    control = NULL) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of family ----
  check_family_(family)

  # Check validity of control + Extract control list ----
  control <- check_control_(control)

  # Update formula and do further validity check ----
  formula <- update_formula_(formula)

  # Generate model.frame
  lhs <- NA # just to avoid global variable warning
  nobs_na <- NA
  nobs_full <- NA
  model_frame_(data, formula, weights)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables and sort ----
  k_vars <- attr(terms(formula, rhs = 2L), "term.labels")

  # Generate temporary variable ----
  tmp_var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp_var, k_vars, control)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, k_vars)

  # Determine the number of dropped observations ----
  nt <- nrow(data)
  nobs <- nobs_(nobs_full, nobs_na, nt)

  # Extract model response and regressor matrix ----
  nms_sp <- NA
  p <- NA
  model_response_(data, formula)

  # Check for linear dependence in 'x' ----
  check_linear_dependence_(x, p)

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else {
    wt <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(wt)

  # Compute and check starting guesses ----
  start_guesses_(beta_start, eta_start, y, x, beta, nt, wt, p, family)

  # Get names and number of levels in each fixed effects category ----
  nms_fe <- lapply(select(data, all_of(k_vars)), levels)
  lvls_k <- vapply(nms_fe, length, integer(1))

  # Generate auxiliary list of indexes for different sub panels ----
  k_list <- get_index_list_(k_vars, data)

  # Fit generalized linear model ----
  if (is.integer(y)) {
    y <- as.numeric(y)
  }
  fit <- feglm_fit_(
    beta, eta, y, x, wt, 0.0, family[["family"]], control, k_list
  )

  y <- NULL
  x <- NULL
  eta <- NULL

  # Add names to beta, hessian, and mx (if provided) ----
  names(fit[["coefficients"]]) <- nms_sp
  if (control[["keep_mx"]]) {
    colnames(fit[["mx"]]) <- nms_sp
  }
  dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)

  # Add to fit list ----
  fit[["nobs"]] <- nobs
  fit[["lvls_k"]] <- lvls_k
  fit[["nms_fe"]] <- nms_fe
  fit[["formula"]] <- formula
  fit[["data"]] <- data
  fit[["family"]] <- family
  fit[["control"]] <- control

  # Return result list ----
  structure(fit, class = "feglm")
}
