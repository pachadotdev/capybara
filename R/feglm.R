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
#'  in the model.
#' @param family the link function to be used in the model. Similar to
#'  \code{\link[stats]{glm.fit}} this has to be the result of a call to a family
#'  function. Default is \code{gaussian()}. See \code{\link[stats]{family}} for
#'  details of family functions.
#' @param weights an optional string with the name of the 'prior weights'
#'  variable in \code{data}.
#' @param beta.start an optional vector of starting values for the structural
#'  parameters in the linear predictor. Default is
#'  \eqn{\boldsymbol{\beta} = \mathbf{0}}{\beta = 0}.
#' @param eta.start an optional vector of starting values for the linear
#'  predictor.
#' @param control a named list of parameters for controlling the fitting
#'  process. See \code{\link{feglm_control}} for details.
#'
#' @details If \code{\link{feglm}} does not converge this is often a sign of
#'  linear dependence between one or more regressors and a fixed effects
#'  category. In this case, you should carefully inspect your model
#'  specification.
#'
#' @return A named list of class \code{"feglm"}.
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
#' mod <- feglm(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_panel,
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
    beta.start = NULL,
    eta.start = NULL,
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
  nobs.na <- NA
  nobs.full <- NA
  model_frame_(data, formula, weights)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables and sort ----
  k.vars <- attr(terms(formula, rhs = 2L), "term.labels")
  k <- length(k.vars)

  # Generate temporary variable ----
  tmp.var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp.var, k.vars, control)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, k.vars)

  # Determine the number of dropped observations ----
  nt <- nrow(data)
  nobs <- nobs_(nobs.full, nobs.na, nt)

  # Extract model response and regressor matrix ----
  nms.sp <- NA
  p <- NA
  model_response_(data, formula)

  # Check for linear dependence in 'X' ----
  check_linear_dependence_(X, p)

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else {
    wt <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(wt)

  # Compute and check starting guesses ----
  start_guesses_(beta.start, eta.start, y, X, beta, nt, wt, p, family)

  # Get names and number of levels in each fixed effects category ----
  nms.fe <- lapply(select(data, all_of(k.vars)), levels)
  lvls.k <- vapply(nms.fe, length, integer(1))

  # Generate auxiliary list of indexes for different sub panels ----
  k.list <- get_index_list_(k.vars, data)

  # Fit generalized linear model ----
  fit <- feglm_fit_(
    beta, eta, y, X, wt, k.list, family, control
  )

  y <- NULL
  X <- NULL
  eta <- NULL

  # Add names to beta, Hessian, and MX (if provided) ----
  names(fit[["coefficients"]]) <- nms.sp
  if (control[["keep.mx"]]) {
    colnames(fit[["MX"]]) <- nms.sp
  }
  dimnames(fit[["Hessian"]]) <- list(nms.sp, nms.sp)

  # Generate result list ----
  reslist <- c(
    fit, list(
      nobs    = nobs,
      lvls.k  = lvls.k,
      nms.fe  = nms.fe,
      formula = formula,
      data    = data,
      family  = family,
      control = control
    )
  )

  # Return result list ----
  structure(reslist, class = "feglm")
}
