#' srr_stats
#' @srrstats {G1.0} The implementation is aligned with established methods, including those described in Stammann (2018), Fernández-Val and Weidner (2016), and others.
#' @srrstats {G2.1a} Ensures input objects are of the expected class (`feglm`).
#' @srrstats {G2.3a} Validates string arguments like `panel_structure` using `match.arg()` for predefined values.
#' @srrstats {G2.14a} Provides errors for missing or invalid inputs, such as non-`feglm` objects.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G3.1a} Supports structured panels (`classic` or `network`) for analyzing fixed effects.
#' @srrstats {RE5.0} Efficient handling of computational scaling for large panels through fixed-effect groupings.
#' @srrstats {G5.2a} Produces unique and informative error messages for all stopping conditions.
#' @srrstats {G5.4a} Includes logical checks for computational edge cases, such as unsupported models.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing imputation methods would bias the estimation (i.e., replacing all missing values with the median).
#' @noRd
NULL

#' @title Asymptotic bias correction after fitting binary choice models with a
#'  1,2,3-way error component
#'
#' @description Post-estimation routine to substantially reduce the incidental
#'  parameter bias problem. Applies the analytical bias correction derived by
#'  Fernández-Val and Weidner (2016) and Hinz, Stammann, and Wanner (2020) to
#'  obtain bias-corrected estimates of the structural parameters and is
#'  currently restricted to \code{\link[stats]{binomial}} with 1,2,3-way fixed
#'  effects.
#'
#' @param object an object of class \code{"feglm"}.
#' @param l unsigned integer indicating a bandwidth for the estimation of
#'  spectral densities proposed by Hahn and Kuersteiner (2011). The default is
#'  zero, which should be used if all regressors are assumed to be strictly
#'  exogenous with respect to the idiosyncratic error term. In the presence of
#'  weakly exogenous regressors, e.g. lagged outcome variables, we suggest to
#'  choose a bandwidth between one and four. Note that the order of factors to
#'  be partialed out is important for bandwidths larger than zero.
#' @param panel_structure a string equal to \code{"classic"} or \code{"network"}
#'  which determines the structure of the panel used. \code{"classic"} denotes
#'  panel structures where for example the same cross-sectional units are
#'  observed several times (this includes pseudo panels). \code{"network"}
#'  denotes panel structures where for example bilateral trade flows are
#'  observed for several time periods. Default is \code{"classic"}.
#' @param weights a numeric vector of observation weights. If \code{NULL}
#'  (default), unit weights are used.
#'
#' @return A named list of classes \code{"bias_corr"} and \code{"feglm"}.
#'
#' @references Czarnowske, D. and A. Stammann (2020). "Fixed Effects Binary
#'  Choice Models: Estimation and Inference with Long Panels". ArXiv e-prints.
#' @references Fernández-Val, I. and M. Weidner (2016). "Individual and time
#'  effects in nonlinear panel models with large N, T". Journal of Econometrics,
#'  192(1), 291-312.
#' @references Fernández-Val, I. and M. Weidner (2018). "Fixed effects
#'  estimation of large-t panel data models". Annual Review of Economics, 10,
#'  109-138.
#' @references Hahn, J. and G. Kuersteiner (2011). "Bias reduction for dynamic
#'  nonlinear panel models with fixed effects". Econometric Theory, 27(6),
#'  1152-1191.
#' @references Hinz, J., A. Stammann, and J. Wanner (2020). "State Dependence
#'  and Unobserved Heterogeneity in the Extensive Margin of Trade". ArXiv
#'  e-prints.
#' @references Neyman, J. and E. L. Scott (1948). "Consistent estimates based on
#'  partially consistent observations". Econometrica, 16(1), 1-32.
#'
#' @seealso \code{\link{feglm}}
#'
#' @examples
#' mtcars2 <- mtcars
#' mtcars2$mpg01 <- ifelse(mtcars2$mpg > mean(mtcars2$mpg), 1L, 0L)
#'
#' # Fit 'feglm()'
#' mod <- feglm(mpg01 ~ wt | cyl, mtcars2, family = binomial())
#'
#' # Apply analytical bias correction
#' mod_bc <- bias_corr(mod)
#' summary(mod_bc)
#'
#' @export
bias_corr <- function(
    object = NULL,
    l = 0L,
    panel_structure = c("classic", "network"),
    weights = NULL) {
  # Check validity of 'object'
  apes_bias_check_object_(object, fun = "bias_corr")

  # Check validity of 'panel_structure'
  panel_structure <- match.arg(panel_structure)

  # Extract model information
  beta_uncorr <- object[["coefficients"]]
  control <- object[["control"]]
  data <- object[["data"]]
  family <- object[["family"]]
  formula <- object[["formula"]]
  fe.levels <- object[["fe.levels"]]
  nms_sp <- names(beta_uncorr)
  nt <- object[["nobs"]][["nobs"]]
  fe_names <- names(fe.levels)
  k <- length(fe.levels)

  # Check if binary choice model
  apes_bias_check_binary_model_(family, fun = "bias_corr")

  # Check if the number of FEs is > 3
  bias_corr_check_fixed_effects_(k)

  # Check if provided object matches requested panel structure
  apes_bias_check_panel_(panel_structure, k)

  # Extract model response, regressor matrix, and weights
  y <- data[[1L]]
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  attr(x, "dimnames") <- NULL

  # Extract weights - default to unit weights if not provided
  if (is.null(weights)) {
    w <- rep(1.0, nrow(data))
  } else {
    # If weights is a character string (column name), extract from data
    if (is.character(weights) && length(weights) == 1) {
      if (!(weights %in% names(data))) {
        stop("Weight variable '", weights, "' not found in data.", call. = FALSE)
      }
      w <- data[[weights]]
    } else if (is.numeric(weights)) {
      # If weights is a numeric vector, use directly
      if (length(weights) != nrow(data)) {
        stop("Length of weights must equal number of observations.", call. = FALSE)
      }
      w <- weights
    } else {
      stop("'weights' must be NULL, a column name, or a numeric vector.", call. = FALSE)
    }
  }

  # Generate auxiliary list of indexes for different sub panels
  FEs <- get_index_list_(fe_names, data)

  # Compute derivatives and weights
  eta <- object[["eta"]]
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  v <- w * (y - mu)
  w_working <- w * mu_eta # Working weights for GLM
  z <- w * partial_mu_eta_(eta, family, 2L)
  if (family[["link"]] != "logit") {
    h <- mu_eta / family[["variance"]](mu)
    v <- h * v
    w_working <- h * w_working
    z <- h * z
    rm(h)
  }

  # Center regressor matrix (if required) - use working weights
  if (control[["keep_dmx"]]) {
    x <- object[["X_dm"]]
  } else {
    x <- demean_variables_(
      x, w_working, FEs, control[["demean_tol"]],
      control[["iter_max"]], control[["iter_interrupt"]],
      control[["iter_ssr"]], "gaussian"
    )
  }

  # Compute bias terms for requested bias correction
  if (panel_structure == "classic") {
    # Compute \hat{B} and \hat{D}
    b <- as.vector(group_sums_(x * z, w_working, FEs[[1L]])) / 2.0 / nt
    if (k > 1L) {
      b <- b + as.vector(group_sums_(x * z, w_working, FEs[[2L]])) / 2.0 / nt
    }

    # Compute spectral density part of \hat{B}
    if (l > 0L) {
      b <- (b + group_sums_spectral_(x * w_working, v, w_working, l, FEs[[1L]])) / nt
    }
  } else {
    # Compute \hat{D}_{1}, \hat{D}_{2}, and \hat{B}
    b <- group_sums_(x * z, w_working, FEs[[1L]]) / (2.0 * nt)
    b <- (b + group_sums_(x * z, w_working, FEs[[2L]])) / (2.0 * nt)
    if (k > 2L) {
      b <- (b + group_sums_(x * z, w_working, FEs[[3L]])) / (2.0 * nt)
    }

    # Compute spectral density part of \hat{B}
    if (k > 2L && l > 0L) {
      b <- (b + group_sums_spectral_(x * w_working, v, w_working, l, FEs[[3L]])) / nt
    }
  }

  # Compute bias-corrected structural parameters
  beta <- beta_uncorr - solve(object[["hessian"]] / nt, b)
  names(beta) <- nms_sp

  # Update \eta and first- and second-order derivatives
  eta <- feglm_offset_(object, x %*% beta)
  mu <- family[["linkinv"]](eta)
  mu_eta <- family[["mu.eta"]](eta)
  v <- w * (y - mu) # Use original weights
  w_working <- w * mu_eta # Recompute working weights
  if (family[["link"]] != "logit") {
    h <- mu_eta / family[["variance"]](mu)
    v <- h * v
    w_working <- h * w_working
    rm(h)
  }

  # Update centered regressor matrix
  x <- demean_variables_(
    x, w_working, FEs, control[["demean_tol"]],
    control[["iter_max"]], control[["iter_interrupt"]],
    control[["iter_ssr"]], "gaussian"
  )
  colnames(x) <- nms_sp

  # Update hessian
  h <- crossprod(x * sqrt(w_working))
  dimnames(h) <- list(nms_sp, nms_sp)

  # Update result list
  object[["coefficients"]] <- beta
  object[["eta"]] <- eta
  if (control[["keep_dmx"]]) object[["X_dm"]] <- x
  object[["hessian"]] <- h
  object[["coefficients_uncorr"]] <- beta_uncorr
  object[["bias_term"]] <- b
  object[["panel_structure"]] <- panel_structure
  object[["bandwidth"]] <- l

  # Add additional class to result list
  attr(object, "class") <- c("feglm", "bias_corr")

  # Return updated list
  object
}

bias_corr_check_fixed_effects_ <- function(fe.levels) {
  if (length(fe.levels) > 3) {
    stop(
      "bias_corr() only supports models with up to three-way fixed effects.",
      call. = FALSE
    )
  }
}
